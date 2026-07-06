#include "TestFramework.h"

#include <private/GigaLearnCPP/SAC/SACLearner.h>
#include <private/GigaLearnCPP/SAC/ReplayBuffer.h>
#include <private/GigaLearnCPP/Util/PolicyInference.h>

using namespace GGL;

static SACLearnerConfig MakeTestSACConfig() {
	SACLearnerConfig config = {};
	config.policy.layerSizes = { 32 };
	config.qNet.layerSizes = { 32 };
	config.sharedHead = {}; // No shared head
	config.batchSize = 256;
	config.replayBufferSize = 8192;
	config.gradientStepsPerItr = 50;
	return config;
}

// Fills a replay buffer with a contextual-free bandit problem:
//	random states, uniform random actions, reward 1 only for REWARDED_ACTION, episodes end every step
static constexpr int BANDIT_OBS_SIZE = 4, BANDIT_NUM_ACTIONS = 3, BANDIT_REWARDED_ACTION = 2;
static void FillBanditReplay(ReplayBuffer& buffer, int64_t amount) {
	ReplayTransitions t = {};
	t.states = torch::rand({ amount, BANDIT_OBS_SIZE });
	t.actions = torch::randint(BANDIT_NUM_ACTIONS, { amount }, torch::kInt64);
	t.rewards = (t.actions == BANDIT_REWARDED_ACTION).to(torch::kFloat32);
	t.nextStates = torch::rand({ amount, BANDIT_OBS_SIZE });
	t.actionMasks = torch::ones({ amount, BANDIT_NUM_ACTIONS }, torch::kUInt8);
	t.nextActionMasks = torch::ones({ amount, BANDIT_NUM_ACTIONS }, torch::kUInt8);
	t.dones = torch::ones({ amount }, torch::kFloat32); // Pure bandit: every step ends the episode
	buffer.Append(t);
}

TEST_CASE(SACLearner_QTargetsMatchReference) {
	RG_NO_GRAD;

	constexpr float GAMMA = 0.9f, ALPHA = 0.5f;

	auto rewards = torch::tensor({ 1.f, 2.f });
	auto dones = torch::tensor({ 0.f, 1.f });
	auto nextProbs = torch::tensor({ { 0.5f, 0.3f, 0.2f }, { 0.6f, 0.2f, 0.2f } });
	auto nextLogProbs = nextProbs.log();
	auto minNextQ = torch::tensor({ { 1.f, 2.f, 3.f }, { 4.f, 5.f, 6.f } });
	auto masks = torch::ones({ 2, 3 }, torch::kUInt8);
	auto entCoef = torch::tensor(ALPHA);

	auto targets = SACLearner::ComputeQTargets(rewards, dones, nextProbs, nextLogProbs, minNextQ, masks, entCoef, GAMMA);
	CHECK_EQ(targets.dim(), 1);
	CHECK_EQ(targets.size(0), 2);

	// Hand-computed: y = r + gamma * (1 - done) * sum_a[ pi_a * (minQ_a - alpha * log pi_a) ]
	double v0 = 0;
	double probs0[3] = { 0.5, 0.3, 0.2 }, qs0[3] = { 1, 2, 3 };
	for (int a = 0; a < 3; a++)
		v0 += probs0[a] * (qs0[a] - ALPHA * std::log(probs0[a]));
	double expected0 = 1 + GAMMA * v0;

	// The second transition is done: no bootstrap at all
	double expected1 = 2;

	CHECK_NEAR(targets[0].item<float>(), expected0, 1e-5);
	CHECK_NEAR(targets[1].item<float>(), expected1, 1e-5);
}

TEST_CASE(SACLearner_QTargetsIgnoreMaskedActions) {
	RG_NO_GRAD;

	auto rewards = torch::tensor({ 0.f });
	auto dones = torch::tensor({ 0.f });

	// The masked action has a huge Q value and (near) zero probability;
	//	it must contribute exactly nothing to the target
	auto nextProbs = torch::tensor({ { 0.5f, 0.5f, 1e-11f } });
	auto nextLogProbs = nextProbs.log();
	auto minNextQ = torch::tensor({ { 1.f, 2.f, 1e9f } });
	auto masks = torch::tensor({ { 1, 1, 0 } }).to(torch::kUInt8);
	auto entCoef = torch::tensor(0.25f);

	auto targets = SACLearner::ComputeQTargets(rewards, dones, nextProbs, nextLogProbs, minNextQ, masks, entCoef, 1.0f);

	double expected =
		0.5 * (1 - 0.25 * std::log(0.5)) +
		0.5 * (2 - 0.25 * std::log(0.5));
	CHECK_NEAR(targets[0].item<float>(), expected, 1e-4);
}

TEST_CASE(SACLearner_BuildTransitionsLayout) {
	RG_NO_GRAD;

	// Flattened complete episodes, exactly like the collection loop produces:
	//	Episode A: rows 0-2 (ends TRUNCATED), episode B: rows 3-4 (ends NORMAL), episode C: row 5 (ends TRUNCATED)
	auto states = torch::arange(0, 12, torch::kFloat32).reshape({ 6, 2 });
	auto masks = torch::ones({ 6, 2 }, torch::kUInt8);
	masks[1][0] = 0; // Distinguishable mask row (row 1)
	auto actions = torch::zeros({ 6 }, torch::kInt64);
	auto rewards = torch::arange(0, 6, torch::kFloat32);

	auto terminals = torch::tensor(
		{ (int8_t)RLGC::TerminalType::NOT_TERMINAL, (int8_t)RLGC::TerminalType::NOT_TERMINAL, (int8_t)RLGC::TerminalType::TRUNCATED,
		  (int8_t)RLGC::TerminalType::NOT_TERMINAL, (int8_t)RLGC::TerminalType::NORMAL,
		  (int8_t)RLGC::TerminalType::TRUNCATED },
		torch::kInt8
	);

	// Truncation bootstrap states, in episode-finish order (row 2's first, then row 5's)
	auto truncStates = torch::tensor({ { 100.f, 101.f }, { 200.f, 201.f } });
	auto truncMasks = torch::zeros({ 2, 2 }, torch::kUInt8);
	truncMasks[1][1] = 1;

	auto t = SACLearner::BuildTransitions(states, masks, actions, rewards, terminals, truncStates, truncMasks);

	// Mid-episode rows: next state is simply the following row
	CHECK_TRUE(torch::equal(t.nextStates[0], states[1]));
	CHECK_TRUE(torch::equal(t.nextStates[1], states[2]));
	CHECK_TRUE(torch::equal(t.nextStates[3], states[4]));
	CHECK_TRUE(torch::equal(t.nextActionMasks[0], masks[1]));

	// Truncated rows bootstrap from their saved states/masks, consumed in order
	CHECK_TRUE(torch::equal(t.nextStates[2], truncStates[0]));
	CHECK_TRUE(torch::equal(t.nextStates[5], truncStates[1]));
	CHECK_TRUE(torch::equal(t.nextActionMasks[2], truncMasks[0]));
	CHECK_TRUE(torch::equal(t.nextActionMasks[5], truncMasks[1]));

	// Only the NORMAL episode end sets done (truncations bootstrap instead)
	auto dones = TENSOR_TO_VEC<float>(t.dones);
	std::vector<float> expectedDones = { 0, 0, 0, 0, 1, 0 };
	CHECK_TRUE(dones == expectedDones);

	// Everything else passes through unchanged
	CHECK_TRUE(torch::equal(t.states, states));
	CHECK_TRUE(torch::equal(t.actions, actions));
	CHECK_TRUE(torch::equal(t.rewards, rewards));

	// Mismatched truncation-state count must be rejected
	CHECK_THROWS(SACLearner::BuildTransitions(states, masks, actions, rewards, terminals, truncStates.slice(0, 0, 1), truncMasks.slice(0, 0, 1)));

	// Experience not ending in a terminal must be rejected
	auto badTerminals = terminals.clone();
	badTerminals[5] = (int8_t)RLGC::TerminalType::NOT_TERMINAL;
	CHECK_THROWS(SACLearner::BuildTransitions(states, masks, actions, rewards, badTerminals, truncStates.slice(0, 0, 1), truncMasks.slice(0, 0, 1)));
}

TEST_CASE(SACLearner_LearnRejectsDeterministic) {
	torch::manual_seed(123);

	SACLearnerConfig config = MakeTestSACConfig();
	config.deterministic = true;

	auto learner = SACLearner(BANDIT_OBS_SIZE, BANDIT_NUM_ACTIONS, config, torch::kCPU);

	ReplayBuffer buffer = ReplayBuffer(1024, BANDIT_OBS_SIZE, BANDIT_NUM_ACTIONS);
	FillBanditReplay(buffer, 1024);

	Report report = {};
	CHECK_THROWS(learner.Learn(buffer, report));
}

TEST_CASE(SACLearner_TargetNetsStartAsCopies) {
	RG_NO_GRAD;
	torch::manual_seed(123);

	auto learner = SACLearner(BANDIT_OBS_SIZE, BANDIT_NUM_ACTIONS, MakeTestSACConfig(), torch::kCPU);

	auto obs = torch::rand({ 8, BANDIT_OBS_SIZE });
	CHECK_TRUE(torch::allclose(learner.models["q1"]->Forward(obs, false), learner.targetModels["q1_target"]->Forward(obs, false)));
	CHECK_TRUE(torch::allclose(learner.models["q2"]->Forward(obs, false), learner.targetModels["q2_target"]->Forward(obs, false)));
}

TEST_CASE(SACLearner_PolyakUpdateMath) {
	RG_NO_GRAD;
	torch::manual_seed(123);

	auto learner = SACLearner(BANDIT_OBS_SIZE, BANDIT_NUM_ACTIONS, MakeTestSACConfig(), torch::kCPU);

	// Shift the live q1 away from its target
	for (auto& param : learner.models["q1"]->parameters())
		param += 1.0f;

	auto liveParams = learner.models["q1"]->parameters();
	auto targetParams = learner.targetModels["q1_target"]->parameters();
	auto targetBefore = targetParams[0].clone();

	constexpr float TAU = 0.25f;
	learner.UpdateTargets(TAU);

	// target = tau * live + (1 - tau) * target
	auto expected = liveParams[0] * TAU + targetBefore * (1 - TAU);
	CHECK_TRUE(torch::allclose(targetParams[0], expected, 1e-5, 1e-6));

	// Live params must be untouched
	CHECK_TRUE(torch::allclose(liveParams[0], targetBefore + 1.0f, 1e-5, 1e-6));
}

TEST_CASE(SACLearner_LearnsBanditProblem) {
	torch::manual_seed(123);

	SACLearnerConfig config = MakeTestSACConfig();
	config.autoEntCoef = false;
	config.entCoef = 0.01f; // Tiny entropy bonus so the policy can commit to the best action
	config.policyLR = 3e-3f;
	config.qLR = 3e-3f;
	config.tau = 0.05f;
	config.gradientStepsPerItr = 100;

	auto learner = SACLearner(BANDIT_OBS_SIZE, BANDIT_NUM_ACTIONS, config, torch::kCPU);

	ReplayBuffer buffer = ReplayBuffer(8192, BANDIT_OBS_SIZE, BANDIT_NUM_ACTIONS);
	FillBanditReplay(buffer, 8192);

	Report report = {};
	for (int i = 0; i < 3; i++) {
		report = {};
		learner.Learn(buffer, report);
	}

	// All learn metrics must be present and finite
	for (const char* key : { "Policy Entropy", "Policy Loss", "SAC/Q1 Loss", "SAC/Q2 Loss", "SAC/Avg Q", "SAC/Entropy Coef" }) {
		CHECK_TRUE(report.Has(key));
		CHECK_TRUE(std::isfinite(report[key]));
	}
	CHECK_EQ((int64_t)report["SAC/Total Gradient Steps"], (int64_t)300);

	RG_NO_GRAD;

	// The policy must strongly prefer the rewarded action
	auto obs = torch::rand({ 64, BANDIT_OBS_SIZE });
	auto masks = torch::ones({ 64, BANDIT_NUM_ACTIONS }, torch::kUInt8);
	auto probs = PolicyInference::InferProbs(learner.models, obs, masks, 1, false);

	float rewardedProb = probs.index({ torch::indexing::Slice(), BANDIT_REWARDED_ACTION }).mean().item<float>();
	CHECK_TRUE(rewardedProb > 0.6f);

	// The Q nets must approximate the true action values (1 for rewarded, 0 otherwise)
	auto q1 = learner.models["q1"]->Forward(obs, false);
	float rewardedQ = q1.index({ torch::indexing::Slice(), BANDIT_REWARDED_ACTION }).mean().item<float>();
	float unrewardedQ = q1.index({ torch::indexing::Slice(), 0 }).mean().item<float>();
	CHECK_NEAR(rewardedQ, 1, 0.25);
	CHECK_NEAR(unrewardedQ, 0, 0.25);
}

TEST_CASE(SACLearner_AutoEntCoefMovesTowardTarget) {
	torch::manual_seed(123);

	SACLearnerConfig config = MakeTestSACConfig();
	config.autoEntCoef = true;
	config.entCoef = 0.2f;
	config.entCoefLR = 1e-2f;
	// A fresh policy is near uniform (entropy ~ log(numActions)), so with a target of half that,
	//	the auto-tuner must push alpha DOWN
	config.targetEntropyScale = 0.5f;

	auto learner = SACLearner(BANDIT_OBS_SIZE, BANDIT_NUM_ACTIONS, config, torch::kCPU);
	float alphaBefore = learner.GetEntCoef();
	CHECK_NEAR(alphaBefore, 0.2f, 1e-5);

	ReplayBuffer buffer = ReplayBuffer(8192, BANDIT_OBS_SIZE, BANDIT_NUM_ACTIONS);
	FillBanditReplay(buffer, 4096);

	Report report = {};
	learner.Learn(buffer, report);

	float alphaAfter = learner.GetEntCoef();
	CHECK_TRUE(alphaAfter < alphaBefore * 0.95f);
	CHECK_TRUE(alphaAfter > 0);
	CHECK_TRUE(report.Has("SAC/Entropy Coef Loss"));
}

TEST_CASE(SACLearner_SaveLoadRoundtrip) {
	torch::manual_seed(123);

	namespace fs = std::filesystem;
	auto saveFolder = fs::temp_directory_path() / "ggl_test_sac_save";
	fs::remove_all(saveFolder);
	fs::create_directories(saveFolder);

	SACLearnerConfig config = MakeTestSACConfig();
	config.gradientStepsPerItr = 10;

	auto learner = SACLearner(BANDIT_OBS_SIZE, BANDIT_NUM_ACTIONS, config, torch::kCPU);

	{ // Run some learning so the models, targets, and alpha all move off their initial values
		ReplayBuffer buffer = ReplayBuffer(1024, BANDIT_OBS_SIZE, BANDIT_NUM_ACTIONS);
		FillBanditReplay(buffer, 1024);
		Report report = {};
		learner.Learn(buffer, report);
	}

	learner.SaveTo(saveFolder);

	// A fresh learner (different random weights), loading the checkpoint, must match exactly
	auto loaded = SACLearner(BANDIT_OBS_SIZE, BANDIT_NUM_ACTIONS, config, torch::kCPU);
	loaded.LoadFrom(saveFolder);

	RG_NO_GRAD;
	auto obs = torch::rand({ 16, BANDIT_OBS_SIZE });
	auto masks = torch::ones({ 16, BANDIT_NUM_ACTIONS }, torch::kUInt8);

	CHECK_TRUE(torch::allclose(
		PolicyInference::InferProbs(learner.models, obs, masks, 1, false),
		PolicyInference::InferProbs(loaded.models, obs, masks, 1, false)
	));
	CHECK_TRUE(torch::allclose(
		learner.models["q1"]->Forward(obs, false),
		loaded.models["q1"]->Forward(obs, false)
	));
	CHECK_TRUE(torch::allclose(
		learner.targetModels["q2_target"]->Forward(obs, false),
		loaded.targetModels["q2_target"]->Forward(obs, false)
	));
	CHECK_NEAR(loaded.GetEntCoef(), learner.GetEntCoef(), 1e-6);
	CHECK_EQ(loaded.totalGradSteps, learner.totalGradSteps);

	// Target nets must stay frozen after loading (only Polyak averaging writes them)
	for (auto& param : loaded.targetModels["q1_target"]->parameters())
		CHECK_FALSE(param.requires_grad());

	fs::remove_all(saveFolder);
}

TEST_CASE(SACLearner_RejectsBadConfigs) {
	auto fnMakeWith = [](std::function<void(SACLearnerConfig&)> mutate) {
		SACLearnerConfig config = MakeTestSACConfig();
		mutate(config);
		return SACLearner(BANDIT_OBS_SIZE, BANDIT_NUM_ACTIONS, config, torch::kCPU);
	};

	CHECK_THROWS(fnMakeWith([](SACLearnerConfig& c) { c.tau = 0; }));
	CHECK_THROWS(fnMakeWith([](SACLearnerConfig& c) { c.tau = 1.5f; }));
	CHECK_THROWS(fnMakeWith([](SACLearnerConfig& c) { c.gamma = 1; }));
	CHECK_THROWS(fnMakeWith([](SACLearnerConfig& c) { c.replayBufferSize = c.batchSize - 1; }));
	CHECK_THROWS(fnMakeWith([](SACLearnerConfig& c) { c.entCoef = 0; }));
	CHECK_THROWS(fnMakeWith([](SACLearnerConfig& c) { c.gradientStepsPerItr = 0; }));
	CHECK_THROWS(fnMakeWith([](SACLearnerConfig& c) { c.targetEntropyScale = 2; }));
	CHECK_THROWS(fnMakeWith([](SACLearnerConfig& c) { c.batchSize = 0; }));
}
