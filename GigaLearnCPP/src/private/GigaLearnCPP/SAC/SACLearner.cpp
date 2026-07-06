#include "SACLearner.h"

#include "../Util/PolicyInference.h"
#include "../Util/MetricAccum.h"

#include <torch/nn/utils/clip_grad.h>
#include <torch/nn/modules/loss.h>
#include <torch/csrc/api/include/torch/serialize.h>

using namespace torch;

// SAC state that isn't part of any Model: the entropy temperature and the gradient step counter
// (named entries in one archive, so more can be added without changing the file layout)
constexpr const char* EXTRA_STATE_FILE_NAME = "SAC_STATE.lt";
constexpr const char* ENT_COEF_OPTIM_FILE_NAME = "SAC_ENT_COEF_OPTIM.lt";

GGL::SACLearner::SACLearner(int obsSize, int numActions, SACLearnerConfig _config, Device _device)
	: AlgoLearner(_device), config(_config), numActions(numActions) {

	{ // Validate config
		if (config.tsPerItr <= 0 || config.batchSize <= 0)
			RG_ERR_CLOSE("SACLearner: config.tsPerItr and config.batchSize must be positive");

		if (config.replayBufferSize < config.batchSize)
			RG_ERR_CLOSE(
				"SACLearner: config.replayBufferSize (" << config.replayBufferSize << ") " <<
				"must be at least config.batchSize (" << config.batchSize << ")"
			);

		if (config.gradientStepsPerItr <= 0)
			RG_ERR_CLOSE("SACLearner: config.gradientStepsPerItr must be positive");

		if (config.tau <= 0 || config.tau > 1)
			RG_ERR_CLOSE("SACLearner: config.tau (" << config.tau << ") must be within (0, 1]");

		if (config.gamma < 0 || config.gamma >= 1)
			RG_ERR_CLOSE("SACLearner: config.gamma (" << config.gamma << ") must be within [0, 1)");

		if (config.targetUpdateInterval <= 0)
			RG_ERR_CLOSE("SACLearner: config.targetUpdateInterval must be positive");

		if (config.entCoef <= 0)
			RG_ERR_CLOSE(
				"SACLearner: config.entCoef (" << config.entCoef << ") must be positive " <<
				"(use a tiny value like 1e-8 for an effectively-zero entropy bonus)"
			);

		if (config.autoEntCoef) {
			if (config.entCoefLR <= 0)
				RG_ERR_CLOSE("SACLearner: config.entCoefLR must be positive when config.autoEntCoef is enabled");

			if (config.targetEntropyScale < 0 || config.targetEntropyScale > 1)
				RG_ERR_CLOSE(
					"SACLearner: config.targetEntropyScale (" << config.targetEntropyScale << ") must be within [0, 1] " <<
					"(it is a fraction of the maximum possible entropy)"
				);
		}
	}

	PolicyInference::MakePolicyModels(obsSize, numActions, config.sharedHead, config.policy, device, models);

	{ // Make the twin Q nets and their target copies
		ModelConfig qConfig = config.qNet;
		qConfig.numInputs = obsSize;
		qConfig.numOutputs = numActions;

		models.Add(new Model("q1", qConfig, device));
		models.Add(new Model("q2", qConfig, device));

		auto fnMakeTarget = [&](const char* liveName, const char* targetName) {
			Model* target = new Model(targetName, qConfig, device);

			RG_NO_GRAD;
			auto fromParams = models[liveName]->parameters();
			auto toParams = target->parameters();
			for (int i = 0; i < fromParams.size(); i++) {
				toParams[i].copy_(fromParams[i], true);
				// Target nets are only ever written by Polyak averaging
				toParams[i].set_requires_grad(false);
			}

			targetModels.Add(target);
		};
		fnMakeTarget("q1", "q1_target");
		fnMakeTarget("q2", "q2_target");
	}

	{ // Make the entropy temperature (alpha)
		targetEntropy = config.targetEntropyScale * logf(numActions);

		logEntCoef = torch::tensor(
			logf(config.entCoef),
			torch::TensorOptions().dtype(torch::kFloat32).device(device).requires_grad(config.autoEntCoef)
		);

		if (config.autoEntCoef)
			entCoefOptim = new torch::optim::Adam({ logEntCoef }, config.entCoefLR);
	}

	SetLearningRates(config.policyLR, config.qLR);

	// Print param counts (target nets are copies, so they aren't counted)
	RG_LOG("Model parameter counts:");
	uint64_t total = 0;
	for (auto model : this->models) {
		uint64_t count = model->GetParamCount();
		RG_LOG("\t\"" << model->modelName << "\": " << Utils::NumToStr(count));
		total += count;
	}
	RG_LOG("\t[Total]: " << Utils::NumToStr(total));
}

void GGL::SACLearner::InferActions(torch::Tensor obs, torch::Tensor actionMasks, torch::Tensor* outActions, torch::Tensor* outLogProbs, ModelSet* models) {
	PolicyInference::InferActions(models ? *models : this->models, obs, actionMasks, config.deterministic, config.policyTemperature, config.useHalfPrecision, outActions, outLogProbs);
}

torch::Tensor GGL::SACLearner::ComputeQTargets(
	torch::Tensor rewards, torch::Tensor dones,
	torch::Tensor nextProbs, torch::Tensor nextLogProbs,
	torch::Tensor minNextTargetQ, torch::Tensor nextActionMasks,
	torch::Tensor entCoef, float gamma) {

	// Invalid actions are multiplied out exactly
	//	(their probabilities are already near zero, but their Q values could be anything)
	auto nextMasksF = nextActionMasks.to(torch::kFloat32);

	// Soft state value of the next state, exact over the discrete action space
	auto nextV = (nextMasksF * nextProbs * (minNextTargetQ - entCoef * nextLogProbs)).sum(-1);

	return rewards + gamma * (1 - dones) * nextV;
}

GGL::ReplayTransitions GGL::SACLearner::BuildTransitions(
	torch::Tensor states, torch::Tensor actionMasks,
	torch::Tensor actions, torch::Tensor rewards, torch::Tensor terminals,
	torch::Tensor truncNextStates, torch::Tensor truncNextMasks) {

	RG_NO_GRAD;

	int64_t numSteps = states.size(0);
	RG_ASSERT(numSteps > 0);

	{ // The experience must only contain complete episodes (matches GAE's requirement)
		int8_t lastTerminal = terminals[numSteps - 1].item<int8_t>();
		if (lastTerminal == RLGC::TerminalType::NOT_TERMINAL)
			RG_ERR_CLOSE(
				"SACLearner::BuildTransitions(): The last timestep must end an episode (terminal or truncated), " <<
				"but it is not terminal. Experience must only contain complete episodes."
			);
	}

	// Each timestep's next state is simply the following row of the same episode
	// The last row of each episode is fixed up below (it has no following row):
	//	- Truncated episodes bootstrap from their saved truncation state
	//	- Normally-ended episodes have their (rolled, wrong) next state ignored via done = 1
	torch::Tensor nextStates = torch::roll(states, -1, 0);
	torch::Tensor nextActionMasks = torch::roll(actionMasks, -1, 0);

	// NOTE: index_copy_() requires int64 indices (nonzero() already returns them)
	torch::Tensor truncIndices = (terminals == RLGC::TerminalType::TRUNCATED).nonzero().flatten();
	int64_t numTruncSaved = truncNextStates.defined() ? truncNextStates.size(0) : 0;
	if (truncIndices.size(0) != numTruncSaved)
		RG_ERR_CLOSE(
			"SACLearner::BuildTransitions(): Experience has " << truncIndices.size(0) << " truncated timestep(s), " <<
			"but " << numTruncSaved << " truncation bootstrap state(s) were saved"
		);

	if (numTruncSaved > 0) {
		// Truncation states were appended in episode-finish order,
		//	which is exactly the order truncated rows appear in the flattened experience
		nextStates.index_copy_(0, truncIndices, truncNextStates);
		nextActionMasks.index_copy_(0, truncIndices, truncNextMasks);
	}

	ReplayTransitions transitions = {};
	transitions.states = states;
	transitions.actions = actions;
	transitions.rewards = rewards;
	transitions.nextStates = nextStates;
	transitions.actionMasks = actionMasks;
	transitions.nextActionMasks = nextActionMasks;
	// Only real episode ends stop the bootstrap (truncations bootstrap from their next state)
	transitions.dones = (terminals == RLGC::TerminalType::NORMAL).to(torch::kFloat32);
	return transitions;
}

void GGL::SACLearner::UpdateTargets(float tau) {
	RG_NO_GRAD;

	constexpr const char* PAIRS[2][2] = { { "q1", "q1_target" }, { "q2", "q2_target" } };
	for (auto& pair : PAIRS) {
		auto liveParams = models[pair[0]]->parameters();
		auto targetParams = targetModels[pair[1]]->parameters();
		RG_ASSERT(liveParams.size() == targetParams.size());

		for (int i = 0; i < liveParams.size(); i++)
			targetParams[i].mul_(1 - tau).add_(liveParams[i], tau);
	}
}

void GGL::SACLearner::Learn(ReplayBuffer& replay, Report& report) {
	if (config.deterministic)
		RG_ERR_CLOSE(
			"SACLearner::Learn(): Cannot run a learn iteration in deterministic mode.\n" <<
			"Deterministic mode is only for inference/rendering (it removes the exploration SAC needs to learn)."
		);

	if (replay.Size() < config.batchSize)
		RG_ERR_CLOSE(
			"SACLearner::Learn(): Not enough replay data for a single batch " <<
			"(have " << replay.Size() << ", batch size is " << config.batchSize << ")"
		);

	auto mseLoss = torch::nn::MSELoss();

	MetricAccum
		avgQ1Loss, avgQ2Loss,
		avgPolicyLoss,
		avgEntropy,
		avgEntCoef, avgEntCoefLoss,
		avgQ, avgQTarget;

	// Save parameters first, to measure the update magnitudes
	auto policyBefore = models["policy"]->CopyParams();
	auto q1Before = models["q1"]->CopyParams();

	bool hasSharedHead = models["shared_head"] != NULL;

	for (int step = 0; step < config.gradientStepsPerItr; step++) {
		auto batch = replay.Sample(config.batchSize);

		// Send everything to the device
		auto obs = batch.states.to(device, true);
		auto acts = batch.actions.to(device, true);
		auto rews = batch.rewards.to(device, true);
		auto nextObs = batch.nextStates.to(device, true);
		auto masks = batch.actionMasks.to(device, true);
		auto nextMasks = batch.nextActionMasks.to(device, true);
		auto dones = batch.dones.to(device, true);

		// Kept as a device tensor so metrics/targets don't force a device sync every step
		auto entCoef = logEntCoef.detach().exp();

		// ~~~ Q-net update ~~~
		torch::Tensor qTargets;
		{
			RG_NO_GRAD;
			auto nextProbs = PolicyInference::InferProbs(models, nextObs, nextMasks, config.policyTemperature, false);
			auto nextLogProbs = nextProbs.log();
			auto minNextTargetQ = torch::min(
				targetModels["q1_target"]->Forward(nextObs, false),
				targetModels["q2_target"]->Forward(nextObs, false)
			);
			qTargets = ComputeQTargets(rews, dones, nextProbs, nextLogProbs, minNextTargetQ, nextMasks, entCoef, config.gamma);
		}

		// Q values of the actions that were actually taken
		auto q1 = models["q1"]->Forward(obs, false).gather(-1, acts.unsqueeze(-1)).flatten();
		auto q2 = models["q2"]->Forward(obs, false).gather(-1, acts.unsqueeze(-1)).flatten();

		auto q1Loss = mseLoss(q1, qTargets);
		auto q2Loss = mseLoss(q2, qTargets);
		(q1Loss + q2Loss).backward();

		if (config.gradClipNorm > 0) {
			nn::utils::clip_grad_norm_(models["q1"]->parameters(), config.gradClipNorm);
			nn::utils::clip_grad_norm_(models["q2"]->parameters(), config.gradClipNorm);
		}

		models["q1"]->StepOptim();
		models["q2"]->StepOptim();

		// ~~~ Policy update ~~~
		auto probs = PolicyInference::InferProbs(models, obs, masks, config.policyTemperature, false);
		auto logProbs = probs.log();

		torch::Tensor minQ;
		{
			// Uses the just-updated live Q nets, without tracking gradients through them
			RG_NO_GRAD;
			minQ = torch::min(
				models["q1"]->Forward(obs, false),
				models["q2"]->Forward(obs, false)
			);
		}

		auto masksF = masks.to(torch::kFloat32);

		// Minimize E_{a~pi}[ alpha * log pi(a|s) - minQ(s,a) ], exact over the discrete action space
		//	(no sampling/reparameterization needed)
		auto policyLoss = (masksF * probs * (entCoef * logProbs - minQ)).sum(-1).mean();
		policyLoss.backward();

		if (config.gradClipNorm > 0) {
			nn::utils::clip_grad_norm_(models["policy"]->parameters(), config.gradClipNorm);
			if (hasSharedHead)
				nn::utils::clip_grad_norm_(models["shared_head"]->parameters(), config.gradClipNorm);
		}

		models["policy"]->StepOptim();
		if (hasSharedHead)
			models["shared_head"]->StepOptim();

		// Exact entropy of each sampled state's action distribution, in nats
		auto entropy = -(masksF * probs * logProbs).sum(-1).detach();

		// ~~~ Entropy temperature (alpha) update ~~~
		if (config.autoEntCoef) {
			// Standard temperature loss on log(alpha):
			//	pushes alpha up when entropy is below the target, down when it is above
			auto entCoefLoss = -(logEntCoef * (targetEntropy - entropy)).mean();

			entCoefOptim->zero_grad();
			entCoefLoss.backward();
			entCoefOptim->step();

			avgEntCoefLoss.Add(entCoefLoss);
		}

		// ~~~ Target net update ~~~
		totalGradSteps++;
		if (totalGradSteps % config.targetUpdateInterval == 0)
			UpdateTargets(config.tau);

		avgQ1Loss.Add(q1Loss);
		avgQ2Loss.Add(q2Loss);
		avgPolicyLoss.Add(policyLoss);
		avgEntropy.Add(entropy.mean());
		avgEntCoef.Add(entCoef);
		avgQ.Add(q1.detach().mean());
		avgQTarget.Add(qTargets.mean());
	}

	// Compute magnitude of updates made to the policy and Q estimators
	auto policyAfter = models["policy"]->CopyParams();
	auto q1After = models["q1"]->CopyParams();

	// Assemble the report
	// (This is where the metric accumulators actually synchronize with the device)
	float rawEntropy = avgEntropy.Get();
	report["Policy Entropy"] = rawEntropy / logf(numActions); // Normalized like PPO's, for comparability
	report["Policy Loss"] = avgPolicyLoss.Get();
	report["SAC/Entropy"] = rawEntropy;
	report["SAC/Target Entropy"] = targetEntropy;
	report["SAC/Entropy Coef"] = avgEntCoef.Get();
	if (config.autoEntCoef)
		report["SAC/Entropy Coef Loss"] = avgEntCoefLoss.Get();
	report["SAC/Q1 Loss"] = avgQ1Loss.Get();
	report["SAC/Q2 Loss"] = avgQ2Loss.Get();
	report["SAC/Avg Q"] = avgQ.Get();
	report["SAC/Avg Q Target"] = avgQTarget.Get();
	report["SAC/Total Gradient Steps"] = totalGradSteps;
	report["Policy Update Magnitude"] = (policyBefore - policyAfter).norm().item<float>();
	report["Q Update Magnitude"] = (q1Before - q1After).norm().item<float>();
}

float GGL::SACLearner::GetEntCoef() const {
	return expf(logEntCoef.detach().cpu().item<float>());
}

void GGL::SACLearner::SetEntCoef(float entCoef) {
	if (entCoef <= 0)
		RG_ERR_CLOSE("SACLearner::SetEntCoef(): entCoef (" << entCoef << ") must be positive");

	RG_NO_GRAD;
	logEntCoef.fill_(logf(entCoef));
}

void GGL::SACLearner::SaveTo(std::filesystem::path folderPath) {
	models.Save(folderPath);
	targetModels.Save(folderPath, false); // Target nets have no meaningful optimizer state

	{ // Save the non-Model state: the entropy coef and the gradient step counter
		torch::serialize::OutputArchive stateArchive;
		stateArchive.write("log_ent_coef", logEntCoef.detach().cpu());
		stateArchive.write("total_grad_steps", torch::tensor(totalGradSteps, torch::kInt64));
		stateArchive.save_to((folderPath / EXTRA_STATE_FILE_NAME).string());

		if (entCoefOptim) {
			torch::serialize::OutputArchive optimArchive;
			entCoefOptim->save(optimArchive);
			optimArchive.save_to((folderPath / ENT_COEF_OPTIM_FILE_NAME).string());
		}
	}
}

void GGL::SACLearner::LoadFrom(std::filesystem::path folderPath) {
	if (!std::filesystem::is_directory(folderPath))
		RG_ERR_CLOSE("SACLearner::LoadFrom(): Path " << folderPath << " is not a valid directory");

	models.Load(folderPath, true, true);

	{ // Load target nets; if they weren't saved, re-sync them from the live Q nets
		RG_NO_GRAD;

		constexpr const char* PAIRS[2][2] = { { "q1", "q1_target" }, { "q2", "q2_target" } };
		for (auto& pair : PAIRS) {
			Model* target = targetModels[pair[1]];

			if (std::filesystem::exists(target->GetSavePath(folderPath))) {
				target->Load(folderPath, false, false);
			} else {
				RG_LOG("Warning: Model \"" << pair[1] << "\" does not exist in " << folderPath << ", copying from \"" << pair[0] << "\"");
				auto fromParams = models[pair[0]]->parameters();
				auto toParams = target->parameters();
				for (int i = 0; i < fromParams.size(); i++)
					toParams[i].copy_(fromParams[i], true);
			}

			// Loading can replace the parameter tensors, so re-freeze them
			//	(target nets are only ever written by Polyak averaging)
			for (auto& param : target->parameters())
				param.set_requires_grad(false);
		}
	}

	{ // Load the non-Model state (entropy coef + gradient step counter)
		auto statePath = folderPath / EXTRA_STATE_FILE_NAME;
		if (std::filesystem::exists(statePath)) {
			torch::serialize::InputArchive stateArchive;
			stateArchive.load_from(statePath.string(), torch::kCPU);

			torch::Tensor loadedLogEntCoef, loadedGradSteps;
			stateArchive.read("log_ent_coef", loadedLogEntCoef);
			stateArchive.read("total_grad_steps", loadedGradSteps);

			RG_NO_GRAD;
			logEntCoef.copy_(loadedLogEntCoef.to(device));
			totalGradSteps = loadedGradSteps.item<int64_t>();
		} else {
			RG_LOG("Warning: No saved SAC state found at " << statePath << ", the entropy coef will use the config value");
		}

		auto entCoefOptimPath = folderPath / ENT_COEF_OPTIM_FILE_NAME;
		if (entCoefOptim && std::filesystem::exists(entCoefOptimPath)) {
			torch::serialize::InputArchive optimArchive;
			optimArchive.load_from(entCoefOptimPath.string(), device);
			entCoefOptim->load(optimArchive);
		}
	}

	SetLearningRates(config.policyLR, config.qLR);
}

void GGL::SACLearner::SetLearningRates(float policyLR, float qLR) {
	config.policyLR = policyLR;
	config.qLR = qLR;

	models["policy"]->SetOptimLR(policyLR);
	if (models["shared_head"])
		models["shared_head"]->SetOptimLR(policyLR); // The shared head only feeds the policy
	models["q1"]->SetOptimLR(qLR);
	models["q2"]->SetOptimLR(qLR);

	RG_LOG("SACLearner: " << RS_STR(std::scientific << "Set learning rate to [" << policyLR << ", " << qLR << "]"));
}

GGL::ModelSet GGL::SACLearner::GetPolicyModels() {
	ModelSet result = {};
	for (Model* model : models) {
		if (model->modelName == std::string("q1") || model->modelName == std::string("q2"))
			continue;

		result.Add(model);
	}
	return result;
}
