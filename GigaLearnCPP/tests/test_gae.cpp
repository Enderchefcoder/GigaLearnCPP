#include "TestFramework.h"

#include <private/GigaLearnCPP/PPO/GAE.h>
#include <RLGymCPP/TerminalConditions/TerminalCondition.h>

using namespace GGL;

// Reference implementation of GAE for a single episode, straight from the paper
// (Bootstraps with truncValPred at the end if truncated)
static void ReferenceGAE(
	const std::vector<float>& rews, const std::vector<float>& valPreds,
	bool truncated, float truncValPred,
	float gamma, float lambda,
	std::vector<float>& outAdvantages
) {
	int n = (int)rews.size();
	outAdvantages.assign(n, 0);

	float nextAdvantage = 0;
	for (int t = n - 1; t >= 0; t--) {
		float nextVal;
		if (t == n - 1) {
			nextVal = truncated ? truncValPred : 0;
		} else {
			nextVal = valPreds[t + 1];
		}

		float delta = rews[t] + gamma * nextVal - valPreds[t];
		float adv = delta + gamma * lambda * nextAdvantage;
		outAdvantages[t] = adv;
		nextAdvantage = adv;
	}
}

TEST_CASE(GAE_SingleTerminalEpisode) {
	constexpr float GAMMA = 0.9f, LAMBDA = 0.85f;

	std::vector<float> rews = { 1, -2, 3, 0.5f };
	std::vector<float> valPreds = { 0.3f, -0.1f, 2, 1 };
	std::vector<int8_t> terminals = { 0, 0, 0, RLGC::TerminalType::NORMAL };

	torch::Tensor tAdvantages, tTargetVals, tReturns;
	float clipPortion;
	GAE::Compute(
		VEC_TO_TENSOR(rews), VEC_TO_TENSOR(terminals), VEC_TO_TENSOR(valPreds), torch::Tensor(),
		tAdvantages, tTargetVals, tReturns, clipPortion,
		GAMMA, LAMBDA, 0, 0
	);

	std::vector<float> expected;
	ReferenceGAE(rews, valPreds, false, 0, GAMMA, LAMBDA, expected);

	auto advantages = TENSOR_TO_VEC<float>(tAdvantages);
	CHECK_EQ(advantages.size(), expected.size());
	for (int i = 0; i < expected.size(); i++)
		CHECK_NEAR(advantages[i], expected[i], 1e-5f);

	// Target values = value preds + advantages
	auto targetVals = TENSOR_TO_VEC<float>(tTargetVals);
	for (int i = 0; i < expected.size(); i++)
		CHECK_NEAR(targetVals[i], valPreds[i] + expected[i], 1e-5f);
}

TEST_CASE(GAE_MultipleEpisodes) {
	constexpr float GAMMA = 0.99f, LAMBDA = 0.95f;

	// Two episodes back-to-back in the same buffer
	std::vector<float> rews1 = { 1, 0, 2 }, rews2 = { -1, 3 };
	std::vector<float> vals1 = { 0.5f, 0.25f, 1 }, vals2 = { 0.75f, -0.5f };

	std::vector<float> rews = rews1;
	rews.insert(rews.end(), rews2.begin(), rews2.end());
	std::vector<float> valPreds = vals1;
	valPreds.insert(valPreds.end(), vals2.begin(), vals2.end());

	std::vector<int8_t> terminals = { 0, 0, RLGC::TerminalType::NORMAL, 0, RLGC::TerminalType::NORMAL };

	torch::Tensor tAdvantages, tTargetVals, tReturns;
	float clipPortion;
	GAE::Compute(
		VEC_TO_TENSOR(rews), VEC_TO_TENSOR(terminals), VEC_TO_TENSOR(valPreds), torch::Tensor(),
		tAdvantages, tTargetVals, tReturns, clipPortion,
		GAMMA, LAMBDA, 0, 0
	);

	// Each episode's advantages must match the isolated reference
	//	(episode boundaries must fully reset the GAE accumulators)
	std::vector<float> expected1, expected2;
	ReferenceGAE(rews1, vals1, false, 0, GAMMA, LAMBDA, expected1);
	ReferenceGAE(rews2, vals2, false, 0, GAMMA, LAMBDA, expected2);

	auto advantages = TENSOR_TO_VEC<float>(tAdvantages);
	for (int i = 0; i < 3; i++)
		CHECK_NEAR(advantages[i], expected1[i], 1e-5f);
	for (int i = 0; i < 2; i++)
		CHECK_NEAR(advantages[3 + i], expected2[i], 1e-5f);
}

// Regression test: truncated value predictions are stored in episode-completion order,
//	but GAE iterates backwards, so it must consume them from the back
TEST_CASE(GAE_TruncationOrdering) {
	constexpr float GAMMA = 0.9f, LAMBDA = 0.95f;

	// Two truncated episodes with different bootstrap values
	std::vector<float> rews = { 0, 0, 0, 0 };
	std::vector<float> valPreds = { 0, 0, 0, 0 };
	std::vector<int8_t> terminals = {
		0, RLGC::TerminalType::TRUNCATED, // Episode 1 (finished first)
		0, RLGC::TerminalType::TRUNCATED, // Episode 2 (finished second)
	};

	// Episode 1 was truncated at value 10, episode 2 at value 20
	std::vector<float> truncValPreds = { 10, 20 };

	torch::Tensor tAdvantages, tTargetVals, tReturns;
	float clipPortion;
	GAE::Compute(
		VEC_TO_TENSOR(rews), VEC_TO_TENSOR(terminals), VEC_TO_TENSOR(valPreds), VEC_TO_TENSOR(truncValPreds),
		tAdvantages, tTargetVals, tReturns, clipPortion,
		GAMMA, LAMBDA, 0, 0
	);

	auto advantages = TENSOR_TO_VEC<float>(tAdvantages);

	// With zero rewards and zero value predictions, the advantage at each truncated step
	//	is just gamma * bootstrapValue
	CHECK_NEAR(advantages[1], GAMMA * 10, 1e-4f);
	CHECK_NEAR(advantages[3], GAMMA * 20, 1e-4f);

	// Also validate against the reference implementation
	std::vector<float> expected1, expected2;
	ReferenceGAE({ 0, 0 }, { 0, 0 }, true, 10, GAMMA, LAMBDA, expected1);
	ReferenceGAE({ 0, 0 }, { 0, 0 }, true, 20, GAMMA, LAMBDA, expected2);
	for (int i = 0; i < 2; i++) {
		CHECK_NEAR(advantages[i], expected1[i], 1e-4f);
		CHECK_NEAR(advantages[2 + i], expected2[i], 1e-4f);
	}
}

TEST_CASE(GAE_RewardStandardizationAndClipping) {
	// Rewards standardized by returnStd = 2, clipped to +-1
	std::vector<float> rews = { 4, -6 };
	std::vector<float> valPreds = { 0, 0 };
	std::vector<int8_t> terminals = { 0, RLGC::TerminalType::NORMAL };

	torch::Tensor tAdvantages, tTargetVals, tReturns;
	float clipPortion;
	GAE::Compute(
		VEC_TO_TENSOR(rews), VEC_TO_TENSOR(terminals), VEC_TO_TENSOR(valPreds), torch::Tensor(),
		tAdvantages, tTargetVals, tReturns, clipPortion,
		1.0f, 1.0f, 2 /* returnStd */, 1 /* clipRange */
	);

	// Standardized rewards: 2, -3 -> clipped: 1, -1
	// Total |rew| = 5, total clipped |rew| = 2, so clip portion = 3/5
	CHECK_NEAR(clipPortion, 0.6f, 1e-5f);

	// Advantage of the last step: clipped reward - valPred = -1
	auto advantages = TENSOR_TO_VEC<float>(tAdvantages);
	CHECK_NEAR(advantages[1], -1, 1e-5f);

	// Returns use unstandardized rewards
	auto returns = TENSOR_TO_VEC<float>(tReturns);
	CHECK_NEAR(returns[1], -6, 1e-5f);
	CHECK_NEAR(returns[0], 4 + -6, 1e-5f);
}

// Property test: random episode structures must always match the reference implementation
TEST_CASE(GAE_RandomizedAgainstReference) {
	std::default_random_engine rng(0xC0FFEE);
	auto randFloat = [&](float min, float max) {
		return min + (rng() / (float)rng.max()) * (max - min);
	};
	auto randInt = [&](int min, int max) { // Max exclusive
		return min + (int)(rng() % (max - min));
	};

	constexpr int NUM_CASES = 100;
	for (int testCase = 0; testCase < NUM_CASES; testCase++) {
		float gamma = randFloat(0.8f, 1.0f);
		float lambda = randFloat(0.8f, 1.0f);

		// Build a random set of episodes
		int numEpisodes = randInt(1, 6);
		std::vector<float> rews = {}, valPreds = {}, truncValPreds = {};
		std::vector<int8_t> terminals = {};

		struct EpisodeRef {
			std::vector<float> rews, valPreds;
			bool truncated;
			float truncValPred;
		};
		std::vector<EpisodeRef> episodes = {};

		for (int ep = 0; ep < numEpisodes; ep++) {
			int len = randInt(1, 12);
			bool truncated = randInt(0, 2) == 1;

			EpisodeRef ref = {};
			ref.truncated = truncated;

			for (int t = 0; t < len; t++) {
				float rew = randFloat(-2, 2), valPred = randFloat(-2, 2);
				ref.rews.push_back(rew);
				ref.valPreds.push_back(valPred);
				rews.push_back(rew);
				valPreds.push_back(valPred);
				terminals.push_back(
					(t == len - 1)
					? (truncated ? RLGC::TerminalType::TRUNCATED : RLGC::TerminalType::NORMAL)
					: RLGC::TerminalType::NOT_TERMINAL
				);
			}

			if (truncated) {
				ref.truncValPred = randFloat(-2, 2);
				truncValPreds.push_back(ref.truncValPred);
			}

			episodes.push_back(ref);
		}

		torch::Tensor tAdvantages, tTargetVals, tReturns;
		float clipPortion;
		GAE::Compute(
			VEC_TO_TENSOR(rews), VEC_TO_TENSOR(terminals), VEC_TO_TENSOR(valPreds),
			truncValPreds.empty() ? torch::Tensor() : VEC_TO_TENSOR(truncValPreds),
			tAdvantages, tTargetVals, tReturns, clipPortion,
			gamma, lambda, 0, 0
		);

		auto advantages = TENSOR_TO_VEC<float>(tAdvantages);
		auto targetVals = TENSOR_TO_VEC<float>(tTargetVals);

		// Each episode must independently match the reference implementation
		int offset = 0;
		for (auto& ep : episodes) {
			std::vector<float> expected;
			ReferenceGAE(ep.rews, ep.valPreds, ep.truncated, ep.truncValPred, gamma, lambda, expected);

			for (int t = 0; t < expected.size(); t++) {
				CHECK_NEAR(advantages[offset + t], expected[t], 2e-4f);
				CHECK_NEAR(targetVals[offset + t], ep.valPreds[t] + expected[t], 2e-4f);
			}
			offset += ep.rews.size();
		}

		CHECK_EQ(offset, (int)advantages.size());
	}
}

TEST_CASE(GAE_RejectsIncompleteEpisodes) {
	// The buffer must only contain complete episodes,
	//	otherwise the last step would bootstrap from out-of-bounds memory
	std::vector<float> rews = { 1, 1 };
	std::vector<float> valPreds = { 0, 0 };
	std::vector<int8_t> terminals = { 0, RLGC::TerminalType::NOT_TERMINAL };

	torch::Tensor tAdvantages, tTargetVals, tReturns;
	float clipPortion;
	CHECK_THROWS(
		GAE::Compute(
			VEC_TO_TENSOR(rews), VEC_TO_TENSOR(terminals), VEC_TO_TENSOR(valPreds), torch::Tensor(),
			tAdvantages, tTargetVals, tReturns, clipPortion,
			0.99f, 0.95f, 0, 0
		)
	);
}

TEST_CASE(GAE_RejectsWrongTruncationCount) {
	std::vector<float> rews = { 1, 1 };
	std::vector<float> valPreds = { 0, 0 };
	std::vector<int8_t> terminals = { 0, RLGC::TerminalType::TRUNCATED };

	// Truncation present but no trunc val preds given
	torch::Tensor tAdvantages, tTargetVals, tReturns;
	float clipPortion;
	CHECK_THROWS(
		GAE::Compute(
			VEC_TO_TENSOR(rews), VEC_TO_TENSOR(terminals), VEC_TO_TENSOR(valPreds), torch::Tensor(),
			tAdvantages, tTargetVals, tReturns, clipPortion,
			0.99f, 0.95f, 0, 0
		)
	);

	// Too many trunc val preds given
	std::vector<float> truncValPreds = { 1, 2 };
	CHECK_THROWS(
		GAE::Compute(
			VEC_TO_TENSOR(rews), VEC_TO_TENSOR(terminals), VEC_TO_TENSOR(valPreds), VEC_TO_TENSOR(truncValPreds),
			tAdvantages, tTargetVals, tReturns, clipPortion,
			0.99f, 0.95f, 0, 0
		)
	);
}
