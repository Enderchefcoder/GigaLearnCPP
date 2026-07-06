#pragma once
#include <RLGymCPP/BasicTypes/Lists.h>

#include "../Util/ModelConfig.h"

namespace GGL {

	// Configuration for Soft Actor-Critic with discrete actions (SAC-Discrete)
	// https://arxiv.org/abs/1910.07207
	//
	// SAC is off-policy: it keeps a large replay buffer of past transitions and
	//	learns from random samples of it, instead of only learning from the newest experience like PPO
	// The policy has the same structure as PPO's (masked-softmax over discrete actions),
	//	so checkpointed SAC policies work everywhere PPO policies do (InferUnit, RLBot, old-version play)
	struct SACLearnerConfig {

		// Env timesteps to collect per iteration (between learn phases)
		int64_t tsPerItr = 10'000;

		// Maximum transitions kept in the replay buffer (oldest are overwritten)
		// Memory usage is roughly: replayBufferSize * (2*obsSize*4 + 2*numActions + 16) bytes
		//	(e.g. ~500 MB for the default with an obs size of ~100 and ~90 actions)
		int64_t replayBufferSize = 500'000;

		// Replay transitions sampled per gradient step
		int64_t batchSize = 512;

		// Gradient steps to run per iteration
		// The replay ratio (average times each collected timestep is learned from) is
		//	gradientStepsPerItr * batchSize / tsPerItr; higher learns more per timestep but is slower
		//	and can destabilize training if pushed too far
		int gradientStepsPerItr = 64;

		// No learning happens until this many timesteps have been collected
		//	(gives the replay buffer some diverse data before the first update)
		int64_t learningStartTimesteps = 20'000;

		double maxEpisodeDuration = 120; // In seconds

		// Actions with the highest probability are always chosen, instead of being more likely
		// This will make your bot play better (usually), but is horrible for learning
		// Trying to run a SAC learn iteration with deterministic mode will throw an exception
		bool deterministic = false;

		// Use half-precision models for inference
		// This is much faster on GPU, not so much for CPU
		bool useHalfPrecision = false;

		// Temperature of the policy's softmax distribution
		float policyTemperature = 1;

		// Model configs
		// The Q nets both use qNet's config (obs -> one Q value per action)
		// sharedHead (optional) feeds only the policy, NOT the Q nets
		//	(a trunk shared between the actor and critics is a common source of SAC instability)
		PartialModelConfig policy, qNet, sharedHead;

		float policyLR = 3e-4f; // Policy learning rate
		float qLR = 3e-4f; // Q-net learning rate (both Q nets)

		// Rate of reward decay
		// NOTE: SAC does not use GAE; this is the plain discount on the Q-learning target
		float gamma = 0.99f;

		// Polyak averaging coefficient for the target Q nets
		//	(each update: target = tau * live + (1 - tau) * target)
		float tau = 0.005f;

		// Gradient steps between target net updates (1 = update every step, the usual choice)
		int targetUpdateInterval = 1;

		// ~~~ Entropy temperature (alpha) ~~~
		// Alpha scales the entropy bonus: bigger alpha = more exploration
		// With autoEntCoef, alpha is tuned automatically so the policy's entropy
		//	tracks targetEntropy = targetEntropyScale * log(numActions)

		bool autoEntCoef = true;
		float entCoef = 0.2f; // Fixed alpha if !autoEntCoef, otherwise the initial alpha
		float entCoefLR = 3e-4f; // Learning rate of the alpha auto-tuner
		// Fraction of the maximum possible entropy (log of the action count) to target
		// The SAC-Discrete paper used 0.98, which tends to over-explore;
		//	lower it if your bot stays too random, raise it if it collapses to a few actions
		float targetEntropyScale = 0.7f;

		// Maximum gradient norm per model per gradient step
		// Set to 0 to disable clipping (SAC usually doesn't need it)
		float gradClipNorm = 0;

		SACLearnerConfig() {
			policy = {};
			policy.layerSizes = { 256, 256, 256 };
			qNet = {};
			qNet.layerSizes = { 256, 256, 256 };
			// No shared head by default (see the note above)
			sharedHead = {};
			sharedHead.addOutputLayer = false;
		}
	};
}
