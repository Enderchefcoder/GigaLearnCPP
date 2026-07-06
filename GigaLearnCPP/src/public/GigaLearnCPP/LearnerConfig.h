#pragma once
#include <RLGymCPP/BasicTypes/Lists.h>
#include "PPO/PPOLearnerConfig.h"
#include "SAC/SACLearnerConfig.h"
#include "SkillTrackerConfig.h"

namespace GGL {
	enum class LearnerDeviceType {
		AUTO,
		CPU,
		GPU_CUDA
	};

	enum class LearningAlgorithmType {
		// Proximal Policy Optimization: on-policy, the well-tested default for Rocket League ML
		PPO,

		// Soft Actor-Critic with discrete actions (SAC-Discrete): off-policy,
		//	learns from a replay buffer of past experience with automatic entropy tuning
		SAC
	};

	// https://github.com/AechPro/rlgym-ppo/blob/main/rlgym_ppo/learner.py
	struct LearnerConfig {
		int numGames = 300;

		int tickSkip = 8;
		int actionDelay = 7;

		// Number of torch intra-op threads to use during collection (0 = leave at torch's default)
		// Torch's idle worker threads spin-wait, which starves the environment threads during collection,
		//	so limiting torch to 1 thread during collection is usually much faster on CPU
		//	(the full thread count is restored for the learn phase, which needs it)
		// On this library's test machine, 1 makes CPU collection ~2.4x faster than torch's default
		int collectionTorchThreads = 1;

		bool renderMode = false;
		// If renderMode, this is the scaling of time for the game
		// 1.0 = Run the game at real time
		// 2.0 = Run the game twice as fast as real time
		float renderTimeScale = 1.0f; 

		// Which learning algorithm to train with
		// Only the matching config (cfg.ppo or cfg.sac) is used
		// NOTE: A checkpoint folder belongs to one algorithm; switching algorithms needs a new folder
		//	(policies transfer between algorithms for *inference*, but optimizer/value state does not)
		LearningAlgorithmType algorithm = LearningAlgorithmType::PPO;

		PPOLearnerConfig ppo = {}; // Only used if algorithm == PPO
		SACLearnerConfig sac = {}; // Only used if algorithm == SAC

		// Stop training once this many total timesteps have been collected
		//	(a final checkpoint is saved first, if saving is enabled)
		// Set to 0 to train forever (default)
		int64_t timestepLimit = 0;

		// Checkpoints are saved here as timestep-numbered subfolders
		//	e.g. a checkpoint at 20,000 steps will save to a subfolder called "20000"
		// Set empty to disable saving
		std::filesystem::path checkpointFolder = "checkpoints"; 

		// Which checkpoint (timestep subfolder) to load at startup
		// -1 loads the newest checkpoint (default)
		// Set to a specific checkpoint's timestep number to roll back after a bad
		//	training period. NOTE: Delete the newer checkpoint subfolders (and newer
		//	policy versions, if using them) afterwards, otherwise auto-cleanup and
		//	version loading will misbehave around them (a warning will tell you).
		int64_t checkpointToLoad = -1;

		// Save every timestep
		// Set to zero to just use timestepsPerIteration
		int64_t tsPerSave = 1'000'000;

		int64_t randomSeed = -1; // Set to -1 to use the current time
		int checkpointsToKeep = 8; // Checkpoint storage limit before old checkpoints are deleted, set to -1 to disable
		LearnerDeviceType deviceType = LearnerDeviceType::AUTO; // Auto will use your CUDA GPU if available

		// Standardize the obs values (doesn't seem to help much from my testing)
		bool standardizeObs = false;
		float minObsSTD = 1 / 10.f;
		float maxObsMeanRange = 3;
		int maxObsSamples = 100;

		// Standardize the returns to help the critic (don't disable this unless you know what you're doing)
		bool standardizeReturns = true;
		int maxReturnSamples = 150;

		// Will automatically add the rewards to metrics
		bool addRewardsToMetrics = true;
		int maxRewardSamples = 50; // Maximum reward samples per step for reward metrics
		int rewardSampleRandInterval = 8; // Randomized interval range between sampling rewards (per step)

		// Send metrics to the python metrics receiver
		// The receiver can then log them to wandb or whatever
		bool sendMetrics = true;
		std::string metricsProjectName = "gigalearncpp"; // Project name for the python metrics receiver
		std::string metricsGroupName = "unnamed-runs"; // Group name for the python metrics receiver
		std::string metricsRunName = "gigalearncpp-run"; // Run name for the python metrics receiver

		bool savePolicyVersions = false;
		int64_t tsPerVersion = 25'000'000;
		int maxOldVersions = 32;

		bool trainAgainstOldVersions = false;
		float trainAgainstOldChance = 0.15f; // Chance (from 0 - 1) that an iteration will train against an old version

		// Biases old-version opponent selection toward recent versions
		// 0 = uniform over all saved versions (default)
		// Otherwise, each step back in version history is (1 - bias) times as likely to be picked
		//	(e.g. 0.25 means a version is picked 75% as often as the version after it)
		float oldVersionRecencyBias = 0;

		SkillTrackerConfig skillTracker = {};
	};
}