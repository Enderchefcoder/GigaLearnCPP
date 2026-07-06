#pragma once

#include <RLGymCPP/EnvSet/EnvSet.h>
#include "Util/MetricSender.h"
#include "Util/RenderSender.h"
#include "LearnerConfig.h"
#include "PPO/TransferLearnConfig.h"

namespace GGL {

	typedef std::function<void(class Learner*, const std::vector<RLGC::GameState>& states, Report& report)> StepCallbackFn;

	// https://github.com/AechPro/rlgym-ppo/blob/main/rlgym_ppo/learner.py
	class RG_IMEXPORT Learner {
	public:
		LearnerConfig config;

		RLGC::EnvSet* envSet;

		// The active algorithm (see config.algorithm); the non-active one is NULL
		class PPOLearner* ppo;
		class SACLearner* sac;
		// The active learner, through the algorithm-agnostic interface
		class AlgoLearner* algo;

		class PolicyVersionManager* versionMgr;

		RLGC::EnvCreateFn envCreateFn;
		MetricSender* metricSender;
		RenderSender* renderSender;

		int obsSize;
		int numActions;

		struct WelfordStat* returnStat;
		struct BatchedWelfordStat* obsStat;

		std::string runID = {};

		uint64_t
			totalTimesteps = 0,
			totalIterations = 0;

		// Whether this learner started the embedded Python interpreter (and thus must finalize it)
		bool _ownsPyInterpreter = false;

		StepCallbackFn stepCallback = NULL;

		Learner(RLGC::EnvCreateFn envCreateFunc, LearnerConfig config, StepCallbackFn stepCallback = NULL);
		void Start();

		// NOTE: Transfer learning is currently only supported with the PPO algorithm
		void StartTransferLearn(const TransferLearnConfig& transferLearnConfig);

		// Runtime training-parameter adjustment (e.g. for schedules driven from the step callback)
		// Takes effect from the next learn phase onwards
		// For SAC, the second learning rate is the Q-net learning rate (SAC has no critic)
		void SetLearningRates(float policyLR, float criticLR);
		// For PPO, this is the normalized-entropy bonus scale (cfg.ppo.entropyScale)
		// For SAC, this sets the entropy temperature alpha; only allowed when
		//	cfg.sac.autoEntCoef is off (alpha is auto-tuned otherwise)
		void SetEntropyScale(float entropyScale);

		float GetPolicyLR() const;
		float GetCriticLR() const; // For SAC, returns the Q-net learning rate
		float GetEntropyScale() const; // For SAC, returns the current (possibly auto-tuned) alpha

		void StartQuitKeyThread(bool& quitPressed, std::thread& outThread);

		void Save();
		void Load();
		void SaveStats(std::filesystem::path path);
		void LoadStats(std::filesystem::path path);

		RG_NO_COPY(Learner);

		~Learner();
	};
}