#pragma once
#include "ReplayBuffer.h"
#include <GigaLearnCPP/Util/Report.h>
#include <GigaLearnCPP/SAC/SACLearnerConfig.h>

#include "../AlgoLearner.h"
#include "../Util/Models.h"

#include <torch/optim/adam.h>

namespace GGL {

	// Soft Actor-Critic for discrete action spaces (SAC-Discrete)
	// https://arxiv.org/abs/1910.07207
	//
	// The policy is the same masked-softmax categorical policy PPO uses
	// Value estimation uses twin Q nets (obs -> one Q value per action) with
	//	Polyak-averaged target copies, and the policy maximizes
	//	E[Q - alpha * log pi] exactly over the discrete action distribution
	//	(no sampling/reparameterization needed, unlike continuous SAC)
	class SACLearner : public AlgoLearner {
	public:
		// Trained models: "policy" (+ optional "shared_head"), "q1", "q2"
		ModelSet models = {};
		// Polyak-averaged target copies: "q1_target", "q2_target" (never directly trained)
		ModelSet targetModels = {};

		SACLearnerConfig config;
		int numActions;

		// log(alpha); optimized toward targetEntropy if config.autoEntCoef
		torch::Tensor logEntCoef;
		torch::optim::Adam* entCoefOptim = NULL;
		float targetEntropy; // In nats

		int64_t totalGradSteps = 0;

		SACLearner(
			int obsSize, int numActions,
			SACLearnerConfig config, torch::Device device
		);

		// If models is null, this->models will be used
		virtual void InferActions(torch::Tensor obs, torch::Tensor actionMasks, torch::Tensor* outActions, torch::Tensor* outLogProbs, ModelSet* models = NULL) override;

		// The Q-learning target: y = r + gamma * (1 - done) * V(s'),
		//	with the soft state value V(s') = E_{a~pi}[ minTargetQ(s',a) - alpha * log pi(a|s') ]
		//	computed exactly over the valid actions of s'
		// Pure tensor math (static so it can be unit-tested directly)
		static torch::Tensor ComputeQTargets(
			torch::Tensor rewards, torch::Tensor dones,
			torch::Tensor nextProbs, torch::Tensor nextLogProbs,
			torch::Tensor minNextTargetQ, torch::Tensor nextActionMasks,
			torch::Tensor entCoef, float gamma
		);

		// Runs config.gradientStepsPerItr gradient updates on samples from the replay buffer
		void Learn(ReplayBuffer& replay, Report& report);

		// Polyak-averages the live Q nets into the target Q nets
		void UpdateTargets(float tau);

		float GetEntCoef() const; // Current alpha (NOTE: synchronizes the device)
		void SetEntCoef(float entCoef);

		virtual void SaveTo(std::filesystem::path folderPath) override;
		virtual void LoadFrom(std::filesystem::path folderPath) override;
		void SetLearningRates(float policyLR, float qLR);

		// NOTE: The returned set is a non-owning view of this learner's models
		virtual ModelSet GetPolicyModels() override;

		virtual float GetPolicyTemperature() const override {
			return config.policyTemperature;
		}

		virtual bool GetUseHalfPrecision() const override {
			return config.useHalfPrecision;
		}

		RG_NO_COPY(SACLearner);

		~SACLearner() {
			models.Free();
			targetModels.Free();
			delete entCoefOptim;
		}
	};
}
