#pragma once
#include "ExperienceBuffer.h"
#include <GigaLearnCPP/Util/Report.h>
#include <GigaLearnCPP/Util/Timer.h>
#include <GigaLearnCPP/PPO/PPOLearnerConfig.h>
#include <GigaLearnCPP/PPO/TransferLearnConfig.h>

#include "../AlgoLearner.h"
#include "../Util/Models.h"

#include <torch/optim/adam.h>
#include <torch/nn/modules/loss.h>
#include <torch/nn/modules/container/sequential.h>

namespace GGL {

	// https://github.com/AechPro/rlgym-ppo/blob/main/rlgym_ppo/ppo/ppo_learner.py
	class PPOLearner : public AlgoLearner {
	public:
		ModelSet models = {};
		ModelSet guidingPolicyModels = {};

		PPOLearnerConfig config;

		PPOLearner(
			int obsSize, int numActions,
			PPOLearnerConfig config, torch::Device device
		);

		// If models is null, this->models will be used
		virtual void InferActions(torch::Tensor obs, torch::Tensor actionMasks, torch::Tensor* outActions, torch::Tensor* outLogProbs, ModelSet* models = NULL) override;
		torch::Tensor InferCritic(torch::Tensor obs);

		void Learn(ExperienceBuffer& experience, Report& report, bool isFirstIteration);

		void TransferLearn(
			ModelSet& oldModels, 
			torch::Tensor newObs, torch::Tensor oldObs, 
			torch::Tensor newActionMasks, torch::Tensor oldActionMasks, 
			torch::Tensor actionMaps,
			Report& report, 
			const TransferLearnConfig& transferLearnConfig
		);

		virtual void SaveTo(std::filesystem::path folderPath) override;
		virtual void LoadFrom(std::filesystem::path folderPath) override;
		void SetLearningRates(float policyLR, float criticLR);

		// NOTE: The returned set is a non-owning view of this learner's models
		virtual ModelSet GetPolicyModels() override;

		virtual float GetPolicyTemperature() const override {
			return config.policyTemperature;
		}

		virtual bool GetUseHalfPrecision() const override {
			return config.useHalfPrecision;
		}

		RG_NO_COPY(PPOLearner);

		~PPOLearner() {
			models.Free();
			guidingPolicyModels.Free();
		}
	};
}
