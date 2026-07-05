#pragma once

#include "ModelConfig.h"

namespace GGL {
	struct RG_IMEXPORT InferUnit {
		int obsSize;
		RLGC::ObsBuilder* obsBuilder;
		RLGC::ActionParser* actionParser;
		struct ModelSet* models;
		bool useGPU;

		// NOTE: Reset() will never be called on your obs builder
		// NOTE: If one InferUnit is shared between multiple bots calling from different threads,
		//	the obs builder must be stateless (e.g. AdvancedObs); otherwise create one InferUnit per bot
		InferUnit(
			RLGC::ObsBuilder* obsBuilder, int obsSize, RLGC::ActionParser* actionParser,
			PartialModelConfig sharedHeadConfig, PartialModelConfig policyConfig,
			std::filesystem::path modelsFolder, bool useGPU);


		RLGC::Action InferAction(const RLGC::Player& player, const RLGC::GameState& state, bool deterministic, float temperature = 1);
		std::vector<RLGC::Action> BatchInferActions(const std::vector<RLGC::Player>& players, const std::vector<RLGC::GameState>& states, bool deterministic, float temperature = 1);

		RG_NO_COPY(InferUnit);

		// NOTE: Does not free the obs builder or action parser (they are owned by the caller)
		~InferUnit();
	};
}