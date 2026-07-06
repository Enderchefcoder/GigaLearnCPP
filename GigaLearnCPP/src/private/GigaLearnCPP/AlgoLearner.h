#pragma once
#include "Util/Models.h"

namespace GGL {

	// Common interface over the learning algorithms (PPO, SAC)
	// Shared infrastructure (checkpointing, policy versions, skill matches, the collection loop)
	//	works through this instead of a concrete algorithm
	// Both algorithms use the same discrete policy structure (see Util/PolicyInference.h),
	//	which is what makes saved policies interchangeable between them for inference
	class AlgoLearner {
	public:
		torch::Device device;

		AlgoLearner(torch::Device device) : device(device) {}

		// Runs the policy on a batch of obs, sampling actions (and their log probs, if requested)
		// If models is null, the learner's own models are used
		//	(pass e.g. an old policy version's models to infer with them instead)
		virtual void InferActions(
			torch::Tensor obs, torch::Tensor actionMasks,
			torch::Tensor* outActions, torch::Tensor* outLogProbs,
			ModelSet* models = NULL
		) = 0;

		// NOTE: The returned set is a non-owning view of this learner's policy models
		//	("policy" + optional "shared_head", without any value/Q models)
		virtual ModelSet GetPolicyModels() = 0;

		virtual float GetPolicyTemperature() const = 0;
		virtual bool GetUseHalfPrecision() const = 0;

		virtual void SaveTo(std::filesystem::path folderPath) = 0;
		virtual void LoadFrom(std::filesystem::path folderPath) = 0;

		RG_NO_COPY(AlgoLearner);

		virtual ~AlgoLearner() {}
	};
}
