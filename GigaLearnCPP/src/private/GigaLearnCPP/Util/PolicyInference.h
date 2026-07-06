#pragma once
#include "Models.h"

// Shared helpers for discrete masked-softmax policies
// Both learning algorithms (PPO and SAC) use the same policy structure:
//	an optional "shared_head" model feeding a "policy" model that outputs one logit per action,
//	turned into a categorical distribution with invalid actions masked out
// Keeping this logic in one place guarantees that saved policies behave identically
//	everywhere they are loaded (training, old-version play, skill matches, InferUnit)
namespace GGL::PolicyInference {

	// Probabilities are clamped to this minimum so log() can never produce -inf
	constexpr float ACTION_MIN_PROB = 1e-11f;
	constexpr float ACTION_DISABLED_LOGIT = -1e10f;

	// Makes the models of a discrete policy ("policy", plus "shared_head" if configured)
	void MakePolicyModels(
		int obsSize, int numActions,
		PartialModelConfig sharedHeadConfig, PartialModelConfig policyConfig,
		torch::Device device,
		ModelSet& outModels
	);

	// Returns the masked softmax action probabilities, shape [batch, numActions]
	// Invalid actions get (effectively) zero probability, all probs are clamped to at least ACTION_MIN_PROB
	torch::Tensor InferProbs(
		ModelSet& models,
		torch::Tensor obs, torch::Tensor actionMasks,
		float temperature,
		bool halfPrec
	);

	// Samples actions from the policy (or takes the argmax if deterministic)
	// outLogProbs is only written when non-null and not deterministic
	void InferActions(
		ModelSet& models,
		torch::Tensor obs, torch::Tensor actionMasks,
		bool deterministic, float temperature, bool halfPrec,
		torch::Tensor* outActions, torch::Tensor* outLogProbs
	);

	// Returns the mean normalized entropy of the given action probabilities (a scalar tensor)
	// If maskEntropy, entropy is normalized by each state's valid action count instead of the total
	torch::Tensor ComputeEntropy(torch::Tensor probs, torch::Tensor actionMasks, bool maskEntropy);
}
