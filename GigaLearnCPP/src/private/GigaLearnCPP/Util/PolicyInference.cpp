#include "PolicyInference.h"

void GGL::PolicyInference::MakePolicyModels(
	int obsSize, int numActions,
	PartialModelConfig sharedHeadConfig, PartialModelConfig policyConfig,
	torch::Device device,
	ModelSet& outModels) {

	ModelConfig fullPolicyConfig = policyConfig;
	fullPolicyConfig.numInputs = obsSize;
	fullPolicyConfig.numOutputs = numActions;

	if (sharedHeadConfig.IsValid()) {
		ModelConfig fullSharedHeadConfig = sharedHeadConfig;
		fullSharedHeadConfig.numInputs = obsSize;
		fullSharedHeadConfig.numOutputs = 0;

		RG_ASSERT(!sharedHeadConfig.addOutputLayer);

		fullPolicyConfig.numInputs = fullSharedHeadConfig.layerSizes.back();

		outModels.Add(new Model("shared_head", fullSharedHeadConfig, device));
	}

	outModels.Add(new Model("policy", fullPolicyConfig, device));
}

torch::Tensor GGL::PolicyInference::InferProbs(
	ModelSet& models,
	torch::Tensor obs, torch::Tensor actionMasks,
	float temperature, bool halfPrec) {

	actionMasks = actionMasks.to(torch::kBool);

	if (models["shared_head"])
		obs = models["shared_head"]->Forward(obs, halfPrec);

	auto logits = models["policy"]->Forward(obs, halfPrec) / temperature;

	auto result = torch::softmax(logits.masked_fill(actionMasks.logical_not(), ACTION_DISABLED_LOGIT), -1);
	return result.view({ -1, models["policy"]->config.numOutputs }).clamp(ACTION_MIN_PROB, 1);
}

void GGL::PolicyInference::InferActions(
	ModelSet& models,
	torch::Tensor obs, torch::Tensor actionMasks,
	bool deterministic, float temperature, bool halfPrec,
	torch::Tensor* outActions, torch::Tensor* outLogProbs) {

	auto probs = InferProbs(models, obs, actionMasks, temperature, halfPrec);

	if (deterministic) {
		auto action = probs.argmax(1);
		if (outActions)
			*outActions = action.flatten();
	} else {
		auto action = torch::multinomial(probs, 1, true);
		if (outActions)
			*outActions = action.flatten();

		if (outLogProbs) {
			// Gather before log so we only compute log() on the selected actions
			auto logProb = probs.gather(-1, action).log();
			*outLogProbs = logProb.flatten();
		}
	}
}

torch::Tensor GGL::PolicyInference::ComputeEntropy(torch::Tensor probs, torch::Tensor actionMasks, bool maskEntropy) {
	// Compute log probs and entropy
	auto entropy = -(probs.log() * probs).sum(-1);

	if (maskEntropy) {
		// Account for action masking in entropy
		// We will effectively narrow the entropy to the scope of the valid actions
		// This way states with more masked actions don't just have inherently lower entropy
		// NOTE: Clamped to a minimum of 2 valid actions,
		//	otherwise a state with 1 valid action would divide by log(1) = 0
		//	(the entropy of such a state is always 0 anyway)
		auto numValidActions = actionMasks.to(torch::kFloat32).sum(-1).clamp_min(2);
		entropy /= numValidActions.log();
	} else {
		entropy /= logf(actionMasks.size(-1));
	}

	return entropy.mean();
}
