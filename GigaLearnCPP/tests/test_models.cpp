#include "TestFramework.h"

#include <private/GigaLearnCPP/Util/Models.h>
#include <private/GigaLearnCPP/Util/PolicyInference.h>
#include <private/GigaLearnCPP/PPO/PPOLearner.h>

using namespace GGL;

static ModelConfig MakeTestConfig(int inputs, std::vector<int> layers, int outputs) {
	PartialModelConfig partial = {};
	partial.layerSizes = layers;
	partial.addLayerNorm = true;

	ModelConfig config = ModelConfig(partial);
	config.numInputs = inputs;
	config.numOutputs = outputs;
	return config;
}

TEST_CASE(Model_ForwardShape) {
	RG_NO_GRAD;

	auto model = Model("test_model", MakeTestConfig(6, { 16, 16 }, 4), torch::kCPU);

	auto input = torch::rand({ 10, 6 });
	auto output = model.Forward(input, false);

	CHECK_EQ(output.dim(), 2);
	CHECK_EQ(output.size(0), 10);
	CHECK_EQ(output.size(1), 4);

	// Deterministic: same input, same output
	auto output2 = model.Forward(input, false);
	CHECK_TRUE(torch::allclose(output, output2));
}

TEST_CASE(Model_SaveLoadRoundtrip) {
	RG_NO_GRAD;

	namespace fs = std::filesystem;
	auto saveFolder = fs::temp_directory_path() / "ggl_test_model_save";
	fs::remove_all(saveFolder);
	fs::create_directories(saveFolder);

	auto config = MakeTestConfig(8, { 12 }, 3);
	auto model = Model("test_model", config, torch::kCPU);
	model.Save(saveFolder);

	// A fresh model with the same config but different random weights
	auto loaded = Model("test_model", config, torch::kCPU);
	auto input = torch::rand({ 4, 8 });

	// (Different random init should give different outputs)
	CHECK_FALSE(torch::allclose(model.Forward(input, false), loaded.Forward(input, false)));

	loaded.Load(saveFolder, false);
	CHECK_TRUE(torch::allclose(model.Forward(input, false), loaded.Forward(input, false)));

	fs::remove_all(saveFolder);
}

TEST_CASE(Model_LoadRejectsWrongArchitecture) {
	RG_NO_GRAD;

	namespace fs = std::filesystem;
	auto saveFolder = fs::temp_directory_path() / "ggl_test_model_mismatch";
	fs::remove_all(saveFolder);
	fs::create_directories(saveFolder);

	auto model = Model("test_model", MakeTestConfig(8, { 12 }, 3), torch::kCPU);
	model.Save(saveFolder);

	// Different architecture must refuse to load
	auto loaded = Model("test_model", MakeTestConfig(8, { 24 }, 3), torch::kCPU);
	CHECK_THROWS(loaded.Load(saveFolder, false));

	fs::remove_all(saveFolder);
}

TEST_CASE(Model_CloneMatches) {
	RG_NO_GRAD;

	auto model = Model("test_model", MakeTestConfig(5, { 8, 8 }, 2), torch::kCPU);
	Model* clone = model.MakeClone();

	auto input = torch::rand({ 3, 5 });
	CHECK_TRUE(torch::allclose(model.Forward(input, false), clone->Forward(input, false)));

	delete clone;
}

TEST_CASE(Model_HalfPrecisionTracksFullPrecision) {
	RG_NO_GRAD;

	auto model = Model("test_model", MakeTestConfig(6, { 16 }, 4), torch::kCPU);
	auto input = torch::rand({ 8, 6 });

	// Half-precision (bfloat16) inference should approximate full precision
	auto full = model.Forward(input, false);
	auto half = model.Forward(input, true);
	CHECK_TRUE(torch::allclose(full, half, 0.1, 0.05));

	// After the parameters change, the half-precision copy must refresh
	//	(it is lazily rebuilt whenever the optimizer steps)
	for (auto& param : model.parameters())
		param += 0.5f;
	model._seqHalfOutdated = true;

	auto fullUpdated = model.Forward(input, false);
	auto halfUpdated = model.Forward(input, true);

	// The outputs must have changed, and half must track the change
	CHECK_FALSE(torch::allclose(fullUpdated, full, 0.1, 0.05));
	CHECK_TRUE(torch::allclose(fullUpdated, halfUpdated, 0.1, 0.05));
}

TEST_CASE(PolicyInference_ActionMaskingWorks) {
	RG_NO_GRAD;

	constexpr int OBS_SIZE = 4, NUM_ACTIONS = 6, NUM_STATES = 32;

	ModelSet models = {};
	PolicyInference::MakePolicyModels(
		OBS_SIZE, NUM_ACTIONS,
		{}, // No shared head
		PartialModelConfig{ .layerSizes = { 16, 16 } },
		torch::kCPU, models
	);

	auto obs = torch::rand({ NUM_STATES, OBS_SIZE });

	// Mask off actions 0 and 3 for all states
	auto masks = torch::ones({ NUM_STATES, NUM_ACTIONS }, torch::kUInt8);
	masks.index_put_({ torch::indexing::Slice(), 0 }, 0);
	masks.index_put_({ torch::indexing::Slice(), 3 }, 0);

	auto probs = PolicyInference::InferProbs(models, obs, masks, 1, false);

	CHECK_EQ(probs.size(0), NUM_STATES);
	CHECK_EQ(probs.size(1), NUM_ACTIONS);

	// Probabilities of masked actions must be (effectively) zero
	auto maxMaskedProb = RS_MAX(
		probs.index({ torch::indexing::Slice(), 0 }).max().item<float>(),
		probs.index({ torch::indexing::Slice(), 3 }).max().item<float>()
	);
	CHECK_TRUE(maxMaskedProb < 1e-9f);

	// Each row must still sum to ~1
	auto rowSums = probs.sum(-1);
	CHECK_TRUE(torch::allclose(rowSums, torch::ones({ NUM_STATES }), 1e-4, 1e-4));

	// Sampled actions must never be masked
	torch::Tensor actions, logProbs;
	PolicyInference::InferActions(models, obs, masks, false, 1, false, &actions, &logProbs);
	auto actionVec = TENSOR_TO_VEC<int>(actions);
	for (int action : actionVec) {
		CHECK_TRUE(action != 0 && action != 3);
		CHECK_TRUE(action >= 0 && action < NUM_ACTIONS);
	}

	// Log probs must be finite and negative
	auto logProbVec = TENSOR_TO_VEC<float>(logProbs);
	for (float logProb : logProbVec) {
		CHECK_TRUE(std::isfinite(logProb));
		CHECK_TRUE(logProb <= 0);
	}

	// Deterministic inference must pick the highest-probability valid action
	torch::Tensor detActions;
	PolicyInference::InferActions(models, obs, masks, true, 1, false, &detActions, NULL);
	auto detActionVec = TENSOR_TO_VEC<int>(detActions);
	auto argmax = TENSOR_TO_VEC<int>(probs.argmax(-1).to(torch::kInt32));
	for (int i = 0; i < NUM_STATES; i++)
		CHECK_EQ(detActionVec[i], argmax[i]);

	models.Free();
}

TEST_CASE(PPOLearner_RejectsBadBatchSizes) {
	// batchSize must be a multiple of miniBatchSize
	PPOLearnerConfig config = {};
	config.batchSize = 100;
	config.miniBatchSize = 33;

	CHECK_THROWS(PPOLearner(8, 4, config, torch::kCPU));
}
