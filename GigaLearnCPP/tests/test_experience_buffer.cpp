#include "TestFramework.h"

#include <private/GigaLearnCPP/PPO/ExperienceBuffer.h>

using namespace GGL;

// Fills an experience buffer where every field of sample i has the value i
static void FillArangeExperience(ExperienceBuffer& buffer, int64_t numSamples) {
	auto arange = torch::arange(numSamples, torch::kFloat32);
	buffer.data.states = arange.clone();
	buffer.data.actions = arange.clone();
	buffer.data.logProbs = arange.clone();
	buffer.data.targetValues = arange.clone();
	buffer.data.actionMasks = arange.clone();
	buffer.data.advantages = arange.clone();
}

TEST_CASE(ExperienceBuffer_BatchSizesExact) {
	constexpr int64_t NUM_SAMPLES = 12, BATCH_SIZE = 4;

	ExperienceBuffer buffer = ExperienceBuffer(123, torch::kCPU);
	FillArangeExperience(buffer, NUM_SAMPLES);

	auto batches = buffer.GetAllBatchesShuffled(BATCH_SIZE, false);
	CHECK_EQ(batches.size(), 3);
	for (auto& batch : batches)
		CHECK_EQ(batch.states.size(0), BATCH_SIZE);
}

TEST_CASE(ExperienceBuffer_Overbatching) {
	constexpr int64_t NUM_SAMPLES = 14, BATCH_SIZE = 4;

	ExperienceBuffer buffer = ExperienceBuffer(123, torch::kCPU);
	FillArangeExperience(buffer, NUM_SAMPLES);

	// Without overbatching: 3 batches of 4 (2 samples dropped)
	{
		auto batches = buffer.GetAllBatchesShuffled(BATCH_SIZE, false);
		CHECK_EQ(batches.size(), 3);
		int64_t total = 0;
		for (auto& batch : batches)
			total += batch.states.size(0);
		CHECK_EQ(total, 12);
	}

	// With overbatching: last batch is extended to cover all samples
	{
		auto batches = buffer.GetAllBatchesShuffled(BATCH_SIZE, true);
		CHECK_EQ(batches.size(), 3);
		CHECK_EQ(batches[0].states.size(0), BATCH_SIZE);
		CHECK_EQ(batches[1].states.size(0), BATCH_SIZE);
		CHECK_EQ(batches[2].states.size(0), BATCH_SIZE + 2);
	}
}

TEST_CASE(ExperienceBuffer_ShuffleCoversAllSamplesOnce) {
	constexpr int64_t NUM_SAMPLES = 32, BATCH_SIZE = 8;

	ExperienceBuffer buffer = ExperienceBuffer(456, torch::kCPU);
	FillArangeExperience(buffer, NUM_SAMPLES);

	auto batches = buffer.GetAllBatchesShuffled(BATCH_SIZE, true);

	std::vector<int> seenCounts(NUM_SAMPLES, 0);
	bool anyShuffled = false;
	for (auto& batch : batches) {
		auto states = TENSOR_TO_VEC<float>(batch.states);
		for (int i = 0; i < states.size(); i++) {
			int sampleIdx = (int)states[i];
			CHECK_TRUE(sampleIdx >= 0 && sampleIdx < NUM_SAMPLES);
			seenCounts[sampleIdx]++;
			if (sampleIdx != i)
				anyShuffled = true;
		}
	}

	// Every sample appears exactly once
	for (int count : seenCounts)
		CHECK_EQ(count, 1);

	// And the order isn't just 0,1,2,... (astronomically unlikely with a working shuffle)
	CHECK_TRUE(anyShuffled);
}

TEST_CASE(ExperienceBuffer_FieldsStayAligned) {
	constexpr int64_t NUM_SAMPLES = 16, BATCH_SIZE = 4;

	ExperienceBuffer buffer = ExperienceBuffer(789, torch::kCPU);
	FillArangeExperience(buffer, NUM_SAMPLES);

	auto batches = buffer.GetAllBatchesShuffled(BATCH_SIZE, true);

	// Sample i has value i in every field, so after shuffling,
	//	all fields of a given row must still hold the same value
	for (auto& batch : batches) {
		auto states = TENSOR_TO_VEC<float>(batch.states);
		auto actions = TENSOR_TO_VEC<float>(batch.actions);
		auto logProbs = TENSOR_TO_VEC<float>(batch.logProbs);
		auto advantages = TENSOR_TO_VEC<float>(batch.advantages);

		for (int i = 0; i < states.size(); i++) {
			CHECK_EQ(states[i], actions[i]);
			CHECK_EQ(states[i], logProbs[i]);
			CHECK_EQ(states[i], advantages[i]);
		}
	}
}

TEST_CASE(ExperienceBuffer_RejectsTooSmallExperience) {
	ExperienceBuffer buffer = ExperienceBuffer(1, torch::kCPU);
	FillArangeExperience(buffer, 4);

	CHECK_THROWS(buffer.GetAllBatchesShuffled(10, true));
}
