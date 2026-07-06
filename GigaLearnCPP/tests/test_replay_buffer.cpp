#include "TestFramework.h"

#include <private/GigaLearnCPP/SAC/ReplayBuffer.h>

using namespace GGL;

// Makes a batch of transitions where every field of transition i is filled with (base + i),
//	so tests can identify exactly which transitions ended up where
static ReplayTransitions MakeTransitions(int64_t amount, int obsSize, int numActions, float base) {
	ReplayTransitions t = {};

	auto ar = torch::arange(base, base + amount, torch::kFloat32);

	t.states = ar.unsqueeze(-1).repeat({ 1, obsSize });
	t.actions = ar.to(torch::kInt64).remainder(numActions);
	t.rewards = ar.clone();
	t.nextStates = (ar + 0.5f).unsqueeze(-1).repeat({ 1, obsSize });
	t.actionMasks = torch::ones({ amount, numActions }, torch::kUInt8);
	t.nextActionMasks = torch::ones({ amount, numActions }, torch::kUInt8);
	t.dones = ar.remainder(2);

	return t;
}

TEST_CASE(ReplayBuffer_AppendAndSize) {
	ReplayBuffer buffer = ReplayBuffer(10, 3, 4);
	CHECK_EQ(buffer.Size(), 0);

	buffer.Append(MakeTransitions(4, 3, 4, 0));
	CHECK_EQ(buffer.Size(), 4);
	CHECK_EQ(buffer.totalAdded, 4);

	buffer.Append(MakeTransitions(4, 3, 4, 100));
	CHECK_EQ(buffer.Size(), 8);

	// Exceeding capacity caps the size but keeps counting totalAdded
	buffer.Append(MakeTransitions(4, 3, 4, 200));
	CHECK_EQ(buffer.Size(), 10);
	CHECK_EQ(buffer.totalAdded, 12);
}

TEST_CASE(ReplayBuffer_WraparoundOverwritesOldest) {
	constexpr int CAPACITY = 6;
	ReplayBuffer buffer = ReplayBuffer(CAPACITY, 2, 3);

	// 0..3, then 100..103: buffer holds 8 > 6, so rewards 0 and 1 must be gone
	buffer.Append(MakeTransitions(4, 2, 3, 0));
	buffer.Append(MakeTransitions(4, 2, 3, 100));

	auto rewards = TENSOR_TO_VEC<float>(buffer.data.rewards);
	std::set<float> held = std::set<float>(rewards.begin(), rewards.end());

	std::set<float> expected = { 2, 3, 100, 101, 102, 103 };
	CHECK_TRUE(held == expected);
}

TEST_CASE(ReplayBuffer_OversizedBatchKeepsNewest) {
	constexpr int CAPACITY = 4;
	ReplayBuffer buffer = ReplayBuffer(CAPACITY, 2, 3);

	// A single batch bigger than the whole buffer: only the newest transitions must remain
	buffer.Append(MakeTransitions(10, 2, 3, 0));
	CHECK_EQ(buffer.Size(), CAPACITY);

	auto rewards = TENSOR_TO_VEC<float>(buffer.data.rewards);
	std::set<float> held = std::set<float>(rewards.begin(), rewards.end());

	std::set<float> expected = { 6, 7, 8, 9 };
	CHECK_TRUE(held == expected);
}

TEST_CASE(ReplayBuffer_SampleShapesAndValues) {
	constexpr int OBS_SIZE = 3, NUM_ACTIONS = 5;
	ReplayBuffer buffer = ReplayBuffer(64, OBS_SIZE, NUM_ACTIONS);
	buffer.Append(MakeTransitions(20, OBS_SIZE, NUM_ACTIONS, 0));

	auto sample = buffer.Sample(32);

	CHECK_EQ(sample.Size(), 32);
	CHECK_EQ(sample.states.size(1), OBS_SIZE);
	CHECK_EQ(sample.nextStates.size(1), OBS_SIZE);
	CHECK_EQ(sample.actionMasks.size(1), NUM_ACTIONS);
	CHECK_EQ(sample.nextActionMasks.size(1), NUM_ACTIONS);
	CHECK_TRUE(sample.actions.dtype() == torch::kInt64);

	// Every sampled transition must be one that was inserted, with its fields still aligned
	auto rewards = TENSOR_TO_VEC<float>(sample.rewards);
	auto states = sample.states;
	auto nextStates = sample.nextStates;
	auto dones = TENSOR_TO_VEC<float>(sample.dones);
	for (int i = 0; i < 32; i++) {
		float reward = rewards[i];
		CHECK_TRUE(reward >= 0 && reward < 20);

		// All fields of transition k are derived from k (see MakeTransitions)
		CHECK_NEAR(states[i][0].item<float>(), reward, 1e-6f);
		CHECK_NEAR(nextStates[i][0].item<float>(), reward + 0.5f, 1e-6f);
		CHECK_NEAR(dones[i], (float)((int)reward % 2), 1e-6f);
	}

	// Sampling never returns rows beyond Size() (the unwritten region is all zeros;
	//	a zero row would have nextStates[0] == 0 instead of reward + 0.5)
}

TEST_CASE(ReplayBuffer_SampleEmptyThrows) {
	ReplayBuffer buffer = ReplayBuffer(16, 2, 3);
	CHECK_THROWS(buffer.Sample(4));
}

TEST_CASE(ReplayBuffer_RejectsMismatchedBatch) {
	ReplayBuffer buffer = ReplayBuffer(16, 2, 3);

	// Wrong obs size must be rejected
	auto bad = MakeTransitions(4, 5, 3, 0);
	CHECK_THROWS(buffer.Append(bad));
}
