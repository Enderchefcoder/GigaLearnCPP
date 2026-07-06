#include "ReplayBuffer.h"

GGL::ReplayBuffer::ReplayBuffer(int64_t capacity, int obsSize, int numActions)
	: capacity(capacity), obsSize(obsSize), numActions(numActions) {

	RG_ASSERT(capacity > 0 && obsSize > 0 && numActions > 0);

	// Preallocate all storage up front so we fail fast if the buffer doesn't fit in memory
	data.states = torch::zeros({ capacity, obsSize }, torch::kFloat32);
	data.actions = torch::zeros({ capacity }, torch::kInt64);
	data.rewards = torch::zeros({ capacity }, torch::kFloat32);
	data.nextStates = torch::zeros({ capacity, obsSize }, torch::kFloat32);
	data.actionMasks = torch::zeros({ capacity, numActions }, torch::kUInt8);
	data.nextActionMasks = torch::zeros({ capacity, numActions }, torch::kUInt8);
	data.dones = torch::zeros({ capacity }, torch::kFloat32);
}

void GGL::ReplayBuffer::Append(const ReplayTransitions& transitions) {
	RG_NO_GRAD;

	int64_t amount = transitions.Size();
	if (amount == 0)
		return;

	{ // Validate incoming batch
		RG_ASSERT(transitions.states.size(0) == amount);
		RG_ASSERT(transitions.states.size(1) == obsSize);
		RG_ASSERT(transitions.nextStates.size(0) == amount && transitions.nextStates.size(1) == obsSize);
		RG_ASSERT(transitions.actionMasks.size(0) == amount && transitions.actionMasks.size(1) == numActions);
		RG_ASSERT(transitions.nextActionMasks.size(0) == amount && transitions.nextActionMasks.size(1) == numActions);
		RG_ASSERT(transitions.actions.size(0) == amount);
		RG_ASSERT(transitions.rewards.size(0) == amount);
		RG_ASSERT(transitions.dones.size(0) == amount);
	}

	int64_t srcStart = 0;
	if (amount > capacity) {
		// The batch alone exceeds the whole buffer; only the newest transitions fit
		srcStart = amount - capacity;
		amount = capacity;
	}

	// Copy in up to two contiguous chunks (the second one wraps around to the start)
	int64_t written = 0;
	while (written < amount) {
		int64_t chunk = RS_MIN(amount - written, capacity - nextIdx);

		auto* toItr = data.begin();
		auto* fromItr = transitions.begin();
		for (; toItr != data.end(); toItr++, fromItr++) {
			toItr->slice(0, nextIdx, nextIdx + chunk)
				.copy_(fromItr->slice(0, srcStart + written, srcStart + written + chunk));
		}

		written += chunk;
		nextIdx = (nextIdx + chunk) % capacity;
	}

	totalAdded += amount;
}

GGL::ReplayTransitions GGL::ReplayBuffer::Sample(int64_t amount) const {
	RG_NO_GRAD;

	int64_t curSize = Size();
	if (curSize == 0)
		RG_ERR_CLOSE("ReplayBuffer::Sample(): Cannot sample from an empty buffer");

	torch::Tensor tIndices = torch::randint(curSize, { amount }, torch::kInt64);

	ReplayTransitions result = {};
	auto* toItr = result.begin();
	auto* fromItr = data.begin();
	for (; toItr != result.end(); toItr++, fromItr++)
		*toItr = torch::index_select(*fromItr, 0, tIndices);

	return result;
}
