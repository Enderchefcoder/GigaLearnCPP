#pragma once
#include "../FrameworkTorch.h"

namespace GGL {

	// A batch of SAC transitions (first dim = transition index)
	// dones is 1 only for real episode ends (not truncations, which bootstrap from their next state)
	struct ReplayTransitions {
		torch::Tensor
			states,          // [n, obsSize] float32
			actions,         // [n] int64 (gather() indices)
			rewards,         // [n] float32
			nextStates,      // [n, obsSize] float32
			actionMasks,     // [n, numActions] uint8
			nextActionMasks, // [n, numActions] uint8
			dones;           // [n] float32

		auto begin() { return &states; }
		auto end() { return &dones + 1; }
		auto begin() const { return &states; }
		auto end() const { return &dones + 1; }

		int64_t Size() const {
			return states.defined() ? states.size(0) : 0;
		}
	};

	// Fixed-capacity circular replay buffer for off-policy learning (SAC)
	// Storage always lives on the CPU; sampled batches are moved to the learn device by the caller
	class ReplayBuffer {
	public:
		int64_t capacity;
		int obsSize, numActions;

		// Total transitions ever added (can exceed capacity; oldest get overwritten)
		int64_t totalAdded = 0;
		int64_t nextIdx = 0;

		// Preallocated [capacity, ...] storage
		ReplayTransitions data;

		ReplayBuffer(int64_t capacity, int obsSize, int numActions);

		int64_t Size() const {
			return RS_MIN(totalAdded, capacity);
		}

		// Appends a batch of transitions, overwriting the oldest data once full
		void Append(const ReplayTransitions& transitions);

		// Uniformly samples transitions (with replacement), using torch's global RNG
		ReplayTransitions Sample(int64_t amount) const;
	};
}
