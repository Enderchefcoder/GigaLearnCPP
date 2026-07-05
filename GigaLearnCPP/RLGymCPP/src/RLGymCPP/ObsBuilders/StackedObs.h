#pragma once
#include "ObsBuilder.h"

#include <deque>

namespace RLGC {
	// Wraps another obs builder and stacks the last N observations per player
	//	(newest first), giving the policy temporal context (velocity changes,
	//	dodge timings, etc.) without recurrence.
	// The obs size becomes numStacks * childObsSize.
	// NOTE: Takes ownership of the child obs builder
	// NOTE: Avoid wrapping obs builders that randomize their layout per call
	//	(e.g. DefaultObsPadded's slot shuffling), as slots won't correspond
	//	across the stacked frames
	class StackedObs : public ObsBuilder {
	public:
		ObsBuilder* child;
		int numStacks;

		// History of previous obs per player (car ID), newest at the front
		std::map<uint32_t, std::deque<FList>> playerHistories;

		StackedObs(ObsBuilder* child, int numStacks) : child(child), numStacks(numStacks) {
			RG_ASSERT(child != NULL);
			RG_ASSERT(numStacks >= 1);
		}

		virtual void Reset(const GameState& initialState) override {
			child->Reset(initialState);

			// Histories re-seed lazily on the first BuildObs() of the new episode
			playerHistories.clear();
		}

		virtual FList BuildObs(const Player& player, const GameState& state) override {
			FList curObs = child->BuildObs(player, state);

			auto& history = playerHistories[player.carId];

			// At the start of an episode, seed the history by repeating the current obs
			//	(the standard approach for frame stacking)
			if (history.empty())
				history.assign(numStacks - 1, curObs);

			FList result = {};
			result.reserve(curObs.size() * numStacks);
			result += curObs;
			for (auto& prevObs : history)
				result += prevObs;

			if (numStacks > 1) {
				history.push_front(curObs);
				history.pop_back();
			}

			return result;
		}

		virtual ~StackedObs() {
			delete child;
		}
	};
}
