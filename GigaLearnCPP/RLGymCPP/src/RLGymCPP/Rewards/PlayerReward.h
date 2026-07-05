#pragma once
#include "Reward.h"

// https://github.com/AechPro/rocket-league-gym-sim/blob/main/rlgym_sim/utils/reward_functions/reward_function.py
namespace RLGC {
	// Maintains a separate instance of a reward for each player
	// Useful for rewards that track per-player state internally
	// T must be default-constructible
	template<typename T>
	class PlayerReward : public Reward {
	private:
		std::vector<T*> _instances;
		
	public:
		virtual void Reset(const GameState& initialState) override {
			if (_instances.size() != initialState.players.size()) {
				for (auto inst : _instances)
					delete inst;
				_instances.clear();

				// Generate instances
				for (int i = 0; i < initialState.players.size(); i++)
					_instances.push_back(new T());
			}

			for (auto inst : _instances)
				inst->Reset(initialState);
		}

		virtual void PreStep(const GameState& state) override {
			for (auto inst : _instances)
				inst->PreStep(state);
		}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			return _instances[player.index]->GetReward(player, state, isFinal);
		}

		virtual std::string GetName() override {
			// Use the name of the inner reward type
			static T nameInstance = {};
			return nameInstance.GetName();
		}

		virtual ~PlayerReward() {
			for (auto inst : _instances)
				delete inst;
		};
	};
}
