#include "TestFramework.h"

#include <RLGymCPP/ActionParsers/DefaultAction.h>

using namespace RLGC;

static Player MakePlayer(bool onGround, float boost, bool hasFlipOrJump) {
	Player player = {};
	player.isOnGround = onGround;
	player.boost = boost;

	// Configure jump/flip state
	if (hasFlipOrJump) {
		player.hasJumped = false;
		player.hasFlipped = false;
		player.hasDoubleJumped = false;
	} else {
		player.hasJumped = true;
		player.hasFlipped = true;
		player.hasDoubleJumped = true;
		player.airTime = 10;
		player.airTimeSinceJump = 10;
	}

	player.worldContact.hasContact = false;
	return player;
}

TEST_CASE(DefaultAction_MaskAlwaysHasValidAction) {
	DefaultAction parser = {};
	GameState state = {};

	for (bool onGround : { true, false }) {
		for (float boost : { 0.f, 50.f }) {
			for (bool hasFlip : { true, false }) {
				auto player = MakePlayer(onGround, boost, hasFlip);
				auto mask = parser.GetActionMask(player, state);
				CHECK_EQ(mask.size(), parser.GetActionAmount());

				int numValid = 0;
				for (uint8_t v : mask)
					numValid += (v != 0);
				CHECK_TRUE(numValid > 0);
			}
		}
	}
}

TEST_CASE(DefaultAction_NoBoostActionsWithoutBoost) {
	DefaultAction parser = {};
	GameState state = {};

	for (bool onGround : { true, false }) {
		for (bool hasFlip : { true, false }) {
			auto player = MakePlayer(onGround, 0, hasFlip);
			auto mask = parser.GetActionMask(player, state);

			// No available action may use boost, since the player has none
			// (This includes jump actions, which used to be re-enabled after the boost mask was applied)
			for (int i = 0; i < parser.actions.size(); i++)
				if (mask[i])
					CHECK_EQ(parser.actions[i].boost, 0.f);
		}
	}
}

TEST_CASE(DefaultAction_JumpActionsRequireFlipOrJump) {
	DefaultAction parser = {};
	GameState state = {};

	// Without a flip/jump available (and not turtled), no jump actions should be available
	{
		auto player = MakePlayer(false, 100, false);
		auto mask = parser.GetActionMask(player, state);
		for (int i = 0; i < parser.actions.size(); i++)
			if (mask[i])
				CHECK_EQ(parser.actions[i].jump, 0.f);
	}

	// With a flip/jump available, at least one jump action should be available
	{
		auto player = MakePlayer(false, 100, true);
		auto mask = parser.GetActionMask(player, state);
		int numJumpActions = 0;
		for (int i = 0; i < parser.actions.size(); i++)
			if (mask[i] && parser.actions[i].jump == 1)
				numJumpActions++;
		CHECK_TRUE(numJumpActions > 0);
	}
}

// Regression test: an off-by-one used to permanently mask the first aerial action
TEST_CASE(DefaultAction_NoPermanentlyDeadActions) {
	DefaultAction parser = {};
	GameState state = {};

	int numActions = parser.GetActionAmount();
	std::vector<bool> everAvailable(numActions, false);

	for (bool onGround : { true, false }) {
		for (bool hasFlip : { true, false }) {
			auto player = MakePlayer(onGround, 100, hasFlip);
			auto mask = parser.GetActionMask(player, state);
			for (int i = 0; i < numActions; i++)
				if (mask[i])
					everAvailable[i] = true;
		}
	}

	for (int i = 0; i < numActions; i++) {
		if (!everAvailable[i]) {
			_TEST_FAIL(
				"Action " << i << " (" << parser.actions[i] << ") is not available in any tested state"
			);
		}
	}
}

TEST_CASE(DefaultAction_GroundActionsOnGround) {
	DefaultAction parser = {};
	GameState state = {};

	auto player = MakePlayer(true, 100, true);
	auto mask = parser.GetActionMask(player, state);

	// All full-throttle non-jump ground actions should be available
	// (Ground actions have pitch == 0 and roll == 0)
	for (int i = 0; i < parser.actions.size(); i++) {
		auto& action = parser.actions[i];
		if (action.throttle == 1 && action.pitch == 0 && action.roll == 0 && action.jump == 0)
			CHECK_TRUE(mask[i]);
	}
}

TEST_CASE(DefaultAction_NoDuplicateActions) {
	DefaultAction parser = {};

	for (int i = 0; i < parser.actions.size(); i++) {
		for (int j = i + 1; j < parser.actions.size(); j++) {
			auto& a = parser.actions[i];
			auto& b = parser.actions[j];

			bool identical = true;
			for (int k = 0; k < Action::ELEM_AMOUNT; k++)
				identical &= (a[k] == b[k]);

			if (identical)
				_TEST_FAIL("Actions " << i << " and " << j << " are identical: " << a);
		}
	}
}
