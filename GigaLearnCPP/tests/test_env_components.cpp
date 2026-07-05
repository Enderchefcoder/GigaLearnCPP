#include "TestFramework.h"

#include <RLGymCPP/Gamestates/GameState.h>
#include <RLGymCPP/Gamestates/StateUtil.h>
#include <RLGymCPP/ObsBuilders/AdvancedObs.h>
#include <RLGymCPP/ObsBuilders/DefaultObsPadded.h>
#include <RLGymCPP/TerminalConditions/NoTouchCondition.h>
#include <RLGymCPP/StateSetters/CombinedState.h>

using namespace RLGC;

static GameState MakeState(int playersPerTeam) {
	GameState state = {};
	state.players.resize(playersPerTeam * 2);
	for (int i = 0; i < state.players.size(); i++) {
		auto& player = state.players[i];
		player.index = i;
		player.carId = i + 1;
		player.team = (i < playersPerTeam) ? Team::BLUE : Team::ORANGE;
		player.pos = Vec(100.f * i, -200.f * i, 17);
		player.rotMat = RotMat::GetIdentity();
		player.boost = 33;
	}
	state.ball.pos = Vec(500, 1000, 200);
	state.ball.vel = Vec(100, -50, 25);
	return state;
}

TEST_CASE(GameState_BoostPadPerspectives) {
	GameState state = MakeState(1);

	// Give the normal and inverted pad data distinct values
	state.boostPads.assign(CommonValues::BOOST_LOCATIONS_AMOUNT, true);
	state.boostPadsInv.assign(CommonValues::BOOST_LOCATIONS_AMOUNT, false);
	state.boostPadTimers.assign(CommonValues::BOOST_LOCATIONS_AMOUNT, 1.f);
	state.boostPadTimersInv.assign(CommonValues::BOOST_LOCATIONS_AMOUNT, 2.f);

	// Non-inverted perspective gets the normal data
	CHECK_EQ(state.GetBoostPads(false)[0], true);
	CHECK_EQ(state.GetBoostPadTimers(false)[0], 1.f);

	// Inverted perspective gets the inverted data
	// (Regression: GetBoostPadTimers used to return the wrong perspective)
	CHECK_EQ(state.GetBoostPads(true)[0], false);
	CHECK_EQ(state.GetBoostPadTimers(true)[0], 2.f);
}

TEST_CASE(InvertPhys_RoundTrip) {
	PhysState phys = {};
	phys.pos = Vec(100, -200, 300);
	phys.vel = Vec(-50, 60, -70);
	phys.angVel = Vec(1, -2, 3);
	phys.rotMat = Angle(0.5f, 0.25f, -0.75f).ToRotMat();

	// Inverting twice returns the original
	PhysState twice = InvertPhys(InvertPhys(phys, true), true);
	CHECK_NEAR(twice.pos.Dist(phys.pos), 0, 1e-4f);
	CHECK_NEAR(twice.vel.Dist(phys.vel), 0, 1e-4f);
	CHECK_NEAR(twice.angVel.Dist(phys.angVel), 0, 1e-4f);
	for (int i = 0; i < 3; i++)
		CHECK_NEAR(twice.rotMat[i].Dist(phys.rotMat[i]), 0, 1e-4f);

	// Inverting flips x and y but not z
	PhysState once = InvertPhys(phys, true);
	CHECK_EQ(once.pos.x, -phys.pos.x);
	CHECK_EQ(once.pos.y, -phys.pos.y);
	CHECK_EQ(once.pos.z, phys.pos.z);

	// Passing false is a no-op
	PhysState unchanged = InvertPhys(phys, false);
	CHECK_EQ(unchanged.pos.x, phys.pos.x);
}

TEST_CASE(AdvancedObs_ConsistentSizeAndMirroring) {
	AdvancedObs obs = {};

	auto state = MakeState(2);
	int obsSize = obs.BuildObs(state.players[0], state).size();
	CHECK_TRUE(obsSize > 0);

	// Same size for every player
	for (auto& player : state.players)
		CHECK_EQ(obs.BuildObs(player, state).size(), obsSize);

	// All values finite
	for (auto& player : state.players)
		for (float v : obs.BuildObs(player, state))
			CHECK_TRUE(std::isfinite(v));

	// A perfectly mirrored state must produce identical obs for mirrored players
	auto mirrorState = MakeState(1);
	auto& blue = mirrorState.players[0];
	auto& orange = mirrorState.players[1];
	blue.pos = Vec(300, -1000, 17);
	orange.pos = Vec(-300, 1000, 17);
	blue.rotMat = Angle(0.5f, 0, 0).ToRotMat();
	orange.rotMat = Angle(0.5f + M_PI, 0, 0).ToRotMat();
	blue.vel = Vec(100, 200, 0);
	orange.vel = Vec(-100, -200, 0);
	mirrorState.ball.pos = Vec(0, 0, 200);
	mirrorState.ball.vel = Vec(0, 0, 0);

	auto blueObs = obs.BuildObs(blue, mirrorState);
	auto orangeObs = obs.BuildObs(orange, mirrorState);
	CHECK_EQ(blueObs.size(), orangeObs.size());
	for (int i = 0; i < blueObs.size(); i++)
		CHECK_NEAR(blueObs[i], orangeObs[i], 1e-4f);
}

TEST_CASE(DefaultObsPadded_SizeInvariantAcrossTeamSizes) {
	constexpr int MAX_PLAYERS = 3;
	DefaultObsPadded obs = DefaultObsPadded(MAX_PLAYERS);

	int size1v1 = obs.BuildObs(MakeState(1).players[0], MakeState(1)).size();
	int size2v2 = obs.BuildObs(MakeState(2).players[0], MakeState(2)).size();
	int size3v3 = obs.BuildObs(MakeState(3).players[0], MakeState(3)).size();

	CHECK_EQ(size1v1, size2v2);
	CHECK_EQ(size2v2, size3v3);
}

TEST_CASE(NoTouchCondition_TruncatesAfterTimeout) {
	NoTouchCondition cond = NoTouchCondition(2.0f); // 2 second timeout
	CHECK_TRUE(cond.IsTruncation());

	auto state = MakeState(1);
	state.deltaTime = 0.5f;

	cond.Reset(state);

	// 3 steps of 0.5s without touches -> 1.5s, not yet terminal
	for (int i = 0; i < 3; i++)
		CHECK_FALSE(cond.IsTerminal(state));

	// A touch resets the timer
	state.players[0].ballTouchedStep = true;
	CHECK_FALSE(cond.IsTerminal(state));
	state.players[0].ballTouchedStep = false;

	// 4 steps of 0.5s = 2.0s -> terminal
	CHECK_FALSE(cond.IsTerminal(state));
	CHECK_FALSE(cond.IsTerminal(state));
	CHECK_FALSE(cond.IsTerminal(state));
	CHECK_TRUE(cond.IsTerminal(state));
}

TEST_CASE(CombinedState_WeightedSelection) {
	// Counting setters that don't touch the arena
	struct CountingSetter : StateSetter {
		int* counter;
		CountingSetter(int* counter) : counter(counter) {}
		void ResetArena(Arena* arena) override { (*counter)++; }
	};

	int countA = 0, countB = 0, countC = 0;

	CombinedState combined = CombinedState({
		{ new CountingSetter(&countA), 1.0f },
		{ new CountingSetter(&countB), 1.0f },
		{ new CountingSetter(&countC), 0.0f }, // Zero weight, should (almost) never run
	});

	constexpr int RUNS = 2000;
	for (int i = 0; i < RUNS; i++)
		combined.ResetArena(NULL);

	CHECK_EQ(countA + countB + countC, RUNS);

	// With equal weights, both A and B should get roughly half
	CHECK_TRUE(countA > RUNS / 4);
	CHECK_TRUE(countB > RUNS / 4);

	// C's zero weight means it can only be picked on an exact-boundary roll (~never)
	CHECK_TRUE(countC <= 1);
}
