#include "TestFramework.h"

#include <RLGymCPP/BasicTypes/Lists.h>
#include <RLGymCPP/Rewards/ZeroSumReward.h>
#include <RLGymCPP/Rewards/CommonRewards.h>

using namespace RLGC;

// Test reward that returns a fixed value per player index
class FixedReward : public Reward {
public:
	FList values;
	FixedReward(const FList& values) : values(values) {}

	virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
		return values[player.index];
	}
};

static GameState MakeState2v2() {
	GameState state = {};
	state.players.resize(4);
	for (int i = 0; i < 4; i++) {
		state.players[i].index = i;
		state.players[i].carId = i + 1;
		state.players[i].team = (i < 2) ? Team::BLUE : Team::ORANGE;
	}
	return state;
}

TEST_CASE(ZeroSumReward_FullySelfish) {
	auto state = MakeState2v2();

	// teamSpirit = 0: own reward minus average opponent reward
	Reward* zeroSum = new ZeroSumReward(new FixedReward({ 1, 3, 5, 7 }), 0, 1);

	auto rewards = zeroSum->GetAllRewards(state, false);

	float blueAvg = (1 + 3) / 2.f, orangeAvg = (5 + 7) / 2.f;
	CHECK_NEAR(rewards[0], 1 - orangeAvg, 1e-6f);
	CHECK_NEAR(rewards[1], 3 - orangeAvg, 1e-6f);
	CHECK_NEAR(rewards[2], 5 - blueAvg, 1e-6f);
	CHECK_NEAR(rewards[3], 7 - blueAvg, 1e-6f);

	delete zeroSum;
}

TEST_CASE(ZeroSumReward_TeamSpirit) {
	auto state = MakeState2v2();

	constexpr float TEAM_SPIRIT = 0.3f;
	Reward* zeroSum = new ZeroSumReward(new FixedReward({ 1, 3, 5, 7 }), TEAM_SPIRIT, 1);

	auto rewards = zeroSum->GetAllRewards(state, false);

	float blueAvg = 2, orangeAvg = 6;
	CHECK_NEAR(rewards[0], 1 * (1 - TEAM_SPIRIT) + blueAvg * TEAM_SPIRIT - orangeAvg, 1e-6f);
	CHECK_NEAR(rewards[3], 7 * (1 - TEAM_SPIRIT) + orangeAvg * TEAM_SPIRIT - blueAvg, 1e-6f);

	delete zeroSum;
}

TEST_CASE(ZeroSumReward_SumIsZero) {
	auto state = MakeState2v2();

	Reward* zeroSum = new ZeroSumReward(new FixedReward({ 2, -4, 8, 1.5f }), 0.7f, 1);
	auto rewards = zeroSum->GetAllRewards(state, false);

	float sum = 0;
	for (float r : rewards)
		sum += r;
	CHECK_NEAR(sum, 0, 1e-5f);

	delete zeroSum;
}

TEST_CASE(Reward_GetNameIsReadable) {
	// GetName() must return a human-readable class name on all compilers
	//	(typeid names are mangled on GCC/Clang and must be demangled)
	AirReward airReward = {};
	CHECK_EQ(airReward.GetName(), "AirReward");

	VelocityPlayerToBallReward velReward = {};
	CHECK_EQ(velReward.GetName(), "VelocityPlayerToBallReward");

	// Event rewards are templates and use explicit names
	PlayerGoalReward goalReward = {};
	CHECK_EQ(goalReward.GetName(), "PlayerGoalReward");

	DemoedPenalty demoedPenalty = {};
	CHECK_EQ(demoedPenalty.GetName(), "DemoedPenalty");

	// Wrappers inherit the name of their child
	Reward* wrapped = new ZeroSumReward(new AirReward(), 0.5f);
	CHECK_EQ(wrapped->GetName(), "AirReward");
	delete wrapped;
}

TEST_CASE(CommonRewards_BasicValues) {
	auto state = MakeState2v2();
	state.ball.pos = Vec(0, 0, CommonValues::BALL_RADIUS);

	auto& player = state.players[0];
	player.pos = Vec(0, -1000, 17);
	player.vel = Vec(0, CommonValues::CAR_MAX_SPEED, 0); // Driving straight at the ball at max speed
	player.rotMat = Angle(M_PI / 2, 0, 0).ToRotMat(); // Facing the ball
	player.isOnGround = true;

	{
		VelocityPlayerToBallReward reward = {};
		float value = reward.GetReward(player, state, false);
		CHECK_NEAR(value, 1, 0.01f);
	}

	{
		FaceBallReward reward = {};
		float value = reward.GetReward(player, state, false);
		CHECK_NEAR(value, 1, 0.01f);
	}

	{
		AirReward reward = {};
		CHECK_NEAR(reward.GetReward(player, state, false), 0, 1e-6f);
		player.isOnGround = false;
		CHECK_NEAR(reward.GetReward(player, state, false), 1, 1e-6f);
		player.isOnGround = true;
	}

	{
		// SaveBoostReward is sqrt(boost fraction)
		SaveBoostReward reward = {};
		player.boost = 25;
		CHECK_NEAR(reward.GetReward(player, state, false), 0.5f, 1e-4f);
		player.boost = 100;
		CHECK_NEAR(reward.GetReward(player, state, false), 1.f, 1e-4f);
	}
}
