#pragma once

#include <rlbot/bot.h>
#include <RLGymCPP/ObsBuilders/ObsBuilder.h>
#include <RLGymCPP/ActionParsers/ActionParser.h>
#include <GigaLearnCPP/Util/InferUnit.h>

struct RLBotParams {
	// Set this to the same port used in rlbot/port.cfg
	int port;

	int tickSkip; // Your tick skip
	int actionDelay; // Your action delay

	GGL::InferUnit* inferUnit = NULL;
};

class RLBotBot : public rlbot::Bot {
public:

	// Parameters to define the bot
	RLBotParams params;

	// Queued action and current action
	RLGC::Action
		action = {}, 
		controls = {};

	// Persistent info
	bool updateAction = true;
	float prevTime = 0;
	int ticks = -1;

	// Maps RLBot's boost pad ordering to RLGymCPP's (RLBot's field info pad order
	//	does not match CommonValues::BOOST_LOCATIONS)
	// Empty if the mapping hasn't been built or the arena's pads don't match
	std::vector<int> boostPadMap = {};
	bool triedBuildingBoostPadMap = false;

	RLBotBot(int _index, int _team, std::string _name, const RLBotParams& params);
	~RLBotBot();

	rlbot::Controller GetOutput(rlbot::GameTickPacket gameTickPacket) override;
};

namespace RLBotClient {
	void Run(const RLBotParams& params);
}