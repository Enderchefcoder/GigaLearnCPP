#include "RLBotClient.h"

#include <rlbot/platform.h>
#include <rlbot/botmanager.h>

using namespace RLGC;
using namespace GGL;

// Global variable so that we can pass params to the bot factory
// TODO: This is a lame solution
RLBotParams g_RLBotParams = {};

rlbot::Bot* BotFactory(int index, int team, std::string name) {
	return new RLBotBot(index, team, name, g_RLBotParams);
}

RLBotBot::RLBotBot(int _index, int _team, std::string _name, const RLBotParams& params) 
	: rlbot::Bot(_index, _team, _name), params(params) {

	RG_LOG("Created RLBot bot: index " << _index << ", name: " << name << "...");
}

RLBotBot::~RLBotBot() {
	// NOTE: The InferUnit is shared between all bots (it lives in g_RLBotParams),
	//	so it must not be deleted here (other bots may still be using it)
}

Vec ToVec(const rlbot::flat::Vector3* rlbotVec) {
	return Vec(rlbotVec->x(), rlbotVec->y(), rlbotVec->z());
}

PhysState ToPhysObj(const rlbot::flat::Physics* phys) {
	PhysState obj = {};
	obj.pos = ToVec(phys->location());

	Angle ang = Angle(phys->rotation()->yaw(), phys->rotation()->pitch(), phys->rotation()->roll());
	obj.rotMat = ang.ToRotMat();

	obj.vel = ToVec(phys->velocity());
	obj.angVel = ToVec(phys->angularVelocity());

	return obj;
}

Player ToPlayer(const rlbot::flat::PlayerInfo* playerInfo) {
	Player pd = {};
	
	static_cast<PhysState&>(pd) = ToPhysObj(playerInfo->physics());

	pd.carId = playerInfo->spawnId();

	pd.team = (Team)playerInfo->team();

	pd.boost = playerInfo->boost();
	pd.isOnGround = playerInfo->hasWheelContact();
	pd.hasJumped = playerInfo->jumped();
	pd.hasDoubleJumped = playerInfo->doubleJumped();
	pd.isDemoed = playerInfo->isDemolished();

	return pd;
}

// Builds a mapping from RLBot's boost pad ordering to RLGymCPP's ordering
//	(they don't match, so pads must be matched by position)
// Returns an empty vector if the arena's pads don't match the standard layout
static std::vector<int> BuildBoostPadMap(const rlbot::FieldInfo& fieldInfo) {
	constexpr float MAX_MATCH_DIST_SQ = 100 * 100;

	auto pads = fieldInfo->boostPads();
	if (!pads || pads->size() != CommonValues::BOOST_LOCATIONS_AMOUNT)
		return {};

	std::vector<int> map = std::vector<int>(pads->size(), -1);
	std::vector<bool> used = std::vector<bool>(CommonValues::BOOST_LOCATIONS_AMOUNT, false);

	for (int i = 0; i < pads->size(); i++) {
		Vec padPos = ToVec(pads->Get(i)->location());

		int bestIdx = -1;
		float bestDistSq = MAX_MATCH_DIST_SQ;
		for (int j = 0; j < CommonValues::BOOST_LOCATIONS_AMOUNT; j++) {
			if (used[j])
				continue;

			float distSq = padPos.DistSq2D(CommonValues::BOOST_LOCATIONS[j]);
			if (distSq < bestDistSq) {
				bestDistSq = distSq;
				bestIdx = j;
			}
		}

		if (bestIdx == -1)
			return {}; // Not a standard soccar pad layout

		map[i] = bestIdx;
		used[bestIdx] = true;
	}

	return map;
}

GameState ToGameState(rlbot::GameTickPacket& gameTickPacket, const std::vector<int>& boostPadMap) {
	GameState gs = {};

	auto players = gameTickPacket->players();
	for (int i = 0; i < players->size(); i++)
		gs.players.push_back(ToPlayer(players->Get(i)));

	static_cast<PhysState&>(gs.ball) = ToPhysObj(gameTickPacket->ball()->physics());

	auto boostPadStates = gameTickPacket->boostPadStates();
	if (boostPadStates->size() != CommonValues::BOOST_LOCATIONS_AMOUNT || boostPadMap.empty()) {
		if (rand() % 20 == 0) { // Don't spam-log as that will lag the bot
			RG_LOG(
				"RLBotClient ToGameState(): Bad boost pad amount or non-standard pad layout " <<
				"(expected " << CommonValues::BOOST_LOCATIONS_AMOUNT << ", got " << boostPadStates->size() << ")"
			);
		}

		// Just set all boost pads to on
		std::fill(gs.boostPads.begin(), gs.boostPads.end(), 1);
		std::fill(gs.boostPadsInv.begin(), gs.boostPadsInv.end(), 1);
	} else {
		for (int i = 0; i < CommonValues::BOOST_LOCATIONS_AMOUNT; i++) {
			int rsIdx = boostPadMap[i];
			int rsIdxInv = CommonValues::BOOST_LOCATIONS_AMOUNT - rsIdx - 1;

			gs.boostPads[rsIdx] = boostPadStates->Get(i)->isActive();
			gs.boostPadsInv[rsIdxInv] = gs.boostPads[rsIdx];

			gs.boostPadTimers[rsIdx] = boostPadStates->Get(i)->timer();
			gs.boostPadTimersInv[rsIdxInv] = gs.boostPadTimers[rsIdx];
		}
	}

	return gs;
}

rlbot::Controller RLBotBot::GetOutput(rlbot::GameTickPacket gameTickPacket) {

	float curTime = gameTickPacket->gameInfo()->secondsElapsed();
	float deltaTime = curTime - prevTime;
	prevTime = curTime;

	int ticksElapsed = roundf(deltaTime * 120);
	ticks += ticksElapsed;

	if (!triedBuildingBoostPadMap) {
		triedBuildingBoostPadMap = true;
		boostPadMap = BuildBoostPadMap(GetFieldInfo());
		if (boostPadMap.empty())
			RG_LOG("RLBotClient: Failed to match boost pads to the standard layout, all pads will be treated as active");
	}

	GameState gs = ToGameState(gameTickPacket, boostPadMap);
	auto& localPlayer = gs.players[index];
	localPlayer.prevAction = controls;

	if (updateAction) {
		updateAction = false;
		action = params.inferUnit->InferAction(localPlayer, gs, true);
	}

	if (ticks >= (params.actionDelay - 1) || ticks == -1) {
		// Apply new action
		controls = action;
	}

	if (ticks >= params.tickSkip || ticks == -1) {
		
		// Trigger action update next tick
		ticks = 0;
		updateAction = true;
	}

	auto rc = rlbot::Controller();
	{
		rc.throttle = controls.throttle;
		rc.steer = controls.steer;

		rc.pitch = controls.pitch;
		rc.yaw = controls.yaw;
		rc.roll = controls.roll;

		rc.boost = controls.boost;
		rc.jump = controls.jump;
		rc.handbrake = controls.handbrake;
	}

	return rc;
}

void RLBotClient::Run(const RLBotParams& params) {
	g_RLBotParams = params;

	rlbot::platform::SetWorkingDirectory(
		rlbot::platform::GetExecutableDirectory()
	);

	rlbot::BotManager botManager(BotFactory);
	botManager.StartBotServer(params.port);
}