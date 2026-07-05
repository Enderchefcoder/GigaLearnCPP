// Example RLBot entry point: runs a trained GigaLearn policy in real Rocket League matches.
//
// Build it with -DGGL_BUILD_RLBOT_EXAMPLE=ON (produces the "GigaLearnRLBot" executable),
//	then see rlbot/README.md for hooking it up to the RLBot framework.
//
// EVERYTHING here must match how the policy was trained:
//	obs builder, action parser, model architecture, tick skip, and action delay.

#include "../RLBotClient.h"

#include <RLGymCPP/ObsBuilders/AdvancedObs.h>
#include <RLGymCPP/ActionParsers/DefaultAction.h>

using namespace GGL;
using namespace RLGC;

int main(int argc, char* argv[]) {

	// === EDIT THESE to match your training setup ===

	// The checkpoint folder containing POLICY.lt (and SHARED_HEAD.lt if you use shared layers)
	std::filesystem::path modelsFolder = "checkpoint_to_run";

	// Same obs builder and action parser as training
	auto obsBuilder = new AdvancedObs();
	auto actionParser = new DefaultAction();

	// Same model architecture as training (cfg.ppo.sharedHead / cfg.ppo.policy)
	PartialModelConfig sharedHeadConfig = {};
	sharedHeadConfig.layerSizes = { 256, 256 };
	sharedHeadConfig.addOutputLayer = false;

	PartialModelConfig policyConfig = {};
	policyConfig.layerSizes = { 256, 256, 256 };

	// Determine the obs size from a synthetic state
	//	(must use the same team sizes the obs builder was trained with)
	int obsSize;
	{
		GameState testState = {};
		testState.players.resize(2);
		for (int i = 0; i < 2; i++) {
			auto& player = testState.players[i];
			player.index = i;
			player.carId = i + 1;
			player.team = (i == 0) ? Team::BLUE : Team::ORANGE;
			player.rotMat = RotMat::GetIdentity();
		}
		obsSize = obsBuilder->BuildObs(testState.players[0], testState).size();
		RG_LOG("Obs size: " << obsSize);
	}

	RLBotParams params = {};
	params.port = 23233; // Must match rlbot/port.cfg
	params.tickSkip = 8; // Same as cfg.tickSkip in training
	params.actionDelay = 7; // Same as cfg.actionDelay in training
	params.inferUnit = new InferUnit(
		obsBuilder, obsSize, actionParser,
		sharedHeadConfig, policyConfig,
		modelsFolder,
		false // Inference on CPU (a single bot doesn't need a GPU)
	);

	// ===============================================

	RLBotClient::Run(params);

	return EXIT_SUCCESS;
}
