// End-to-end integration test:
//	- Initializes RocketSim with a synthetic in-memory arena mesh (no game files needed)
//	- Trains a tiny model for a few iterations on CPU
//	- Verifies checkpoints save, load, and resume
//	- Verifies InferUnit can load the trained policy and produce valid actions
// Exits 0 on success, non-zero on failure.

#include <GigaLearnCPP/Learner.h>
#include <GigaLearnCPP/Util/InferUnit.h>

#include <RLGymCPP/Rewards/CommonRewards.h>
#include <RLGymCPP/Rewards/ZeroSumReward.h>
#include <RLGymCPP/TerminalConditions/NoTouchCondition.h>
#include <RLGymCPP/TerminalConditions/GoalScoreCondition.h>
#include <RLGymCPP/ObsBuilders/DefaultObsPadded.h>
#include <RLGymCPP/StateSetters/KickoffState.h>
#include <RLGymCPP/ActionParsers/DefaultAction.h>

using namespace GGL;
using namespace RLGC;

// Builds a .cmf-format collision mesh of a rounded arena-sized box, in memory
// (Format: int32 numTris, int32 numVerts, tris as 3x int32, verts as 3x float)
//
// The edges of the box are rounded: sharp 90-degree seams (which real arenas
//	don't have) let cars wedge into corners and can blow up the physics solver.
// Faces are also tessellated, since giant triangles destabilize collisions.
static RocketSim::FileData MakeBoxMeshFile() {
	// Inset from the real arena bounds so the mesh is never coplanar with RocketSim's
	//	built-in boundary planes (coplanar surfaces create degenerate contacts)
	constexpr float INSET = 16;
	constexpr float X = 4096 - INSET, Y = 5120 - INSET, Z = 2044 - INSET;
	constexpr float TRI_SIZE = 256; // Tessellation resolution
	constexpr float ROUNDING = 300; // Edge rounding radius

	std::vector<float> verts = {};
	std::vector<int32_t> tris = {};

	// Projects a point of the sharp box onto the rounded box
	auto fnRound = [&](Vec pos) {
		Vec clamped = Vec(
			RS_CLAMP(pos.x, -X + ROUNDING, X - ROUNDING),
			RS_CLAMP(pos.y, -Y + ROUNDING, Y - ROUNDING),
			RS_CLAMP(pos.z, ROUNDING, Z - ROUNDING)
		);

		Vec dir = pos - clamped;
		float dist = dir.Length();
		if (dist < 1e-6f)
			return pos;

		return clamped + dir * (ROUNDING / dist);
	};

	// Adds a tessellated quad defined by an origin corner and two edge vectors,
	//	with all vertices projected onto the rounded box
	auto fnAddFace = [&](Vec origin, Vec edgeU, Vec edgeV) {
		int divsU = (int)ceilf(edgeU.Length() / TRI_SIZE);
		int divsV = (int)ceilf(edgeV.Length() / TRI_SIZE);

		int32_t baseIdx = verts.size() / 3;
		for (int u = 0; u <= divsU; u++) {
			for (int v = 0; v <= divsV; v++) {
				Vec pos = fnRound(origin + edgeU * (u / (float)divsU) + edgeV * (v / (float)divsV));
				verts.insert(verts.end(), { pos.x, pos.y, pos.z });
			}
		}

		for (int u = 0; u < divsU; u++) {
			for (int v = 0; v < divsV; v++) {
				int32_t
					i00 = baseIdx + u * (divsV + 1) + v,
					i01 = i00 + 1,
					i10 = i00 + (divsV + 1),
					i11 = i10 + 1;

				tris.insert(tris.end(), { i00, i10, i11 });
				tris.insert(tris.end(), { i00, i11, i01 });
			}
		}
	};

	fnAddFace(Vec(-X, -Y, 0), Vec(2 * X, 0, 0), Vec(0, 2 * Y, 0)); // Floor
	fnAddFace(Vec(-X, -Y, Z), Vec(2 * X, 0, 0), Vec(0, 2 * Y, 0)); // Ceiling
	fnAddFace(Vec(-X, -Y, 0), Vec(2 * X, 0, 0), Vec(0, 0, Z)); // -Y wall
	fnAddFace(Vec(-X,  Y, 0), Vec(2 * X, 0, 0), Vec(0, 0, Z)); // +Y wall
	fnAddFace(Vec(-X, -Y, 0), Vec(0, 2 * Y, 0), Vec(0, 0, Z)); // -X wall
	fnAddFace(Vec( X, -Y, 0), Vec(0, 2 * Y, 0), Vec(0, 0, Z)); // +X wall

	RocketSim::FileData data = {};
	auto fnWrite = [&](const void* ptr, size_t size) {
		data.insert(data.end(), (const byte*)ptr, (const byte*)ptr + size);
	};

	int32_t numTris = tris.size() / 3, numVerts = verts.size() / 3;
	fnWrite(&numTris, sizeof(numTris));
	fnWrite(&numVerts, sizeof(numVerts));
	fnWrite(tris.data(), tris.size() * sizeof(int32_t));
	fnWrite(verts.data(), verts.size() * sizeof(float));
	return data;
}

constexpr int MAX_PLAYERS_PER_TEAM = 2;

static EnvCreateResult EnvCreateFunc(int index) {
	std::vector<WeightedReward> rewards = {
		{ new VelocityPlayerToBallReward(), 1.f },
		{ new ZeroSumReward(new VelocityBallToGoalReward(), 1), 1.f },
		{ new GoalReward(), 50 },
	};

	std::vector<TerminalCondition*> terminalConditions = {
		new NoTouchCondition(6),
		new GoalScoreCondition()
	};

	// Mix 1v1 and 2v2 arenas in the same env set
	//	(exercises the variable-player-count bookkeeping and padded obs)
	int playersPerTeam = (index % MAX_PLAYERS_PER_TEAM) + 1;

	auto arena = Arena::Create(GameMode::SOCCAR);
	for (int i = 0; i < playersPerTeam; i++) {
		arena->AddCar(Team::BLUE);
		arena->AddCar(Team::ORANGE);
	}

	EnvCreateResult result = {};
	result.actionParser = new DefaultAction();
	result.obsBuilder = new DefaultObsPadded(MAX_PLAYERS_PER_TEAM);
	result.stateSetter = new KickoffState();
	result.terminalConditions = terminalConditions;
	result.rewards = rewards;
	result.arena = arena;
	return result;
}

static LearnerConfig MakeTestConfig(std::filesystem::path checkpointFolder) {
	LearnerConfig cfg = {};
	cfg.deviceType = LearnerDeviceType::CPU;
	cfg.numGames = 4;
	cfg.randomSeed = 123;

	cfg.ppo.tsPerItr = 500;
	cfg.ppo.batchSize = 500;
	cfg.ppo.miniBatchSize = 500;
	cfg.ppo.epochs = 1;

	cfg.ppo.sharedHead.layerSizes = { 32 };
	cfg.ppo.policy.layerSizes = { 32 };
	cfg.ppo.critic.layerSizes = { 32 };

	cfg.checkpointFolder = checkpointFolder;
	cfg.tsPerSave = 1000;
	cfg.timestepLimit = 2000;

	cfg.sendMetrics = false; // No wandb/python receiver needed
	cfg.addRewardsToMetrics = true;

	return cfg;
}

#define INTEG_CHECK(cond) { \
	if (!(cond)) { \
		std::cout << "INTEGRATION TEST FAILED at " << __FILE__ << ":" << __LINE__ << ": " << #cond << std::endl; \
		return 1; \
	} \
}

int main(int argc, char* argv[]) {
	auto checkpointFolder = std::filesystem::temp_directory_path() / "ggl_integration_checkpoints";
	std::filesystem::remove_all(checkpointFolder);

	// Initialize RocketSim from the in-memory box mesh
	{
		std::map<GameMode, std::vector<RocketSim::FileData>> meshMap = {};
		meshMap[GameMode::SOCCAR] = { MakeBoxMeshFile() };
		RocketSim::InitFromMem(meshMap, true);
	}

	uint64_t timestepsAfterFirstRun = 0;

	{ // Phase 1: train from scratch until the timestep limit
		Learner* learner = new Learner(EnvCreateFunc, MakeTestConfig(checkpointFolder));
		learner->Start(); // Returns at cfg.timestepLimit

		timestepsAfterFirstRun = learner->totalTimesteps;
		INTEG_CHECK(timestepsAfterFirstRun >= 2000);
		INTEG_CHECK(learner->totalIterations >= 2);

		delete learner;
	}

	// A checkpoint must exist on disk
	INTEG_CHECK(std::filesystem::is_directory(checkpointFolder));
	INTEG_CHECK(!Utils::FindNumberedDirs(checkpointFolder).empty());

	{ // Phase 2: resume from the checkpoint
		auto cfg = MakeTestConfig(checkpointFolder);
		cfg.timestepLimit = timestepsAfterFirstRun + 1000;

		Learner* learner = new Learner(EnvCreateFunc, cfg);

		// The checkpoint must have loaded (timesteps carried over)
		INTEG_CHECK(learner->totalTimesteps > 0);
		INTEG_CHECK(learner->totalTimesteps <= timestepsAfterFirstRun);

		learner->Start();
		INTEG_CHECK(learner->totalTimesteps >= timestepsAfterFirstRun + 1000);

		delete learner;
	}

	{ // Phase 3: load the trained policy with InferUnit and infer actions
		int64_t newestCheckpoint = *Utils::FindNumberedDirs(checkpointFolder).rbegin();

		auto obsBuilder = new DefaultObsPadded(MAX_PLAYERS_PER_TEAM);
		auto actionParser = new DefaultAction();

		// Determine the obs size using a synthetic 1v1 state
		GameState testState = {};
		testState.players.resize(2);
		for (int i = 0; i < 2; i++) {
			testState.players[i].index = i;
			testState.players[i].carId = i + 1;
			testState.players[i].team = (i == 0) ? Team::BLUE : Team::ORANGE;
			testState.players[i].pos = Vec(0, -1000 + 2000 * i, 17);
			testState.players[i].rotMat = RotMat::GetIdentity();
		}
		testState.ball.pos = Vec(0, 0, 93);
		int obsSize = obsBuilder->BuildObs(testState.players[0], testState).size();

		PartialModelConfig sharedHeadConfig = {};
		sharedHeadConfig.layerSizes = { 32 };
		sharedHeadConfig.addOutputLayer = false;
		PartialModelConfig policyConfig = {};
		policyConfig.layerSizes = { 32 };

		InferUnit* inferUnit = new InferUnit(
			obsBuilder, obsSize, actionParser,
			sharedHeadConfig, policyConfig,
			checkpointFolder / std::to_string(newestCheckpoint), false
		);

		// Both deterministic and stochastic inference must produce valid actions
		for (bool deterministic : { true, false }) {
			Action action = inferUnit->InferAction(testState.players[0], testState, deterministic);

			bool matchesAny = false;
			for (auto& parserAction : ((DefaultAction*)actionParser)->actions) {
				bool matches = true;
				for (int i = 0; i < Action::ELEM_AMOUNT; i++)
					matches &= (action[i] == parserAction[i]);
				matchesAny |= matches;
			}
			INTEG_CHECK(matchesAny);
		}

		delete inferUnit;
		delete obsBuilder;
		delete actionParser;
	}

	std::filesystem::remove_all(checkpointFolder);

	std::cout << std::string(40, '=') << std::endl;
	std::cout << "INTEGRATION TEST PASSED" << std::endl;
	return 0;
}
