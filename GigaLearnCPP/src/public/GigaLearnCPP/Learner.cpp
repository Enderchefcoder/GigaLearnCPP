#include "Learner.h"

#include <GigaLearnCPP/PPO/PPOLearner.h>
#include <GigaLearnCPP/PPO/ExperienceBuffer.h>

#include <torch/cuda.h>
#include <ATen/Parallel.h>
#include <nlohmann/json.hpp>
#include <pybind11/embed.h>

#ifdef RG_CUDA_SUPPORT
#include <c10/cuda/CUDACachingAllocator.h>
#endif
#include <private/GigaLearnCPP/PPO/ExperienceBuffer.h>
#include <private/GigaLearnCPP/PPO/GAE.h>
#include <private/GigaLearnCPP/PolicyVersionManager.h>

#include "Util/KeyPressDetector.h"
#include <private/GigaLearnCPP/Util/WelfordStat.h>
#include "Util/AvgTracker.h"

#include <csignal>

using namespace RLGC;

// Set by SIGINT/SIGTERM so training can save and exit at the end of the iteration
// (Must be a plain flag: only async-signal-safe operations are allowed in handlers)
static volatile std::sig_atomic_t g_StopSignalReceived = 0;

static void _StopSignalHandler(int signum) {
	if (g_StopSignalReceived) {
		// Second signal: the user really wants out, stop immediately
		std::signal(signum, SIG_DFL);
		std::raise(signum);
		return;
	}

	g_StopSignalReceived = 1;
}

static void _InstallStopSignalHandlers() {
	std::signal(SIGINT, _StopSignalHandler);
	std::signal(SIGTERM, _StopSignalHandler);
}

GGL::Learner::Learner(EnvCreateFn envCreateFn, LearnerConfig config, StepCallbackFn stepCallback) :
	envCreateFn(envCreateFn), config(config), stepCallback(stepCallback)
{
	// The interpreter may already be running (e.g. a previous Learner whose constructor
	//	threw, or a host application that embeds Python itself)
	if (!Py_IsInitialized()) {
		pybind11::initialize_interpreter();
		_ownsPyInterpreter = true;
	}

	{
		// Make sure Python can find our scripts (e.g. "python_scripts/metric_receiver.py"),
		//	regardless of what directory we were launched from
		auto sysPath = pybind11::module::import("sys").attr("path");
		sysPath.attr("insert")(0, std::filesystem::current_path().string());
		sysPath.attr("insert")(0, Utils::GetExecutableDir().string());
	}

#ifndef NDEBUG
	RG_LOG("===========================");
	RG_LOG("WARNING: GigaLearn runs extremely slowly in debug, and there are often bizzare issues with debug-mode torch.");
	RG_LOG("It is recommended that you compile in release mode without optimization for debugging.");
	RG_SLEEP(1000);
#endif

	if (config.tsPerSave == 0)
		config.tsPerSave = config.ppo.tsPerItr;

	{ // Validate config
		if (config.numGames <= 0)
			RG_ERR_CLOSE("Learner: config.numGames must be positive");

		if (config.tickSkip <= 0)
			RG_ERR_CLOSE("Learner: config.tickSkip must be positive");

		if (config.actionDelay < 0 || config.actionDelay > config.tickSkip)
			RG_ERR_CLOSE(
				"Learner: config.actionDelay (" << config.actionDelay << ") must be from 0 to config.tickSkip (" << config.tickSkip << ")"
			);

		if (config.ppo.tsPerItr <= 0 || config.ppo.batchSize <= 0)
			RG_ERR_CLOSE("Learner: config.ppo.tsPerItr and config.ppo.batchSize must be positive");

		if (config.ppo.batchSize > config.ppo.tsPerItr)
			RG_LOG(
				"WARNING: config.ppo.batchSize (" << config.ppo.batchSize << ") is larger than config.ppo.tsPerItr (" << config.ppo.tsPerItr << "), " <<
				"iterations are only guaranteed to collect tsPerItr timesteps, so learning may fail"
			);

		if (config.ppo.epochs <= 0)
			RG_ERR_CLOSE("Learner: config.ppo.epochs must be positive");
	}

	RG_LOG("Learner::Learner():");

	if (config.randomSeed == -1)
		config.randomSeed = RS_CUR_MS();

	RG_LOG("\tCheckpoint Save/Load Dir: " << config.checkpointFolder);

	torch::manual_seed(config.randomSeed);

	at::Device device = at::Device(at::kCPU);
	if (
		config.deviceType == LearnerDeviceType::GPU_CUDA || 
		(config.deviceType == LearnerDeviceType::AUTO && torch::cuda::is_available())
		) {
		RG_LOG("\tUsing CUDA GPU device...");

		// Test out moving a tensor to GPU and back to make sure the device is working
		torch::Tensor t;
		bool deviceTestFailed = false;
		try {
			t = torch::tensor(0);
			t = t.to(at::Device(at::kCUDA));
			t = t.cpu();
		} catch (...) {
			deviceTestFailed = true;
		}

		if (!torch::cuda::is_available() || deviceTestFailed)
			RG_ERR_CLOSE(
				"Learner::Learner(): Can't use CUDA GPU because " <<
				(torch::cuda::is_available() ? "libtorch cannot access the GPU" : "CUDA is not available to libtorch") << ".\n" <<
				"Make sure your libtorch comes with CUDA support, and that CUDA is installed properly."
			)
		device = at::Device(at::kCUDA);
	} else {
		RG_LOG("\tUsing CPU device...");
		device = at::Device(at::kCPU);
	}

	if (RocketSim::GetStage() != RocketSimStage::INITIALIZED) {
		constexpr const char* MESHES_FOLDER = "collision_meshes";

		RG_LOG("\tInitializing RocketSim...");

		if (!std::filesystem::is_directory(MESHES_FOLDER)) {
			RG_LOG(
				"\tWARNING: No \"" << MESHES_FOLDER << "\" folder found in the working directory (" << std::filesystem::current_path() << ").\n" <<
				"\tWithout arena collision meshes, cars and the ball will not collide with walls or ramps properly!\n" <<
				"\tDump the meshes with https://github.com/ZealanL/RLArenaCollisionDumper, or copy the folder next to your executable.\n" <<
				"\t(You can also call RocketSim::Init(\"path/to/collision_meshes\") yourself before creating the Learner)"
			);
		}

		RocketSim::Init(MESHES_FOLDER, true);
	}

	{
		RG_LOG("\tCreating envs...");
		EnvSetConfig envSetConfig = {};
		envSetConfig.envCreateFn = envCreateFn;
		envSetConfig.numArenas = config.renderMode ? 1 : config.numGames;
		envSetConfig.tickSkip = config.tickSkip;
		envSetConfig.actionDelay = config.actionDelay;
		envSetConfig.saveRewards = config.addRewardsToMetrics;
		envSet = new RLGC::EnvSet(envSetConfig);
		obsSize = envSet->state.obs.size[1];
		numActions = envSet->actionParsers[0]->GetActionAmount();
	}

	{
		if (config.standardizeReturns) {
			this->returnStat = new WelfordStat();
		} else {
			this->returnStat = NULL;
		}

		if (config.standardizeObs) {
			this->obsStat = new BatchedWelfordStat(obsSize);
		} else {
			this->obsStat = NULL;
		}
	}

	try {
		RG_LOG("\tMaking PPO learner...");
		ppo = new PPOLearner(obsSize, numActions, config.ppo, device);
	} catch (std::exception& e) {
		RG_ERR_CLOSE("Failed to create PPO learner: " << e.what());
	}

	if (config.renderMode) {
		renderSender = new RenderSender(config.renderTimeScale);
	} else {
		renderSender = NULL;
	}

	if (config.skillTracker.enabled || config.trainAgainstOldVersions)
		config.savePolicyVersions = true;

	if (config.savePolicyVersions && !config.renderMode) {
		if (config.checkpointFolder.empty())
			RG_ERR_CLOSE("Cannot save/load old policy versions with no checkpoint save folder");
		versionMgr = new PolicyVersionManager(
			config.checkpointFolder / "policy_versions", config.maxOldVersions, config.tsPerVersion,
			config.skillTracker, envSet->config
		);
	} else {
		versionMgr = NULL;
	}

	if (!config.checkpointFolder.empty())
		Load();

	if (config.savePolicyVersions && !config.renderMode) {
		if (config.checkpointFolder.empty())
			RG_ERR_CLOSE("Cannot save/load old policy versions with no checkpoint save folder");
		auto models = ppo->GetPolicyModels();
		versionMgr->LoadVersions(models, totalTimesteps);
	}

	if (config.sendMetrics && !config.renderMode) {
		if (!runID.empty())
			RG_LOG("\tRun ID: " << runID);
		metricSender = new MetricSender(config.metricsProjectName, config.metricsGroupName, config.metricsRunName, runID);
	} else {
		metricSender = NULL;
	}

	RG_LOG(RG_DIVIDER);
}

void GGL::Learner::SaveStats(std::filesystem::path path) {
	using namespace nlohmann;

	constexpr const char* ERROR_PREFIX = "Learner::SaveStats(): ";

	std::ofstream fOut(path);
	if (!fOut.good())
		RG_ERR_CLOSE(ERROR_PREFIX << "Can't open file at " << path);

	json j = {};
	j["total_timesteps"] = totalTimesteps;
	j["total_iterations"] = totalIterations;

	// Environment metadata, checked on load to catch obs builder/action parser changes early
	j["obs_size"] = obsSize;
	j["num_actions"] = numActions;
	j["tick_skip"] = config.tickSkip;

	if (config.sendMetrics)
		j["run_id"] = metricSender->curRunID;

	if (returnStat)
		j["return_stat"] = returnStat->ToJSON();
	if (obsStat)
		j["obs_stat"] = obsStat->ToJSON();

	if (versionMgr)
		versionMgr->AddRunningStatsToJSON(j);

	std::string jStr = j.dump(4);
	fOut << jStr;
}

void GGL::Learner::LoadStats(std::filesystem::path path) {
	// TODO: Repetitive code, merge repeated code into one function called from both SaveStats() and LoadStats()

	using namespace nlohmann;
	constexpr const char* ERROR_PREFIX = "Learner::LoadStats(): ";

	std::ifstream fIn(path);
	if (!fIn.good())
		RG_ERR_CLOSE(ERROR_PREFIX << "Can't open file at " << path);

	json j = json::parse(fIn);
	totalTimesteps = j["total_timesteps"];
	totalIterations = j["total_iterations"];

	// Catch incompatible env changes before the cryptic model-size error would
	if (j.contains("obs_size") && (int)j["obs_size"] != obsSize)
		RG_ERR_CLOSE(
			ERROR_PREFIX << "This checkpoint was trained with obs size " << j["obs_size"] << ", " <<
			"but the current obs builder produces obs of size " << obsSize << ".\n" <<
			"Your obs builder (or team sizes) changed; either revert it, use transfer learning, or start a new checkpoint folder."
		);

	if (j.contains("num_actions") && (int)j["num_actions"] != numActions)
		RG_ERR_CLOSE(
			ERROR_PREFIX << "This checkpoint was trained with " << j["num_actions"] << " actions, " <<
			"but the current action parser has " << numActions << ".\n" <<
			"Your action parser changed; either revert it, use transfer learning, or start a new checkpoint folder."
		);

	if (j.contains("tick_skip") && (int)j["tick_skip"] != config.tickSkip)
		RG_LOG(
			"WARNING: This checkpoint was trained with tickSkip = " << j["tick_skip"] <<
			", but the current config uses tickSkip = " << config.tickSkip << " (the game speed the policy sees has changed)"
		);

	if (j.contains("run_id"))
		runID = j["run_id"];

	if (returnStat)
		returnStat->ReadFromJSON(j["return_stat"]);
	if (obsStat)
		obsStat->ReadFromJSON(j["obs_stat"]);

	if (versionMgr)
		versionMgr->LoadRunningStatsFromJSON(j);
}

// Different than RLGym-PPO to show that they are not compatible
constexpr const char* STATS_FILE_NAME = "RUNNING_STATS.json";

void GGL::Learner::Save() {
	if (config.checkpointFolder.empty())
		RG_ERR_CLOSE("Learner::Save(): Cannot save because config.checkpointSaveFolder is not set");

	std::filesystem::path saveFolder = config.checkpointFolder / std::to_string(totalTimesteps);

	// Save to a temporary folder first, then rename it into place once complete
	// This way, a crash mid-save can't leave a corrupt checkpoint behind
	//	(incomplete "~incomplete" folders are ignored by checkpoint loading)
	std::filesystem::path tmpSaveFolder = saveFolder;
	tmpSaveFolder += "~incomplete";

	std::filesystem::remove_all(tmpSaveFolder);
	std::filesystem::create_directories(tmpSaveFolder);

	RG_LOG("Saving to folder " << saveFolder << "...");
	SaveStats(tmpSaveFolder / STATS_FILE_NAME);
	ppo->SaveTo(tmpSaveFolder);

	std::filesystem::remove_all(saveFolder);
	std::filesystem::rename(tmpSaveFolder, saveFolder);

	// Remove old checkpoints
	if (config.checkpointsToKeep != -1) {
		std::set<int64_t> allSavedTimesteps = Utils::FindNumberedDirs(config.checkpointFolder);
		while (allSavedTimesteps.size() > config.checkpointsToKeep) {
			int64_t lowestCheckpointTS = INT64_MAX;
			for (int64_t savedTimesteps : allSavedTimesteps)
				lowestCheckpointTS = RS_MIN(lowestCheckpointTS, savedTimesteps);

			std::filesystem::path removePath = config.checkpointFolder / std::to_string(lowestCheckpointTS);
			try {
				std::filesystem::remove_all(removePath);
			} catch (std::exception& e) {
				RG_ERR_CLOSE("Failed to remove old checkpoint from " << removePath << ", exception: " << e.what());
			}
			allSavedTimesteps.erase(lowestCheckpointTS);
		}
	}

	if (versionMgr)
		versionMgr->SaveVersions();

	RG_LOG(" > Done.");
}

void GGL::Learner::Load() {
	if (config.checkpointFolder.empty())
		RG_ERR_CLOSE("Learner::Load(): Cannot load because config.checkpointLoadFolder is not set");

	RG_LOG("Loading most recent checkpoint in " << config.checkpointFolder << "...");

	int64_t highest = -1;
	std::set<int64_t> allSavedTimesteps = Utils::FindNumberedDirs(config.checkpointFolder);
	for (int64_t timesteps : allSavedTimesteps)
		highest = RS_MAX(timesteps, highest);

	if (highest != -1) {
		std::filesystem::path loadFolder = config.checkpointFolder / std::to_string(highest);
		RG_LOG(" > Loading checkpoint " << loadFolder << "...");
		LoadStats(loadFolder / STATS_FILE_NAME);
		ppo->LoadFrom(loadFolder);
		RG_LOG(" > Done.");
	} else {
		RG_LOG(" > No checkpoints found, starting new model.")
	}
}

void GGL::Learner::SetLearningRates(float policyLR, float criticLR) {
	config.ppo.policyLR = policyLR;
	config.ppo.criticLR = criticLR;
	ppo->SetLearningRates(policyLR, criticLR);
}

void GGL::Learner::SetEntropyScale(float entropyScale) {
	config.ppo.entropyScale = entropyScale;
	ppo->config.entropyScale = entropyScale;
}

float GGL::Learner::GetPolicyLR() const {
	return ppo->config.policyLR;
}

float GGL::Learner::GetCriticLR() const {
	return ppo->config.criticLR;
}

float GGL::Learner::GetEntropyScale() const {
	return ppo->config.entropyScale;
}

void GGL::Learner::StartQuitKeyThread(bool& quitPressed, std::thread& outThread) {
	quitPressed = false;

	RG_LOG("Press 'Q' to save and quit!");
	outThread = std::thread(
		[&] {
			while (true) {
				char pressed = KeyPressDetector::GetPressedChar();

				if (pressed == KeyPressDetector::CHAR_UNAVAILABLE) {
					// Input is unavailable (e.g. headless server or stdin closed),
					//	stop polling so we don't spin forever
					RG_LOG("Learner: Input unavailable, 'Q'-to-quit is disabled.");
					return;
				}

				if (toupper(pressed) == 'Q') {
					RG_LOG("Save queued, will save and exit next iteration.");
					quitPressed = true;
				}
			}
		}
	);

	outThread.detach();
}
void GGL::Learner::StartTransferLearn(const TransferLearnConfig& tlConfig) {

	RG_LOG("Starting transfer learning...");

	// TODO: Lots of manual obs builder stuff going on which is quite volatile
	//	Although I can't really think another way to do this

	std::vector<ObsBuilder*> oldObsBuilders = {};
	for (int i = 0; i < envSet->arenas.size(); i++)
		oldObsBuilders.push_back(tlConfig.makeOldObsFn());

	// Reset all obs builders initially, each with its own arena's state
	for (int i = 0; i < envSet->arenas.size(); i++)
		oldObsBuilders[i]->Reset(envSet->state.gameStates[i]);

	std::vector<ActionParser*> oldActionParsers = {};
	for (int i = 0; i < envSet->arenas.size(); i++)
		oldActionParsers.push_back(tlConfig.makeOldActFn());

	int oldNumActions = oldActionParsers[0]->GetActionAmount();

	if (oldNumActions != numActions) {
		if (!tlConfig.mapActsFn) {
			RG_ERR_CLOSE(
				"StartTransferLearn: Old and new action parsers have a different number of actions, but tlConfig.mapActsFn is NULL.\n" <<
				"You must implement this function to translate the action indices."
			);
		};
	}

	// Determine old obs size
	int oldObsSize;
	{
		GameState testState = envSet->state.gameStates[0];
		oldObsSize = oldObsBuilders[0]->BuildObs(testState.players[0], testState).size();
	}

	ModelSet oldModels = {};
	{
		RG_NO_GRAD;
		PPOLearner::MakeModels(false, oldObsSize, oldNumActions, tlConfig.oldSharedHeadConfig, tlConfig.oldPolicyConfig, {}, ppo->device, oldModels);

		oldModels.Load(tlConfig.oldModelsPath, false, false);
	}

	try {
		bool saveQueued;
		std::thread keyPressThread;
		StartQuitKeyThread(saveQueued, keyPressThread);

		// Ctrl+C / SIGTERM also save-and-quit at the end of the iteration
		_InstallStopSignalHandlers();

		// See config.collectionTorchThreads (same reasoning as the main training loop)
		int defaultTorchThreads = at::get_num_threads();
		bool limitCollectionTorchThreads = (config.collectionTorchThreads > 0);

		while (true) {
			Report report = {};

			// Collect obs
			std::vector<float> allNewObs = {};
			std::vector<float> allOldObs = {};
			std::vector<uint8_t> allNewActionMasks = {};
			std::vector<uint8_t> allOldActionMasks = {};
			std::vector<int> allActionMaps = {};
			int stepsCollected;
			{
				RG_NO_GRAD;

				if (limitCollectionTorchThreads)
					at::set_num_threads(config.collectionTorchThreads);
				for (stepsCollected = 0; stepsCollected < tlConfig.batchSize; stepsCollected += envSet->state.numPlayers) {
					
					auto terminals = envSet->state.terminals; // Backup
					envSet->Reset();
					for (int i = 0; i < envSet->arenas.size(); i++) // Manually reset old obs builders
						if (terminals[i])
							oldObsBuilders[i]->Reset(envSet->state.gameStates[i]);

					torch::Tensor tActions, tLogProbs;
					torch::Tensor tStates = DIMLIST2_TO_TENSOR<float>(envSet->state.obs);
					torch::Tensor tActionMasks = DIMLIST2_TO_TENSOR<uint8_t>(envSet->state.actionMasks);

					envSet->StepFirstHalf(true);

					allNewObs += envSet->state.obs.data;
					allNewActionMasks += envSet->state.actionMasks.data;

					// Run all old obs and old action parser on each player
					// TODO: Could be multithreaded
					for (int arenaIdx = 0; arenaIdx < envSet->arenas.size(); arenaIdx++) {
						auto& gs = envSet->state.gameStates[arenaIdx];
						for (auto& player : gs.players) {
							allOldObs += oldObsBuilders[arenaIdx]->BuildObs(player, gs);
							allOldActionMasks += oldActionParsers[arenaIdx]->GetActionMask(player, gs);

							if (tlConfig.mapActsFn) {
								auto curMap = tlConfig.mapActsFn(player, gs);
								if (curMap.size() != numActions)
									RG_ERR_CLOSE("StartTransferLearn: Your action map must have the same size as the new action parser's actions");
								allActionMaps += curMap;
							}
						}
					}

					ppo->InferActions(
						tStates.to(ppo->device, true), tActionMasks.to(ppo->device, true), 
						&tActions, &tLogProbs
					);

					auto curActions = TENSOR_TO_VEC<int>(tActions);

					envSet->Sync();
					envSet->StepSecondHalf(curActions, false);

					if (stepCallback)
						stepCallback(this, envSet->state.gameStates, report);
				}

				// Learning gets torch's full thread count back
				if (limitCollectionTorchThreads)
					at::set_num_threads(defaultTorchThreads);
			}

			uint64_t prevTimesteps = totalTimesteps;
			totalTimesteps += stepsCollected;
			report["Total Timesteps"] = totalTimesteps;
			report["Collected Timesteps"] = stepsCollected;
			totalIterations++;
			report["Total Iterations"] = totalIterations;

			// Make tensors
			torch::Tensor tNewObs = VEC_TO_TENSOR(allNewObs).reshape({ -1, obsSize }).to(ppo->device);
			torch::Tensor tOldObs = VEC_TO_TENSOR(allOldObs).reshape({ -1, oldObsSize }).to(ppo->device);
			torch::Tensor tNewActionMasks = VEC_TO_TENSOR(allNewActionMasks).reshape({ -1, numActions }).to(ppo->device);
			torch::Tensor tOldActionMasks = VEC_TO_TENSOR(allOldActionMasks).reshape({ -1, oldNumActions }).to(ppo->device);

			torch::Tensor tActionMaps = {};
			if (!allActionMaps.empty()) {
				// NOTE: gather() requires int64 indices
				tActionMaps = VEC_TO_TENSOR(allActionMaps).reshape({ -1, numActions }).to(torch::kInt64).to(ppo->device);
			}

			// Transfer learn
			ppo->TransferLearn(oldModels, tNewObs, tOldObs, tNewActionMasks, tOldActionMasks, tActionMaps, report, tlConfig);

			if (versionMgr)
				versionMgr->OnIteration(ppo, report, totalTimesteps, prevTimesteps);

			if (g_StopSignalReceived) {
				RG_LOG("Stop signal received (Ctrl+C/SIGTERM), saving and exiting...");
				saveQueued = true;
			}

			if (saveQueued) {
				if (!config.checkpointFolder.empty())
					Save();
				exit(0);
			}

			if (!config.checkpointFolder.empty()) {
				if (totalTimesteps / config.tsPerSave > prevTimesteps / config.tsPerSave) {
					// Auto-save
					Save();
				}
			}

			report.Finish();

			if (metricSender)
				metricSender->Send(report);

			report.Display(
				{
					"Transfer Learn Accuracy",
					"Transfer Learn Loss",
					"",
					"Policy Entropy",
					"Old Policy Entropy",
					"Policy Update Magnitude",
					"",
					"Collected Timesteps",
					"Total Timesteps",
					"Total Iterations"
				}
			);

			bool timestepLimitReached =
				(config.timestepLimit > 0) && (totalTimesteps >= (uint64_t)config.timestepLimit);
			if (timestepLimitReached) {
				if (!config.checkpointFolder.empty())
					Save();
				RG_LOG("Learner: Timestep limit of " << config.timestepLimit << " reached, stopping transfer learning.");
				return;
			}
		}

	} catch (std::exception& e) {
		RG_ERR_CLOSE("Exception thrown during transfer learn loop: " << e.what());
	}
}

// Checks a chunk of floats for NaN/inf values
// Summing into a double cannot overflow (or produce NaN) unless the input already contains NaN/inf,
//	so this is a branchless, auto-vectorizable check
static bool ContainsNonFinite(const float* data, size_t size) {
	double sum = 0;
	for (size_t i = 0; i < size; i++)
		sum += data[i];
	return !std::isfinite(sum);
}

void GGL::Learner::Start() {

	bool render = config.renderMode;

	RG_LOG("Learner::Start():");
	RG_LOG("\tObs size: " << obsSize);
	RG_LOG("\tAction amount: " << numActions);

	if (render)
		RG_LOG("\t(Render mode enabled)");

	if (config.ppo.deterministic && !render)
		RG_ERR_CLOSE(
			"Learner::Start(): Cannot train with config.ppo.deterministic enabled.\n" <<
			"Deterministic mode is only for inference/rendering (it does not produce the log probs PPO needs to learn)."
		);

	try {
		bool saveQueued;
		std::thread keyPressThread;
		StartQuitKeyThread(saveQueued, keyPressThread);

		// Ctrl+C / SIGTERM also save-and-quit at the end of the iteration
		//	(a second signal force-quits immediately)
		_InstallStopSignalHandlers();

		ExperienceBuffer experience = ExperienceBuffer(config.randomSeed, torch::kCPU);

		int numPlayers = envSet->state.numPlayers;

		struct Trajectory {
			FList states, nextStates, rewards, logProbs;
			std::vector<uint8_t> actionMasks;
			std::vector<int8_t> terminals;
			std::vector<int32_t> actions;

			void Clear() {
				*this = Trajectory();
			}

			void Append(const Trajectory& other) {
				states += other.states;
				nextStates += other.nextStates;
				rewards += other.rewards;
				logProbs += other.logProbs;
				actionMasks += other.actionMasks;
				terminals += other.terminals;
				actions += other.actions;
			}

			size_t Length() const {
				return actions.size();
			}
		};

		auto trajectories = std::vector<Trajectory>(numPlayers, Trajectory{});
		int maxEpisodeLength = (int)(config.ppo.maxEpisodeDuration * (120.f / config.tickSkip));

		// Torch's idle intra-op worker threads spin-wait, which starves the env-stepping threads
		//	during collection, so torch threads are limited while collecting and restored for learning
		// (See config.collectionTorchThreads)
		int defaultTorchThreads = at::get_num_threads();
		bool limitCollectionTorchThreads = (config.collectionTorchThreads > 0) && !render;

		// Which players were controlled by the current policy (and thus recorded) last iteration
		auto prevRecordedMask = std::vector<bool>(numPlayers, true);

		// Trajectories that were force-truncated because their player switched to/from old-version control
		// These are added to the next iteration's experience
		Trajectory pendingTruncated = {};

		// The last mean/STD used for obs standardization (only used if obsStat is enabled)
		std::vector<double> lastObsMean, lastObsStd;

		// Standardizes an obs row in-place with the last-used mean/STD
		auto fnStandardizeRow = [&](FList& row) {
			if (!obsStat || lastObsMean.empty())
				return;
			for (int j = 0; j < obsSize; j++)
				row[j] = (row[j] - lastObsMean[j]) / lastObsStd[j];
		};

		while (true) {
			Report report = {};

			bool isFirstIteration = (totalTimesteps == 0);

			// Arenas reset due to non-finite states this iteration (see the NaN check below)
			int nanResetsThisIteration = 0;

			GGL::PolicyVersion* oldVersion = NULL;
			std::vector<int> newPlayerIndices = {}, oldPlayerIndices = {};
			torch::Tensor tNewPlayerIndices, tOldPlayerIndices;

			if (config.trainAgainstOldVersions && !render) {
				RG_ASSERT(config.trainAgainstOldChance >= 0 && config.trainAgainstOldChance <= 1);
				bool shouldTrainAgainstOld =
					(RocketSim::Math::RandFloat() < config.trainAgainstOldChance)
					&& !versionMgr->versions.empty();

				if (shouldTrainAgainstOld) {
					// Set up training against old versions

					int numVersions = versionMgr->versions.size();
					int oldVersionIdx;
					if (config.oldVersionRecencyBias > 0) {
						// Weighted selection favoring recent versions:
						//	weight = (1 - bias) ^ (version age), newest version has age 0
						float keepChance = 1 - RS_CLAMP(config.oldVersionRecencyBias, 0, 1);

						float totalWeight = 0;
						auto weights = std::vector<float>(numVersions);
						float curWeight = 1;
						for (int age = 0; age < numVersions; age++) {
							weights[numVersions - 1 - age] = curWeight;
							totalWeight += curWeight;
							curWeight *= keepChance;
						}

						oldVersionIdx = numVersions - 1;
						float roll = RocketSim::Math::RandFloat(0, totalWeight);
						for (int i = 0; i < numVersions; i++) {
							roll -= weights[i];
							if (roll <= 0) {
								oldVersionIdx = i;
								break;
							}
						}
					} else {
						oldVersionIdx = RocketSim::Math::RandInt(0, numVersions);
					}

					oldVersion = &versionMgr->versions[oldVersionIdx];

					Team oldVersionTeam = Team(RocketSim::Math::RandInt(0, 2)); 
					
					int i = 0;
					for (auto& state : envSet->state.gameStates) {
						for (auto& player : state.players) {
							if (player.team == oldVersionTeam) {
								oldPlayerIndices.push_back(i);
							} else {
								newPlayerIndices.push_back(i);
							}
							i++;
						}
					}

					// NOTE: Index tensors must be int64 for index_copy_()
					tNewPlayerIndices = torch::tensor(newPlayerIndices, torch::kInt64);
					tOldPlayerIndices = torch::tensor(oldPlayerIndices, torch::kInt64);
				}
			}

			if (!oldVersion) {
				newPlayerIndices.reserve(numPlayers);
				for (int i = 0; i < numPlayers; i++)
					newPlayerIndices.push_back(i);
			}

			if (config.trainAgainstOldVersions && !render)
				report["Trained Against Old Version"] = (oldVersion != NULL);

			{
				// Players that stopped being recorded (i.e. switched to old-version control) have their
				//	unfinished trajectories force-truncated, otherwise those trajectories would resume later
				//	with a gap in the middle and corrupt learning
				auto recordedMask = std::vector<bool>(numPlayers, false);
				for (int newPlayerIdx : newPlayerIndices)
					recordedMask[newPlayerIdx] = true;

				for (int i = 0; i < numPlayers; i++) {
					if (prevRecordedMask[i] && !recordedMask[i] && trajectories[i].Length() > 0) {
						auto& traj = trajectories[i];

						// Truncation requires the next state for the critic
						FList nextState = envSet->state.obs.GetRow(i);

						if (ContainsNonFinite(nextState.data(), nextState.size())) {
							// The env diverged at the iteration boundary, this trajectory can't be bootstrapped
							traj.Clear();
							continue;
						}

						// The trajectory always ends mid-episode here (otherwise it would have been consumed already)
						traj.terminals.back() = RLGC::TerminalType::TRUNCATED;

						fnStandardizeRow(nextState);
						traj.nextStates += nextState;

						pendingTruncated.Append(traj);
						traj.Clear();
					}
				}

				prevRecordedMask = recordedMask;
			}

			int numRealPlayers = oldVersion ? newPlayerIndices.size() : envSet->state.numPlayers;

			int stepsCollected = 0;
			{ // Generate experience

				// Only contains complete episodes
				auto combinedTraj = Trajectory();

				// Include trajectories that were truncated by old-version switching
				// (Their timesteps were already counted in the iteration they were collected)
				if (pendingTruncated.Length() > 0) {
					combinedTraj.Append(pendingTruncated);
					pendingTruncated.Clear();
				}

				Timer collectionTimer = {};
				{ // Collect timesteps
					RG_NO_GRAD;

					if (limitCollectionTorchThreads)
						at::set_num_threads(config.collectionTorchThreads);

					float inferTime = 0;
					float envStepTime = 0;

					for (int step = 0; combinedTraj.Length() < config.ppo.tsPerItr || render; step++, stepsCollected += numRealPlayers) {
						Timer stepTimer = {};
						envSet->Reset();
						envStepTime += stepTimer.Elapsed();

						if (ContainsNonFinite(envSet->state.obs.data.data(), envSet->state.obs.data.size())) {
							// One or more arenas produced non-finite obs
							// This is either a physics divergence (extreme collisions can very rarely
							//	make RocketSim produce NaN states) or a bugged obs builder
							// Physics divergences are recovered from by resetting the affected arenas,
							//	a bugged obs builder is a fatal error (see below)

							// Find and reset the affected arenas, discarding their poisoned trajectories
							int numBadArenas = 0;
							for (int arenaIdx = 0; arenaIdx < envSet->arenas.size(); arenaIdx++) {
								int playerStartIdx = envSet->state.arenaPlayerStartIdx[arenaIdx];
								int playersInArena = envSet->state.gameStates[arenaIdx].players.size();

								bool arenaBad = ContainsNonFinite(
									&envSet->state.obs.At(playerStartIdx, 0),
									(size_t)playersInArena * obsSize
								);
								if (!arenaBad)
									continue;

								numBadArenas++;
								nanResetsThisIteration++;
								RG_LOG(
									"WARNING: Non-finite values in the obs of arena " << arenaIdx << ", resetting it " <<
									"(extreme collisions can very rarely diverge the physics; " <<
									"the arena's in-progress episode data will be discarded)"
								);

								for (int i = 0; i < playersInArena; i++)
									trajectories[playerStartIdx + i].Clear();

								envSet->ResetArena(arenaIdx);
								envSet->state.terminals[arenaIdx] = 0;
							}

							report.Add("Env NaN Resets", numBadArenas);

							// If the obs are STILL bad after resetting, the obs builder itself is broken
							if (ContainsNonFinite(envSet->state.obs.data.data(), envSet->state.obs.data.size())) {
								for (int i = 0; i < envSet->state.numPlayers; i++)
									for (int j = 0; j < obsSize; j++)
										if (!std::isfinite(envSet->state.obs.At(i, j)))
											RG_ERR_CLOSE(
												"Obs builder produced a NaN/inf value at obs index " << j <<
												" (player index " << i << ", value: " << envSet->state.obs.At(i, j) << "), " <<
												"even for a freshly-reset state.\n" <<
												"Check your obs builder for divisions by zero, normalizations of zero-length vectors, etc."
											);
							}

							// A reasonable training setup should only ever hit NaN resets very rarely
							// Hitting many in one iteration means something is deterministically broken
							if (nanResetsThisIteration > RS_MAX(envSet->arenas.size(), 16))
								RG_ERR_CLOSE(
									"Env state repeatedly contained NaN/inf values (" << nanResetsThisIteration << " arena resets this iteration).\n" <<
									"Something is deterministically broken (bugged state setter, reward, or physics-breaking custom setup)."
								);
						}

						if (!render && obsStat) {
							// TODO: This samples from old versions too
							int numSamples = RS_MIN(envSet->state.numPlayers, config.maxObsSamples);
							for (int i = 0; i < numSamples; i++) {
								int idx = Math::RandInt(0, envSet->state.numPlayers);
								obsStat->IncrementRow(&envSet->state.obs.At(idx, 0));
							}

							lastObsMean = obsStat->GetMean();
							lastObsStd = obsStat->GetSTD();
							for (double& f : lastObsMean)
								f = RS_CLAMP(f, -config.maxObsMeanRange, config.maxObsMeanRange);
							for (double& f : lastObsStd)
								f = RS_MAX(f, config.minObsSTD);
							for (int i = 0; i < envSet->state.numPlayers; i++) {
								for (int j = 0; j < obsSize; j++) {
									float& obsVal = envSet->state.obs.At(i, j);
									obsVal = (obsVal - lastObsMean[j]) / lastObsStd[j];
								}
							}
						}

						torch::Tensor tActions, tLogProbs;
						torch::Tensor tStates = DIMLIST2_TO_TENSOR<float>(envSet->state.obs);
						torch::Tensor tActionMasks = DIMLIST2_TO_TENSOR<uint8_t>(envSet->state.actionMasks);

						if (!render) {
							for (int newPlayerIdx : newPlayerIndices) {
								envSet->state.obs.AppendRowTo(newPlayerIdx, trajectories[newPlayerIdx].states);
								envSet->state.actionMasks.AppendRowTo(newPlayerIdx, trajectories[newPlayerIdx].actionMasks);
							}
						}

						envSet->StepFirstHalf(true);

						Timer inferTimer = {};

						if (oldVersion) {
							torch::Tensor tdNewStates = tStates.index_select(0, tNewPlayerIndices).to(ppo->device, true);
							torch::Tensor tdOldStates = tStates.index_select(0, tOldPlayerIndices).to(ppo->device, true);
							torch::Tensor tdNewActionMasks = tActionMasks.index_select(0, tNewPlayerIndices).to(ppo->device, true);
							torch::Tensor tdOldActionMasks = tActionMasks.index_select(0, tOldPlayerIndices).to(ppo->device, true);

							torch::Tensor tNewActions;
							torch::Tensor tOldActions;

							ppo->InferActions(tdNewStates, tdNewActionMasks, &tNewActions, &tLogProbs);
							ppo->InferActions(tdOldStates, tdOldActionMasks, &tOldActions, NULL, &oldVersion->models);

							tActions = torch::zeros(numPlayers, tNewActions.dtype());
							tActions.index_copy_(0, tNewPlayerIndices, tNewActions.cpu());
							tActions.index_copy_(0, tOldPlayerIndices, tOldActions.cpu());
						} else {
							torch::Tensor tdStates = tStates.to(ppo->device, true);
							torch::Tensor tdActionMasks = tActionMasks.to(ppo->device, true);
							ppo->InferActions(tdStates, tdActionMasks, &tActions, &tLogProbs);
							tActions = tActions.cpu();
						}
						inferTime += inferTimer.Elapsed();

						auto curActions = TENSOR_TO_VEC<int>(tActions);
						FList newLogProbs;
						if (tLogProbs.defined() && !render)
							newLogProbs = TENSOR_TO_VEC<float>(tLogProbs);	

						stepTimer.Reset();
						envSet->Sync(); // Make sure the first half is done
						envSet->StepSecondHalf(curActions, false);
						envStepTime += stepTimer.Elapsed();

						if (stepCallback)
							stepCallback(this, envSet->state.gameStates, report);

						if (render) {
							renderSender->Send(envSet->state.gameStates[0]);
							continue;
						}

						// Calc average rewards
						if (config.addRewardsToMetrics && (Math::RandInt(0, config.rewardSampleRandInterval) == 0)) {
							int numSamples = RS_MIN(envSet->arenas.size(), config.maxRewardSamples);
							std::unordered_map<std::string, AvgTracker> avgRewards = {};
							for (int i = 0; i < numSamples; i++) {
								int arenaIdx = Math::RandInt(0, envSet->arenas.size());
								auto& prevRewards = envSet->state.lastRewards[arenaIdx];
								if (prevRewards.empty())
									continue; // This arena hasn't stepped yet

								for (int j = 0; j < envSet->rewards[arenaIdx].size(); j++) {
									std::string rewardName = envSet->rewards[arenaIdx][j].reward->GetName();
									avgRewards[rewardName] += prevRewards[j];
								}
							}

							for (auto& pair : avgRewards)
								report.AddAvg("Rewards/" + pair.first, pair.second.Get());
						}

						// Now that we've inferred and stepped the env, we can add that stuff to the trajectories
						int i = 0;
						for (int newPlayerIdx : newPlayerIndices) {
							trajectories[newPlayerIdx].actions.push_back(curActions[newPlayerIdx]);
							trajectories[newPlayerIdx].rewards += envSet->state.rewards[newPlayerIdx];
							trajectories[newPlayerIdx].logProbs += newLogProbs[i];
							i++;
						}

						auto curTerminals = std::vector<uint8_t>(numPlayers, 0);
						for (int idx = 0; idx < envSet->arenas.size(); idx++) {
							uint8_t terminalType = envSet->state.terminals[idx];
							if (!terminalType)
								continue;

							auto playerStartIdx = envSet->state.arenaPlayerStartIdx[idx];
							int playersInArena = envSet->state.gameStates[idx].players.size();
							for (int i = 0; i < playersInArena; i++)
								curTerminals[playerStartIdx + i] = terminalType;
						}

						for (int newPlayerIdx : newPlayerIndices) {
							int8_t terminalType = curTerminals[newPlayerIdx];
							auto& traj = trajectories[newPlayerIdx];

							if (!terminalType && traj.Length() >= maxEpisodeLength) {
								// Episode is too long, truncate it here
								// This won't actually reset the env, but rather will just add it to experience buffer as truncated
								terminalType = RLGC::TerminalType::TRUNCATED;
							}

							traj.terminals.push_back(terminalType);
							if (terminalType) {

								if (terminalType == RLGC::TerminalType::TRUNCATED) {
									// Truncation requires an additional next state for the critic
									// NOTE: Standardized to match the states the critic is trained on
									FList nextState = envSet->state.obs.GetRow(newPlayerIdx);
									fnStandardizeRow(nextState);
									traj.nextStates += nextState;
								}

								combinedTraj.Append(traj);
								traj.Clear();
							}
						}
					}

					report["Inference Time"] = inferTime;
					report["Env Step Time"] = envStepTime;

					// Consumption gets torch's full thread count back
					if (limitCollectionTorchThreads)
						at::set_num_threads(defaultTorchThreads);
				}
				float collectionTime = collectionTimer.Elapsed();

				Timer consumptionTimer = {};
				{ // Process timesteps
					RG_NO_GRAD;

					// Make and transpose tensors
					torch::Tensor tStates = VEC_TO_TENSOR(combinedTraj.states).reshape({ -1, obsSize });
					torch::Tensor tActionMasks = VEC_TO_TENSOR(combinedTraj.actionMasks).reshape({ -1, numActions });
					// NOTE: Actions are used as gather() indices during learning, which requires int64
					torch::Tensor tActions = VEC_TO_TENSOR(combinedTraj.actions).to(torch::kInt64);
					torch::Tensor tLogProbs = VEC_TO_TENSOR(combinedTraj.logProbs);
					torch::Tensor tRewards = VEC_TO_TENSOR(combinedTraj.rewards);
					torch::Tensor tTerminals = VEC_TO_TENSOR(combinedTraj.terminals);

					// States we truncated at (there could be none)
					torch::Tensor tNextTruncStates;
					if (!combinedTraj.nextStates.empty())
						tNextTruncStates = VEC_TO_TENSOR(combinedTraj.nextStates).reshape({ -1, obsSize });

					report["Average Step Reward"] = tRewards.mean().item<float>();
					report["Collected Timesteps"] = stepsCollected;

					// Runs the critic over a tensor of states, minibatched to limit device memory usage
					auto fnInferCriticBatched = [&](torch::Tensor tInStates) {
						int64_t numStates = tInStates.size(0);

						if (ppo->device.is_cpu() || numStates <= ppo->config.miniBatchSize)
							return ppo->InferCritic(tInStates.to(ppo->device, true, true)).cpu();

						torch::Tensor tOutPreds = torch::zeros({ numStates });
						for (int64_t i = 0; i < numStates; i += ppo->config.miniBatchSize) {
							int64_t start = i;
							int64_t end = RS_MIN(i + ppo->config.miniBatchSize, numStates);
							torch::Tensor tStatesPart = tInStates.slice(0, start, end);

							auto valPredsPart = ppo->InferCritic(tStatesPart.to(ppo->device, true, true)).cpu();
							RG_ASSERT(valPredsPart.size(0) == (end - start));
							tOutPreds.slice(0, start, end).copy_(valPredsPart, true);
						}
						return tOutPreds;
					};

					torch::Tensor tValPreds = fnInferCriticBatched(tStates);
					torch::Tensor tTruncValPreds;
					if (tNextTruncStates.defined())
						tTruncValPreds = fnInferCriticBatched(tNextTruncStates);

					float normalTerminalPortion = (tTerminals == RLGC::TerminalType::NORMAL).to(torch::kFloat32).mean().item<float>();
					if (normalTerminalPortion > 0)
						report["Episode Length"] = 1.f / normalTerminalPortion;

					Timer gaeTimer = {};
					// Run GAE
					torch::Tensor tAdvantages, tTargetVals, tReturns;
					float rewClipPortion;
					GAE::Compute(
						tRewards, tTerminals, tValPreds, tTruncValPreds,
						tAdvantages, tTargetVals, tReturns, rewClipPortion,
						config.ppo.gaeGamma, config.ppo.gaeLambda, returnStat ? returnStat->GetSTD() : 1, config.ppo.rewardClipRange
					);
					report["GAE Time"] = gaeTimer.Elapsed();
					report["Clipped Reward Portion"] = rewClipPortion;

					if (returnStat) {
						report["GAE/Returns STD"] = returnStat->GetSTD();

						int numToIncrement = RS_MIN(config.maxReturnSamples, tReturns.size(0));
						if (numToIncrement > 0) {
							auto selectedReturns = tReturns.index_select(0, torch::randint(tReturns.size(0), { (int64_t)numToIncrement }));
							returnStat->Increment(TENSOR_TO_VEC<float>(selectedReturns));
						}
					}
					report["GAE/Avg Return"] = tReturns.abs().mean().item<float>();
					report["GAE/Avg Advantage"] = tAdvantages.abs().mean().item<float>();
					report["GAE/Avg Val Target"] = tTargetVals.abs().mean().item<float>();

					// Set experience buffer
					experience.data.actions = tActions;
					experience.data.logProbs = tLogProbs;
					experience.data.actionMasks = tActionMasks;
					experience.data.states = tStates;
					experience.data.advantages = tAdvantages;
					experience.data.targetValues = tTargetVals;

					if (config.ppo.experienceOnDevice && ppo->device.is_cuda()) {
						// Upload the whole iteration's experience once,
						//	instead of once per minibatch per epoch
						for (auto* t = experience.data.begin(); t != experience.data.end(); t++)
							*t = t->to(ppo->device, true);
					}
				}

				// Free CUDA cache
#ifdef RG_CUDA_SUPPORT
				if (ppo->device.is_cuda())
					c10::cuda::CUDACachingAllocator::emptyCache();
#endif

				// Learn
				Timer learnTimer = {};
				ppo->Learn(experience, report, isFirstIteration);
				report["PPO Learn Time"] = learnTimer.Elapsed();

				// Set metrics
				float consumptionTime = consumptionTimer.Elapsed();
				report["Collection Time"] = collectionTime;
				report["Consumption Time"] = consumptionTime;
				report["Collection Steps/Second"] = stepsCollected / collectionTime;
				report["Consumption Steps/Second"] = stepsCollected / consumptionTime;
				report["Overall Steps/Second"] = stepsCollected / (collectionTime + consumptionTime);

				uint64_t prevTimesteps = totalTimesteps;
				totalTimesteps += stepsCollected;
				report["Total Timesteps"] = totalTimesteps;
				totalIterations++;
				report["Total Iterations"] = totalIterations;

				if (versionMgr)
					versionMgr->OnIteration(ppo, report, totalTimesteps, prevTimesteps);

				bool timestepLimitReached =
					(config.timestepLimit > 0) && (totalTimesteps >= (uint64_t)config.timestepLimit);

				if (g_StopSignalReceived) {
					RG_LOG("Stop signal received (Ctrl+C/SIGTERM), saving and exiting...");
					saveQueued = true;
				}

				if (saveQueued) {
					if (!config.checkpointFolder.empty())
						Save();
					exit(0);
				}

				if (!config.checkpointFolder.empty()) {
					if (timestepLimitReached || (totalTimesteps / config.tsPerSave > prevTimesteps / config.tsPerSave)) {
						// Auto-save
						// A failed auto-save (e.g. disk full) is not worth killing the run over,
						//	we can just try again at the next save interval
						try {
							Save();
						} catch (std::exception& e) {
							RG_LOG(
								"WARNING: Failed to save checkpoint (training continues, will retry at the next save interval).\n" <<
								"Exception: " << e.what()
							);
						}
					}
				}

				report.Finish();

				if (metricSender)
					metricSender->Send(report);

				report.Display(
					{
						"Average Step Reward",
						"Policy Entropy",
						"Mean KL Divergence",
						"Policy Loss",
						"Critic Loss",
						"",
						"Policy Update Magnitude",
						"Critic Update Magnitude",
						"",
						"Collection Steps/Second",
						"Consumption Steps/Second",
						"Overall Steps/Second",
						"",
						"Collection Time",
						"-Inference Time",
						"-Env Step Time",
						"Consumption Time",
						"-GAE Time",
						"-PPO Learn Time",
						"",
						"Collected Timesteps",
						"Total Timesteps",
						"Total Iterations"
					}
				);

				if (timestepLimitReached) {
					RG_LOG("Learner: Timestep limit of " << config.timestepLimit << " reached, stopping training.");
					return;
				}
			}
		}
		
	} catch (std::exception& e) {
		RG_ERR_CLOSE("Exception thrown during main learner loop: " << e.what());
	}
}

GGL::Learner::~Learner() {
	delete ppo;
	delete versionMgr;
	delete metricSender;
	delete renderSender;
	delete envSet;
	delete returnStat;
	delete obsStat;

	if (_ownsPyInterpreter)
		pybind11::finalize_interpreter();
}