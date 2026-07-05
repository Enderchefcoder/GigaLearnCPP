# Getting Started

This walks through [`src/ExampleMain.cpp`](../src/ExampleMain.cpp) — a complete training setup — and explains each part. Once you understand it, copy it and make it your own.

## The environment creation function

Every game instance ("arena") is created by a function you provide. It is called once per game at startup, in parallel:

```cpp
EnvCreateResult EnvCreateFunc(int index) {
	// Rewards: pairs of (reward function, weight)
	std::vector<WeightedReward> rewards = {
		{ new FaceBallReward(), 0.25f },
		{ new VelocityPlayerToBallReward(), 4.f },
		{ new ZeroSumReward(new VelocityBallToGoalReward(), 1), 2.0f },
		{ new GoalReward(), 150 },
		// ...
	};

	// Episodes end when any of these trigger
	std::vector<TerminalCondition*> terminalConditions = {
		new NoTouchCondition(10),   // Truncate if nobody touches the ball for 10s
		new GoalScoreCondition()    // End the episode on goals
	};

	// Build the RocketSim arena yourself: game mode, team sizes, mutators...
	auto arena = Arena::Create(GameMode::SOCCAR);
	arena->AddCar(Team::BLUE);
	arena->AddCar(Team::ORANGE);

	EnvCreateResult result = {};
	result.actionParser = new DefaultAction();      // Maps policy outputs to car controls
	result.obsBuilder = new AdvancedObs();          // Builds the policy's input vector
	result.stateSetter = new KickoffState();        // Resets arenas between episodes
	result.terminalConditions = terminalConditions;
	result.rewards = rewards;
	result.arena = arena;
	return result;
}
```

Things you can vary per-arena using `index`: game mode, team sizes, mutators, state setters — mixing 1v1/2v2/3v3 arenas in one run is supported (the obs builder must produce the same obs size for all of them, e.g. `DefaultObsPadded`).

The `EnvSet` takes ownership of everything you return here.

## The step callback (custom metrics)

The step callback runs once per collection step with all game states. Use it to log custom metrics through the `Report`:

```cpp
void StepCallback(Learner* learner, const std::vector<GameState>& states, Report& report) {
	for (auto& state : states) {
		for (auto& player : state.players) {
			report.AddAvg("Player/In Air Ratio", !player.isOnGround);
			report.AddAvg("Player/Speed", player.vel.Length());
		}

		if (state.goalScored)
			report.AddAvg("Game/Goal Speed", state.ball.vel.Length());
	}
}
```

`report.AddAvg()` accumulates an average that is finalized at the end of the iteration and sent to your metrics receiver (wandb by default).

## The learner configuration

```cpp
LearnerConfig cfg = {};

cfg.deviceType = LearnerDeviceType::AUTO; // CUDA GPU if available, else CPU

// Simulation timing
cfg.tickSkip = 8;                    // Policy acts every 8 physics ticks (15Hz)
cfg.actionDelay = cfg.tickSkip - 1;  // Delay before actions apply (see docs/CONFIGURATION.md)

// Parallel games; more games = better GPU utilization, more RAM
cfg.numGames = 256;

// Iteration sizing
int tsPerItr = 50'000;
cfg.ppo.tsPerItr = tsPerItr;      // Timesteps collected per iteration
cfg.ppo.batchSize = tsPerItr;     // Timesteps per learning batch
cfg.ppo.miniBatchSize = 50'000;   // Lower if you run out of VRAM
cfg.ppo.epochs = 1;               // Learning passes per iteration

// Learning rates and entropy
cfg.ppo.policyLR = 1.5e-4;
cfg.ppo.criticLR = 1.5e-4;
cfg.ppo.entropyScale = 0.035f;    // Higher = more exploration

// Model architecture
cfg.ppo.sharedHead.layerSizes = { 256, 256 };  // Shared layers feeding both networks
cfg.ppo.policy.layerSizes = { 256, 256, 256 };
cfg.ppo.critic.layerSizes = { 256, 256, 256 };
```

Then create the learner and go:

```cpp
Learner* learner = new Learner(EnvCreateFunc, cfg, StepCallback);
learner->Start();
```

## What you'll see

Each iteration prints a report:

```
Average Step Reward: 0.0231
Policy Entropy: 0.9847
Mean KL Divergence: 0.0021
...
Collection Steps/Second: 14,382
Consumption Steps/Second: 61,204
Overall Steps/Second: 11,648
...
Total Timesteps: 12,050,000
```

Key things to watch:
- **Average Step Reward** trending up = learning is working
- **Policy Entropy** near 1.0 = random play; slowly decreasing = policy is committing to behavior; crashing to ~0 = entropy collapse (raise `entropyScale`)
- **Collection vs Consumption Steps/Second** tells you whether env stepping or PPO learning is your bottleneck

## Checkpoints

- Saved every `cfg.tsPerSave` timesteps into `cfg.checkpointFolder` (default `checkpoints/`), as timestep-numbered subfolders
- The newest checkpoint is loaded automatically at startup
- `cfg.checkpointsToKeep` controls how many old checkpoints are kept (default 8)
- Press `Q` in the console to save and quit cleanly

## Watching your bot play

Set `cfg.renderMode = true` to run a single game in real time and stream states to [RocketSimVis](https://github.com/ZealanL/RocketSimVis) (run its receiver first). `cfg.renderTimeScale` controls playback speed. Render mode doesn't train — use it to inspect behavior of a checkpoint.

## Playing in Rocket League with RLBot

`src/RLBotClient.h` runs your trained model as an [RLBot](https://rlbot.org/) bot:

```cpp
RLBotParams params = {};
params.port = 23233; // Match rlbot/port.cfg
params.tickSkip = 8;
params.actionDelay = 7;
params.inferUnit = new InferUnit(
	new AdvancedObs(), obsSize, new DefaultAction(),
	sharedHeadConfig, policyConfig,
	"path/to/checkpoint", false
);
RLBotClient::Run(params);
```

See `rlbot/README.md` for setting up the RLBot side (config files, auto-start).

## Where to go next

- [CONFIGURATION.md](CONFIGURATION.md) — every option explained, including self-play, skill rating, and transfer learning
- [ARCHITECTURE.md](ARCHITECTURE.md) — what actually happens each iteration
- [MIGRATING.md](MIGRATING.md) — porting code from RLGymPPO-CPP
