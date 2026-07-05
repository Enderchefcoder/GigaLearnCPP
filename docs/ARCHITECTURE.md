# Architecture

How GigaLearn actually works, end to end. Useful when tuning performance or extending the framework.

## The big picture

GigaLearn runs everything in **one process**:

```
┌─────────────────────────────── Learner ────────────────────────────────┐
│                                                                        │
│   ┌── EnvSet (RLGymCPP) ──┐      ┌── PPOLearner (libtorch) ─────────┐  │
│   │ Arena 0 ─ RocketSim   │ obs  │ shared head ─┬─ policy (actions) │  │
│   │ Arena 1 ─ RocketSim   │ ───> │              └─ critic (values)  │  │
│   │ ...      (thread pool)│ <─── │                                  │  │
│   │ Arena N ─ RocketSim   │ acts │ Learn(): PPO on collected data   │  │
│   └───────────────────────┘      └──────────────────────────────────┘  │
│                                                                        │
│   PolicyVersionManager (old versions, skill rating)                    │
│   MetricSender / RenderSender (embedded Python)                        │
└────────────────────────────────────────────────────────────────────────┘
```

There is no inter-process communication and no per-env model copy: all players in all arenas are batched into **one inference call** per step.

## The iteration loop

`Learner::Start()` loops over iterations, each with two phases:

### 1. Collection

Repeats until `tsPerItr` timesteps of *complete episodes* are gathered:

1. `EnvSet::Reset()` — arenas whose episode ended are reset by their state setters (parallelized on the thread pool)
2. Obs are validated (NaN/inf check) and optionally standardized
3. `StepFirstHalf()` — arenas step `actionDelay` ticks with their *previous* actions, asynchronously on the thread pool
4. **While arenas are stepping**, the policy infers actions for every player from the pre-step obs (this is the collection/inference overlap that hides inference latency)
5. `StepSecondHalf()` — arenas apply the new actions for the remaining `tickSkip - actionDelay` ticks, then compute rewards, terminals, next obs, and action masks
6. Each player's `(state, mask, action, logProb, reward, terminal)` is appended to its **trajectory**; finished trajectories move to the iteration's experience

The `actionDelay` mechanic reproduces Rocket League's real input latency: the policy sees a state, but its action only takes effect `actionDelay` ticks later.

Episodes longer than `maxEpisodeDuration` are *truncated in the buffer* — the env keeps running, but the trajectory is split, with the critic bootstrapping the cut point's value from the next state.

### 2. Consumption

1. Trajectories are flattened into tensors
2. The critic predicts values for all states (minibatched on GPU)
3. **GAE** computes advantages and value targets; rewards are scaled by the running return STD and clipped to `rewardClipRange`
4. `PPOLearner::Learn()` runs `epochs` passes; each pass shuffles the experience into batches, splits batches into minibatches (gradient accumulation), computes the clipped PPO objective + entropy bonus + critic MSE loss, clips gradient norms, and steps the optimizers

Metrics accumulate as device tensors and synchronize once per iteration.

## Models

Three networks (see `Util/Models.h`):

- **shared_head** *(optional, on by default)*: obs → shared representation
- **policy**: representation → one logit per action
- **critic**: representation → state value

Action masking sets invalid actions' logits to a huge negative value before softmax, so they get ~zero probability during collection, learning, and inference.

Checkpoints store each model (`POLICY.lt`, `CRITIC.lt`, `SHARED_HEAD.lt`) plus optimizer state (`*_OPTIM.lt`) plus `RUNNING_STATS.json` (timesteps, return/obs stats, wandb run ID). Model files are raw libtorch serialization and can be inspected from Python (`tools/checkpoint_converter.py`).

## Environment framework (RLGymCPP)

Each arena bundles what `EnvCreateFunc` returned:
- **Arena** — the RocketSim simulation
- **ObsBuilder** — builds each player's observation vector
- **ActionParser** — defines the discrete action table + per-state action masks
- **Rewards** — list of weighted reward functions
- **TerminalConditions** — decide when episodes end (terminal vs truncation)
- **StateSetter** — sets up arenas on reset

`GameState` wraps the arena state each step, adding previous-state access (`state.prev`, `player.prev`), per-step events (`player.eventState.goal/shot/save/bump/demo/...` driven by RocketSim's event tracker), boost pad state (normal + inverted perspectives), and scoring info.

All arena stepping, resetting, and obs building runs on a shared work-stealing thread pool sized to your hardware concurrency.

## Self-play systems

- **PolicyVersionManager** snapshots the policy every `tsPerVersion` timesteps (stored under `checkpoints/policy_versions/`)
- **Skill tracker**: every `updateInterval` iterations, plays the current policy against a random version on a separate set of arenas (kickoff-only, goal-terminal) and updates ELO-style ratings per game mode (`Rating/1v1` etc.). Games that don't finish continue next time ("continuation")
- **Training against old versions**: with probability `trainAgainstOldChance`, an iteration assigns one team to a random old version. Only current-policy players are recorded into experience. When players switch between recorded/unrecorded across iterations, their partial trajectories are force-truncated (with proper value bootstraps) to keep the experience stream clean

## Embedded Python

The learner embeds a Python interpreter (pybind11) for:
- `python_scripts/metric_receiver.py` — wandb logging (JSONL fallback)
- `python_scripts/render_receiver.py` — streams states to RocketSimVis over UDP

Both are plain Python files next to the executable — edit them freely; the exe directory and CWD are on `sys.path`.

## Performance notes

- **Torch thread management**: torch's idle intra-op workers spin-wait, starving the env threads during collection. The learner limits torch to `collectionTorchThreads` (default 1) while collecting and restores the full count for learning — on a 4-core CPU test machine this made collection 2.4x faster (+58% overall). Set it to 0 for torch's default behavior.
- **Collection throughput** scales with `numGames` and CPU threads until inference becomes the bottleneck; GPU + `useHalfPrecision` helps large models
- **Consumption throughput** is GPU-bound; `miniBatchSize` trades VRAM for speed
- **`-DGGL_NATIVE_ARCH=ON`** compiles the simulation for your exact CPU (faster stepping, non-portable binaries)
- The obs NaN check, tensor conversions, and GAE are deliberately flat, vectorizable loops — profile before "optimizing" them further
- Half precision (bfloat16) is only used for collection/inference; learning always runs fp32
