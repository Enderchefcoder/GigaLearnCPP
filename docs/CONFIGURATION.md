# Configuration Reference

All learner behavior is controlled through `LearnerConfig` (which contains `PPOLearnerConfig` and `SkillTrackerConfig`). This documents every field with guidance.

## LearnerConfig

### Simulation

| Field | Default | Description |
| --- | --- | --- |
| `numGames` | 300 | Parallel arenas. Scale to your CPU/RAM; higher improves GPU inference batching. |
| `tickSkip` | 8 | Physics ticks per policy action. 8 = 15 actions/sec (the standard). |
| `actionDelay` | 7 | Ticks after a policy decision before the action takes effect. Rocket League itself has input delay, so `tickSkip - 1` matches other RLGym frameworks. Lower values react faster in sim but transfer worse to the real game. |
| `randomSeed` | -1 | -1 seeds from the current time. The seed strongly affects early training. |
| `deviceType` | `AUTO` | `AUTO` uses a CUDA GPU if libtorch can access one, else CPU. Force with `CPU`/`GPU_CUDA`. |

### Render mode

| Field | Default | Description |
| --- | --- | --- |
| `renderMode` | false | Run a single arena in real time and stream it to the render receiver ([RocketSimVis](https://github.com/ZealanL/RocketSimVis)). No training happens. |
| `renderTimeScale` | 1.0 | Game speed multiplier in render mode. |

### Checkpoints

| Field | Default | Description |
| --- | --- | --- |
| `checkpointFolder` | `"checkpoints"` | Checkpoints save into timestep-numbered subfolders. Empty disables saving. The newest checkpoint is auto-loaded at startup. |
| `tsPerSave` | 1,000,000 | Timesteps between auto-saves. 0 = save every iteration. |
| `checkpointsToKeep` | 8 | Older checkpoints are deleted beyond this count. -1 keeps everything. |

### Standardization

| Field | Default | Description |
| --- | --- | --- |
| `standardizeReturns` | true | Track return statistics and scale rewards by the return STD for the critic. Don't disable unless you know what you're doing. |
| `maxReturnSamples` | 150 | Return samples per iteration for the running STD. |
| `standardizeObs` | false | Standardize observations with running mean/STD per obs index. Usually unnecessary if your obs builder outputs sane ranges. |
| `minObsSTD` | 0.1 | Lower clamp for obs STD (prevents huge multipliers on near-constant obs). |
| `maxObsMeanRange` | 3 | Clamp for obs mean correction. |
| `maxObsSamples` | 100 | Max obs rows sampled per step for the running stats. |

### Metrics

| Field | Default | Description |
| --- | --- | --- |
| `sendMetrics` | true | Send reports to `python_scripts/metric_receiver.py` (wandb, or local JSONL fallback). |
| `metricsProjectName` | `"gigalearncpp"` | wandb project name. |
| `metricsGroupName` | `"unnamed-runs"` | wandb group name. |
| `metricsRunName` | `"gigalearncpp-run"` | wandb run name. Runs auto-resume via the run ID stored in checkpoints. |
| `addRewardsToMetrics` | true | Log each reward's average as `Rewards/<name>`. |
| `maxRewardSamples` | 50 | Max arenas sampled per reward logging step. |
| `rewardSampleRandInterval` | 8 | Random interval range between reward sampling steps. |

### Self-play (policy versions)

| Field | Default | Description |
| --- | --- | --- |
| `savePolicyVersions` | false | Periodically snapshot the policy (forced on by the skill tracker / training against old versions). Versions save inside the checkpoint folder. |
| `tsPerVersion` | 25,000,000 | Timesteps between version snapshots. |
| `maxOldVersions` | 32 | Version storage cap (oldest deleted first). |
| `trainAgainstOldVersions` | false | Some iterations pit the current policy against a random old version (one team each). Only the current policy's experience is learned from. |
| `trainAgainstOldChance` | 0.15 | Chance that an iteration trains against an old version. |
| `skillTracker` | see below | ELO-style rating of the current policy vs old versions. |

## PPOLearnerConfig (`cfg.ppo`)

### Iteration sizing

| Field | Default | Description |
| --- | --- | --- |
| `tsPerItr` | 50,000 | Timesteps collected per iteration. |
| `batchSize` | 50,000 | Timesteps per PPO batch. Must be ≤ collected timesteps. |
| `miniBatchSize` | 0 | Splits batches into minibatches for gradient accumulation (reduces VRAM). 0 = use `batchSize`. Must divide `batchSize` evenly. |
| `overbatching` | true | The final batch absorbs leftover experience (up to 2x batch size) instead of discarding it. |
| `epochs` | 2 | Learning passes over each iteration's experience. 1-3 is typical. |
| `maxEpisodeDuration` | 120 | Max episode length in seconds before the episode is truncated in the experience buffer (the env keeps running). |

### Core PPO

| Field | Default | Description |
| --- | --- | --- |
| `policyLR` / `criticLR` | 3e-4 | Learning rates. Lowering as the bot matures is standard practice. Setting one to 0 freezes that model. |
| `entropyScale` | 0.018 | Scale of the normalized entropy bonus. Unlike `ent_coef` in other frameworks, entropy here is normalized by the action count, so this doesn't need retuning when you change action parsers. |
| `maskEntropy` | false | If true, entropy is normalized only over currently-valid actions. |
| `clipRange` | 0.2 | PPO ratio clip range. |
| `normalizeAdvantages` | false | Normalize advantages within each minibatch (common PPO trick; discards advantage magnitude information but can stabilize training). |
| `gaeGamma` | 0.99 | Reward discount rate. Lower values favor short-term reward; starting lower (0.99) and raising later is common. |
| `gaeLambda` | 0.95 | GAE smoothing parameter. |
| `rewardClipRange` | 10 | Clip range for standardized rewards. 0 disables. |
| `policyTemperature` | 1 | Softmax temperature of the policy distribution. |

### Models (`policy`, `critic`, `sharedHead`)

| Field | Default | Description |
| --- | --- | --- |
| `layerSizes` | `{256,256,256}` (head: `{256}`) | Hidden layer sizes. Set `sharedHead.layerSizes = {}` to disable shared layers. |
| `activationType` | `RELU` | `RELU`, `LEAKY_RELU`, `SIGMOID`, or `TANH`. |
| `optimType` | `ADAM` | `ADAM`, `ADAMW`, `ADAGRAD`, `RMSPROP`, or `MAGSGD`. |
| `addLayerNorm` | true | LayerNorm after each hidden layer (recommended). |

The shared head feeds both the policy and critic; its learning rate is the minimum of the two.

### Inference

| Field | Default | Description |
| --- | --- | --- |
| `deterministic` | false | Always pick the highest-probability action. Better play, terrible learning — only for rendering/evaluation. Training with this enabled throws an error. |
| `useHalfPrecision` | false | bfloat16 inference during collection. Significantly faster on GPU; learning still happens in fp32. |

### Guiding policy

| Field | Default | Description |
| --- | --- | --- |
| `useGuidingPolicy` | false | Adds a loss pulling the policy's distribution toward a frozen guiding policy (same obs/action space). Useful to gently steer training with a previously trained bot. |
| `guidingPolicyPath` | `"guiding_policy/"` | Folder containing the guiding policy model(s). |
| `guidingStrength` | 0.03 | Scale of the guiding loss. |

## SkillTrackerConfig (`cfg.skillTracker`)

Plays rating matches between the current policy and saved versions on separate arenas, maintaining an ELO-style rating per game mode (logged as `Rating/<mode>`).

| Field | Default | Description |
| --- | --- | --- |
| `enabled` | false | Enable skill tracking (forces `savePolicyVersions`). |
| `numArenas` | 16 | Evaluation arenas (keep at or below CPU thread count). |
| `simTime` | 45 | Seconds simulated per rating run, per arena. |
| `maxSimTime` | 240 | Max total simulated seconds before games are force-reset. |
| `updateInterval` | 16 | Iterations between rating runs. |
| `ratingInc` | 5 | Rating increment scale per goal. |
| `initialRating` | 0 | Rating of the first version. |
| `deterministic` | false | Evaluate policies deterministically. Off by default since training optimizes the stochastic policy. |

## Transfer learning (`Learner::StartTransferLearn`)

Instead of `Start()`, `StartTransferLearn(TransferLearnConfig)` trains the current (new) policy to imitate an old policy that may use a **different obs builder and/or action parser**:

| Field | Default | Description |
| --- | --- | --- |
| `makeOldObsFn` / `makeOldActFn` | — | Factories for the old obs builder / action parser. |
| `mapActsFn` | NULL | Maps new action indices to old ones when action spaces differ (may be state-dependent). |
| `oldPolicyConfig` / `oldSharedHeadConfig` | — | The old model's architecture. |
| `oldModelsPath` | — | Folder with the old policy model(s). |
| `lr` | 3e-4 | Imitation learning rate. |
| `batchSize` | 50,000 | Timesteps per imitation iteration (no minibatching). |
| `epochs` | 5 | Learning passes per iteration. |
| `useKLDiv` | false | Use KL-divergence loss instead of absolute probability difference. |
| `lossScale` | 500 | Loss multiplier (keeps optimizers alive at naturally small loss values). |
| `lossExponent` | 1 | Exponent applied to the loss. |

Watch `Transfer Learn Accuracy` — it is the fraction of states where the new policy's top action matches the old policy's. Once it plateaus (often >90%), save and switch to normal training.
