# Configuration Reference

All learner behavior is controlled through `LearnerConfig` (which contains `PPOLearnerConfig`, `SACLearnerConfig`, and `SkillTrackerConfig`). This documents every field with guidance.

## LearnerConfig

### Algorithm

| Field | Default | Description |
| --- | --- | --- |
| `algorithm` | `PPO` | Which learning algorithm to train with: `LearningAlgorithmType::PPO` or `LearningAlgorithmType::SAC`. Only the matching sub-config (`cfg.ppo` or `cfg.sac`) is used. A checkpoint folder belongs to one algorithm — switching algorithms needs a new folder (trained policies still work everywhere for *inference* via `InferUnit`, regardless of the algorithm that produced them). |

PPO is the well-tested default for Rocket League ML. SAC (Soft Actor-Critic with discrete actions, [arXiv:1910.07207](https://arxiv.org/abs/1910.07207)) is off-policy: it keeps a large replay buffer of past transitions and learns from random samples of it, which reuses experience more aggressively and can be more sample-efficient, at the cost of more compute per collected timestep and different tuning behavior.

### Simulation

| Field | Default | Description |
| --- | --- | --- |
| `numGames` | 300 | Parallel arenas. Scale to your CPU/RAM; higher improves GPU inference batching. |
| `timestepLimit` | 0 | Stop training (with a final save) once this many total timesteps are reached. 0 = train forever. |
| `tickSkip` | 8 | Physics ticks per policy action. 8 = 15 actions/sec (the standard). |
| `actionDelay` | 7 | Ticks after a policy decision before the action takes effect. Rocket League itself has input delay, so `tickSkip - 1` matches other RLGym frameworks. Lower values react faster in sim but transfer worse to the real game. |
| `randomSeed` | -1 | -1 seeds from the current time. The seed strongly affects early training. |
| `deviceType` | `AUTO` | `AUTO` uses a CUDA GPU if libtorch can access one, else CPU. Force with `CPU`/`GPU_CUDA`. |
| `collectionTorchThreads` | 1 | Torch intra-op threads during collection (full count is restored for learning). Torch's idle workers spin-wait and starve the env threads, so 1 is usually much faster (+53% overall on the library's CPU test machine). 0 leaves torch's default. |

### Render mode

| Field | Default | Description |
| --- | --- | --- |
| `renderMode` | false | Run a single arena in real time and stream it to the render receiver ([RocketSimVis](https://github.com/ZealanL/RocketSimVis)). No training happens. |
| `renderTimeScale` | 1.0 | Game speed multiplier in render mode. |

### Checkpoints

| Field | Default | Description |
| --- | --- | --- |
| `checkpointFolder` | `"checkpoints"` | Checkpoints save into timestep-numbered subfolders. Empty disables saving. The newest checkpoint is auto-loaded at startup. |
| `checkpointToLoad` | -1 | Load a specific checkpoint (by its timestep number) instead of the newest — for rolling back after a bad training period. Delete the newer checkpoint subfolders (and newer policy versions) when rolling back; a warning reminds you. |
| `tsPerSave` | 1,000,000 | Timesteps between auto-saves. 0 = save every iteration. |
| `checkpointsToKeep` | 8 | Older checkpoints are deleted beyond this count. -1 keeps everything. |

### Standardization

| Field | Default | Description |
| --- | --- | --- |
| `standardizeReturns` | true | Track return statistics and scale rewards by the return STD for the critic. Don't disable unless you know what you're doing. **PPO only** (SAC learns on raw rewards; its auto-tuned entropy temperature absorbs scale differences). |
| `maxReturnSamples` | 150 | Return samples per iteration for the running STD. |
| `standardizeObs` | false | Standardize observations with running mean/STD per obs index. Usually unnecessary if your obs builder outputs sane ranges. With SAC, note that replayed transitions keep the standardization from when they were collected (slightly stale for old replay data). |
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
| `oldVersionRecencyBias` | 0 | Biases opponent selection toward recent versions: each step back in history is `(1 - bias)` times as likely. 0 = uniform. |
| `skillTracker` | see below | ELO-style rating of the current policy vs old versions. |

## PPOLearnerConfig (`cfg.ppo`)

### Iteration sizing

| Field | Default | Description |
| --- | --- | --- |
| `tsPerItr` | 50,000 | Timesteps collected per iteration. |
| `batchSize` | 50,000 | Timesteps per PPO batch. Must be ≤ collected timesteps. |
| `miniBatchSize` | 0 | Splits batches into minibatches for gradient accumulation (reduces VRAM). 0 = use `batchSize`. Must divide `batchSize` evenly. |
| `overbatching` | true | The final batch absorbs leftover experience (up to 2x batch size) instead of discarding it. |
| `experienceOnDevice` | true | Keep the whole iteration's experience in VRAM for the learn phase instead of re-uploading every minibatch each epoch. Disable if VRAM-constrained with very large `tsPerItr`/obs. No effect on CPU. |
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
| `targetKLDiv` | 0 | If > 0, remaining epochs are skipped once an epoch's mean KL divergence exceeds 1.5x this value (protects against destructively large updates; reported as `Epochs Ran`). |
| `gradClipNorm` | 0.5 | Max gradient norm per model per batch. 0 disables clipping. |
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

## SACLearnerConfig (`cfg.sac`)

Used when `cfg.algorithm == LearningAlgorithmType::SAC`. SAC-Discrete keeps the same masked-softmax policy structure as PPO, but estimates values with twin Q nets (one Q value per action) and learns off-policy from a replay buffer.

### Iteration sizing & replay

| Field | Default | Description |
| --- | --- | --- |
| `tsPerItr` | 10,000 | Timesteps collected per iteration (between learn phases). |
| `replayBufferSize` | 500,000 | Max transitions kept (oldest overwritten). Memory is roughly `replayBufferSize * (2*obsSize*4 + 2*numActions + 16)` bytes — about 500 MB at the defaults with obs size ~100. The buffer lives in RAM (not VRAM) and is **not** saved in checkpoints (a resumed run refills it before learning resumes). |
| `batchSize` | 512 | Replay transitions sampled per gradient step. |
| `gradientStepsPerItr` | 64 | Gradient updates per iteration. The replay ratio (how often each collected timestep is learned from, on average) is `gradientStepsPerItr * batchSize / tsPerItr` — ~3.3 at the defaults. Raising it learns more per timestep but costs compute and can destabilize training. |
| `learningStartTimesteps` | 20,000 | No learning until this many timesteps exist (gives the buffer diverse data first). `SAC/Learning Active` reports whether learning has started. |
| `maxEpisodeDuration` | 120 | Max episode length in seconds before truncation (same as PPO's). |

### Core SAC

| Field | Default | Description |
| --- | --- | --- |
| `policyLR` / `qLR` | 3e-4 | Learning rates of the policy and the (twin) Q nets. |
| `gamma` | 0.99 | Reward discount on the Q-learning target (SAC does not use GAE). |
| `tau` | 0.005 | Polyak averaging coefficient for the target Q nets: `target = tau * live + (1 - tau) * target` each update. |
| `targetUpdateInterval` | 1 | Gradient steps between target net updates. |
| `gradClipNorm` | 0 | Max gradient norm per model per gradient step. 0 disables (SAC usually doesn't need it). |
| `policyTemperature` | 1 | Softmax temperature of the policy distribution. |

### Entropy temperature (alpha)

Alpha scales the entropy bonus: bigger alpha = more exploration. By default it is auto-tuned so the policy's entropy tracks a target.

| Field | Default | Description |
| --- | --- | --- |
| `autoEntCoef` | true | Auto-tune alpha toward `targetEntropy = targetEntropyScale * log(numActions)`. |
| `entCoef` | 0.2 | Fixed alpha if `autoEntCoef` is off, otherwise the initial alpha. |
| `entCoefLR` | 3e-4 | Learning rate of the alpha auto-tuner. |
| `targetEntropyScale` | 0.7 | Fraction of the maximum possible entropy (`log(numActions)`, over the *full* action table) to target. The SAC-Discrete paper used 0.98, which tends to over-explore; lower it if your bot stays too random, raise it if the policy collapses to a few actions. Note that action masking lowers the achievable entropy in masked states, so with heavily-masked action parsers, prefer lower scales. Watch `SAC/Entropy` vs `SAC/Target Entropy` and `SAC/Entropy Coef` in the metrics. |

### Models (`policy`, `qNet`, `sharedHead`)

Same fields as the PPO model configs (`layerSizes`, `activationType`, `optimType`, `addLayerNorm`). Both Q nets use `qNet`'s config and output one Q value per action. `sharedHead` is **optional and feeds only the policy** — a trunk shared between the actor and critics is a common source of SAC instability, so the Q nets always read the raw obs. The default has no shared head (`sharedHead.layerSizes = {}`).

### Inference

`deterministic` and `useHalfPrecision` behave exactly like their PPO counterparts (deterministic mode is inference/render-only and throws if used for training).

### Not supported with SAC (yet)

- **Transfer learning** (`StartTransferLearn`) and the **guiding policy** are PPO-only. To migrate obs/action spaces, transfer-learn with PPO first, then start a SAC run from the resulting policy.
- `standardizeReturns` has no effect (SAC learns on raw rewards).

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

## Runtime schedules

Learning rate and entropy can be adjusted while training runs (e.g. decayed by total timesteps) from your step callback:

```cpp
void StepCallback(Learner* learner, const std::vector<GameState>& states, Report& report) {
	// Example: drop the learning rate after 500M steps
	if (learner->totalTimesteps > 500'000'000 && learner->GetPolicyLR() > 1e-4f)
		learner->SetLearningRates(1e-4f, 1e-4f);
}
```

Available: `SetLearningRates(policyLR, criticLR)`, `SetEntropyScale(scale)`, `GetPolicyLR()`, `GetCriticLR()`, `GetEntropyScale()`. Changes take effect from the next learn phase.

With SAC, the same functions map naturally: `SetLearningRates(policyLR, qLR)` sets the policy and Q-net rates, `GetEntropyScale()` returns the current (possibly auto-tuned) alpha, and `SetEntropyScale(alpha)` sets a fixed alpha (only allowed when `autoEntCoef` is off — it throws otherwise; adjust `targetEntropyScale` instead).

## Transfer learning (`Learner::StartTransferLearn`)

**PPO only.** Instead of `Start()`, `StartTransferLearn(TransferLearnConfig)` trains the current (new) policy to imitate an old policy that may use a **different obs builder and/or action parser**:

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
