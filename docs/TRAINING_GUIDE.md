# Training Guide

A practical playbook for taking a bot from random movement to competent play. This is guidance, not law — reward design and hyperparameters interact, and every run is a bit different. Watch your metrics and adjust.

## Before you start: know your metrics

Train with metrics on (wandb, or the local JSONL fallback — visualize those with `python tools/plot_metrics.py metrics/<run>.jsonl`) and watch these:

| Metric | Healthy | Warning signs |
| --- | --- | --- |
| `Average Step Reward` | Trending up over hours | Flat forever (rewards too sparse or conflicting), or spiking then crashing (reward hacking found and lost) |
| `Policy Entropy` | Starts near 1.0, declines *slowly* over hundreds of millions of steps | Crashing toward 0 early = entropy collapse: raise `entropyScale`. Stuck at ~1.0 forever = policy isn't learning: check rewards/LR |
| `Mean KL Divergence` | Small and stable (~1e-3 order) | Large spikes = destructive updates: lower LR or set `targetKLDiv` |
| `Critic Loss` | Declining, then stabilizing at a modest value | Growing without bound = check reward scales / return standardization |
| `Rewards/<name>` | Each reward stream behaving as designed | A shaping reward dominating everything else = re-weight |
| `Env NaN Resets` | Absent or extremely rare | Frequent = something in your setup is breaking the physics |

The console report shows collection/consumption speed; keep an eye on `Overall Steps/Second` when you change `numGames`, model sizes, or minibatch sizes.

### SAC metrics

When training with `cfg.algorithm = LearningAlgorithmType::SAC`, the PPO-specific rows (KL divergence, critic loss, clip fraction) are replaced by:

| Metric | Healthy | Warning signs |
| --- | --- | --- |
| `SAC/Q1 Loss` / `SAC/Q2 Loss` | Settling to a stable band after warmup (they track each other closely) | Growing without bound = Q divergence: lower `qLR`, lower the replay ratio (`gradientStepsPerItr`), or check reward scales |
| `SAC/Avg Q` | Drifting up smoothly as play improves, magnitude consistent with your reward scale and gamma | Exploding = Q divergence (see above); pinned near 0 forever = not learning |
| `SAC/Entropy` vs `SAC/Target Entropy` | Entropy hovering near the target once alpha stabilizes | Entropy stuck far above target = policy can't commit (target too high? LR too low?); far below = raise `targetEntropyScale` |
| `SAC/Entropy Coef` | Settling into a stable range after early movement | Racing toward 0 = entropy target too low; growing forever = policy can't reach the entropy target |
| `SAC/Learning Active` | 1 after `learningStartTimesteps` | Stuck at 0 = `learningStartTimesteps`/`batchSize` never satisfied |

`Policy Entropy` stays comparable across algorithms (both report entropy normalized by the action count).

## Stage 1: Ballchasing (0 → ~100M steps)

The goal is dense, immediate feedback so random actions can bootstrap into intent:

- **Rewards**: heavy on `VelocityPlayerToBallReward` (or `LiuDistancePlayerToBallReward`), some `FaceBallReward`, a strong `StrongTouchReward`, `ZeroSumReward(VelocityBallToGoalReward)`, and a large `GoalReward`. Add small `PickupBoostReward`/`SaveBoostReward` so boost habits form early. The example's reward set is exactly this stage.
- **Gamma**: start around `0.99` (tickSkip 8). Short-horizon credit assignment learns faster early.
- **Entropy**: `0.02 - 0.035`. If the bot commits to driving in circles, raise it.
- **LR**: `1.5e-4` to `3e-4` for ~256-neuron networks.
- **State setters**: `KickoffState` is fine; `RandomState` (or a `CombinedState` mix) exposes more of the state space faster.
- **Expect**: touching the ball within the first few million steps, deliberate shots by ~50-100M.

## Stage 2: Competence (~100M → ~1B steps)

The bot scores on an empty net and contests the ball. Now shaping rewards become crutches:

- **Re-weight down** the ballchasing rewards (`VelocityPlayerToBall`, `FaceBall`) — by this point they fight against learning to rotate, shadow, and wait.
- **Keep event rewards strong** (`GoalReward`, `ShotReward`, `SaveReward`, demos if you want them).
- **Raise gamma** toward `0.995+` as behavior stabilizes — longer credit horizons enable positioning play.
- **Decay LR** (e.g. halve it) once entropy is declining steadily and updates look stable. You can automate this from the step callback with `learner->SetLearningRates(...)` (see [CONFIGURATION.md](CONFIGURATION.md#runtime-schedules)).
- **Team sizes**: if you started 1v1, this is where 2v2/3v3 arenas (mixed in one run via `DefaultObsPadded`) start mattering for game sense. Zero-sum wrappers with `teamSpirit > 0` encourage team play.

## Stage 3: Self-play refinement (1B+ steps)

- Enable `savePolicyVersions` + `trainAgainstOldVersions` (~0.15 chance is a good default; add `oldVersionRecencyBias` ≈ 0.2-0.4 to favor recent opponents). This fights catastrophic forgetting and rock-paper-scissors drift: the bot must still beat its past selves.
- Enable the `skillTracker` to get an objective `Rating/<mode>` curve — average step reward stops being meaningful once opponents improve alongside the bot.
- Keep entropy from collapsing entirely; even at high level, some stochasticity protects against exploitable determinism.
- Consider `normalizeAdvantages` and/or `targetKLDiv` if updates get spiky at low LR.

## Common failure modes

**Entropy collapse** (entropy → 0 early): raise `entropyScale`; check for a single overwhelming reward; check LR isn't too high.

**Reward hacking**: the bot maximizes your reward, not your intent. Classic examples: farming `TouchBallReward` by dribbling against a wall, or hovering near the ball for distance rewards without ever shooting. Use event/outcome rewards (`StrongTouchReward`, `TouchAccelReward`, `GoalReward`) over raw proximity/touch counts, and watch per-reward metrics to catch farming.

**Plateau**: reward flat, entropy still healthy. Usually reward design, not hyperparameters — the current reward set no longer distinguishes better play from current play. Add/re-weight rewards for the *next* skill you want.

**Sudden regression after many stable hours**: look at `Mean KL Divergence` and `Policy Update Magnitude` for a destructive update (lower LR / set `targetKLDiv`), and check whether a reward started being exploited (per-reward metrics).

**Forgetting how to beat older strategies**: that's what `trainAgainstOldVersions` is for.

## Scaling up

- Bigger models (512-1024+ wide, more layers) learn richer play but cost collection speed — GPU + `useHalfPrecision` recommended.
- More `numGames` improves GPU batching efficiency and experience diversity; scale until RAM/collection speed says stop.
- Bigger `tsPerItr`/`batchSize` stabilize updates at the cost of wall-clock per iteration; keep `miniBatchSize` within VRAM.
- Long runs: leave `standardizeReturns` on, keep auto-saves frequent enough (`tsPerSave`), and rely on the built-in protections (atomic saves, NaN recovery, Ctrl+C save-and-exit).

## Changing obs/actions mid-project

A trained policy is welded to its obs layout and action table. To move to a new obs builder or action parser without starting over, use transfer learning (`Learner::StartTransferLearn`) to imitate the old policy into the new spaces — see [CONFIGURATION.md](CONFIGURATION.md#transfer-learning-learnerstarttransferlearn). Expect >90% action-match accuracy before switching to normal training. (Transfer learning is PPO-only; for a SAC project, transfer-learn with PPO, then point a new SAC run at the resulting policy.)

## Choosing between PPO and SAC

- **PPO** is the battle-tested default for Rocket League: cheap per timestep, tolerant of huge throughput, and the configuration wisdom above was written for it. Start here.
- **SAC** replays each collected timestep multiple times (`gradientStepsPerItr * batchSize / tsPerItr`), which can extract more learning per env step — attractive when collection (not the GPU) is your bottleneck. It brings its own knobs: the replay ratio, the entropy target, and `tau`. Its exploration is driven by the auto-tuned temperature instead of an entropy bonus scale.
- The two share policies for inference, but **not** training state: a checkpoint folder belongs to one algorithm.
