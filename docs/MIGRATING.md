# Migrating from RLGymPPO-CPP

GigaLearn's environment API is a redesign of [RLGymPPO-CPP](https://github.com/ZealanL/RLGymPPO-CPP)'s. This is the mapping for porting your rewards, obs builders, and configs.

## State & player access

| RLGymPPO-CPP | GigaLearn |
| --- | --- |
| `PlayerData` | `Player` |
| `player.phys.pos` / `player.carState.pos` | `player.pos` (Player *is* a `CarState`) |
| `player.phys.rotMat` | `player.rotMat` |
| `player.carState.isFlipping` | `player.isFlipping` (all `CarState` fields inherited) |
| `state.ballState` | `state.ball` |
| `player.ballTouched` | `player.ballTouchedStep` (any tick this step) or `player.ballTouchedTick` (final tick) |
| separate inverted fields | `state.GetBoostPads(inv)`, `InvertPhys(phys, inv)` |
| — | `player.prev->...` / `state.prev->...` (previous step's state) |
| — | `player.eventState.goal/assist/shot/save/bump/demo/...` |

There are no more duplicate/inverted state fields; invert on demand with `InvertPhys` and the `GetBoostPads(inverted)` helpers.

## Rewards

Old signature:

```cpp
float GetReward(const PlayerData& player, const GameState& state, const Action& prevAction);
float GetFinalReward(const PlayerData& player, const GameState& state, const Action& prevAction);
```

New signature (one function, `isFinal` argument):

```cpp
float GetReward(const Player& player, const GameState& state, bool isFinal) override;
```

- `prevAction` argument → `player.prevAction`
- `GetFinalReward()` → check `isFinal`
- Per-player state without arrays → wrap your reward in `PlayerReward<T>`

## Metrics

| RLGymPPO-CPP | GigaLearn |
| --- | --- |
| `metrics.AccumAvg(k, v)` | `report.AddAvg(k, v)` |
| `metrics.GetAvg(k)` | (finalized automatically each iteration) |

## Learner config

| RLGymPPO-CPP | GigaLearn |
| --- | --- |
| `cfg.numThreads`, `cfg.numGamesPerThread` | `cfg.numGames` (threading is automatic) |
| `cfg.ppo.policyLayerSizes` | `cfg.ppo.policy.layerSizes` |
| `cfg.ppo.criticLayerSizes` | `cfg.ppo.critic.layerSizes` |
| — | `cfg.ppo.sharedHead.layerSizes` (new; set `{}` to disable) |
| `cfg.expBufferSize` | removed (each iteration learns from its own experience) |
| `cfg.timestepsPerIteration` | `cfg.ppo.tsPerItr` |
| `cfg.ppo.entCoef` | `cfg.ppo.entropyScale` (normalized — retuning needed, defaults are sane) |

## Environment setup

Instead of returning lists of rewards/conditions/etc. plus match settings, you now build the arena yourself in `EnvCreateFunc` (game mode, cars, mutators) and return everything in one `EnvCreateResult`. State setters receive the `Arena*` directly and use RocketSim state setting.

## Checkpoints

Checkpoint formats are **not compatible** (different stats file, different model layout, and GigaLearn's shared head has no RLGymPPO-CPP equivalent).

- From Python rlgym-ppo: `tools/checkpoint_converter.py` converts policy/critic weights in either direction (optimizers reset)
- From RLGymPPO-CPP with a different obs/action setup: use transfer learning (`Learner::StartTransferLearn`) to imitate your old policy into a fresh model — see [CONFIGURATION.md](CONFIGURATION.md#transfer-learning-learnerstarttransferlearn)
