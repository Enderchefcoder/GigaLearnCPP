# Built-in Rewards

All rewards live in `RLGymCPP/Rewards/CommonRewards.h` unless noted. Combine them with weights in your env creation function:

```cpp
std::vector<WeightedReward> rewards = {
	{ new FaceBallReward(), 0.25f },
	{ new VelocityPlayerToBallReward(), 4.f },
	{ new ZeroSumReward(new VelocityBallToGoalReward(), 1), 2.0f },
	{ new GoalReward(), 150 },
};
```

Reward design is the highest-leverage part of training a good bot: rewards define *what the bot wants*. The example's set produces a scoring 1v1 bot in ~100M steps; start there and iterate.

## Movement & possession

| Reward | Output | Notes |
| --- | --- | --- |
| `VelocityPlayerToBallReward` | [-1, 1] | Speed toward the ball. The classic "go to the ball" shaping signal. |
| `LiuDistancePlayerToBallReward` | (0, 1] | Exponential falloff with distance to the ball. Denser than velocity-based shaping; good very early. |
| `FaceBallReward` | [-1, 1] | Facing direction alignment with the ball. |
| `SpeedReward` / `VelocityReward` | [0, 1] / [-1, 1] | Raw speed (or negated). Mostly for movement warm-up. |
| `AirReward` | {0, 1} | 1 while airborne. Small weights encourage aerial comfort. |
| `WavedashReward` | {0, 1} | 1 on the step a wavedash lands. |
| `FlipResetReward` | {0, 1} | 1 on the step a flip reset is *obtained* (not used). |

## Ball advancement & scoring

| Reward | Output | Notes |
| --- | --- | --- |
| `VelocityBallToGoalReward` | [-1, 1] | Ball speed toward the opponent goal. Usually wrapped in `ZeroSumReward`. |
| `LiuDistanceBallToGoalReward` | (0, 1] | Exponential falloff with the ball's distance to the goal. Gentle gradient across the whole field. |
| `AlignBallGoalReward` | [-1, 1] | Positioning: rewards being between your goal and the ball (defense) and having the opponent goal beyond the ball (offense). Configurable weights. |
| `TouchBallReward` | {0, 1} | 1 on any touch. Beware: bots learn to dribble-farm this; prefer the touch rewards below. |
| `TouchAccelReward` | [0, 1] | Rewards *speeding the ball up* on touch, normalized so putts and shots both pay out fairly. |
| `StrongTouchReward` | [0, 1] | Rewards hit force above a threshold (default 20-130 KPH range). Good anti-dribble-farm shot shaping. |
| `GoalReward` | {-1*, 0, 1} | Team-based goal reward, already zero-sum. `concedeScale` sets the concede penalty (default -1). |

## Boost economy

| Reward | Output | Notes |
| --- | --- | --- |
| `PickupBoostReward` | [0, 1] | Rewards boost gained, sqrt-scaled (small pads matter). |
| `SaveBoostReward` | [0, 1] | Continuous sqrt(boost) signal; teaches boost conservation. |

## Game events (`PlayerDataEventReward` family)

`PlayerGoalReward`, `AssistReward`, `ShotReward`, `ShotPassReward`, `SaveReward`, `BumpReward`, `BumpedPenalty`, `DemoReward`, `DemoedPenalty` — 1 (or -1 for penalties) on the step the event fires, driven by RocketSim's event tracker.

## Wrappers

| Wrapper | Purpose |
| --- | --- |
| `ZeroSumReward(child, teamSpirit, opponentScale)` | Makes a reward team-shared and opponent-mirrored: `own*(1-spirit) + teamAvg*spirit - oppAvg*opponentScale`. Use for anything that shouldn't be farmable by both teams at once (ball-to-goal velocity, demos, bumps). |
| `PlayerReward<T>` | Maintains a separate instance of a stateful reward per player. |

## Custom rewards

Subclass `Reward` and override `GetReward` (per player) or `GetAllRewards` (whole state at once):

```cpp
class BallHeightReward : public RLGC::Reward {
public:
	float GetReward(const Player& player, const GameState& state, bool isFinal) override {
		return state.ball.pos.z / RLGC::CommonValues::CEILING_Z;
	}
};
```

Useful state access inside rewards:
- `player.prev->...` / `state.prev->...` — previous step (nullable on the first step of an episode)
- `player.eventState.shot/goal/save/bump/demo/...` — events this step
- `player.ballTouchedStep`, `player.isOnGround`, `player.HasFlipOrJump()`, all other `CarState` fields
- `isFinal` — whether this is the last step of the episode (goal scored / truncated)

Per-reward averages are logged automatically to your metrics as `Rewards/<ClassName>` (pre-weight values), so you can watch each signal's behavior during training.
