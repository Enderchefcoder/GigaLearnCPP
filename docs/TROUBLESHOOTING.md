# Troubleshooting

Common problems, roughly in the order you might hit them.

## CMake / build

**`Could not find a package configuration file provided by "Torch"`**
libtorch isn't where the build expects. Either place it at `GigaLearnCPP/libtorch/` or pass `-DCMAKE_PREFIX_PATH=/path/to/libtorch`.

**`Compatibility with CMake < 3.10 has been removed` or similar policy errors**
Your CMake is fine — this repo requires 3.18+. If you see this from an old checkout, update the checkout; the build files have been modernized.

**Linker error: `cannot find -lstdc++`**
Your default `c++` may point at Clang without its GCC runtime dev packages. Either install them or configure with `-DCMAKE_CXX_COMPILER=g++`.

**`relocation R_X86_64_TPOFF32 ... recompile with -fPIC`**
You're building with stale CMake caches from an old checkout. Delete the build folder and reconfigure — RocketSim/RLGymCPP now build with position-independent code.

**MSVC can't find Python or Torch DLLs at runtime**
The build copies both next to the executable post-build. If you moved the exe, copy `GigaLearnCPP/libtorch/lib/*.dll` and the Python DLLs with it.

**Python headers not found**
Install the dev package (`python3-dev` on Debian/Ubuntu, `python3-devel` on Fedora). On Windows, use the python.org installer.

## Startup

**`No "collision_meshes" folder found ...` warning**
You must dump arena meshes with [RLArenaCollisionDumper](https://github.com/ZealanL/RLArenaCollisionDumper) and put the `collision_meshes` folder in the working directory. Training without them means no walls/ramps — don't ignore this.

**`ROCKETSIM FATAL ERROR: No arena meshes found for gamemode soccar`**
Same cause as above — the meshes folder is missing or in the wrong place (it must contain a `soccar/` subfolder with `.cmf` files).

**`MetricSender: Failed to import metrics receiver`**
The `python_scripts` folder must be next to the executable (the build copies it automatically). If you run from another directory, it's also searched via the working directory.

**`[metric_receiver] Failed to import wandb`**
Not fatal — metrics fall back to JSONL files in `metrics/`. To use wandb, `pip install wandb` into the Python that CMake found (printed during configure), or set `cfg.sendMetrics = false` to silence metrics entirely.

**`Can't use CUDA GPU because ...`**
Your libtorch is CPU-only, or CUDA/driver versions don't match the libtorch build. Check `nvidia-smi`, and download the libtorch build matching your CUDA version.

**Crash at startup in debug mode**
Debug-mode libtorch is extremely slow and has known issues. Build Release (optionally with debug info) instead.

## Training

**`Cannot train with config.ppo.deterministic enabled`**
Deterministic mode is for inference/rendering only; PPO needs the stochastic log probs. Disable it for training.

**`ExperienceBuffer: Not enough experience for a single batch`**
`cfg.ppo.batchSize` must be ≤ the timesteps collected per iteration (`cfg.ppo.tsPerItr`).

**`PPOLearner: config.batchSize must be a multiple of config.miniBatchSize`**
Exactly what it says — pick a minibatch size that divides the batch size.

**`WARNING: Non-finite values in the obs of arena N, resetting it`**
Extreme collisions can very rarely diverge RocketSim's physics to NaN. The learner recovers automatically: the affected arena is reset and its in-progress episode data is discarded (`Env NaN Resets` metric). Seeing this occasionally is harmless; seeing it constantly means your setup is breaking the physics (e.g. a state setter spawning objects inside each other).

**`Obs builder produced a NaN/inf value at obs index N, even for a freshly-reset state`**
Your obs builder produced garbage — the index tells you which obs element. Common causes: normalizing a zero-length vector, dividing by a value that can be zero, uninitialized fields.

**Out of VRAM during learning**
Lower `cfg.ppo.miniBatchSize` (gradient accumulation keeps the math identical). Collection VRAM scales with `numGames`; lower it if inference itself runs out.

**Entropy crashes toward 0 / bot plays the same action**
Raise `entropyScale`. Also check your rewards aren't wildly imbalanced (one huge reward dominating).

**Reward magnitudes look wrong in wandb**
`Rewards/<name>` logs the *pre-weight* reward output, sampled from a random player. Zero-sum wrappers log the inner reward's value.

**Training resumed but wandb started a new run**
The run ID is stored in `RUNNING_STATS.json` inside the checkpoint. If you deleted checkpoints (or changed `checkpointFolder`), a new run starts.

**`Tried to load saved policy version that is newer than our current model`**
You deleted recent checkpoints but kept newer policy versions. Delete the newer folders under `checkpoints/policy_versions/` too.

## Render mode

**Nothing shows up**
Start the [RocketSimVis](https://github.com/ZealanL/RocketSimVis) receiver first; `render_receiver.py` sends UDP to `127.0.0.1:9273`. Both must run on the same machine (or edit the script).

## RLBot

**Bot connects but doesn't move**
Check that obs builder, action parser, model configs, and checkpoint path passed to `InferUnit` exactly match training, and that `tickSkip`/`actionDelay` match too.

**`InferUnit: Obs builder produced an obs that differs from the provided size`**
The obs size you passed to `InferUnit` doesn't match what the obs builder produced — usually a team-size mismatch (train with the same team sizes, or use a padded obs builder).
