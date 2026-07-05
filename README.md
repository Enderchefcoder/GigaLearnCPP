# GigaLearnCPP

**GigaLearn** is a high-performance C++ machine learning framework for training Rocket League bots with [PPO](https://en.wikipedia.org/wiki/Proximal_policy_optimization), built on [RocketSim](https://github.com/ZealanL/RocketSim).

It is the successor to [RLGymPPO-CPP](https://github.com/ZealanL/RLGymPPO-CPP), with a redesigned environment API, a monolithic single-process inference model, and far higher throughput.

> **Note on this repository's origin:** This library was originally private. After a former user with source access leaked it, this specific version was formally published for anyone to use.

## Highlights

- **Fast**: Collection speeds are around 2x faster than RLGymPPO-CPP and around 10x faster than the Python RLGym-PPO (measured on the author's machine, hardware-dependent)
- **Single-process**: One process, one model, batched inference across all game instances — no inter-process copying of observations
- **Complete action masking**: Invalid actions are masked out of the policy distribution during both collection and learning
- **Self-play infrastructure built in**: Policy version saving, ELO-based skill tracking, and training against older versions

## Feature Overview

**Core learning:**
- Fast PPO implementation with configurable epochs, batch/minibatch sizes, entropy, and clip range
- Shared layers ("shared head") between policy and critic (enabled by default)
- Configurable model layer sizes, activation functions, optimizers, and layer normalization
- Return standardization and observation standardization
- Optional advantage normalization
- Half-precision (bfloat16) inference for faster collection on GPU
- Checkpoint saving/loading with automatic cleanup of old checkpoints
- Transfer learning from an old policy with a different obs builder or action parser ("brain surgery")
- Optional guiding policy to nudge training toward an existing policy's behavior

**Environment & state:**
- Multithreaded environment stepping over any number of RocketSim arenas
- Access to previous states (e.g. `player.prev->pos`)
- Inherited access to all `CarState` and `BallState` fields (e.g. `player.isFlipping`)
- Current-step event access (e.g. `if (player.eventState.shot) ...`) for goals, assists, shots, saves, bumps, and demos
- User-led arena setup during environment creation, including RocketSim-based state setting
- Configurable tick skip and action delay

**Self-play & evaluation:**
- Policy version saving system
- Built-in configurable ELO-based skill tracking against saved versions
- Training against older versions of the policy

**Tooling:**
- Metric reporting to [Weights & Biases](https://wandb.ai/) (falls back to local JSONL logging if wandb is unavailable)
- Built-in visualization support via [RocketSimVis](https://github.com/ZealanL/RocketSimVis)
- Easy custom metrics from a step callback
- Built-in per-reward logging
- [RLBot](https://rlbot.org/) client for running your trained bot in-game (see `src/RLBotClient.h` and `rlbot/`)
- Checkpoint conversion to/from Python rlgym-ppo (`tools/checkpoint_converter.py`)

## Quick Start

```bash
git clone <this repository>
cd GigaLearnCPP-Leak

# Place libtorch at GigaLearnCPP/libtorch (see docs/INSTALLATION.md)

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Then look at [`src/ExampleMain.cpp`](src/ExampleMain.cpp) — it is a complete, commented training setup with rewards, terminal conditions, and a learner configuration. Copy it and start experimenting.

For the full walkthrough (prerequisites, collision meshes, CUDA, wandb), read:

| Document | Contents |
| --- | --- |
| [docs/INSTALLATION.md](docs/INSTALLATION.md) | Prerequisites, libtorch setup, building on Windows & Linux |
| [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) | Your first training run, explained line by line |
| [docs/CONFIGURATION.md](docs/CONFIGURATION.md) | Every config option, with guidance |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | How collection, learning, and self-play work internally |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | Fixes for common setup and runtime issues |
| [docs/MIGRATING.md](docs/MIGRATING.md) | Porting rewards/obs builders from RLGymPPO-CPP |

## Project Structure

```
├── src/                  # Your bot: example training main + RLBot client
├── GigaLearnCPP/         # The learning framework (PPO, models, checkpoints, metrics)
│   ├── RLGymCPP/         # The environment framework (arenas, obs, rewards, state setters)
│   │   └── RocketSim/    # Rocket League physics simulation
│   ├── python_scripts/   # Embedded Python receivers for metrics & rendering
│   └── tests/            # Unit tests (enable with -DGGL_BUILD_TESTS=ON)
├── RLBotCPP/             # RLBot framework bindings for playing in-game
├── rlbot/                # RLBot bot folder (configs + Python agent)
└── tools/                # Checkpoint converter for rlgym-ppo interop
```

## Requirements

- CMake 3.18+
- A C++20 compiler (MSVC 2019+, GCC 11+, or Clang 14+)
- [libtorch](https://pytorch.org/get-started/locally/) (CUDA build strongly recommended for training)
- Python 3.8+ (embedded for metrics/rendering; `wandb` optional)
- Rocket League arena collision meshes, dumped with [RLArenaCollisionDumper](https://github.com/ZealanL/RLArenaCollisionDumper)

## Credits

- [ZealanL](https://github.com/ZealanL) — GigaLearn, RocketSim, RLGymPPO-CPP
- [RLGym](https://rlgym.org/) & [rlgym-ppo](https://github.com/AechPro/rlgym-ppo) — the API and algorithms this framework is based on
- [kipje13/RLBotCPP](https://github.com/kipje13/RLBotCPP) — RLBot C++ bindings
- [DeveloperPaul123/thread-pool](https://github.com/DeveloperPaul123/thread-pool) — thread pool library
- [nlohmann/json](https://github.com/nlohmann/json), [pybind11](https://github.com/pybind/pybind11)
