# Installation

This guide takes you from a fresh clone to a working training build on Windows or Linux.

## 1. Prerequisites

| Requirement | Version | Notes |
| --- | --- | --- |
| CMake | 3.18+ | https://cmake.org/download/ |
| C++ compiler | C++20 capable | MSVC 2019+, GCC 11+, or Clang 14+ |
| Python | 3.8+ (64-bit) | Must include development headers (see below) |
| libtorch | 2.x | CUDA build strongly recommended for training |
| CUDA toolkit | Matching your libtorch build | Only needed for GPU training |

**Python development headers:**
- Windows: included with the standard python.org installer
- Ubuntu/Debian: `sudo apt install python3-dev`
- Fedora: `sudo dnf install python3-devel`

## 2. Get the code

```bash
git clone https://github.com/Enderchefcoder/GigaLearnCPP
cd GigaLearnCPP
```

All dependencies (RocketSim, RLGymCPP, pybind11, RLBotCPP, nlohmann/json, thread-pool) are vendored in the repository — there are no submodules to initialize.

## 3. Install libtorch

Download libtorch from https://pytorch.org/get-started/locally/ (select **LibTorch** as the package and **C++/Java** as the language):

- **For NVIDIA GPU training (recommended):** pick the CUDA version matching your installed CUDA toolkit
- **For CPU-only training or inference:** pick the CPU version
- On Windows, download the **Release** build (the Debug build is separate)

Extract the archive so that the `libtorch` folder sits inside the `GigaLearnCPP` folder:

```
GigaLearnCPP/
├── libtorch/
│   ├── bin/
│   ├── include/
│   ├── lib/
│   └── share/
├── src/
└── ...
```

The build automatically detects `GigaLearnCPP/libtorch/`. Alternatively, install libtorch anywhere and pass `-DCMAKE_PREFIX_PATH=/path/to/libtorch` when configuring.

> **Windows + CUDA note:** If you have multiple CUDA versions installed, make sure the one matching your libtorch build is first in your `PATH`.

## 4. Configure and build

### Windows (Visual Studio)

```bat
cmake -S . -B build
cmake --build build --config Release -j
```

You can also open the folder directly in Visual Studio (or generate a `.sln` with `cmake .`) — just make sure you build in **Release**. Debug-mode libtorch is extremely slow and often has bizarre issues.

### Linux

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

If your default `c++` compiler is misconfigured, explicitly select GCC with `-DCMAKE_CXX_COMPILER=g++`.

### Optional build flags

| Flag | Effect |
| --- | --- |
| `-DGGL_NATIVE_ARCH=ON` | Optimize the simulation for your exact CPU (`-march=native`). Faster collection, but the binaries won't run on other machines. |
| `-DGGL_BUILD_TESTS=ON` | Build the unit test suite (`GigaLearnTests`). |

The build produces:
- `GigaLearnBot` (or `GigaLearnBot.exe`) — the example training executable built from `src/`
- `GigaLearnCPP` shared library
- `python_scripts/` — copied beside the executable (needed at runtime for metrics/rendering)

## 5. Get collision meshes

RocketSim needs Rocket League's arena collision meshes, which cannot be distributed with this repository.

1. Dump them from your Rocket League installation using [RLArenaCollisionDumper](https://github.com/ZealanL/RLArenaCollisionDumper)
2. Place the resulting `collision_meshes` folder in the **working directory** you run the bot from (usually next to the executable):

```
build/
├── GigaLearnBot
├── python_scripts/
└── collision_meshes/
    └── soccar/
        ├── mesh_0.cmf
        └── ...
```

If the meshes are missing, the learner prints a prominent warning — training will technically run, but cars and the ball will not collide with walls or ramps properly, so don't skip this.

## 6. (Optional) Set up Weights & Biases metrics

Training metrics are sent to [wandb](https://wandb.ai/) through the embedded Python interpreter:

```bash
pip install wandb
wandb login
```

If wandb is not installed, metrics automatically fall back to local JSONL files in `metrics/` (a warning is printed). You can also disable metrics entirely with `cfg.sendMetrics = false`.

> The embedded interpreter uses the same Python installation that CMake found at configure time (printed as `Python_EXECUTABLE` during configuration). Install wandb into **that** Python.

## 7. (Optional) Build and run the tests

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGL_BUILD_TESTS=ON
cmake --build build -j
./build/GigaLearnCPP/tests/GigaLearnTests
```

## 8. Run

```bash
cd build
./GigaLearnBot
```

You should see the learner initialize, print model parameter counts, and start printing training reports. Press `Q` to save and quit.

If anything goes wrong, check [TROUBLESHOOTING.md](TROUBLESHOOTING.md).
