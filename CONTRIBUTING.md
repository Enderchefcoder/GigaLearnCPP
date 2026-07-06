# Contributing

## Development setup

Follow [docs/INSTALLATION.md](docs/INSTALLATION.md), but configure with tests enabled:

```bash
./scripts/setup_libtorch.sh   # CPU libtorch is fine for development
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGL_BUILD_TESTS=ON
cmake --build build -j
```

## Running the tests

```bash
./build/GigaLearnTests            # Unit tests (~1 second)
./build/GigaLearnIntegrationTest  # End-to-end: real training on CPU (~10 seconds)
# or: cd build && ctest
```

The integration test needs no game files — it generates a synthetic arena mesh in memory.

CI runs three jobs on every PR: Linux (GCC), Linux with UndefinedBehaviorSanitizer, and Windows (MSVC). All must pass. You can reproduce the sanitizer job locally with:

```bash
cmake -S . -B build-ubsan -DCMAKE_BUILD_TYPE=RelWithDebInfo -DGGL_BUILD_TESTS=ON \
  "-DCMAKE_CXX_FLAGS=-fsanitize=undefined -fno-sanitize-recover=all" \
  "-DCMAKE_EXE_LINKER_FLAGS=-fsanitize=undefined" \
  "-DCMAKE_SHARED_LINKER_FLAGS=-fsanitize=undefined"
cmake --build build-ubsan -j && ./build-ubsan/GigaLearnTests
```

## Code expectations

- C++20, match the existing style (tabs, brace placement, `RG_LOG`/`RG_ERR_CLOSE` for output/errors)
- New learning-math or env-logic code should come with unit tests (see `GigaLearnCPP/tests/`)
- Changes to defaults must preserve existing training behavior unless the old behavior was a bug
- Performance claims need before/after measurements (see the benchmark methodology in past PRs)
- Vendored third-party code (`RocketSim/libsrc`, `pybind11`, `RLBotCPP/lib`, `thread_pool`) should only be touched to fix genuine bugs, with a comment marking the deviation from upstream

## Project layout

See the structure table in the [README](README.md#project-structure) and the internals walkthrough in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
