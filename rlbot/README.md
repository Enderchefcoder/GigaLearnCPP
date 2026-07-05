# RLBot Integration

This folder is the [RLBot](https://rlbot.org/) bot definition that lets your trained GigaLearn model play in actual Rocket League matches. It is based on [kipje13's CPPExampleBot](https://github.com/kipje13/CPPExampleBot).

## How it works

- `CppPythonAgent.py` is a thin Python agent that RLBot launches; it forwards match info to your C++ executable over a local socket
- Your executable calls `RLBotClient::Run(params)` (see `src/RLBotClient.h`) with an `InferUnit` that loads your trained policy
- `port.cfg` must contain the same port you pass in `RLBotParams::port`

## Setup

1. Install [RLBot](https://rlbot.org/) (RLBotGUI is the easiest way)
2. Build your bot executable with an `RLBotClient::Run()` entry point:
   - Use the **same obs builder, action parser, model architecture, tick skip, and action delay** as training
   - Point the `InferUnit` at your checkpoint's model files
3. Add this `rlbot` folder as a bot in RLBotGUI (it reads `CppPythonAgent.cfg`)
4. Either start your executable manually before the match, or configure auto-start (below)

## Auto-start

The RLBot framework can launch your bot executable automatically — useful when sharing your bot and usually required for tournaments:

1. Build your bot executable (in **Release** mode)
2. Set the `cpp_executable_path` field in `rlbot/CppPythonAgent.cfg` to point to the executable (copying the executable into this folder keeps the path simple)

## Notes

- Compile in release mode when sharing your bot — debug builds require debug runtimes that other people usually don't have
- Bot appearance is configured in `appearance.cfg`, name/description in `CppPythonAgent.cfg`
