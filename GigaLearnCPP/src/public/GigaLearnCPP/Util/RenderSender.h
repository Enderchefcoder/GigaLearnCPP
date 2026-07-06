#pragma once
#include "Report.h"
#include <RLGymCPP/Gamestates/GameState.h>
#include <RLGymCPP/BasicTypes/Action.h>
#include <GigaLearnCPP/Util/Timer.h>

namespace GGL {
	struct RG_IMEXPORT RenderSender {
		// Hides the embedded Python module from this public header
		// (Also avoids visibility mismatches between the exported struct and pybind11's hidden types)
		struct Impl;
		Impl* impl;

		float timeScale;
		double adaptiveRenderDelay = -1;
		Timer renderTimer = {};

		RenderSender(float timeScale);

		RG_NO_COPY(RenderSender);

		void Send(const RLGC::GameState& state);

		~RenderSender();
	};
}
