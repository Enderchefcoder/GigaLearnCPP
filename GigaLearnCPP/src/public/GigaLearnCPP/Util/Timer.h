#pragma once
#include "../Framework.h"

namespace GGL {
	struct Timer {
		// NOTE: steady_clock is used consistently (high_resolution_clock is not guaranteed
		//	to be steady, and is a different type than steady_clock on some standard libraries)
		std::chrono::steady_clock::time_point startTime;

		Timer() {
			Reset();
		}

		// Returns elapsed time in seconds
		double Elapsed() const {
			auto endTime = std::chrono::steady_clock::now();
			std::chrono::duration<double> elapsed = endTime - startTime;
			return elapsed.count();
		}

		void Reset() {
			startTime = std::chrono::steady_clock::now();
		}
	};
}
