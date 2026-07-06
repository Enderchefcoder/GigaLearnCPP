#pragma once
#include "Report.h"

namespace GGL {
	struct RG_IMEXPORT MetricSender {
		std::string curRunID;
		std::string projectName, groupName, runName;

		// Hides the embedded Python module from this public header
		// (Also avoids visibility mismatches between the exported struct and pybind11's hidden types)
		struct Impl;
		Impl* impl;

		MetricSender(std::string projectName = {}, std::string groupName = {}, std::string runName = {}, std::string runID = {});
		
		RG_NO_COPY(MetricSender);

		void Send(const Report& report);

		~MetricSender();
	};
}
