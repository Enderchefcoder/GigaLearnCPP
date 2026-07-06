#include "MetricSender.h"

#include "Timer.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace GGL;

struct GGL::MetricSender::Impl {
	py::module pyMod;
};

GGL::MetricSender::MetricSender(std::string _projectName, std::string _groupName, std::string _runName, std::string runID) :
	projectName(_projectName), groupName(_groupName), runName(_runName) {

	RG_LOG("Initializing MetricSender...");

	impl = new Impl();

	try {
		impl->pyMod = py::module::import("python_scripts.metric_receiver");
	} catch (std::exception& e) {
		RG_ERR_CLOSE(
			"MetricSender: Failed to import metrics receiver, exception: " << e.what() << "\n" <<
			"Make sure the \"python_scripts\" folder is next to your executable (or in your working directory)."
		);
	}

	try {
		auto returnedRunID = impl->pyMod.attr("init")(PY_EXEC_PATH, projectName, groupName, runName, runID);
		curRunID = returnedRunID.cast<std::string>();
		RG_LOG(" > " << (runID.empty() ? "Starting" : "Continuing") << " run with ID: \"" << curRunID << "\"...");

	} catch (std::exception& e) {
		RG_ERR_CLOSE("MetricSender: Failed to initialize in Python, exception: " << e.what());
	}

	RG_LOG(" > MetricSender initialized.");
}

void GGL::MetricSender::Send(const Report& report) {
	py::dict reportDict = {};

	for (auto& pair : report.data)
		reportDict[pair.first.c_str()] = pair.second;

	try {
		impl->pyMod.attr("add_metrics")(reportDict);
	} catch (std::exception& e) {
		// Metric delivery failures (e.g. wandb network hiccups) should never kill a training run
		RG_LOG("WARNING: MetricSender failed to send metrics (training continues), exception: " << e.what());
	}
}

GGL::MetricSender::~MetricSender() {
	delete impl;
}
