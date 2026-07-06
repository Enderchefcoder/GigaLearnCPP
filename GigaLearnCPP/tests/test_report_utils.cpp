#include "TestFramework.h"

#include <GigaLearnCPP/Util/Report.h>
#include <GigaLearnCPP/Util/Utils.h>
#include <GigaLearnCPP/Util/Timer.h>

using namespace GGL;

TEST_CASE(Report_BasicSetAndGet) {
	Report report = {};
	report["A"] = 1.5;
	report.Add("A", 2.0);
	report.Add("B", 3.0);

	CHECK_TRUE(report.Has("A"));
	CHECK_TRUE(report.Has("B"));
	CHECK_FALSE(report.Has("C"));

	const Report& constReport = report;
	CHECK_NEAR(constReport["A"], 3.5, 1e-12);
	CHECK_NEAR(constReport["B"], 3.0, 1e-12);
}

TEST_CASE(Report_Averages) {
	Report report = {};
	report.AddAvg("Avg", 1);
	report.AddAvg("Avg", 2);
	report.AddAvg("Avg", 6);

	CHECK_FALSE(report.Has("Avg")); // Not finished yet

	report.Finish();
	CHECK_TRUE(report.Has("Avg"));

	const Report& constReport = report;
	CHECK_NEAR(constReport["Avg"], 3.0, 1e-12);
}

TEST_CASE(Report_FinishAvgSingle) {
	Report report = {};
	report.AddAvg("X", 10);
	report.AddAvg("X", 20);
	report.FinishAvg("X");

	const Report& constReport = report;
	CHECK_NEAR(constReport["X"], 15.0, 1e-12);

	CHECK_THROWS(report.FinishAvg("DoesNotExist"));
}

TEST_CASE(Utils_FindNumberedDirs) {
	namespace fs = std::filesystem;

	auto basePath = fs::temp_directory_path() / "ggl_test_numbered_dirs";
	fs::remove_all(basePath);
	fs::create_directories(basePath / "1000");
	fs::create_directories(basePath / "250");
	fs::create_directories(basePath / "not_a_number");
	fs::create_directories(basePath / "123abc");

	auto result = Utils::FindNumberedDirs(basePath);
	CHECK_EQ(result.size(), 2);
	CHECK_TRUE(result.contains(1000));
	CHECK_TRUE(result.contains(250));

	// Non-existent dir returns empty set
	auto emptyResult = Utils::FindNumberedDirs(basePath / "does_not_exist");
	CHECK_TRUE(emptyResult.empty());

	fs::remove_all(basePath);
}

TEST_CASE(Utils_GetExecutableDir) {
	auto dir = Utils::GetExecutableDir();
	CHECK_TRUE(std::filesystem::is_directory(dir));
}

TEST_CASE(Timer_MeasuresTime) {
	Timer timer = {};
	std::this_thread::sleep_for(std::chrono::milliseconds(30));
	double elapsed = timer.Elapsed();

	CHECK_TRUE(elapsed >= 0.02);
	CHECK_TRUE(elapsed < 5.0);

	timer.Reset();
	CHECK_TRUE(timer.Elapsed() < 0.02);
}
