#include "TestFramework.h"

#include <private/GigaLearnCPP/Util/WelfordStat.h>

using namespace GGL;

// Two-pass reference mean/std (sample standard deviation)
static void ReferenceMeanStd(const std::vector<float>& samples, double& outMean, double& outStd) {
	double sum = 0;
	for (float v : samples)
		sum += v;
	outMean = sum / samples.size();

	double sqDevSum = 0;
	for (float v : samples)
		sqDevSum += (v - outMean) * (v - outMean);
	outStd = sqrt(sqDevSum / (samples.size() - 1));
}

TEST_CASE(WelfordStat_MeanAndStd) {
	std::vector<float> samples = { 1, 5, -3, 8, 0.5f, 2, 100, -20 };

	WelfordStat stat = {};
	stat.Increment(samples);

	double refMean, refStd;
	ReferenceMeanStd(samples, refMean, refStd);

	CHECK_NEAR(stat.GetMean(), refMean, 1e-9);
	CHECK_NEAR(stat.GetSTD(), refStd, 1e-9);
}

TEST_CASE(WelfordStat_Defaults) {
	// With too few samples, mean is 0 and std is 1 (identity standardization)
	WelfordStat stat = {};
	CHECK_EQ(stat.GetMean(), 0);
	CHECK_EQ(stat.GetSTD(), 1);

	stat.Increment({ 42 });
	CHECK_EQ(stat.GetMean(), 0);
	CHECK_EQ(stat.GetSTD(), 1);
}

TEST_CASE(WelfordStat_JSONRoundtrip) {
	WelfordStat stat = {};
	stat.Increment({ 1, 2, 3, 4, 5 });

	auto j = stat.ToJSON();

	WelfordStat loaded = {};
	loaded.ReadFromJSON(j);

	CHECK_NEAR(loaded.GetMean(), stat.GetMean(), 1e-12);
	CHECK_NEAR(loaded.GetSTD(), stat.GetSTD(), 1e-12);
	CHECK_EQ(loaded.count, stat.count);
}

TEST_CASE(BatchedWelfordStat_MeanAndStd) {
	constexpr int WIDTH = 3;

	// 4 rows of 3 columns
	std::vector<std::vector<float>> rows = {
		{ 1, 10, 100 },
		{ 2, 20, 200 },
		{ 3, 30, 300 },
		{ 4, 40, 400 },
	};

	BatchedWelfordStat stat = BatchedWelfordStat(WIDTH);
	for (auto& row : rows)
		stat.IncrementRow(row.data());

	for (int col = 0; col < WIDTH; col++) {
		std::vector<float> colSamples = {};
		for (auto& row : rows)
			colSamples.push_back(row[col]);

		double refMean, refStd;
		ReferenceMeanStd(colSamples, refMean, refStd);

		CHECK_NEAR(stat.GetMean()[col], refMean, 1e-9);
		CHECK_NEAR(stat.GetSTD()[col], refStd, 1e-9);
	}
}

// Regression test: ToJSON used to write "means"/"vars" while ReadFromJSON read "mean"/"var",
//	which broke loading any checkpoint with obs standardization enabled
TEST_CASE(BatchedWelfordStat_JSONRoundtrip) {
	constexpr int WIDTH = 2;

	BatchedWelfordStat stat = BatchedWelfordStat(WIDTH);
	float row1[WIDTH] = { 1, -1 };
	float row2[WIDTH] = { 3, -5 };
	stat.IncrementRow(row1);
	stat.IncrementRow(row2);

	auto j = stat.ToJSON();

	BatchedWelfordStat loaded = BatchedWelfordStat(WIDTH);
	loaded.ReadFromJSON(j);

	CHECK_EQ(loaded.count, stat.count);
	for (int i = 0; i < WIDTH; i++) {
		CHECK_NEAR(loaded.GetMean()[i], stat.GetMean()[i], 1e-12);
		CHECK_NEAR(loaded.GetSTD()[i], stat.GetSTD()[i], 1e-12);
	}
}

TEST_CASE(BatchedWelfordStat_LegacyJSONKeys) {
	// Old checkpoints may have used the "mean"/"var" keys
	nlohmann::json j = {};
	j["mean"] = std::vector<double>{ 1.0, 2.0 };
	j["var"] = std::vector<double>{ 0.5, 0.25 };
	j["count"] = 10;

	BatchedWelfordStat loaded = BatchedWelfordStat(2);
	loaded.ReadFromJSON(j);
	CHECK_EQ(loaded.count, 10);
	CHECK_NEAR(loaded.GetMean()[0], 1.0, 1e-12);
	CHECK_NEAR(loaded.GetMean()[1], 2.0, 1e-12);
}

TEST_CASE(BatchedWelfordStat_RejectsWrongWidth) {
	nlohmann::json j = {};
	j["means"] = std::vector<double>{ 1.0, 2.0, 3.0 };
	j["vars"] = std::vector<double>{ 0.5, 0.25, 0.1 };
	j["count"] = 10;

	BatchedWelfordStat loaded = BatchedWelfordStat(2);
	CHECK_THROWS(loaded.ReadFromJSON(j));
}
