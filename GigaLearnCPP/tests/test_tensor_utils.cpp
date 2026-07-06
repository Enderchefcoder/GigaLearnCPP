#include "TestFramework.h"

#include <private/GigaLearnCPP/FrameworkTorch.h>

using namespace GGL;
using namespace RLGC;

TEST_CASE(TensorUtils_VecToTensorRoundtrip) {
	std::vector<float> values = { 1.5f, -2.25f, 0, 1e10f, -1e-10f };

	auto tensor = VEC_TO_TENSOR(values);
	CHECK_EQ(tensor.dim(), 1);
	CHECK_EQ(tensor.size(0), (int64_t)values.size());
	CHECK_TRUE(tensor.dtype() == torch::kFloat32);

	auto roundtrip = TENSOR_TO_VEC<float>(tensor);
	CHECK_EQ(roundtrip.size(), values.size());
	for (int i = 0; i < values.size(); i++)
		CHECK_EQ(roundtrip[i], values[i]);
}

TEST_CASE(TensorUtils_VecToTensorTypes) {
	// int32
	{
		std::vector<int32_t> values = { 1, -5, 100000 };
		auto tensor = VEC_TO_TENSOR(values);
		CHECK_TRUE(tensor.dtype() == torch::kInt32);
		auto roundtrip = TENSOR_TO_VEC<int32_t>(tensor);
		for (int i = 0; i < values.size(); i++)
			CHECK_EQ(roundtrip[i], values[i]);
	}

	// uint8
	{
		std::vector<uint8_t> values = { 0, 1, 255 };
		auto tensor = VEC_TO_TENSOR(values);
		CHECK_TRUE(tensor.dtype() == torch::kUInt8);
		auto roundtrip = TENSOR_TO_VEC<uint8_t>(tensor);
		for (int i = 0; i < values.size(); i++)
			CHECK_EQ(roundtrip[i], values[i]);
	}

	// Empty
	{
		std::vector<float> values = {};
		auto tensor = VEC_TO_TENSOR(values);
		CHECK_EQ(tensor.size(0), 0);
	}
}

TEST_CASE(TensorUtils_VecToTensorIsACopy) {
	// The tensor must own its data (the source vector may be freed/modified after)
	std::vector<float> values = { 1, 2, 3 };
	auto tensor = VEC_TO_TENSOR(values);

	values[0] = 100;
	values.clear();
	values.shrink_to_fit();

	auto result = TENSOR_TO_VEC<float>(tensor);
	CHECK_EQ(result[0], 1);
	CHECK_EQ(result[1], 2);
	CHECK_EQ(result[2], 3);
}

TEST_CASE(TensorUtils_DimList2ToTensor) {
	DimList2<float> list = DimList2<float>(3, 2);
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 2; j++)
			list.At(i, j) = i * 10 + j;

	auto tensor = DIMLIST2_TO_TENSOR(list);
	CHECK_EQ(tensor.dim(), 2);
	CHECK_EQ(tensor.size(0), 3);
	CHECK_EQ(tensor.size(1), 2);

	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 2; j++)
			CHECK_EQ(tensor[i][j].item<float>(), i * 10 + j);
}

TEST_CASE(DimList2_RowOperations) {
	DimList2<float> list = DimList2<float>(2, 3);
	list.Set(0, { 1, 2, 3 });
	list.Set(1, { 4, 5, 6 });

	auto row0 = list.GetRow(0);
	CHECK_EQ(row0.size(), 3);
	CHECK_EQ(row0[0], 1);
	CHECK_EQ(row0[2], 3);

	// AppendRowTo appends without clearing
	std::vector<float> out = { 100 };
	list.AppendRowTo(1, out);
	CHECK_EQ(out.size(), 4);
	CHECK_EQ(out[0], 100);
	CHECK_EQ(out[1], 4);
	CHECK_EQ(out[3], 6);

	// AppendRowTo must match GetRow
	std::vector<float> appended = {};
	list.AppendRowTo(0, appended);
	auto direct = list.GetRow(0);
	CHECK_EQ(appended.size(), direct.size());
	for (int i = 0; i < direct.size(); i++)
		CHECK_EQ(appended[i], direct[i]);
}
