#pragma once
#include <GigaLearnCPP/Framework.h>
#include <RLGymCPP/BasicTypes/Lists.h>

// Include torch
#include <ATen/ATen.h>
#include <ATen/autocast_mode.h>
#include <torch/utils.h>

#define RG_NO_GRAD torch::NoGradGuard _noGradGuard

#define RG_AUTOCAST_ON() { \
at::autocast::set_enabled(true); \
at::autocast::set_autocast_gpu_dtype(torch::kBFloat16); \
at::autocast::set_autocast_cpu_dtype(torch::kFloat); \
}

#define RG_AUTOCAST_OFF() { \
at::autocast::clear_cache(); \
at::autocast::set_enabled(false); \
}

#define RG_HALFPERC_TYPE torch::ScalarType::BFloat16

namespace GGL {
	// Fast vector-to-tensor conversion
	// Uses from_blob() + clone(), which is a memcpy,
	//	unlike torch::tensor() which iterates elementwise
	template <typename T>
	inline torch::Tensor VEC_TO_TENSOR(const std::vector<T>& vec) {
		auto options = torch::TensorOptions().dtype(torch::CppTypeToScalarType<T>::value);

		if (vec.empty())
			return torch::empty({ 0 }, options);

		return torch::from_blob(
			const_cast<T*>(vec.data()), { (int64_t)vec.size() },
			options
		).clone();
	}

	template <typename T>
	inline torch::Tensor DIMLIST2_TO_TENSOR(const RLGC::DimList2<T>& list) {
		return VEC_TO_TENSOR(list.data).reshape({ (int64_t)list.size[0], (int64_t)list.size[1] });
	}

	template <typename T>
	inline std::vector<T> TENSOR_TO_VEC(torch::Tensor tensor) {
		assert(tensor.dim() == 1);
		tensor = tensor.detach().cpu().contiguous().to(torch::CppTypeToScalarType<T>());
		const T* data = tensor.const_data_ptr<T>();
		return std::vector<T>(data, data + tensor.size(0));
	}
}