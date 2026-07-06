#pragma once
#include "../FrameworkTorch.h"

namespace GGL {

	// Accumulates a scalar metric as a device tensor
	// This prevents synchronizing the GPU pipeline every minibatch just to read metrics,
	//	which is a surprisingly large cost when done many times per learn iteration
	struct MetricAccum {
		torch::Tensor sum = {};
		int64_t count = 0;

		void Add(const torch::Tensor& val) {
			auto detached = val.detach();
			if (sum.defined()) {
				sum += detached;
			} else {
				sum = detached.clone();
			}
			count++;
		}

		// NOTE: Synchronizes the device, only call once done accumulating
		float Get() const {
			if (!count)
				return 0;
			return sum.cpu().item<float>() / count;
		}
	};
}
