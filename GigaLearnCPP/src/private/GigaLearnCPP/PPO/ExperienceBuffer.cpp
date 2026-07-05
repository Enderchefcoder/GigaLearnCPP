#include "ExperienceBuffer.h"

using namespace torch;

GGL::ExperienceBuffer::ExperienceBuffer(int seed, torch::Device device) :
	seed(seed), device(device), rng(seed) {

}

GGL::ExperienceTensors GGL::ExperienceBuffer::_GetSamples(torch::Tensor tIndices) const {

	ExperienceTensors result;

	auto* toItr = result.begin();
	auto* fromItr = data.begin();
	for (; toItr != result.end(); toItr++, fromItr++)
		*toItr = torch::index_select(*fromItr, 0, tIndices);

	return result;
}

std::vector<GGL::ExperienceTensors> GGL::ExperienceBuffer::GetAllBatchesShuffled(int64_t batchSize, bool overbatching) {

	RG_NO_GRAD;

	int64_t expSize = data.states.size(0);

	if (expSize < batchSize)
		RG_ERR_CLOSE(
			"ExperienceBuffer::GetAllBatchesShuffled(): Not enough experience for a single batch " <<
			"(have " << expSize << ", batch size is " << batchSize << ").\n" <<
			"Make sure your batch size is not larger than the timesteps collected per iteration."
		);

	// Make a shuffled tensor of sample indices
	Tensor tIndices = torch::empty({ expSize }, torch::kInt64);
	{
		int64_t* indices = tIndices.data_ptr<int64_t>();
		std::iota(indices, indices + expSize, 0); // Fill ascending indices
		std::shuffle(indices, indices + expSize, rng);
	}

	// Get a sample set from each of the batches
	std::vector<ExperienceTensors> result;
	for (int64_t startIdx = 0; startIdx + batchSize <= expSize; startIdx += batchSize) {

		int64_t curBatchSize = batchSize;
		if (overbatching && (startIdx + batchSize * 2 > expSize)) {
			// Last batch of the iteration:
			// Extend batch size to the end of the experience so no timesteps are wasted
			curBatchSize = expSize - startIdx;
		}

		result.push_back(_GetSamples(tIndices.narrow(0, startIdx, curBatchSize)));
	}

	return result;
}
