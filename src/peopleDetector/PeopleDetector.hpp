#pragma once

#include "Counter.hpp"
#include "Detection.hpp"
#include <jetson-inference/detectNet.h>
#include <jetson-inference/tensorConvert.h>
#include <jetson-inference/tensorNet.h>
#include <memory>

/**
 * Default value of the minimum detection threshold
 * @ingroup detectNet
 */
#define DETECTOR_DEFAULT_THRESHOLD 0.9f

/**
 * Default alpha blending value used during overlay
 * @ingroup detectNet
 */
#define DETECTNET_DEFAULT_ALPHA 120

namespace peopleDetector
{

class PeopleDetector : public tensorNet
{

      public:
	PeopleDetector(float meanPixel = 0.0f);
	~PeopleDetector() override;

	static std::unique_ptr<PeopleDetector> Create(const std::string model);
	static std::unique_ptr<PeopleDetector> Create(const char* model, const char* class_labels, float threshold, const char* input,
						      const Dims3& inputDims, const char* output, const char* numDetections, uint32_t maxBatchSize,
						      precisionType precision, deviceType device, bool allowGPUFallback);

	int Detect(void* input, uint32_t width, uint32_t height, imageFormat format, Detection* detections,
		   uint32_t overlay = detectNet::OVERLAY_DEFAULT);

	template <typename T>
	int Detect(T* image, uint32_t width, uint32_t height, Detection** detections, uint32_t overlay = detectNet::OVERLAY_DEFAULT)
	{
		return Detect((void*)image, width, height, imageFormatFromType<T>(), detections, overlay);
	}

	int Detect(void* input, uint32_t width, uint32_t height, imageFormat format, Detection** detections, uint32_t overlay);
	template <typename T>
	void UpdateVisuals(T* input, uint32_t width, uint32_t height, int numDetections, std::vector<std::shared_ptr<Detection>> detections,
			    uint32_t overlay = detectNet::OVERLAY_DEFAULT)
	{
		UpdateVisuals((void*)input, width, height, imageFormatFromType<T>(), numDetections, detections);
	}
	void UpdateVisuals(void* input, uint32_t width, uint32_t height, imageFormat format, int numDetections,
			    std::vector<std::shared_ptr<Detection>> detections, uint32_t overlay = detectNet::OVERLAY_DEFAULT);
	inline void setThreshold(float threshold) { coverageThreshold_ = threshold; }

	/**
	 * Load class descriptions from a label file.
	 */
	static bool LoadClassInfo(const char* filename, std::vector<std::string>& descriptions, int expectedClasses = -1);

	/**
	 * Load class descriptions and synset strings from a label file.
	 */
	static bool LoadClassInfo(const char* filename, std::vector<std::string>& descriptions, std::vector<std::string>& synsets,
				  int expectedClasses = -1);

	/**
	 * Retrieve the number of object classes supported in the detector
	 */
	inline uint32_t GetNumClasses() const { return numClasses_; }

	static void GenerateColor(uint32_t classID, uint8_t* rgb);

	inline const char* GetClassDesc(uint32_t index) const { return classDesc_[index].c_str(); }

	/**
	 * Draw the detected bounding boxes overlayed on an RGBA image.
	 * @note Overlay() will automatically be called by default by Detect(),
	 * if the overlay parameter is true
	 * @param input input image in CUDA device memory.
	 * @param output output image in CUDA device memory.
	 * @param detections Array of detections allocated in CUDA device
	 * memory.
	 */
	bool Overlay(void* input, void* output, uint32_t width, uint32_t height, imageFormat format,
		     std::vector<std::shared_ptr<Detection>> detections, uint32_t numDetections, uint32_t flags = detectNet::OVERLAY_DEFAULT);

	inline uint32_t GetMaxDetections() const { return maxDetections_; }
	inline void setCounter(Counter& setCounter) { counter = &setCounter; }

      protected:
	bool allocDetections();
	bool defaultColors();
	bool loadClassInfo(const char* filename);
	int clusterDetections(Detection* detections, int n, float threshold = DETECTOR_DEFAULT_THRESHOLD);
	// bool isIndoor(Detection& detection);
	void sortDetections(Detection* detections, int numDetections);
	float coverageThreshold_;
	float* classColors_[2];
	float meanPixel_;
	float lineWidth_;

	std::vector<std::string> classDesc_;
	std::vector<std::string> classSynset_;
	std::string classPath_;
	uint32_t numClasses_;

	Detection* detectionSets_[2];		      // list of detections, detectionSets_ *
						      // maxDetections_
	uint32_t detectionSet_;			      // index of next detection set to use
	uint32_t maxDetections_;		      // number of raw detections in the grid
	static const uint32_t numDetectionSets_ = 16; // size of detection ringbuffer
	Counter* counter;
};
} // namespace peopleDetector