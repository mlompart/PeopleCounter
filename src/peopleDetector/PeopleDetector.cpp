#include "PeopleDetector.hpp"
#include "cudaDraw.h"
#include "cudaFont.h"
#include "cudaMappedMemory.h"
#include <jetson-utils/filesystem.h>
#include <jetson-utils/logging.h>

#define OUTPUT_UFF 0 // UFF has primary output containing detection results
#define OUTPUT_NUM 1 // UFF has secondary output containing one detection count

#define OUTPUT_CONF 0 // ONNX SSD-Mobilenet has confidence as first, bbox second
#define CHECK_NULL_STR(x) (x != NULL) ? x : "NULL"

#define DETECTNET_DEFAULT_THRESHOLD2 0.3
namespace peopleDetector
{

PeopleDetector::PeopleDetector(float meanPixel) : tensorNet()
{
	coverageThreshold_ = DETECTOR_DEFAULT_THRESHOLD;
	meanPixel_ = meanPixel;
	lineWidth_ = 2.0f;
	numClasses_ = 0;

	classColors_[0] = NULL; // cpu ptr
	classColors_[1] = NULL; // gpu ptr

	detectionSets_[0] = NULL; // cpu ptr
	detectionSets_[1] = NULL; // gpu ptr
	detectionSet_ = 0;
	maxDetections_ = 0;
}

// destructor
PeopleDetector::~PeopleDetector()
{
	if (detectionSets_ != NULL) {
		CUDA(cudaFreeHost(detectionSets_[0]));

		detectionSets_[0] = NULL;
		detectionSets_[1] = NULL;
	}

	if (classColors_ != NULL) {
		CUDA(cudaFreeHost(classColors_[0]));

		classColors_[0] = NULL;
		classColors_[1] = NULL;
	}
}
std::unique_ptr<PeopleDetector> PeopleDetector::Create(const std::string model)
{
	if (model == "ssd-mobilenet-v2") {
		return Create("networks/SSD-Mobilenet-v2/ssd_mobilenet_v2_coco.uff", "networks/SSD-Mobilenet-v2/ssd_coco_labels.txt",
			      DETECTNET_DEFAULT_THRESHOLD2, "Input", Dims3(3, 300, 300), "NMS", "NMS_1", DEFAULT_MAX_BATCH_SIZE, TYPE_FP32,
			      DEVICE_GPU, true);
	} else if (model == "ssd-mobilenet-v1") {
		return Create("networks/SSD-Mobilenet-v1/ssd_mobilenet_v1_coco.uff", "networks/SSD-Mobilenet-v1/ssd_coco_labels.txt",
			      DETECTNET_DEFAULT_THRESHOLD2, "Input", Dims3(3, 300, 300), "Postprocessor", "Postprocessor_1", DEFAULT_MAX_BATCH_SIZE,
			      TYPE_FP32, DEVICE_GPU, true);
	} else {
		Create("networks/SSD-Inception-v2/ssd_inception_v2_coco.uff", "networks/SSD-Inception-v2/ssd_coco_labels.txt",
		       DETECTNET_DEFAULT_THRESHOLD2, "Input", Dims3(3, 300, 300), "NMS", "NMS_1", DEFAULT_MAX_BATCH_SIZE, TYPE_FP32, DEVICE_GPU,
		       true);
	}
}
// Create (UFF)
std::unique_ptr<PeopleDetector> PeopleDetector::Create(const char* model = "networks/SSD-Mobilenet-v2/ssd_mobilenet_v2_coco.uff",
						       const char* class_labels = "networks/SSD-Mobilenet-v2/ssd_coco_labels.txt",
						       float threshold = DETECTNET_DEFAULT_THRESHOLD, const char* input = "Input",
						       const Dims3& inputDims = Dims3(3, 300, 300), const char* output = "NMS",
						       const char* numDetections = "NMS_1", uint32_t maxBatchSize = DEFAULT_MAX_BATCH_SIZE,
						       precisionType precision = TYPE_FASTEST, deviceType device = DEVICE_GPU,
						       bool allowGPUFallback = true)
{
	std::unique_ptr<PeopleDetector> net = std::make_unique<PeopleDetector>();

	if (!net)
		return NULL;

	LogInfo("\n");
	LogInfo("PeopleDetector -- loading detection network model from:\n");
	LogInfo("          -- model        %s\n", CHECK_NULL_STR(model));
	LogInfo("          -- input_blob   '%s'\n", CHECK_NULL_STR(input));
	LogInfo("          -- output_blob  '%s'\n", CHECK_NULL_STR(output));
	LogInfo("          -- output_count '%s'\n", CHECK_NULL_STR(numDetections));
	LogInfo("          -- class_labels %s\n", CHECK_NULL_STR(class_labels));
	LogInfo("          -- threshold    %f\n", threshold);
	LogInfo("          -- batch_size   %u\n\n", maxBatchSize);

	// create list of output names
	std::vector<std::string> output_blobs;

	if (output != NULL)
		output_blobs.push_back(output);

	if (numDetections != NULL)
		output_blobs.push_back(numDetections);

	// load the model
	if (!net->LoadNetwork(NULL, model, NULL, input, inputDims, output_blobs, maxBatchSize, precision, device, allowGPUFallback)) {
		LogError(LOG_TRT "PeopleDetector -- failed to initialize.\n");
		return NULL;
	}

	// allocate detection sets
	if (!net->allocDetections())
		return NULL;

	// load class descriptions
	net->loadClassInfo(class_labels);

	// set default class colors
	if (!net->defaultColors())
		return NULL;

	// set the specified threshold
	net->setThreshold(threshold);

	return net;
}

// allocDetections
bool PeopleDetector::allocDetections()
{
	// determine max detections

	maxDetections_ = DIMS_H(mOutputs[OUTPUT_UFF].dims) * DIMS_C(mOutputs[OUTPUT_UFF].dims);

	LogVerbose(LOG_TRT "PeopleDetector -- maximum bounding boxes:  %u\n", maxDetections_);

	// allocate array to store detection results
	const size_t det_size = sizeof(Detection) * numDetectionSets_ * maxDetections_;

	if (!cudaAllocMapped((void**)&detectionSets_[0], (void**)&detectionSets_[1], det_size))
		return false;

	memset(detectionSets_[0], 0, det_size);
	return true;
}

// loadClassInfo
bool PeopleDetector::loadClassInfo(const char* filename)
{
	if (!LoadClassInfo(filename, classDesc_, classSynset_, numClasses_))
		return false;

	numClasses_ = classDesc_.size();

	LogInfo(LOG_TRT "PeopleDetector -- number of object classes:  %u\n", numClasses_);
	classPath_ = locateFile(filename);
	return true;
}

// LoadClassInfo
bool PeopleDetector::LoadClassInfo(const char* filename, std::vector<std::string>& descriptions, std::vector<std::string>& synsets,
				   int expectedClasses)
{
	if (!filename)
		return false;

	// locate the file
	const std::string path = locateFile(filename);

	if (path.length() == 0) {
		LogError(LOG_TRT "PeopleDetector -- failed to find %s\n", filename);
		return false;
	}

	// open the file
	FILE* f = fopen(path.c_str(), "r");

	if (!f) {
		LogError(LOG_TRT "PeopleDetector -- failed to open %s\n", path.c_str());
		return false;
	}

	descriptions.clear();
	synsets.clear();

	// read class descriptions
	char str[512];
	uint32_t customClasses = 0;

	while (fgets(str, 512, f) != NULL) {
		const int syn = 9; // length of synset prefix (in characters)
		const int len = strlen(str);

		if (len > syn && str[0] == 'n' && str[syn] == ' ') {
			str[syn] = 0;
			str[len - 1] = 0;

			const std::string a = str;
			const std::string b = (str + syn + 1);

			// printf("a=%s b=%s\n", a.c_str(), b.c_str());

			synsets.push_back(a);
			descriptions.push_back(b);
		} else if (len > 0) // no 9-character synset prefix (i.e. from DIGITS snapshot)
		{
			char a[10];
			sprintf(a, "n%08u", customClasses);

			// printf("a=%s b=%s (custom non-synset)\n", a, str);
			customClasses++;

			if (str[len - 1] == '\n')
				str[len - 1] = 0;

			synsets.push_back(a);
			descriptions.push_back(str);
		}
	}

	fclose(f);

	LogVerbose(LOG_TRT "PeopleDetector -- loaded %zu class info entries\n", synsets.size());

	const int numLoaded = descriptions.size();

	if (numLoaded == 0)
		return false;

	if (expectedClasses > 0) {
		if (numLoaded != expectedClasses)
			LogError(LOG_TRT "PeopleDetector -- didn't load expected number of class "
					 "descriptions  (%i of %i)\n",
				 numLoaded, expectedClasses);

		if (numLoaded < expectedClasses) {
			LogWarning(LOG_TRT "PeopleDetector -- filling in remaining %i class descriptions "
					   "with default labels\n",
				   (expectedClasses - numLoaded));

			for (int n = numLoaded; n < expectedClasses; n++) {
				char synset[10];
				sprintf(synset, "n%08i", n);

				char desc[64];
				sprintf(desc, "Class #%i", n);

				synsets.push_back(synset);
				descriptions.push_back(desc);
			}
		}
	}

	return true;
}

// LoadClassInfo
bool PeopleDetector::LoadClassInfo(const char* filename, std::vector<std::string>& descriptions, int expectedClasses)
{
	std::vector<std::string> synsets;
	return LoadClassInfo(filename, descriptions, synsets, expectedClasses);
}

// defaultColors
bool PeopleDetector::defaultColors()
{
	const uint32_t numClasses = GetNumClasses();

	if (!cudaAllocMapped((void**)&classColors_[0], (void**)&classColors_[1], numClasses * sizeof(float4)))
		return false;

	for (int i = 0; i < numClasses; i++) {
		uint8_t rgb[] = {0, 0, 0};
		GenerateColor(i, rgb);

		classColors_[0][i * 4 + 0] = rgb[0];
		classColors_[0][i * 4 + 1] = rgb[1];
		classColors_[0][i * 4 + 2] = rgb[2];
		classColors_[0][i * 4 + 3] = DETECTNET_DEFAULT_ALPHA;

		// printf(LOG_TRT "color %02i  %3i %3i %3i %3i\n", i, (int)r, (int)g,
		// (int)b, (int)alpha);
	}

	return true;
} // namespace peopleDetector

// GenerateColor
void PeopleDetector::GenerateColor(uint32_t classID, uint8_t* rgb)
{
	if (!rgb)
		return;

	// the first color is black, skip that one
	classID += 1;

// https://github.com/dusty-nv/pytorch-segmentation/blob/16882772bc767511d892d134918722011d1ea771/datasets/sun_remap.py#L90
#define bitget(byteval, idx) ((byteval & (1 << idx)) != 0)

	int r = 0;
	int g = 0;
	int b = 0;
	int c = classID;

	for (int j = 0; j < 8; j++) {
		r = r | (bitget(c, 0) << 7 - j);
		g = g | (bitget(c, 1) << 7 - j);
		b = b | (bitget(c, 2) << 7 - j);
		c = c >> 3;
	}

	rgb[0] = r;
	rgb[1] = g;
	rgb[2] = b;
}

// Detect
int PeopleDetector::Detect(void* input, uint32_t width, uint32_t height, imageFormat format, Detection** detections, uint32_t overlay)
{
	Detection* det = detectionSets_[0] + detectionSet_ * GetMaxDetections();

	if (detections != NULL)
		*detections = det;

	detectionSet_++;

	if (detectionSet_ >= numDetectionSets_)
		detectionSet_ = 0;

	return Detect(input, width, height, format, det, overlay);
}
// Detect
int PeopleDetector::Detect(void* input, uint32_t width, uint32_t height, imageFormat format, Detection* detections, uint32_t overlay)
{
	if (!input || width == 0 || height == 0 || !detections) {
		LogError(LOG_TRT "PeopleDetector::Detect( 0x%p, %u, %u ) -> invalid parameters\n", input, width, height);
		return -1;
	}

	if (!imageFormatIsRGB(format)) {
		LogError(LOG_TRT "PeopleDetector::Detect() -- unsupported image format (%s)\n", imageFormatToStr(format));
		LogError(LOG_TRT "                       supported formats are:\n");
		LogError(LOG_TRT "                          * rgb8\n");
		LogError(LOG_TRT "                          * rgba8\n");
		LogError(LOG_TRT "                          * rgb32f\n");
		LogError(LOG_TRT "                          * rgba32f\n");

		return -1;
	}

	PROFILER_BEGIN(PROFILER_PREPROCESS);

	if (CUDA_FAILED(cudaTensorNormBGR(input, format, width, height, mInputs[0].CUDA, GetInputWidth(), GetInputHeight(), make_float2(-1.0f, 1.0f),
					  GetStream()))) {
		LogError(LOG_TRT "PeopleDetector::Detect() -- cudaTensorNormBGR() failed\n");
		return -1;
	}

	PROFILER_END(PROFILER_PREPROCESS);
	PROFILER_BEGIN(PROFILER_NETWORK);

	if (!ProcessNetwork())
		return -1;

	PROFILER_END(PROFILER_NETWORK);
	PROFILER_BEGIN(PROFILER_POSTPROCESS);

	// post-processing / clustering
	int numDetections = 0;

	const int rawDetections = *(int*)mOutputs[OUTPUT_NUM].CPU;
	const int rawParameters = DIMS_W(mOutputs[OUTPUT_UFF].dims);

#ifdef DEBUG_CLUSTERING
	LogDebug(LOG_TRT "PeopleDetector::Detect() -- %i unfiltered detections\n", rawDetections);
#endif

	// filter the raw detections by thresholding the confidence and choose only
	// person class
	for (int n = 0; n < rawDetections; n++) {
		float* object_data = mOutputs[OUTPUT_UFF].CPU + n * rawParameters;

		uint32_t classId = object_data[1];
		if (classId != 1)
			continue; // beacause classId = 1 only for pedestrian
		if (object_data[2] < coverageThreshold_)
			continue;

		detections[numDetections].Instance = numDetections; //(uint32_t)object_data[0];
		detections[numDetections].ClassID = (uint32_t)object_data[1];
		detections[numDetections].Confidence = object_data[2];
		detections[numDetections].Left = object_data[3] * width;
		detections[numDetections].Top = object_data[4] * height;
		detections[numDetections].Right = object_data[5] * width;
		detections[numDetections].Bottom = object_data[6] * height;

		if (detections[numDetections].ClassID >= numClasses_) {
			LogError(LOG_TRT "PeopleDetector::Detect() -- detected object has invalid "
					 "classID (%u)\n",
				 detections[numDetections].ClassID);
			detections[numDetections].ClassID = 0;
		}
		LogVerbose(LOG_TRT "detections[%i].ClassID = %i\n", numDetections, detections[numDetections].ClassID);
		LogVerbose(LOG_TRT "detections[%i].Confidence = %f\n", numDetections, detections[numDetections].Confidence);
		LogVerbose(LOG_TRT "detections[%i].Left = %f\n", numDetections, detections[numDetections].Left);
		LogVerbose(LOG_TRT "detections[%i].Top = %f\n", numDetections, detections[numDetections].Top);
		LogVerbose(LOG_TRT "detections[%i].Right = %f\n", numDetections, detections[numDetections].Right);
		LogVerbose(LOG_TRT "detections[%i].Bottom = %f\n", numDetections, detections[numDetections].Bottom);

		if (strcmp(GetClassDesc(detections[numDetections].ClassID), "void") == 0)
			continue;

		numDetections += clusterDetections(detections, numDetections);
	}

	// sort the detections by confidence value
	sortDetections(detections, numDetections);

	// verify the bounding boxes are within the bounds of the image
	for (int n = 0; n < numDetections; n++) {
		if (detections[n].Top < 0)
			detections[n].Top = 0;

		if (detections[n].Left < 0)
			detections[n].Left = 0;

		if (detections[n].Right >= width)
			detections[n].Right = width - 1;

		if (detections[n].Bottom >= height)
			detections[n].Bottom = height - 1;
	}

	PROFILER_END(PROFILER_POSTPROCESS);

	// return the number of detections
	return numDetections;
}

void PeopleDetector::UpdateVisuals(void* input, uint32_t width, uint32_t height, imageFormat format, int numDetections,
				    std::vector<std::shared_ptr<Detection>> detections, uint32_t overlay)
{
	// render the overlay
	if (overlay != 0 && numDetections > 0) {
		if (!Overlay(input, input, width, height, format, detections, numDetections, overlay))
			LogError(LOG_TRT "PeopleDetector::Detect() -- failed to render overlay\n");
	}
	const float4 lineColor = {255, 255, 255, 123};
	cudaDrawLine(input, input, width, height, format, 400, 0, 400, height, lineColor, lineWidth_);

	const int2 txtPos = make_int2(5, 5);
	char txt[256];
	sprintf(txt, "Status: %i, In: %i, Out: %i", counter->getStatus(), counter->getEntered(), counter->getLeft());
	static cudaFont* font = NULL;

	// make sure the font object is created
	if (!font) {
		font = cudaFont::Create(adaptFontSize(width));
	}

	std::vector<std::pair<std::string, int2>> labels;
	labels.push_back(std::pair<std::string, int2>(txt, txtPos));
	font->OverlayText(input, format, width, height, labels, make_float4(255, 255, 255, 255));

	// wait for GPU to complete work
	CUDA(cudaDeviceSynchronize());
}

// clusterDetections (UFF/ONNX)
int PeopleDetector::clusterDetections(Detection* detections, int n, float threshold)
{
	if (n == 0)
		return 1;

	// test each detection to see if it intersects
	for (int m = 0; m < n; m++) {
		if (detections[n].Intersects(detections[m],
					     threshold)) // TODO NMS or different threshold for same classes?
		{
			if (detections[n].ClassID != detections[m].ClassID) {
				if (detections[n].Confidence > detections[m].Confidence) {
					detections[m] = detections[n];

					detections[m].Instance = m;
					detections[m].ClassID = detections[n].ClassID;
					detections[m].Confidence = detections[n].Confidence;
				}
			} else {
				detections[m].Expand(detections[n]);
				detections[m].Confidence = fmaxf(detections[n].Confidence, detections[m].Confidence);
			}

			return 0; // merged detection
		}
	}

	return 1; // new detection
}

// sortDetections (UFF)
void PeopleDetector::sortDetections(Detection* detections, int numDetections)
{
	if (numDetections < 2)
		return;

	// order by area (descending) or confidence (ascending)
	for (int i = 0; i < numDetections - 1; i++) {
		for (int j = 0; j < numDetections - i - 1; j++) {
			if (detections[j].Area() < detections[j + 1].Area()) // if( detections[j].Confidence >
									     // detections[j+1].Confidence )
			{
				const Detection det = detections[j];
				detections[j] = detections[j + 1];
				detections[j + 1] = det;
			}
		}
	}

	// renumber the instance ID's
	for (int i = 0; i < numDetections; i++)
		detections[i].Instance = i;
}

// from detectNet.cu
cudaError_t cudaDetectionOverlay(void* input, void* output, uint32_t width, uint32_t height, imageFormat format,
				 std::vector<std::shared_ptr<Detection>> detections, int numDetections, float4* colors);

// Overlay
bool PeopleDetector::Overlay(void* input, void* output, uint32_t width, uint32_t height, imageFormat format,
			     std::vector<std::shared_ptr<Detection>> detections, uint32_t numDetections, uint32_t flags)
{
	PROFILER_BEGIN(PROFILER_VISUALIZE);

	if (flags == 0) {
		LogError(LOG_TRT "PeopleDetector -- Overlay() was called with OVERLAY_NONE, returning "
				 "false\n");
		return false;
	}

	// if input and output are different images, copy the input to the output
	// first then overlay the bounding boxes, ect. on top of the output image
	if (input != output) {
		if (CUDA_FAILED(cudaMemcpy(output, input, imageFormatSize(format, width, height), cudaMemcpyDeviceToDevice))) {
			LogError(LOG_TRT "PeopleDetector -- Overlay() failed to copy input image to output "
					 "image\n");
			return false;
		}
	}

	// make sure there are actually detections
	if (numDetections <= 0) {
		PROFILER_END(PROFILER_VISUALIZE);
		return true;
	}

	// bounding box overlay
	if (flags & detectNet::OVERLAY_BOX) {
		if (CUDA_FAILED(cudaDetectionOverlay(input, output, width, height, format, detections, numDetections, (float4*)classColors_[1])))
			return false;
	}

	// bounding box lines
	if (flags & detectNet::OVERLAY_LINES) {
		for (uint32_t n = 0; n < numDetections; n++) {
			auto& d = detections.at(n);

			const float4& color = ((float4*)classColors_[0])[d->ClassID];
			const float4& color2 = ((float4*)classColors_[0])[d->ClassID + 5];

			CUDA(cudaDrawLine(input, output, width, height, format, d->Left, d->Top, d->Right, d->Top, color, lineWidth_));
			CUDA(cudaDrawLine(input, output, width, height, format, d->Right, d->Top, d->Right, d->Bottom, color, lineWidth_));
			CUDA(cudaDrawLine(input, output, width, height, format, d->Left, d->Bottom, d->Right, d->Bottom, color, lineWidth_));
			CUDA(cudaDrawLine(input, output, width, height, format, d->Left, d->Top, d->Left, d->Bottom, color, lineWidth_));
			CUDA(cudaDrawCircle(input, output, width, height, format, d->Left, d->Top, 5, color2));
		}
	}

	// class label overlay
	if ((flags & detectNet::OVERLAY_LABEL) || (flags & detectNet::OVERLAY_CONFIDENCE)) {
		static cudaFont* font = NULL;

		// make sure the font object is created
		if (!font) {
			font = cudaFont::Create(adaptFontSize(width));

			if (!font) {
				LogError(LOG_TRT "PeopleDetector -- Overlay() was called with OVERLAY_FONT, but "
						 "failed to create cudaFont()\n");
				return false;
			}
		}

		// draw each object's description
		std::vector<std::pair<std::string, int2>> labels;

		for (uint32_t n = 0; n < numDetections; n++) {
			std::string className = GetClassDesc(detections[n]->ClassID);
			const float confidence = detections[n]->Confidence * 100.0f;
			const int trackId = detections[n]->trackId;
			className = className + " " + std::to_string(detections[n]->Instance);
			const int2 position = make_int2(detections[n]->Left + 5, detections[n]->Top + 3);

			if (flags & detectNet::OVERLAY_CONFIDENCE) {
				char str[256];

				if ((flags & detectNet::OVERLAY_LABEL) && (flags & detectNet::OVERLAY_CONFIDENCE))
					if (trackId >= 0)
						sprintf(str, "%i %s %.1f%% ", trackId, className.c_str(), confidence);
					else
						sprintf(str, "%s %.1f%% ", className.c_str(), confidence);
				else
					sprintf(str, "%.1f%%", confidence);

				labels.push_back(std::pair<std::string, int2>(str, position));
			} else {
				// overlay label only
				labels.push_back(std::pair<std::string, int2>(className, position));
			}
		}

		font->OverlayText(output, format, width, height, labels, make_float4(255, 255, 255, 255));
	}

	PROFILER_END(PROFILER_VISUALIZE);
	return true;
}
} // namespace peopleDetector
