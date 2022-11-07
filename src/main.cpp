#include <sstream>
#include <string>

#include <jetson-utils/glDisplay.h>
#include <jetson-utils/gstCamera.h>
#include <jetson-utils/gstDecoder.h>
#include <jetson-utils/gstEncoder.h>

#include "peopleDetector/Counter.hpp"
#include "peopleDetector/Detection.hpp"
#include "peopleDetector/PeopleDetector.hpp"
#include "peopleDetector/Tracker.hpp"

std::mutex cout_mtx_;
using peopleDetector::Counter;
using peopleDetector::PeopleDetector;
using peopleDetector::Tracker;
bool signal_recieved = false;

void sig_handler(int signo)
{
	if (signo == SIGINT) {
		LogVerbose("received SIGINT\n");
		signal_recieved = true;
	}
}

int main()
{
	static Counter counter(0);
	Tracker tracker(counter);
	// gstCamera *input = gstCamera::Create();

	// gstDecoder* input =
	// gstDecoder::Create("rtsp://admin:admin@192.168.1.108:554",
	// videoOptions::CODEC_H264);
	gstDecoder* input = gstDecoder::Create("../my_video1.mp4", videoOptions::CODEC_H264);

	if (!input)
		return -1;

	const std::string model{"ssd-mobilenet-v1"};

	auto net = PeopleDetector::Create(model);
	net->setCounter(counter);

	glDisplay* output = glDisplay::Create();
	// gstEncoder *output2 = gstEncoder::Create("my_video.mp4",
	// videoOptions::CODEC_H264);

	// detect objects in the frame
	peopleDetector::Detection* detections = NULL;

	// Main loop
	int idx = 0;
	while (!tracker._shutdown and !signal_recieved) {

		// 1. Read in frame detections
		uchar3* image{nullptr};
		if (input->Capture(&image, 100) == false) {

			break;
		}
		const int numDetections = net->Detect(image, input->GetWidth(), input->GetHeight(), &detections);

		Tracker::DetectionVec frameDetections;

		std::cout << "Frame idx:" << idx << " numDetections:" << numDetections << std::endl;
		for (int n = 0; n < numDetections; ++n) {
			frameDetections.push_back(std::make_shared<peopleDetector::Detection>(detections[n]));
		}
		tracker.setNewDetections(idx, frameDetections);

		// 2. Associate detections (measurements) to existing tracks
		tracker.associate();
		// Modifies _newDetections, only unassociated new detections
		// remain

		// 3. Create new tracks from unassociated measurements
		tracker.createNewTracks();

		// 5. Update visuals
		net->upadateVisuals(image, input->GetWidth(), input->GetHeight(), numDetections, frameDetections);

		if (output != NULL) {
			output->Render(image, input->GetWidth(), input->GetHeight());
			// output2->Render(image, input->GetWidth(),
			// input->GetHeight()); update the status bar
			char str[256];
			sprintf(str, "TensorRT %i.%i.%i | %s | Network %.0f FPS", NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH,
				precisionTypeToStr(net->GetPrecision()), net->GetNetworkFPS());
			output->SetStatus(str);

			// check if the user quit
			if (!output->IsStreaming())
				signal_recieved = true;
		}

		++idx;
	}
	LogVerbose("detectnet:  shutting down...\n");

	SAFE_DELETE(input);
	SAFE_DELETE(output);
	// SAFE_DELETE(output2);

	LogVerbose("detectnet:  shutdown complete.\n");
	return 0;
}