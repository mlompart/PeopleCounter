#pragma once

#include <condition_variable>
#include <deque>
#include <list>
#include <mutex>
#include <thread>

#include "Counter.hpp"
#include "TrackedObject.hpp"
extern std::mutex cout_mtx_;

namespace peopleDetector
{
class Tracker
{
      public:
	using DetectionVec = std::vector<std::shared_ptr<Detection>>;

	Tracker();
	Tracker(Counter& counter);

	bool _shutdown = false;

	void setNewDetections(int, DetectionVec incomingDetections);
	void associate();
	void createNewTracks();

	// ################### Settings ###################
	const float _assocationDistanceThreshold = 120;
	// ################################################

      private:
	Counter* _counter;
	std::vector<std::thread> _threads;		     // Threads for the TrackedObjects to run in
	std::vector<std::shared_ptr<TrackedObject>> _tracks; // vector of shared_pts to tracks
	DetectionVec _newDetections;			     // vector of shared_pts to detections
};
} // namespace peopleDetector