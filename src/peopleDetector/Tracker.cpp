#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

#include "Tracker.hpp"
namespace peopleDetector
{
Tracker::Tracker() : _counter(nullptr) {}

Tracker::Tracker(Counter& counter) : _counter(&counter) {}

void Tracker::setNewDetections(int idx, DetectionVec incomingDetections)
{
	cout_mtx_.lock();
	std::cout << "//TrackerManager// Running setNewDetections()." << std::endl;
	cout_mtx_.unlock();

	_newDetections.clear();
	for (auto& det : incomingDetections) {
		det->frameId = idx;
		det->x_mid = ((float)(det->Left) + (float)(det->Right)) / 2;
		det->y_mid = ((float)(det->Top) + (float)(det->Bottom)) / 2;
	}
	_newDetections = std::move(incomingDetections);
}

void Tracker::associate()
{
	cout_mtx_.lock();
	std::cout << "//Tracker// Running associate()." << std::endl;
	cout_mtx_.unlock();

	for (int i_track = 0; i_track < _tracks.size(); ++i_track) {
		auto& track = _tracks[i_track];

		if (track->getObjectState() != terminated) {
			cout_mtx_.lock();
			std::cout << "//Tracker// !!!Track!!!: " << track->_id << std::endl;
			cout_mtx_.unlock();

			bool track_associated = false;

			for (int i_det = 0; i_det < _newDetections.size(); ++i_det) {
				auto& det = _newDetections[i_det];
				cout_mtx_.lock();
				std::cout << "//Tracker// !!!Detection!!!: " << det->x_mid << "," << det->y_mid << std::endl;
				cout_mtx_.unlock();

				float distance = track->measureDistance(det);
				cout_mtx_.lock();
				std::cout << "//Tracker// Measured distance: " << distance << std::endl;
				cout_mtx_.unlock();

				if (distance <= _assocationDistanceThreshold) {
					det->associated = true;
					track->sendDetection(det);
					cout_mtx_.lock();
					std::cout << "//Tracker// *****Detection " << i_det << " associated to Track " << i_track << std::endl;
					cout_mtx_.unlock();
					det->trackId = track->_id;
					track_associated = true;
					break;
				}
			}

			if (!track_associated) {
				cout_mtx_.lock();
				std::cout << "//Tracker// ********Track NOT "
					     "associated! Sending "
					     "nullptr..."
					  << std::endl;
				cout_mtx_.unlock();
				track->sendDetection(nullptr);
			}
		}
	}
}

void Tracker::createNewTracks()
{
	cout_mtx_.lock();
	std::cout << "//Tracker// Running createNewTracks()." << std::endl;
	cout_mtx_.unlock();

	for (auto& newDet : _newDetections) {
		// For each remaining unassociated detection, start a new track
		if (!newDet->associated) {
			std::shared_ptr<TrackedObject> newTrack = std::make_shared<TrackedObject>(newDet, _counter);
			_tracks.push_back(newTrack);
			_threads.emplace_back(std::thread(&TrackedObject::run, newTrack));
		}
	}
}
} // namespace peopleDetector