#include "Tracker.hpp"
#include <iostream>
#include <logging.h>
#include <mutex>
#include <thread>
#include <vector>
namespace peopleDetector
{
Tracker::Tracker() : _counter(nullptr) {}

Tracker::Tracker(Counter& counter) : _counter(&counter) {}

void Tracker::setNewDetections(int idx, DetectionVec incomingDetections)
{
	cout_mtx_.lock();
	LogVerbose("//Tracker// Running setNewDetections()\n");
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
	LogVerbose("/Tracker// Running associate()\n");

	cout_mtx_.unlock();

	for (int i_track = 0; i_track < _tracks.size(); ++i_track) {
		auto& track = _tracks[i_track];

		if (track->getObjectState() != terminated) {
			cout_mtx_.lock();
			LogVerbose("//Tracker// !!!Track!!!: %i\n", track->_id);

			cout_mtx_.unlock();

			bool track_associated = false;

			for (int i_det = 0; i_det < _newDetections.size(); ++i_det) {
				auto& det = _newDetections[i_det];
				cout_mtx_.lock();
				LogVerbose("//Tracker// !!!Detection!!!: %f, %f\n", det->x_mid, det->y_mid);

				cout_mtx_.unlock();

				float distance = track->measureDistance(det);
				cout_mtx_.lock();
				LogVerbose("//Tracker// Measured distance: %f\n", distance);

				cout_mtx_.unlock();

				if (distance <= _assocationDistanceThreshold) {
					det->associated = true;
					track->sendDetection(det);
					cout_mtx_.lock();
					LogVerbose("//Tracker// Detection %i associated to Track %i\n", i_det, distance);
					cout_mtx_.unlock();
					det->trackId = track->_id;
					track_associated = true;
					break;
				}
			}

			if (!track_associated) {
				cout_mtx_.lock();

				LogVerbose("//Tracker// Track NOT associated! Sendinng nullptr...\n");

				cout_mtx_.unlock();
				track->sendDetection(nullptr);
			}
		}
	}
}

void Tracker::createNewTracks()
{
	cout_mtx_.lock();
	//std::cout << "//Tracker// Running createNewTracks()." << std::endl;
	LogVerbose("//Tracker// Running createNewTracks()\n");
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