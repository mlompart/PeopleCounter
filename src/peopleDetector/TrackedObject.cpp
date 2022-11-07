#include <cmath>
#include <iostream>
#include <mutex>
#include <thread>

#include "TrackedObject.hpp"
namespace peopleDetector
{
int TrackedObject::_idCount = 0;

// Initialize state transition matrix
Eigen::Matrix<float, 6, 6> TrackedObject::_A = [] {
	Eigen::Matrix<float, 6, 6> tmp;
	tmp << 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
	return tmp;
}();

// Initialize measurement matrix
Eigen::Matrix<float, 2, 6> TrackedObject::_H = [] {
	Eigen::Matrix<float, 2, 6> tmp;
	tmp << 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0;
	return tmp;
}();

template <typename T> T MessageQueue<T>::receive()
{
	std::unique_lock<std::mutex> uLock(_mutex);
	_cond.wait(uLock, [this] { return !_queue.empty(); });
	T msg = std::move(_queue.back());
	_queue.pop_back();

	return msg;
}

template <typename T> void MessageQueue<T>::send(T&& msg)
{
	std::lock_guard<std::mutex> uLock(_mutex);
	_queue.push_back(std::move(msg));
	_cond.notify_one();
}

void TrackedObject::timeUpdate()
{
	cout_mtx_.lock();
	std::cout << "//TrackedObject// Object ID:" << _id << " timeUpdate()" << std::endl;
	cout_mtx_.unlock();

	// Predict state transition
	_X = _A * _X;

	// Update Error covariance matrix
	_P = _A * (_P * _A.transpose()) + _Q;
}

void TrackedObject::measurementUpdate()
{
	cout_mtx_.lock();
	std::cout << "//TrackedObject// Object ID:" << _id << " measurementUpdate()" << std::endl;
	cout_mtx_.unlock();

	// Compute Kalman Gain
	auto _K = _P * _H.transpose() * (_H * _P * _H.transpose() + _R).inverse();

	// Fuse new measurement
	_X = _X + _K * (_Z - _H * _X);

	// Update Error covariance matrix
	_P = (Eigen::Matrix<float, 6, 6>::Identity() - _K * _H) * _P;
}

void TrackedObject::sendDetection(std::shared_ptr<Detection> det) { _detectionQueue.send(std::move(det)); }

TrackedObject::TrackedObject(std::shared_ptr<Detection> newDet, Counter* counter) : _id(_idCount), _counter(counter)
{
	++_idCount;
	cout_mtx_.lock();
	std::cout << _id
		  << "//TrackedObject// Creating new tracked object from "
		     "detection at: ("
		  << newDet->x_mid << "," << newDet->y_mid << ")" << std::endl;
	cout_mtx_.unlock();

	// Initialize state vector with inital position
	_X << newDet->x_mid, newDet->y_mid, 0, 0, 0, 0;

	// Initialize Error Covariance matrix
	_P << _initalErrorCovariance, 0, 0, 0, 0, 0, 0, _initalErrorCovariance, 0, 0, 0, 0, 0, 0, _initalErrorCovariance, 0, 0, 0, 0, 0, 0,
	    _initalErrorCovariance, 0, 0, 0, 0, 0, 0, _initalErrorCovariance, 0, 0, 0, 0, 0, 0, _initalErrorCovariance;

	// Initialize Measurement Covariance matrix
	_R << _measurmantVariance, 0, 0, _measurmantVariance;

	// Initialize Process Covariance matrix
	_Q << _processVariance, 0, 0, 0, 0, 0, 0, _processVariance, 0, 0, 0, 0, 0, 0, _processVariance, 0, 0, 0, 0, 0, 0, _processVariance, 0, 0, 0,
	    0, 0, 0, _processVariance, 0, 0, 0, 0, 0, 0, _processVariance;
}

void TrackedObject::run()
{
	while (_objectState != terminated) {
		cout_mtx_.lock();
		std::cout << "//TrackedObject// Object ID:" << _id << " Main loop." << std::endl;
		cout_mtx_.unlock();

		// Run predict step of Kalman filter
		if (_objectState != init) {
			timeUpdate();
		}

		cout_mtx_.lock();
		std::cout << "//TrackedObject// Object ID:" << _id << "Receiving message..." << std::endl;
		cout_mtx_.unlock();

		auto newDetection = _detectionQueue.receive();

		if (newDetection == nullptr) {
			// If there is no new detection associated while the
			// track is still in init phase, terminate it
			if (_objectState == init) {
				_objectState = terminated;
			} else {
				_objectState = coast;
				++_coastedFrames;
			}

			cout_mtx_.lock();
			std::cout << "//TrackedObject// Track #" << _id
				  << " received Message!!!!! nullptr... "
				     "coasting. _coastedFrames: "
				  << _coastedFrames << std::endl;
			cout_mtx_.unlock();
		} else {
			// if this is the first associated detection when the
			// tract is still in the init phase, initalize the
			// velocity eimste
			if (_objectState == init) {
				float vxEstimate = newDetection->x_mid - _X(0);
				float vyEstimate = newDetection->y_mid - _X(1);

				_X(2) = vxEstimate;
				_X(3) = vyEstimate;

				// Run first time update to catch up
				timeUpdate();
				if (newDetection->x_mid < 400) {
					_firstState = DetectionState::INCOMER;

				} else {
					_firstState = DetectionState::EXITER;
				}
			}

			_objectState = active;
			newDetection->trackId = _id;
			if (newDetection->x_mid > 400 && _firstState == DetectionState::INCOMER) {
				_counter->decrement();
				_firstState = DetectionState::EXITER;
			}
			if (newDetection->x_mid < 400 && _firstState == DetectionState::EXITER) {
				_counter->increment();
				_firstState = DetectionState::INCOMER;
			}
			_coastedFrames = 0;

			cout_mtx_.lock();
			std::cout << "//TrackedObject// Track #" << _id << " received Message!!!!! det at: (" << newDetection->x_mid << ","
				  << newDetection->y_mid << ")" << std::endl;
			cout_mtx_.unlock();

			// Load new measurment into z
			_Z << newDetection->x_mid, newDetection->y_mid;

			measurementUpdate();
		}

		// Prune if track has been coasting too long
		if (_coastedFrames > _maxCoastCount) {
			_objectState = terminated;
			cout_mtx_.lock();
			std::cout << "//TrackedObject// Track #" << _id << " Terminated." << std::endl;
			cout_mtx_.unlock();
		}

		cout_mtx_.lock();
		std::cout << std::endl
			  << "//TrackedObject// "
			     "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% "
			  << std::endl;
		std::cout << "//TrackedObject// %%%%%%%%%%% TRACKID: " << _id << " ---------> END OF THREAD LOOP " << std::endl;
		std::cout << "//TrackedObject// %%%%%%%%%%% _X: " << std::endl << _X << std::endl;
		std::cout << "//TrackedObject// %%%%%%%%%%% _state: " << _objectState << std::endl;
		std::cout << "//TrackedObject// "
			     "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% "
			  << std::endl
			  << std::endl;
		cout_mtx_.unlock();
	}
}

float TrackedObject::measureDistance(std::shared_ptr<Detection> det)
{
	return std::sqrt(std::pow((det->x_mid - _X(0)), 2) + std::pow((det->y_mid - _X(1)), 2));
}
} // namespace peopleDetector