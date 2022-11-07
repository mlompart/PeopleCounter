#pragma once

#include <condition_variable>
#include <deque>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

#include "Counter.hpp"
#include "Detection.hpp"
#ifdef Success
#undef Success
#endif
#include "eigen3/Eigen/Eigen"

extern std::mutex cout_mtx_;
namespace peopleDetector
{

enum class DetectionState : int { UNINITIALIZED = 0, EXITER = 1, INCOMER = 2 };

enum ObjectState { init, active, coast, terminated };

template <class T> class MessageQueue
{
      public:
	T receive();
	void send(T&& msg);

      private:
	std::mutex _mutex;
	std::condition_variable _cond;
	std::deque<T> _queue;
};

class TrackedObject
{
      public:
	const int _id; // Unique constant id for the object

	// ################### Settings ###################
	const int _maxCoastCount = 30;
	const float _initalErrorCovariance = 1.0;
	const float _processVariance = 1.0;
	const float _measurmantVariance = 1.0;
	// ################################################

	TrackedObject(std::shared_ptr<Detection>);
	TrackedObject(std::shared_ptr<Detection>, Counter* counter);

	void run();			       // Main run loop to be activated in thread started by manager
	std::vector<float> getStateEstimate(); // Getter function returns {x, y,
					       // v_x, v_y} for track.
	ObjectState getObjectState() { return _objectState; }
	float measureDistance(std::shared_ptr<Detection>);
	void sendDetection(std::shared_ptr<Detection>);
	inline void setCounter(Counter& counter) { _counter = &counter; };

      private:
	static int _idCount; // Static member increments in constructor and
			     // ensures unique _id for each object
	MessageQueue<std::shared_ptr<Detection>> _detectionQueue;
	Counter* _counter;
	DetectionState _firstState{DetectionState::UNINITIALIZED};
	static Eigen::Matrix<float, 6, 6> _A; // State transition matrix (static)
	static Eigen::Matrix<float, 2, 6> _H; // Measurement matrix (static)

	ObjectState _objectState = init;
	int _coastedFrames = 0;

	// State estimate vector
	Eigen::Matrix<float, 6, 1> _X;

	// Error covariance matrix
	Eigen::Matrix<float, 6, 6> _P;

	// Measurement covariance matrix
	Eigen::Matrix<float, 2, 2> _R;

	// Process covariance matrix
	Eigen::Matrix<float, 6, 6> _Q;

	// Measurement vector
	Eigen::Matrix<float, 2, 1> _Z;

	void timeUpdate();
	void measurementUpdate();
};
} // namespace peopleDetector