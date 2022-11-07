#include "Counter.hpp"
namespace peopleDetector
{
std::atomic<int> Counter::status{0};
std::atomic<int> Counter::entered{0};
std::atomic<int> Counter::left{0};

Counter::Counter(const int startValue = 0) { status = startValue; }

void Counter::decrement()
{
	--status;
	++left;
}

void Counter::increment()
{
	++status;
	++entered;
	;
}

void Counter::reset()
{
	status = 0;
	entered = 0;
	left = 0;
}

void Counter::set(const int value) { status = value; }

int Counter::getStatus() const { return status; }
int Counter::getEntered() const { return entered; }
int Counter::getLeft() const { return left; }

} // namespace peopleDetector