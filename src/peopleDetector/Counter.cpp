#include "Counter.hpp"
namespace peopleDetector
{
std::atomic<int> Counter::counter{0};

Counter::Counter(const int startValue = 0) { counter = startValue; }

void Counter::decrement() { counter = counter - 1; }

void Counter::increment() { counter = counter + 1; }

void Counter::reset() { counter = 0; }

void Counter::set(const int value) { counter = value; }

int Counter::get() const { return counter; }
} // namespace peopleDetector