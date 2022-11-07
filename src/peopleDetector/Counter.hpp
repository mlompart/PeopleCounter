#pragma once
#include <atomic>
namespace peopleDetector
{
class Counter
{
      public:
	Counter(const int startValue);
	void increment();
	void decrement();
	void set(const int value);
	void reset();
	int get() const;

      private:
	static std::atomic<int> counter;
};
} // namespace peopleDetector