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
	int getStatus() const;
	int getEntered() const;
	int getLeft() const;

      private:
	static std::atomic<int> status;
	static std::atomic<int> entered;
	static std::atomic<int> left;
};
} // namespace peopleDetector