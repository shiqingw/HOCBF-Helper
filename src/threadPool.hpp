#ifndef THREADPOOL_HPP
#define THREADPOOL_HPP
#include <iostream>
#include <chrono>
#include <thread>
#include <vector>
#include <queue>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <atomic>

class ThreadPool {
public:
    ThreadPool(int num_threads);
    ~ThreadPool();
    void enqueue(std::function<void()> task);
    void wait();
    void stopAll();

    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    std::atomic<bool> stop;
    std::atomic<int> working_threads;
};

#endif // THREADPOOL_HPP