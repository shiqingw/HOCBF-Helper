#include "threadPool.hpp"

ThreadPool::ThreadPool(int num_threads) : stop(false), working_threads(0) {

    for (int i = 0; i < num_threads; ++i) {
        workers.emplace_back([this] {
            while (true) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(this->queue_mutex);
                    this->condition.wait(lock, [this] { return this->stop || !this->tasks.empty(); });
                    if (this->stop && this->tasks.empty()) return;
                    task = std::move(this->tasks.front());
                    this->tasks.pop();
                    this->working_threads++;
                }
                task();
                {
                    std::unique_lock<std::mutex> lock(this->queue_mutex);
                    this->working_threads--;
                    this->condition.notify_all();
                }
            }
        });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        stop = true;
    }
    condition.notify_all();
    for (std::thread &worker : workers) worker.join();
}

void ThreadPool::enqueue(std::function<void()> task) {
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        tasks.push(std::move(task));
    }
    condition.notify_one();
}

void ThreadPool::wait() {
    std::unique_lock<std::mutex> lock(queue_mutex);
    condition.wait(lock, [this] { return this->tasks.empty() && this->working_threads == 0; });
}