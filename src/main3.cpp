#include "threadPool.hpp"
#include <iostream>
#include <chrono>
#include <thread>

int main() {
    // Create a ThreadPool with 4 threads
    ThreadPool pool(9);

    int N = 10;
    for (int j = 0; j < N; j++) {
        // Enqueue 10 tasks, each task sleeps for 1 second
        for (int i = 0; i < 10; ++i) {
            pool.enqueue([i]() {
                std::this_thread::sleep_for(std::chrono::seconds(1));
                std::cout << "Task " << i << " completed" << std::endl;
            });
        }
        // Wait for all tasks to complete
        pool.wait();
    }

    std::cout << "All tasks completed" << std::endl;

    // Stop the ThreadPool
    pool.stopAll();

    return 0;
}