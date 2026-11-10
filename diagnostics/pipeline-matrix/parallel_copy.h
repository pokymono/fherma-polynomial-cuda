#pragma once
#include <algorithm>
#include <array>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

// A bounded worker team for host staging only. Each copy joins before returning;
// no work involving operands is performed between calls to fherma_run.
class ParallelCopy {
#ifndef HOST_COPY_LANES
#define HOST_COPY_LANES 4
#endif
    const unsigned lanes;
    std::vector<std::thread> workers;
    std::mutex mutex;
    std::condition_variable ready, done;
    bool stopping = false;
    unsigned generation = 0, pending = 0;
    std::array<void*,2> destinations{};
    std::array<const void*,2> sources{};
    size_t bytes = 0;
    unsigned buffers = 0;

    void piece(unsigned lane) {
        size_t first = bytes * lane / lanes;
        size_t last = bytes * (lane + 1) / lanes;
        for (unsigned i = 0; i < buffers; ++i)
            std::memcpy(static_cast<char*>(destinations[i]) + first,
                        static_cast<const char*>(sources[i]) + first, last - first);
    }
    void work(unsigned lane) {
        unsigned seen = 0;
        std::unique_lock<std::mutex> lock(mutex);
        while (true) {
            ready.wait(lock, [&] { return stopping || generation != seen; });
            if (stopping) return;
            seen = generation;
            lock.unlock();
            piece(lane);
            lock.lock();
            if (--pending == 0) done.notify_one();
        }
    }
    void stop() {
        { std::lock_guard<std::mutex> lock(mutex); stopping = true; }
        ready.notify_all();
        for (auto& worker : workers) worker.join();
    }
public:
    explicit ParallelCopy(unsigned count=HOST_COPY_LANES): lanes(count) {
        if(lanes<1||lanes>16) throw std::runtime_error("invalid copy worker count");
        try {
            for (unsigned lane = 1; lane < lanes; ++lane)
                workers.emplace_back([this,lane] { work(lane); });
        } catch (...) { stop(); throw; }
    }
    ~ParallelCopy() { stop(); }
    void copy_pair(void* da, void* db, const void* a, const void* b, size_t size) {
        if (size < 262144) {
            std::memcpy(da,a,size);
            if (db) std::memcpy(db,b,size);
            return;
        }
        {
            std::lock_guard<std::mutex> lock(mutex);
            destinations = {da,db}; sources = {a,b}; bytes = size;
            buffers = db ? 2 : 1; pending = lanes - 1; ++generation;
        }
        ready.notify_all();
        piece(0);
        std::unique_lock<std::mutex> lock(mutex);
        done.wait(lock, [&] { return pending == 0; });
    }
};
