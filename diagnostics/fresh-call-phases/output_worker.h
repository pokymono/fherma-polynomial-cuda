#pragma once
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <stdexcept>
#include <thread>
#include <vector>
#include "cpu_copy.h"
#include "cpu_affinity.h"
#if defined(__x86_64__) && defined(__GNUC__)
#include <immintrin.h>
#endif

// Output construction is timed work. This worker receives only the output
// vector of the current call, and each Request joins before that vector dies.
class OutputWorker {
    std::atomic<bool> stopping{false},ready{false};
    alignas(64) std::atomic<unsigned> generation{0};
    alignas(64) std::atomic<bool> pending{false};
    std::vector<uint32_t>* target=nullptr;
    size_t count=0;
    double allocation_ms=0,wake_ms=0;
    std::chrono::steady_clock::time_point submitted;
    std::exception_ptr error;
    std::thread worker;
    static void pause() {
#if defined(__x86_64__) && defined(__GNUC__)
        _mm_pause();
#else
        std::this_thread::yield();
#endif
    }
    void run() {
        CpuPlacement::pin(CpuPlacement::get().allocator());
        unsigned seen=0;
        ready.store(true,std::memory_order_release);
        while(!stopping.load(std::memory_order_acquire)) {
            unsigned current=generation.load(std::memory_order_acquire);
            if(current==seen) {
#if defined(__x86_64__) && defined(__GNUC__)
                if(__builtin_cpu_supports("waitpkg"))wait_generation(generation,seen);else
#endif
                    pause();
                continue;
            }
            seen=current;
            const auto began=std::chrono::steady_clock::now();
            wake_ms=std::chrono::duration<double,std::milli>(began-submitted).count();
            try {target->resize(count);}catch(...){error=std::current_exception();}
            allocation_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-began).count();
            pending.store(false,std::memory_order_release);
        }
    }
    void join_request() {while(pending.load(std::memory_order_acquire))pause();}
    void submit(std::vector<uint32_t>& output,size_t size) {
        if(pending.load(std::memory_order_acquire)) throw std::runtime_error("overlapping output construction");
        target=&output;count=size;error=nullptr;
        pending.store(true,std::memory_order_relaxed);
        submitted=std::chrono::steady_clock::now();
        generation.fetch_add(1,std::memory_order_release);
    }
public:
    double last_allocation_ms()const{return allocation_ms;}
    double last_wake_ms()const{return wake_ms;}
    OutputWorker() {
        worker=std::thread([this] {run();});
        while(!ready.load(std::memory_order_acquire))pause();
    }
    ~OutputWorker() {
        join_request();stopping.store(true,std::memory_order_release);worker.join();
    }
    class Request {
        OutputWorker& owner;
    public:
        Request(OutputWorker& source,std::vector<uint32_t>& output,size_t count):owner(source) {owner.submit(output,count);}
        ~Request() {owner.join_request();}
        Request(const Request&)=delete;
        Request& operator=(const Request&)=delete;
        void get() {owner.join_request();if(owner.error)std::rethrow_exception(owner.error);}
    };
    Request prepare(std::vector<uint32_t>& output,size_t count) {return Request(*this,output,count);}
};
