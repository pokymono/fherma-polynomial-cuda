#pragma once
#include <atomic>
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
    void (*callback)(void*)=nullptr;
    void* context=nullptr;
    int device=0;
    std::exception_ptr initialization_error;
    std::vector<uint32_t>* target=nullptr;
    size_t count=0;
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
        auto status=cudaSetDevice(device);
        if(status==cudaSuccess)status=cudaFree(nullptr);
        if(status!=cudaSuccess)initialization_error=std::make_exception_ptr(std::runtime_error(cudaGetErrorString(status)));
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
            try {if(callback)callback(context);else target->resize(count);}catch(...){error=std::current_exception();}
            pending.store(false,std::memory_order_release);
        }
    }
    void join_request() {while(pending.load(std::memory_order_acquire))pause();}
    void submit(std::vector<uint32_t>& output,size_t size) {
        if(pending.load(std::memory_order_acquire)) throw std::runtime_error("overlapping output construction");
        target=&output;count=size;error=nullptr;callback=nullptr;context=nullptr;
        pending.store(true,std::memory_order_relaxed);
        generation.fetch_add(1,std::memory_order_release);
    }
    void submit(void (*function)(void*),void* data){
        if(pending.load(std::memory_order_acquire))throw std::runtime_error("overlapping GPU dispatch");
        callback=function;context=data;error=nullptr;
        pending.store(true,std::memory_order_relaxed);
        generation.fetch_add(1,std::memory_order_release);
    }
public:
    OutputWorker() {
        cudaGetDevice(&device);
        worker=std::thread([this] {run();});
        while(!ready.load(std::memory_order_acquire))pause();
        if(initialization_error){stopping.store(true,std::memory_order_release);worker.join();std::rethrow_exception(initialization_error);}
    }
    ~OutputWorker() {
        join_request();stopping.store(true,std::memory_order_release);worker.join();
    }
    class Request {
        OutputWorker& owner;
    public:
        Request(OutputWorker& source,std::vector<uint32_t>& output,size_t count):owner(source) {owner.submit(output,count);}
        Request(OutputWorker& source,void (*function)(void*),void* data):owner(source){owner.submit(function,data);}
        ~Request() {owner.join_request();}
        Request(const Request&)=delete;
        Request& operator=(const Request&)=delete;
        void get() {owner.join_request();if(owner.error)std::rethrow_exception(owner.error);}
    };
    Request prepare(std::vector<uint32_t>& output,size_t count) {return Request(*this,output,count);}
};
