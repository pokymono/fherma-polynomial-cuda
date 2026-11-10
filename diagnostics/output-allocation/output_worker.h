#pragma once
inline unsigned configured_output_huge=0,configured_allocator_sibling=0;
#define OUTPUT_HUGE_PAGES configured_output_huge
#define OUTPUT_ALLOCATOR_SIBLING configured_allocator_sibling
#include <atomic>
#include <chrono>
#include <cstdio>
#include <sys/mman.h>
#ifndef OUTPUT_HUGE_PAGES
#define OUTPUT_HUGE_PAGES 0
#endif
#ifndef OUTPUT_ALLOCATION_REPORT
#define OUTPUT_ALLOCATION_REPORT 0
#endif
#ifndef OUTPUT_ALLOCATOR_SIBLING
#define OUTPUT_ALLOCATOR_SIBLING 0
#endif
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
    static int allocator_cpu(){
        const auto& placement=CpuPlacement::get();
        if(!OUTPUT_ALLOCATOR_SIBLING)return placement.allocator();
        const int peer=OUTPUT_ALLOCATOR_SIBLING==1?placement.caller():placement.copy(0);
        int target_core=-1,target_package=-1;
        const auto read=[&](int cpu,int& core,int& package){
            const std::string path="/sys/devices/system/cpu/cpu"+std::to_string(cpu)+"/topology/";
            std::ifstream(path+"core_id")>>core;
            std::ifstream(path+"physical_package_id")>>package;
        };
        read(peer,target_core,target_package);
        for(int cpu:placement.siblings){
            int core=-2,package=-2;read(cpu,core,package);
            if(core==target_core&&package==target_package)return cpu;
        }
        return placement.allocator();
    }
    void run() {
        CpuPlacement::pin(allocator_cpu());
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
            try {
                if(OUTPUT_HUGE_PAGES){
                    target->reserve(count);
                    constexpr uintptr_t page=2*1024*1024;
                    const uintptr_t base=reinterpret_cast<uintptr_t>(target->data());
                    const uintptr_t first=(base+page-1)&~(page-1);
                    const uintptr_t end=(base+count*sizeof(uint32_t))&~(page-1);
                    // The advised interval is wholly inside this call's owned
                    // allocation. Reserve and zero initialization stay timed.
                    if(end>first)madvise(reinterpret_cast<void*>(first),end-first,
                                        OUTPUT_HUGE_PAGES==1?MADV_HUGEPAGE:MADV_NOHUGEPAGE);
                }
                target->resize(count);
            }catch(...){error=std::current_exception();}
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
        double elapsed_ms() {get();return owner.allocation_ms;}
    };
    Request prepare(std::vector<uint32_t>& output,size_t count) {return Request(*this,output,count);}
};
