#pragma once
inline unsigned configured_output_prefault=0;
#define OUTPUT_PREFAULT configured_output_prefault
#include <atomic>
#include <chrono>
#include <sys/mman.h>
#include <unistd.h>
#ifndef OUTPUT_PREFAULT
#define OUTPUT_PREFAULT 2
#endif
#ifndef MADV_POPULATE_WRITE
#define MADV_POPULATE_WRITE 23
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
    double allocation_ms=0,wake_ms=0,reserve_ms=0,populate_ms=0,resize_ms=0;
    std::atomic<unsigned> population_errors{0};
    std::chrono::steady_clock::time_point submitted;
    std::exception_ptr error;
    std::thread worker;
    // 0: awaiting this call's reservation; 1: helpers may populate; 2: done.
    std::atomic<unsigned> population_phase{0},population_left{0};
    uintptr_t population_begin=0;
    size_t population_pages=0;
    const size_t page_bytes=size_t(sysconf(_SC_PAGESIZE));
    const unsigned helpers=CpuPlacement::get().copy_lanes(HOST_COPY_LANES);
    void populate_slice(unsigned lane,unsigned lanes) noexcept {
        const size_t first=population_pages*lane/lanes;
        const size_t last=population_pages*(lane+1)/lanes;
        // This only faults page tables; it does not write outside vector size
        // through a C++ pointer. The complete interval is owned capacity.
        if(last>first&&madvise(reinterpret_cast<void*>(population_begin+first*page_bytes),
                              (last-first)*page_bytes,MADV_POPULATE_WRITE)!=0)population_errors.fetch_add(1,std::memory_order_relaxed);
    }
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
            reserve_ms=0;populate_ms=0;resize_ms=0;
            try {
                if(OUTPUT_PREFAULT&&count>=(1u<<21)){
                    target->reserve(count);
                    const auto reserved=std::chrono::steady_clock::now();
                    reserve_ms=std::chrono::duration<double,std::milli>(reserved-began).count();
                    const uintptr_t base=reinterpret_cast<uintptr_t>(target->data());
                    const uintptr_t end=(base+count*sizeof(uint32_t))&~(page_bytes-1);
                    population_begin=(base+page_bytes-1)&~(page_bytes-1);
                    population_pages=end>population_begin?(end-population_begin)/page_bytes:0;
                    if(OUTPUT_PREFAULT==3&&population_pages)
                        madvise(reinterpret_cast<void*>(population_begin),population_pages*page_bytes,MADV_HUGEPAGE);
                    if(OUTPUT_PREFAULT>=2){
                        population_left.store(helpers,std::memory_order_relaxed);
                        population_phase.store(1,std::memory_order_release);
                        populate_slice(helpers,helpers+1);
                        while(population_left.load(std::memory_order_acquire))pause();
                    }else populate_slice(0,1);
                    populate_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-reserved).count();
                }
                const auto resizing=std::chrono::steady_clock::now();
                target->resize(count);
                resize_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-resizing).count();
            }catch(...){error=std::current_exception();}
            population_phase.store(2,std::memory_order_release);
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
    void begin_call(){population_errors.store(0,std::memory_order_relaxed);population_phase.store(0,std::memory_order_release);}
    void help_populate(unsigned lane,unsigned lanes,const std::atomic<bool>& cancelled) noexcept {
        if(OUTPUT_PREFAULT<2)return;
        unsigned phase;
        while((phase=population_phase.load(std::memory_order_acquire))==0){
            if(cancelled.load(std::memory_order_acquire)&&!pending.load(std::memory_order_acquire))return;
            pause();
        }
        if(phase==1){
            populate_slice(lane,lanes+1);
            population_left.fetch_sub(1,std::memory_order_acq_rel);
        }
    }
    double last_reserve_ms()const{return reserve_ms;}
    double last_populate_ms()const{return populate_ms;}
    double last_resize_ms()const{return resize_ms;}
    unsigned last_population_errors()const{return population_errors.load(std::memory_order_relaxed);}
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
