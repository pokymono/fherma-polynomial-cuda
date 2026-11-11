#pragma once
#include <atomic>
#include <stdexcept>
#include <thread>
#include <vector>
#include "cpu_copy.h"
#include "cpu_affinity.h"

// One bounded job spans the entire timed multiplication. Workers own fixed,
// disjoint rows and advance without a barrier between strips.
class StreamingTeam {
    unsigned lanes;
    std::vector<std::thread> workers;
    std::atomic<bool> stopping{false};
    alignas(64) std::atomic<unsigned> generation{0};
    alignas(64) std::atomic<unsigned> pending{0};
    alignas(64) std::atomic<unsigned> awake{0};
    void (*function)(void*,unsigned,unsigned)=nullptr;
    void* context=nullptr;
    static void pause(){
#if defined(__x86_64__)
        _mm_pause();
#else
        std::this_thread::yield();
#endif
    }
    void work(unsigned lane){
        CpuPlacement::pin(CpuPlacement::get().copy(lane-1));
        unsigned seen=0;awake.fetch_add(1,std::memory_order_release);
        while(!stopping.load(std::memory_order_acquire)){
            unsigned current=generation.load(std::memory_order_acquire);
            if(current==seen){
#if defined(__x86_64__) && defined(__GNUC__)
                if(__builtin_cpu_supports("waitpkg"))wait_generation(generation,seen);
                else
#endif
                    pause();
                continue;
            }
            seen=current;function(context,lane,lanes);
            pending.fetch_sub(1,std::memory_order_acq_rel);
        }
    }
    void stop(){stopping.store(true,std::memory_order_release);for(auto& thread:workers)thread.join();}
public:
    explicit StreamingTeam(unsigned count):lanes(count){
        if(count<1||count>16)throw std::runtime_error("invalid streaming team size");
        try{for(unsigned lane=1;lane<lanes;++lane)workers.emplace_back([this,lane]{work(lane);});}
        catch(...){stop();throw;}
        while(awake.load(std::memory_order_acquire)!=lanes-1)pause();
    }
    ~StreamingTeam(){finish();stop();}
    void start(void (*callback)(void*,unsigned,unsigned),void* data){
        if(pending.load(std::memory_order_acquire))throw std::runtime_error("overlapping streaming jobs");
        function=callback;context=data;pending.store(lanes-1,std::memory_order_relaxed);
        generation.fetch_add(1,std::memory_order_release);
    }
    void finish(){while(pending.load(std::memory_order_acquire))pause();}
    unsigned size()const{return lanes;}
};
