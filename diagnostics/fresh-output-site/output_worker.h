#pragma once
inline unsigned configured_output_site=0,configured_output_peer=0;
#define OUTPUT_CONSTRUCTION_SITE configured_output_site
#define OUTPUT_ALLOCATOR_PEER configured_output_peer
#include <atomic>
#include <chrono>
#include <sys/resource.h>
#ifndef OUTPUT_ALLOCATOR_PEER
#define OUTPUT_ALLOCATOR_PEER 2
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
    double allocation_ms=0,wake_ms=0,user_ms=0,system_ms=0;
    long minor_faults=0;
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
    static int allocator_cpu(unsigned peer_mode){
        const auto& placement=CpuPlacement::get();
        if(!peer_mode)return placement.allocator();
        const int peer=peer_mode==1?placement.caller():placement.copy(peer_mode==2?0:placement.copy_lanes(HOST_COPY_LANES)-1);
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
        const int cpus[]={allocator_cpu(0),allocator_cpu(1),allocator_cpu(2),allocator_cpu(3)};
        unsigned active_peer=0;
        CpuPlacement::pin(cpus[active_peer]);
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
            if(active_peer!=configured_output_peer){
                active_peer=configured_output_peer;
                CpuPlacement::pin(cpus[active_peer]);
            }
            const auto began=std::chrono::steady_clock::now();
            wake_ms=std::chrono::duration<double,std::milli>(began-submitted).count();
            try {construct_here(*target,count);}catch(...){error=std::current_exception();}
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
    void construct_here(std::vector<uint32_t>& output,size_t size){
        struct rusage before{},after{};
        getrusage(RUSAGE_THREAD,&before);
        const auto start=std::chrono::steady_clock::now();
        output.resize(size);
        allocation_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        getrusage(RUSAGE_THREAD,&after);
        user_ms=(after.ru_utime.tv_sec-before.ru_utime.tv_sec)*1000.0+(after.ru_utime.tv_usec-before.ru_utime.tv_usec)/1000.0;
        system_ms=(after.ru_stime.tv_sec-before.ru_stime.tv_sec)*1000.0+(after.ru_stime.tv_usec-before.ru_stime.tv_usec)/1000.0;
        minor_faults=after.ru_minflt-before.ru_minflt;
    }
    double last_user_ms()const{return user_ms;}
    double last_system_ms()const{return system_ms;}
    long last_minor_faults()const{return minor_faults;}
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
