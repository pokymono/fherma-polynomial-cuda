#pragma once
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>
#if defined(__x86_64__) && defined(__GNUC__)
#include <immintrin.h>

__attribute__((target("avx512f")))
inline void copy_512(void* destination,const void* source,size_t bytes,bool streaming) {
    auto* out=static_cast<unsigned char*>(destination);
    auto* in=static_cast<const unsigned char*>(source);
    size_t prefix=std::min(bytes,(64-(reinterpret_cast<uintptr_t>(out)&63))&63);
    std::memcpy(out,in,prefix);out+=prefix;in+=prefix;bytes-=prefix;
    size_t i=0;
    for(;i+256<=bytes;i+=256) {
        auto a=_mm512_loadu_si512(in+i),b=_mm512_loadu_si512(in+i+64);
        auto c=_mm512_loadu_si512(in+i+128),d=_mm512_loadu_si512(in+i+192);
        if(streaming) {
            _mm512_stream_si512(reinterpret_cast<__m512i*>(out+i),a);
            _mm512_stream_si512(reinterpret_cast<__m512i*>(out+i+64),b);
            _mm512_stream_si512(reinterpret_cast<__m512i*>(out+i+128),c);
            _mm512_stream_si512(reinterpret_cast<__m512i*>(out+i+192),d);
        } else {
            _mm512_store_si512(out+i,a);_mm512_store_si512(out+i+64,b);
            _mm512_store_si512(out+i+128,c);_mm512_store_si512(out+i+192,d);
        }
    }
    for(;i+64<=bytes;i+=64) {
        auto v=_mm512_loadu_si512(in+i);
        if(streaming)_mm512_stream_si512(reinterpret_cast<__m512i*>(out+i),v);
        else _mm512_store_si512(out+i,v);
    }
    std::memcpy(out+i,in+i,bytes-i);
}

__attribute__((target("waitpkg")))
inline void wait_generation(std::atomic<unsigned>& generation,unsigned seen) {
    _umonitor(static_cast<void*>(&generation));
    // Recheck after arming the monitor so a concurrent publication cannot be
    // missed. A short deadline also bounds destruction latency.
    if(generation.load(std::memory_order_acquire)==seen)
        _umwait(1,__rdtsc()+10000); // C0.1: the faster wake-up state.
}
#endif

// Diagnostic team: changing the lane count replaces the workers, so unused
// workers do not continue polling and perturb the comparison.
class ParallelCopy {
    unsigned lanes=8,copy_mode=0;
    bool monitored=false,write_combined=false,input_direction=true,inside_call=false;
    std::vector<std::thread> workers;
    std::atomic<bool> stopping{false};
    alignas(64) std::atomic<unsigned> generation{0};
    alignas(64) std::atomic<unsigned> pending{0};
    alignas(64) std::atomic<unsigned> awake{0};
    std::array<void*,2> destinations{};
    std::array<const void*,2> sources{};
    size_t rows=0,row_bytes=0,destination_stride=0,source_stride=0;
    unsigned buffers=0;
    static void pause() {
#if defined(__x86_64__) && defined(__GNUC__)
        _mm_pause();
#else
        std::this_thread::yield();
#endif
    }
    void piece(unsigned lane) {
        const size_t first=rows*lane/lanes,last=rows*(lane+1)/lanes;
        for(unsigned b=0;b<buffers;++b)for(size_t row=first;row<last;++row) {
            auto* out=static_cast<char*>(destinations[b])+row*destination_stride;
            auto* in=static_cast<const char*>(sources[b])+row*source_stride;
#if defined(__x86_64__) && defined(__GNUC__)
            if(input_direction&&copy_mode&&row_bytes>=64&&__builtin_cpu_supports("avx512f"))
                copy_512(out,in,row_bytes,copy_mode==2);
            else
#endif
                std::memcpy(out,in,row_bytes);
        }
#if defined(__x86_64__) && defined(__GNUC__)
        if(input_direction&&(write_combined||copy_mode==2))_mm_sfence();
#endif
    }
    void work(unsigned lane) {
        unsigned seen=0;awake.fetch_add(1,std::memory_order_release);
        while(!stopping.load(std::memory_order_acquire)) {
            unsigned current=generation.load(std::memory_order_acquire);
            if(current==seen) {
#if defined(__x86_64__) && defined(__GNUC__)
                if(monitored&&__builtin_cpu_supports("waitpkg"))wait_generation(generation,seen);
                else
#endif
                    pause();
                continue;
            }
            seen=current;piece(lane);pending.fetch_sub(1,std::memory_order_acq_rel);
        }
    }
    void stop() {
        stopping.store(true,std::memory_order_release);
        for(auto& worker:workers)worker.join();
        workers.clear();
    }
    void start() {
        stopping.store(false,std::memory_order_relaxed);
        generation.store(0,std::memory_order_relaxed);awake.store(0,std::memory_order_relaxed);
        try {for(unsigned lane=1;lane<lanes;++lane)workers.emplace_back([this,lane]{work(lane);});}
        catch(...){stop();throw;}
        while(awake.load(std::memory_order_acquire)!=lanes-1)pause();
    }
public:
    ParallelCopy(){start();}
    ~ParallelCopy(){stop();}
    void configure(unsigned count,unsigned mode,bool monitor,bool wc) {
        if(inside_call||count<1||count>16||mode>2)throw std::runtime_error("invalid copy configuration");
        stop();lanes=count;copy_mode=mode;monitored=monitor;write_combined=wc;start();
    }
    void set_input(bool input){input_direction=input;}
    class Session {
        ParallelCopy& team;
    public:
        explicit Session(ParallelCopy& copy):team(copy){
            if(team.inside_call)throw std::runtime_error("overlapping copy sessions");team.inside_call=true;
        }
        ~Session(){team.inside_call=false;}
        Session(const Session&)=delete;
        Session& operator=(const Session&)=delete;
    };
    void copy_strided_pair(void* da,void* db,const void* a,const void* b,size_t width,size_t count,size_t to_stride,size_t from_stride) {
        if(!inside_call)throw std::runtime_error("copy outside timed session");
        destinations={da,db};sources={a,b};buffers=db?2:1;
        rows=count;row_bytes=width;destination_stride=to_stride;source_stride=from_stride;
        pending.store(lanes-1,std::memory_order_relaxed);generation.fetch_add(1,std::memory_order_release);
        piece(0);while(pending.load(std::memory_order_acquire))pause();
    }
};
