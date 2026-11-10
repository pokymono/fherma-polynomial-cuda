#pragma once
#include <algorithm>
#include <array>
#include <cstring>
#include <thread>
#include <vector>
#include <cstdint>
#include <atomic>

#if defined(__x86_64__) && defined(__GNUC__)
#include <immintrin.h>
// Streaming stores avoid reading destination cache lines only to overwrite
// them. The fence completes every store before CUDA can consume the buffer.
__attribute__((target("avx2")))
inline void copy_streaming(void* destination,const void* source,size_t bytes) {
    auto* out=static_cast<unsigned char*>(destination);
    auto* in=static_cast<const unsigned char*>(source);
    size_t prefix=(32-(reinterpret_cast<uintptr_t>(out)&31))&31;
    prefix=std::min(prefix,bytes); std::memcpy(out,in,prefix);
    out+=prefix;in+=prefix;bytes-=prefix;
    size_t i=0;
    for(;i+128<=bytes;i+=128) {
        auto a=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(in+i));
        auto b=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(in+i+32));
        auto c=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(in+i+64));
        auto d=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(in+i+96));
        _mm256_stream_si256(reinterpret_cast<__m256i*>(out+i),a);
        _mm256_stream_si256(reinterpret_cast<__m256i*>(out+i+32),b);
        _mm256_stream_si256(reinterpret_cast<__m256i*>(out+i+64),c);
        _mm256_stream_si256(reinterpret_cast<__m256i*>(out+i+96),d);
    }
    for(;i+32<=bytes;i+=32) _mm256_stream_si256(reinterpret_cast<__m256i*>(out+i),_mm256_loadu_si256(reinterpret_cast<const __m256i*>(in+i)));
    std::memcpy(out+i,in+i,bytes-i);
    _mm_sfence();
}
#endif
inline void staging_copy(void* destination,const void* source,size_t bytes,bool streaming) {
#if defined(__x86_64__) && defined(__GNUC__)
    if(streaming&&bytes>=32768&&__builtin_cpu_supports("avx2")) {
        copy_streaming(destination,source,bytes);return;
    }
#endif
    std::memcpy(destination,source,bytes);
}


// A bounded team polls only a job counter between calls. No source or
// destination pointer is read unless the current timed call publishes a job.
// Every job joins before returning, and all threads join at state destruction.
class ParallelCopy {
#ifndef HOST_COPY_LANES
#define HOST_COPY_LANES 8
#endif
#ifndef HOST_STREAMING
#define HOST_STREAMING 0
#endif
    static constexpr unsigned lanes=HOST_COPY_LANES;
    static_assert(lanes>=1&&lanes<=16);
    std::vector<std::thread> workers;
    std::atomic<bool> stopping{false};
    bool inside_call=false;
    alignas(64) std::atomic<unsigned> generation{0};
    alignas(64) std::atomic<unsigned> pending{0};
    alignas(64) std::atomic<unsigned> awake{0};
    std::array<void*,2> destinations{};
    std::array<const void*,2> sources{};
    size_t bytes=0;
    unsigned buffers=0;
    static void pause() {
#if defined(__x86_64__) && defined(__GNUC__)
        _mm_pause();
#else
        std::this_thread::yield();
#endif
    }
    void piece(unsigned lane) {
        size_t first=bytes*lane/lanes,last=bytes*(lane+1)/lanes;
        for(unsigned i=0;i<buffers;++i)
            staging_copy(static_cast<char*>(destinations[i])+first,
                         static_cast<const char*>(sources[i])+first,last-first,HOST_STREAMING);
    }
    void work(unsigned lane) {
        unsigned seen=0;
        awake.fetch_add(1,std::memory_order_release);
        while(!stopping.load(std::memory_order_acquire)) {
            unsigned current=generation.load(std::memory_order_acquire);
            if(current==seen) {pause();continue;}
            seen=current;
            piece(lane);
            pending.fetch_sub(1,std::memory_order_acq_rel);
        }
    }
    void begin() {
        if(inside_call) throw std::runtime_error("overlapping copy sessions");
        inside_call=true;
    }
    void end() {inside_call=false;}
    void stop() {
        stopping.store(true,std::memory_order_release);
        for(auto& worker:workers) worker.join();
    }

public:
    ParallelCopy() {
        try { for(unsigned lane=1;lane<lanes;++lane) workers.emplace_back([this,lane] {work(lane);}); }
        catch(...) {stop();throw;}
        while(awake.load(std::memory_order_acquire)!=lanes-1) pause();
    }
    ~ParallelCopy() {stop();}
    class Session {
        ParallelCopy& team;
    public:
        explicit Session(ParallelCopy& copy):team(copy) {team.begin();}
        ~Session() {team.end();}
        Session(const Session&)=delete;
        Session& operator=(const Session&)=delete;
    };
    void copy_pair(void* da,void* db,const void* a,const void* b,size_t size) {
        if(!inside_call) throw std::runtime_error("copy outside a timed session");
        if(size<262144) {
            staging_copy(da,a,size,HOST_STREAMING);
            if(db) staging_copy(db,b,size,HOST_STREAMING);
            return;
        }
        destinations={da,db};sources={a,b};bytes=size;buffers=db?2:1;
        pending.store(lanes-1,std::memory_order_relaxed);
        generation.fetch_add(1,std::memory_order_release);
        piece(0);
        while(pending.load(std::memory_order_acquire)) pause();
    }
};
