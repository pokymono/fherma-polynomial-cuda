#pragma once
#include <algorithm>
#include <array>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>
#include <cstdint>

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
    bool streaming = false;
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
            staging_copy(static_cast<char*>(destinations[i]) + first,
                         static_cast<const char*>(sources[i]) + first, last - first,streaming);
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
    void set_streaming(bool value) { streaming=value; }
    void copy_pair(void* da, void* db, const void* a, const void* b, size_t size) {
        if (size < 262144) {
            staging_copy(da,a,size,streaming);
            if (db) staging_copy(db,b,size,streaming);
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
