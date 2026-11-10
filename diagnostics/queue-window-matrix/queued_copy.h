#pragma once
#include <array>
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>
#if defined(__x86_64__)
#include <immintrin.h>
#endif

#if defined(__x86_64__) && defined(__GNUC__)
// The caller fences once after all rows in a job, before publishing completion.
__attribute__((target("avx2")))
inline void queued_stream_copy(void* destination,const void* source,size_t bytes) {
    auto* out=static_cast<unsigned char*>(destination);auto* in=static_cast<const unsigned char*>(source);
    size_t prefix=std::min(bytes,(32-(reinterpret_cast<uintptr_t>(out)&31))&size_t(31));
    std::memcpy(out,in,prefix);out+=prefix;in+=prefix;bytes-=prefix;
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
    for(;i+32<=bytes;i+=32)_mm256_stream_si256(reinterpret_cast<__m256i*>(out+i),_mm256_loadu_si256(reinterpret_cast<const __m256i*>(in+i)));
    std::memcpy(out+i,in+i,bytes-i);
}
#endif

// A bounded producer/consumer queue. Each worker owns disjoint rows of every
// copy. A release/acquire completion counter makes a packed strip visible to
// CUDA before it is transferred. All jobs are drained before the call returns.
class QueuedCopy {
    static constexpr unsigned capacity=32;
    const unsigned lanes;
    bool streaming_enabled=true;
#if defined(__x86_64__) && defined(__GNUC__)
    const bool streaming=__builtin_cpu_supports("avx2");
#else
    const bool streaming=false;
#endif
    struct alignas(64) Job {
        std::array<void*,2> destination{};
        std::array<const void*,2> source{};
        size_t rows=0,width=0,destination_stride=0,source_stride=0;
        unsigned buffers=0;
        alignas(64) std::atomic<unsigned> remaining{0};
    };
    std::array<Job,capacity> jobs{};
    std::vector<std::thread> workers;
    alignas(64) std::atomic<uint64_t> published{0};
    std::atomic<unsigned> awake{0};
    std::atomic<bool> stopping{false};
    uint64_t submitted=0;
    bool inside_call=false;
    static void pause() {
#if defined(__x86_64__)
        _mm_pause();
#else
        std::this_thread::yield();
#endif
    }
    void work(unsigned lane) {
        uint64_t consumed=0;
        awake.fetch_add(1,std::memory_order_release);
        while(!stopping.load(std::memory_order_acquire)) {
            if(consumed==published.load(std::memory_order_acquire)) {pause();continue;}
            auto& job=jobs[consumed%capacity];
            size_t first=job.rows*lane/lanes,last=job.rows*(lane+1)/lanes;
            for(unsigned buffer=0;buffer<job.buffers;++buffer)
                for(size_t row=first;row<last;++row) {
                    void* destination=static_cast<char*>(job.destination[buffer])+row*job.destination_stride;
                    const void* source=static_cast<const char*>(job.source[buffer])+row*job.source_stride;
#if defined(__x86_64__) && defined(__GNUC__)
                    if(streaming&&streaming_enabled&&job.width>=512)queued_stream_copy(destination,source,job.width);
                    else
#endif
                        std::memcpy(destination,source,job.width);
                }
#if defined(__x86_64__) && defined(__GNUC__)
            if(streaming&&streaming_enabled&&job.width>=512)_mm_sfence();
#endif
            job.remaining.fetch_sub(1,std::memory_order_acq_rel);
            ++consumed;
        }
    }
    void stop() {stopping.store(true,std::memory_order_release);for(auto& thread:workers)thread.join();}
    void finish() {for(auto& job:jobs)while(job.remaining.load(std::memory_order_acquire))pause();}
public:
    explicit QueuedCopy(unsigned count=8):lanes(count) {
        if(count<1||count>16)throw std::runtime_error("copy lanes must be in [1,16]");
        try {for(unsigned lane=0;lane<lanes;++lane)workers.emplace_back([this,lane]{work(lane);});}
        catch(...) {stop();throw;}
        while(awake.load(std::memory_order_acquire)!=lanes)pause();
    }
    ~QueuedCopy() {finish();stop();}
    void set_streaming(bool value) {if(inside_call)throw std::runtime_error("streaming change during call");streaming_enabled=value;}
    class Session {
        QueuedCopy& queue;
    public:
        explicit Session(QueuedCopy& q):queue(q) {if(q.inside_call)throw std::runtime_error("overlapping copy sessions");q.inside_call=true;}
        ~Session(){queue.finish();queue.inside_call=false;}
        Session(const Session&)=delete;
        Session& operator=(const Session&)=delete;
    };
    uint64_t enqueue(void* da,void* db,const void* a,const void* b,size_t width,size_t rows,size_t destination_stride,size_t source_stride) {
        if(!inside_call)throw std::runtime_error("copy outside a timed session");
        auto ticket=submitted++;auto& job=jobs[ticket%capacity];
        while(job.remaining.load(std::memory_order_acquire))pause();
        job.destination={da,db};job.source={a,b};job.width=width;job.rows=rows;
        job.destination_stride=destination_stride;job.source_stride=source_stride;job.buffers=db?2:1;
        job.remaining.store(lanes,std::memory_order_relaxed);
        published.store(submitted,std::memory_order_release);
        return ticket;
    }
    void wait(uint64_t ticket) const {
        // Callers wait before this ticket's ring slot is reused.
        if(ticket>=submitted||submitted-ticket>capacity)throw std::runtime_error("expired copy ticket");
        while(jobs[ticket%capacity].remaining.load(std::memory_order_acquire))pause();
    }
};
