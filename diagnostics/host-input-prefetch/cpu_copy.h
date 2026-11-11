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

#if defined(__x86_64__) && defined(__GNUC__)
template<size_t Distance,auto Hint>
__attribute__((target("avx512f")))
inline void copy_512_prefetch(void* destination,const void* source,size_t bytes){
    auto* out=static_cast<unsigned char*>(destination);
    auto* in=static_cast<const unsigned char*>(source);
    const size_t prefix=std::min(bytes,(64-(reinterpret_cast<uintptr_t>(out)&63))&63);
    std::memcpy(out,in,prefix);out+=prefix;in+=prefix;bytes-=prefix;
    size_t i=0;
    for(;i+256<=bytes;i+=256){
        // Read ahead only within the current valid source row. These accesses
        // remain inside this multiplication and do not change input values.
        if(i+Distance+256<=bytes){
            _mm_prefetch(reinterpret_cast<const char*>(in+i+Distance),Hint);
            _mm_prefetch(reinterpret_cast<const char*>(in+i+Distance+64),Hint);
            _mm_prefetch(reinterpret_cast<const char*>(in+i+Distance+128),Hint);
            _mm_prefetch(reinterpret_cast<const char*>(in+i+Distance+192),Hint);
        }
        auto a=_mm512_loadu_si512(in+i),b=_mm512_loadu_si512(in+i+64);
        auto c=_mm512_loadu_si512(in+i+128),d=_mm512_loadu_si512(in+i+192);
        _mm512_stream_si512(reinterpret_cast<__m512i*>(out+i),a);
        _mm512_stream_si512(reinterpret_cast<__m512i*>(out+i+64),b);
        _mm512_stream_si512(reinterpret_cast<__m512i*>(out+i+128),c);
        _mm512_stream_si512(reinterpret_cast<__m512i*>(out+i+192),d);
    }
    for(;i+64<=bytes;i+=64){
        auto v=_mm512_loadu_si512(in+i);
        _mm512_stream_si512(reinterpret_cast<__m512i*>(out+i),v);
    }
    std::memcpy(out+i,in+i,bytes-i);
}
inline void copy_input_prefetch(void* destination,const void* source,size_t bytes,unsigned mode){
    switch(mode){
    case 1:copy_512_prefetch<512,_MM_HINT_T0>(destination,source,bytes);break;
    case 2:copy_512_prefetch<2048,_MM_HINT_T0>(destination,source,bytes);break;
    case 3:copy_512_prefetch<2048,_MM_HINT_NTA>(destination,source,bytes);break;
    default:copy_512(destination,source,bytes,true);break;
    }
}
#endif
