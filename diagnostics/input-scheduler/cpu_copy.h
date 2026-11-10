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

#ifndef INPUT_COPY_KIND
#define INPUT_COPY_KIND 1
#endif
static_assert(INPUT_COPY_KIND>=0&&INPUT_COPY_KIND<=2);
#if defined(__x86_64__) && defined(__GNUC__)
__attribute__((target("avx2")))
inline void copy_256_streaming(void* destination,const void* source,size_t bytes){
    auto* out=static_cast<unsigned char*>(destination);
    auto* in=static_cast<const unsigned char*>(source);
    size_t prefix=std::min(bytes,(64-(reinterpret_cast<uintptr_t>(out)&63))&63);
    std::memcpy(out,in,prefix);out+=prefix;in+=prefix;bytes-=prefix;
    size_t i=0;
    for(;i+128<=bytes;i+=128){
        auto a=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(in+i));
        auto b=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(in+i+32));
        auto c=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(in+i+64));
        auto d=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(in+i+96));
        _mm256_stream_si256(reinterpret_cast<__m256i*>(out+i),a);
        _mm256_stream_si256(reinterpret_cast<__m256i*>(out+i+32),b);
        _mm256_stream_si256(reinterpret_cast<__m256i*>(out+i+64),c);
        _mm256_stream_si256(reinterpret_cast<__m256i*>(out+i+96),d);
    }
    for(;i+32<=bytes;i+=32){
        auto v=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(in+i));
        _mm256_stream_si256(reinterpret_cast<__m256i*>(out+i),v);
    }
    std::memcpy(out+i,in+i,bytes-i);
}
__attribute__((target("movdir64b")))
inline void copy_direct64(void* destination,const void* source,size_t bytes){
    auto* out=static_cast<unsigned char*>(destination);
    auto* in=static_cast<const unsigned char*>(source);
    size_t prefix=std::min(bytes,(64-(reinterpret_cast<uintptr_t>(out)&63))&63);
    std::memcpy(out,in,prefix);out+=prefix;in+=prefix;bytes-=prefix;
    size_t i=0;
    for(;i+256<=bytes;i+=256){
        _movdir64b(out+i,in+i);_movdir64b(out+i+64,in+i+64);
        _movdir64b(out+i+128,in+i+128);_movdir64b(out+i+192,in+i+192);
    }
    for(;i+64<=bytes;i+=64)_movdir64b(out+i,in+i);
    std::memcpy(out+i,in+i,bytes-i);
}
#endif
inline unsigned configured_input_copy=0;
inline void copy_input_bytes(void* destination,const void* source,size_t bytes){
#if defined(__x86_64__) && defined(__GNUC__)
    if(configured_input_copy==0&&__builtin_cpu_supports("avx512f")){copy_512(destination,source,bytes,true);return;}
    if(configured_input_copy==1&&__builtin_cpu_supports("avx2")){copy_256_streaming(destination,source,bytes);return;}
    if(configured_input_copy==2&&__builtin_cpu_supports("movdir64b")){copy_direct64(destination,source,bytes);return;}
#endif
    std::memcpy(destination,source,bytes);
}
// The existing per-strip SFENCE in pack()/pack_operand() orders all these
// non-temporal/direct stores before publishing the strip's ready counter.
