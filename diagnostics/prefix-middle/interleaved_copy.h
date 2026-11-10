#pragma once
#include <cstdint>
#include <utility>
#if defined(__x86_64__) && defined(__GNUC__)
#include <immintrin.h>

// A 16-coefficient block is exactly 55 cache lines. Build each output cache
// line from one or two source coefficients and stream it straight to staging.
// Masked loads never read outside the coefficients that belong to this row.
template<unsigned Packet>
__attribute__((target("avx512f")))
inline void copy_interleaved_packet55(uint32_t* destination,const uint32_t* source){
    constexpr unsigned word=Packet*16,coefficient=word/55,limb=word%55;
    const uint32_t* first=source+coefficient*8*55+limb;
    __m512i values;
    if constexpr(limb+16<=55)values=_mm512_loadu_si512(first);
    else{
        constexpr unsigned remaining=55-limb;
        constexpr __mmask16 mask=(1u<<remaining)-1;
        values=_mm512_maskz_loadu_epi32(mask,first);
        const uint32_t* next=source+(coefficient+1)*8*55;
        values=_mm512_mask_loadu_epi32(values,__mmask16(~mask),next-remaining);
    }
    _mm512_stream_si512(reinterpret_cast<__m512i*>(destination+Packet*16),values);
}
template<size_t... Packets>
__attribute__((target("avx512f")))
inline void copy_interleaved_block55(uint32_t* destination,const uint32_t* source,std::index_sequence<Packets...>){
    (copy_interleaved_packet55<Packets>(destination,source),...);
}
__attribute__((target("avx512f")))
inline void copy_interleaved_row55(uint32_t* destination,const uint32_t* source){
    for(unsigned group=0;group<4;++group)
        copy_interleaved_block55(destination+group*16*55,source+group*16*8*55,std::make_index_sequence<55>{});
}
#endif
