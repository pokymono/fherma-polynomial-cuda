#ifndef INPUT_STREAMS
#define INPUT_STREAMS 8
#endif
// Exact negacyclic multiplication using 31-bit NTT fields and integer CRT.
// All operand-dependent work, including transfers, is inside fherma_run.
#include "fherma.h"
#include "big_unsigned.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdio>
#include <cooperative_groups.h>
#include <memory>
#include <optional>
#include <string>
#include "output_worker.h"
#include "streaming_team.h"

#ifndef OUTPUT_CHAINS
#define OUTPUT_CHAINS 2
#endif
#ifndef MERGED_INPUTS
#define MERGED_INPUTS 1
#endif
#ifndef WIDE_SHOUP
#define WIDE_SHOUP 1
#endif
#ifndef WIDE_POINTWISE
#define WIDE_POINTWISE 1
#endif
#ifndef OUTPUT_STAGE_MASK
#define OUTPUT_STAGE_MASK 0
#endif
#ifndef COMPACT_HIGH_ROOTS
#define COMPACT_HIGH_ROOTS 1
#endif
#ifndef OPERAND_WIDE_DIFF
#define OPERAND_WIDE_DIFF 1
#endif
#ifndef OPERAND_WIDE_PRODUCT
#define OPERAND_WIDE_PRODUCT 1
#endif
#ifndef TENSOR_QUOTIENT
#define TENSOR_QUOTIENT 1
#endif
#ifndef TAIL_FUSION
#define TAIL_FUSION 2
#endif
#ifndef FUSION_SKIP_PADDING
#define FUSION_SKIP_PADDING 1
#endif
#ifndef CRT_ROWS
#define CRT_ROWS 64
#endif
#ifndef CRT_WORDS
#define CRT_WORDS 16
#endif
#ifndef CRT_BYTES
#define CRT_BYTES 128
#endif
static_assert(CRT_ROWS==32 || CRT_ROWS==64 || CRT_ROWS==128);
static_assert(CRT_WORDS==8 || CRT_WORDS==16 || CRT_WORDS==32);
static_assert(CRT_BYTES==64 || CRT_BYTES==128);
#ifndef CRT_USE_TMA
#define CRT_USE_TMA 0
#endif
#ifndef CRT_ALIGNED_DOTS
#define CRT_ALIGNED_DOTS 0
#endif
#ifndef OUTPUT_ALLOCATE_AFTER
#define OUTPUT_ALLOCATE_AFTER 3
#endif
#ifndef OPERAND_BOUNDARY
#define OPERAND_BOUNDARY 32
#endif
#ifndef OPERAND_INTERIOR
#define OPERAND_INTERIOR 64
#endif
static_assert(OPERAND_BOUNDARY==32);
static_assert(OPERAND_INTERIOR>=32&&OPERAND_INTERIOR<=256&&OPERAND_INTERIOR%32==0);
constexpr unsigned OPERAND_MIDDLE=512-2*OPERAND_BOUNDARY;
constexpr unsigned OPERAND_STRIPS=2+(OPERAND_MIDDLE+OPERAND_INTERIOR-1)/OPERAND_INTERIOR;
static_assert(2*OPERAND_STRIPS<=32);
#ifndef SPLIT_INPUT_DMA
#define SPLIT_INPUT_DMA 1
#endif
namespace {
constexpr unsigned OPERAND_PARTIAL=OPERAND_MIDDLE%OPERAND_INTERIOR;
__host__ __device__ constexpr unsigned operand_base(unsigned part){
    if(part==0)return 0;
    if(part+1==OPERAND_STRIPS)return 512-OPERAND_BOUNDARY;
    if(part==1)return OPERAND_BOUNDARY;
    return OPERAND_BOUNDARY+(OPERAND_PARTIAL?OPERAND_PARTIAL+(part-2)*OPERAND_INTERIOR:(part-1)*OPERAND_INTERIOR);
}
__host__ __device__ constexpr unsigned operand_columns(unsigned part){
    if(part==0||part+1==OPERAND_STRIPS)return OPERAND_BOUNDARY;
    return part==1&&OPERAND_PARTIAL?OPERAND_PARTIAL:OPERAND_INTERIOR;
}


constexpr uint32_t NTT_TILE=512;
static_assert(MERGED_INPUTS==1);
static_assert(INPUT_STREAMS>=1 && INPUT_STREAMS<=16,"input stream count");
static_assert(OUTPUT_CHAINS>=1 && OUTPUT_CHAINS<=16,"output compute chains");

struct Prime {
    uint32_t p, inverse, radix, radix_shoup, reduce32;
    double reciprocal;
};
// Let R=2^32 and q=floor(r*R/p), with odd p and 0<r<p.
// Then 0<r*R-q*p<p<R, so floor(q*p/R)=r-1 exactly.
// Keeping q alone halves the high-stage root tables; the root is recovered
// with one high-half integer multiply. No approximate arithmetic is used.
#if COMPACT_HIGH_ROOTS
using HighRoot=uint32_t;
HighRoot pack_high_root(uint32_t root,uint32_t p){
    if(!root)throw std::runtime_error("zero high-stage root");
    return (uint64_t(root)<<32)/p;
}
__device__ __forceinline__ uint2 load_high_root(const HighRoot* roots,size_t at,uint32_t p){
    uint32_t quotient=roots[at];return make_uint2(__umulhi(quotient,p)+1,quotient);
}
#else
using HighRoot=uint2;
HighRoot pack_high_root(uint32_t root,uint32_t p){return make_uint2(root,(uint64_t(root)<<32)/p);}
__device__ __forceinline__ uint2 load_high_root(const HighRoot* roots,size_t at,uint32_t){return roots[at];}
#endif
struct State {
    CUtensorMap crt_input_map{},crt_weight_map{};
    bool crt_tma=false;
    uint32_t n=0,limbs=0,width=0,primes=0,wide=0,gap=0,padded_bytes=0;
    uint32_t *a=nullptr,*b=nullptr,*answer=nullptr,*residues=nullptr;
    uint32_t *twist=nullptr,*roots=nullptr,*inverse_roots=nullptr,*finish=nullptr;
    uint32_t *middle_roots=nullptr,*middle_inverse_roots=nullptr,*middle_quotients=nullptr,*middle_inverse_quotients=nullptr;
    uint32_t *crt=nullptr,*product=nullptr,*half_product=nullptr,*q=nullptr,*scratch=nullptr;
    HighRoot *high_roots=nullptr,*high_inverse_roots=nullptr;
    Prime* fields=nullptr;
    uint8_t *byte_inputs=nullptr,*byte_weights=nullptr,*crt_weights=nullptr;
    uint32_t* crt_inputs=nullptr;
    uint64_t* crt_dots=nullptr;
    uint32_t padded_r=0,crt_columns=0;
    bool direct_crt=false;
    uint64_t* fixed_reciprocals=nullptr;
    uint32_t* correction_mod_q=nullptr;
    bool pipelined=false,cluster_inverse=false;
    uint32_t* host_flags=nullptr;
    cudaGraphExec_t pipeline_graph{},first_operand_graph{};
    cudaGraph_t output_plans[16]{},core_plan{};
    uint32_t *startup_host=nullptr,*startup_device=nullptr;
    uint32_t *device_host_inputs=nullptr;
    uint32_t *host_inputs=nullptr,*host_output=nullptr;
    std::unique_ptr<StreamingTeam> host_team;
    mutable OutputWorker output_worker;
    cudaStream_t transfer_streams[INPUT_STREAMS]{},compute_stream{};
    cudaEvent_t inputs_ready[INPUT_STREAMS]{},outputs_ready[16]{};
    cudaGraphExec_t input_graphs[32]{},copy_graphs[32]{};
    cudaStream_t input_copy_stream{};
    cudaEvent_t copied_inputs[32]{};
    ~State() {
        if(input_copy_stream)cudaStreamSynchronize(input_copy_stream);
        if(pipelined){
            for(auto stream:transfer_streams)if(stream)cudaStreamSynchronize(stream);
            if(compute_stream)cudaStreamSynchronize(compute_stream);
        }
        for(auto graph:input_graphs)if(graph)cudaGraphExecDestroy(graph);
        for(auto graph:copy_graphs)if(graph)cudaGraphExecDestroy(graph);
        for(auto event:copied_inputs)if(event)cudaEventDestroy(event);
        if(input_copy_stream)cudaStreamDestroy(input_copy_stream);
        if(pipeline_graph)cudaGraphExecDestroy(pipeline_graph);
        if(first_operand_graph)cudaGraphExecDestroy(first_operand_graph);
        for(auto plan:output_plans)if(plan)cudaGraphDestroy(plan);
        if(core_plan)cudaGraphDestroy(core_plan);
        // Event handles remain alive until every graph referring to them dies.
        for(auto event:inputs_ready)if(event)cudaEventDestroy(event);
        for(auto event:outputs_ready)if(event)cudaEventDestroy(event);
        for(auto stream:transfer_streams)if(stream)cudaStreamDestroy(stream);
        if(compute_stream)cudaStreamDestroy(compute_stream);
        cudaFreeHost(host_flags);
        cudaFreeHost(startup_host);cudaFree(startup_device);
        cudaFreeHost(host_inputs); cudaFreeHost(host_output);
        for(auto ptr:{a,b,answer,residues,twist,roots,inverse_roots,middle_roots,middle_inverse_roots,finish,crt,product,half_product,q,scratch}) cudaFree(ptr);
        cudaFree(middle_quotients);cudaFree(middle_inverse_quotients);
        cudaFree(high_roots);cudaFree(high_inverse_roots);
        cudaFree(fields);
        cudaFree(byte_inputs); cudaFree(byte_weights); cudaFree(crt_weights);
        cudaFree(crt_inputs); cudaFree(crt_dots); cudaFree(fixed_reciprocals); cudaFree(correction_mod_q);
    }
};
void check(cudaError_t result,const char* where) {
    if(result!=cudaSuccess) throw std::runtime_error(std::string(where)+": "+cudaGetErrorString(result));
}
template<class T> void allocate(T*& p,size_t count) { check(cudaMalloc(&p,count*sizeof(T)),"allocate"); }
template<class T> void upload(T*& p,const std::vector<T>& data) {
    allocate(p,data.size()); check(cudaMemcpy(p,data.data(),data.size()*sizeof(T),cudaMemcpyHostToDevice),"upload plan");
}
// For canonical a,b and p<2^31, the wrapped alternative is larger exactly
// when correction is unnecessary. An unsigned minimum selects the residue.
__device__ __forceinline__ uint32_t minimum_u32(uint32_t a,uint32_t b){
    uint32_t value;asm("min.u32 %0,%1,%2;":"=r"(value):"r"(a),"r"(b));return value;
}
__device__ __forceinline__ uint32_t add_mod(uint32_t a,uint32_t b,uint32_t p) {
    uint32_t t=a+b; return minimum_u32(t,t-p);
}
__device__ __forceinline__ uint32_t sub_mod(uint32_t a,uint32_t b,uint32_t p) { uint32_t t=a-b;return minimum_u32(t,t+p); }
// Subtractive Montgomery reduction: m*p and a*b have identical low words.
// For a,b<p<2^31 their high-word difference lies strictly between -p and p.
__device__ __forceinline__ uint32_t mont(uint32_t a,uint32_t b,Prime f) {
    uint32_t low=a*b,high=__umulhi(a,b),m=low*f.inverse;
    uint32_t other=__umulhi(m,f.p),reduced=high-other;
    return minimum_u32(reduced,reduced+f.p);
}
// For canonical a,b and p<2^31, a+p-b is in (0,2p).
// Shoup accepts any unsigned 32-bit first operand: its quotient error
// is below one, so one correction still produces a canonical residue.
__device__ __forceinline__ uint32_t butterfly_difference(uint32_t a,uint32_t b,uint32_t p){
    return WIDE_SHOUP ? a+p-b : sub_mod(a,b,p);
}
__device__ __forceinline__ uint32_t shoup(uint32_t a,uint32_t b,uint32_t bq,uint32_t p) {
    uint32_t q=__umulhi(a,bq),r=a*b-q*p;
    return minimum_u32(r,r-p);
}
__global__ void encode(const uint32_t* a,const uint32_t* b,uint32_t* out,
                       const Prime* fields,const uint32_t* twist,uint32_t n,uint32_t limbs) {
    uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    uint32_t prime=blockIdx.y,batch=blockIdx.z;
    Prime f=fields[prime]; const uint32_t* source=batch?b:a; uint32_t value=0;
    for(uint32_t j=limbs;j-->0;) {
        value=shoup(value,f.radix,f.radix_shoup,f.p);
        uint32_t digit=source[size_t(i)*limbs+j];
        digit-=__umulhi(digit,f.reduce32)*f.p;
        if(digit>=f.p) digit-=f.p;
        value=add_mod(value,digit,f.p);
    }
    // psi^i * R^2 converts a normal value to twisted Montgomery form.
    out[(size_t(batch)*gridDim.y+prime)*n+i]=mont(value,twist[size_t(prime)*n+i],f);
}

__global__ void pad_inputs(const uint32_t* a,const uint32_t* b,uint32_t* padded,
                          uint32_t n,uint32_t limbs,uint32_t padded_words,uint32_t row_offset=0) {
    size_t at=size_t(row_offset)*padded_words+size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    size_t total=size_t(n)*padded_words;
    if(at>=total) return;
    uint32_t row=at/padded_words,col=at%padded_words;
    padded[at]=col<limbs?a[size_t(row)*limbs+col]:0;
    padded[at+total]=col<limbs?b[size_t(row)*limbs+col]:0;
}

// Native INT8 MMA processes 32 byte products per instruction. The register
// layout follows the PTX m16n8k32 fragment contract (SM80 and later).
__device__ __forceinline__ void mma_bytes(uint32_t (&d)[4],const uint32_t (&a)[4],uint32_t b0,uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+r"(d[0]),"+r"(d[1]),"+r"(d[2]),"+r"(d[3])
                 : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));
}
__device__ __forceinline__ uint32_t streamed_index(uint32_t packed,uint32_t n,uint32_t column_log) {
    if(column_log==32) return packed;
    if(n==65536&&column_log==6){
        constexpr unsigned rows=128,boundary_words=OPERAND_BOUNDARY*rows;
        if(packed<boundary_words)return (packed/OPERAND_BOUNDARY)*512+packed%OPERAND_BOUNDARY;
        if(packed>=n-boundary_words){
            const unsigned local=packed-(n-boundary_words);
            return (local/OPERAND_BOUNDARY)*512+512-OPERAND_BOUNDARY+local%OPERAND_BOUNDARY;
        }
        unsigned local=packed-boundary_words;
        if constexpr(OPERAND_PARTIAL){
            if(local<OPERAND_PARTIAL*rows)return (local/OPERAND_PARTIAL)*512+OPERAND_BOUNDARY+local%OPERAND_PARTIAL;
            local-=OPERAND_PARTIAL*rows;
        }
        const unsigned chunk=local/(OPERAND_INTERIOR*rows);local%=OPERAND_INTERIOR*rows;
        return (local/OPERAND_INTERIOR)*512+OPERAND_BOUNDARY+OPERAND_PARTIAL+chunk*OPERAND_INTERIOR+local%OPERAND_INTERIOR;
    }
    uint32_t rows=n/NTT_TILE,row_log=__ffs(rows)-1;
    uint32_t row=(packed>>column_log)&(rows-1),chunk=packed>>(column_log+row_log);
    return row*NTT_TILE+(chunk<<column_log)+(packed&((1u<<column_log)-1));
}
template<bool Encode,uint32_t KnownLimbs=0>
__global__ void tensor_product(const uint8_t* inputs,const uint8_t* weights,
                               uint32_t* out,uint64_t* dots,const Prime* fields,
                               const uint32_t* twist,uint32_t n,uint32_t count,
                               uint32_t k_bytes,uint32_t row_offset=0,uint32_t column_log=32,
                               uint32_t raw_limbs=0,const uint8_t* raw_b=nullptr,uint32_t raw_row_offset=0) {
    if constexpr(KnownLimbs) {raw_limbs=KnownLimbs;k_bytes=(KnownLimbs*4+31)/32*32;}
    constexpr uint32_t tile_bytes=64, tile_words=tile_bytes/4;
    constexpr uint32_t swizzle_shift=1, swizzle_mask=3;
    union Shared {
        struct Matrices { uint32_t a[64][tile_words],b[64][tile_words]; } matrices;
        uint32_t encoded[16][65];
        uint64_t reconstructed[Encode?1:64][17];
    };
    __shared__ __align__(32) Shared shared;
    auto& shared_a=shared.matrices.a;
    auto& shared_b=shared.matrices.b;
    uint32_t warp=threadIdx.x/32,lane=threadIdx.x%32,group=lane/4,member=lane%4;
    uint32_t row_base=row_offset+blockIdx.x*64,word_base=blockIdx.y*16+warp*4;
    uint32_t source_stride=Encode&&raw_limbs?raw_limbs*4:k_bytes;
    const uint8_t* source=Encode&&raw_limbs?(blockIdx.z?raw_b:inputs):inputs+size_t(blockIdx.z)*n*k_bytes;
    uint32_t acc[4][2][4]={};
    #pragma unroll
    for(uint32_t k=0;k<k_bytes;k+=tile_bytes) {
        // XOR the shared-memory column by the row group. A warp's fragment
        // accesses then occupy distinct banks, including at a 128-byte pitch.
        for(uint32_t part=threadIdx.x;part<64*(tile_bytes/16);part+=blockDim.x) {
            uint32_t row=part/(tile_bytes/16),col=(part%(tile_bytes/16))*4;
            uint4 a=make_uint4(0,0,0,0),b=make_uint4(0,0,0,0);
            if(k+col*4<k_bytes) {
                if(row_base+row<n) {
                    if(Encode&&raw_limbs) {
                        uint32_t first=k/4+col;
                        const uint32_t* values=reinterpret_cast<const uint32_t*>(source)+size_t(row_base+row-raw_row_offset)*raw_limbs;
                        a.x=first<raw_limbs?values[first]:0;
                        a.y=first+1<raw_limbs?values[first+1]:0;
                        a.z=first+2<raw_limbs?values[first+2]:0;
                        a.w=first+3<raw_limbs?values[first+3]:0;
                    } else a=*reinterpret_cast<const uint4*>(source+size_t(row_base+row)*source_stride+k+col*4);
                }
                b=*reinterpret_cast<const uint4*>(weights+size_t(blockIdx.y*64+row)*k_bytes+k+col*4);
            }
            uint32_t at=col^(((row>>swizzle_shift)&swizzle_mask)*4);
            *reinterpret_cast<uint4*>(&shared_a[row][at])=a;
            *reinterpret_cast<uint4*>(&shared_b[row][at])=b;
        }
        __syncthreads();
        #pragma unroll
        for(uint32_t inner=0;inner<tile_words;inner+=8) {
            if(k+inner*4>=k_bytes) continue;
            uint32_t b[2][2];
            #pragma unroll
            for(int col=0;col<2;++col) {
                uint32_t row=warp*16+col*8+(lane&7);
                uint32_t word=inner+((lane>>3)&1)*4;
                uint32_t address=static_cast<uint32_t>(__cvta_generic_to_shared(&shared_b[row][word^(((row>>swizzle_shift)&swizzle_mask)*4)]));
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                             : "=r"(b[col][0]),"=r"(b[col][1]) : "r"(address) : "memory");
            }
            #pragma unroll
            for(int m=0;m<4;++m) {
                uint32_t row=m*16+(lane&15),word=inner+(lane>>4)*4;
                uint32_t address=static_cast<uint32_t>(__cvta_generic_to_shared(&shared_a[row][word^(((row>>swizzle_shift)&swizzle_mask)*4)]));
                uint32_t a[4];
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                             : "=r"(a[0]),"=r"(a[1]),"=r"(a[2]),"=r"(a[3]) : "r"(address) : "memory");
                mma_bytes(acc[m][0],a,b[0][0],b[0][1]);
                mma_bytes(acc[m][1],a,b[1][0],b[1][1]);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int m=0;m<4;++m) {
        #pragma unroll
        for(int col=0;col<2;++col) {
            #pragma unroll
            for(int half=0;half<2;++half) {
                uint32_t first=acc[m][col][half*2],second=acc[m][col][half*2+1];
                uint32_t third=__shfl_xor_sync(0xffffffffu,first,1),fourth=__shfl_xor_sync(0xffffffffu,second,1);
                uint32_t i=row_base+m*16+group+half*8,word=word_base+col*2+member/2;
                if((member&1)||i>=n||word>=count) continue;
                uint64_t sum=uint64_t(first)+(uint64_t(second)<<8)+(uint64_t(third)<<16)+(uint64_t(fourth)<<24);
                if(Encode) {
                    Prime f=fields[word];
                    // Each byte dot has at most 32768 terms, so sum<2^55
                    // and high<2^23<p. Reduce sum directly into R^-1 form.
                    // Tensor twists carry R^3, yielding the required psi^i*R.
                    uint32_t high=sum>>32,m=uint32_t(sum)*f.inverse;
                    uint32_t other=__umulhi(m,f.p),value=high-other;
                    if(high<other)value+=f.p;
                    shared.encoded[word-blockIdx.y*16][i-row_base]=mont(value,twist[size_t(word)*n+streamed_index(i,n,column_log)],f);
                } else shared.reconstructed[i-row_base][word-blockIdx.y*16]=sum;
            }
        }
    }
    __syncthreads();
    for(uint32_t at=threadIdx.x;at<64*16;at+=blockDim.x) {
        if(Encode) {
            uint32_t col=at/64,row=at%64,i=row_base+row,word=blockIdx.y*16+col;
            if(i<n&&word<count) out[(size_t(blockIdx.z)*count+word)*n+streamed_index(i,n,column_log)]=shared.encoded[col][row];
        } else {
            uint32_t row=at/16,col=at%16,i=row_base+row,word=blockIdx.y*16+col;
            if(i<n&&word<count) dots[size_t(i)*count+word]=shared.reconstructed[row][col];
        }
    }

}
#ifndef RESIDENT_DEVICE_INPUT
#define RESIDENT_DEVICE_INPUT 0
#endif
#ifndef RESIDENT_GROUPS
#define RESIDENT_GROUPS 8
#endif
static_assert(RESIDENT_GROUPS==1||RESIDENT_GROUPS==2||RESIDENT_GROUPS==4||RESIDENT_GROUPS==8);
#ifndef RESIDENT_ROWS
#define RESIDENT_ROWS 16
#endif
static_assert(RESIDENT_ROWS==16||RESIDENT_ROWS==32||RESIDENT_ROWS==64);
// Load each coefficient from pinned host memory once, then reuse its bytes for
// every RNS field. This avoids the repeated PCIe reads of a field-tiled encoder.
template<unsigned KnownLimbs>
__global__ void resident_input_encode(const uint32_t* input_a,const uint32_t* input_b,
        const uint8_t* weights,uint32_t* output,const Prime* fields,const uint32_t* twist,
        uint32_t n,uint32_t primes,uint32_t row_offset){
    constexpr unsigned Groups=8;
    constexpr unsigned Rows=RESIDENT_ROWS,Words=64,Bytes=(KnownLimbs*4+31)/32*32;
    __shared__ __align__(32) uint32_t raw[Rows][Words];
    union Work {uint32_t weights[64][16];uint32_t encoded[16][Rows+1];};
    __shared__ __align__(32) Work workspaces[Groups];
    auto& work=workspaces[threadIdx.x/128];
    const unsigned tid=threadIdx.x,local=tid%128,warp=local/32,lane=tid%32,group=lane/4,member=lane%4;
    const unsigned first=blockIdx.x*Rows;
    const uint32_t* source=(blockIdx.z?input_b:input_a)+size_t(first)*KnownLimbs;
    for(unsigned at=tid;at<Rows*Words;at+=128*Groups){
        unsigned row=at/Words,word=at%Words;
        if(word>=KnownLimbs)raw[row][word^((row&7)*4)]=0;
    }
    constexpr unsigned Packets=Rows*KnownLimbs/4,PerThread=(Packets+128*Groups-1)/(128*Groups);
    uint4 packets[PerThread];
    // Independent, aligned 16-byte loads expose memory-level parallelism.
    #pragma unroll
    for(unsigned k=0;k<PerThread;++k){
        unsigned packet=tid+k*128*Groups;
        if(packet<Packets)packets[k]=reinterpret_cast<const uint4*>(source)[packet];
    }
    #pragma unroll
    for(unsigned k=0;k<PerThread;++k){
        unsigned packet=tid+k*128*Groups;
        if(packet<Packets){
            const uint32_t words[4]={packets[k].x,packets[k].y,packets[k].z,packets[k].w};
            #pragma unroll
            for(unsigned j=0;j<4;++j){
                unsigned linear=packet*4+j,row=linear/KnownLimbs,word=linear%KnownLimbs;
                raw[row][word^((row&7)*4)]=words[j];
            }
        }
    }
    __syncthreads();
    for(unsigned prime_block=0;prime_block<primes;prime_block+=16*Groups){
        unsigned prime_base=prime_block+(tid/128)*16;
        uint32_t acc[Rows/16][2][4]={};
        #pragma unroll
        for(unsigned k=0;k<Bytes;k+=64){
            for(unsigned part=local;part<256;part+=128){
                unsigned row=part/4,col=(part%4)*4;
                uint4 value=make_uint4(0,0,0,0);
                if(prime_base<(primes+15)/16*16&&k+col*4<Bytes)value=*reinterpret_cast<const uint4*>(weights+size_t(prime_base*4+row)*Bytes+k+col*4);
                *reinterpret_cast<uint4*>(&work.weights[row][col^(((row>>1)&3)*4)])=value;
            }
            __syncthreads();
            #pragma unroll
            for(unsigned inner=0;inner<16;inner+=8){
                if(k+inner*4>=Bytes)continue;
                uint32_t b[2][2];
                #pragma unroll
                for(unsigned col=0;col<2;++col){
                    unsigned row=warp*16+col*8+(lane&7),word=inner+((lane>>3)&1)*4;
                    uint32_t address=static_cast<uint32_t>(__cvta_generic_to_shared(&work.weights[row][word^(((row>>1)&3)*4)]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];":"=r"(b[col][0]),"=r"(b[col][1]):"r"(address):"memory");
                }
                #pragma unroll
                for(unsigned m=0;m<Rows/16;++m){
                    unsigned row=m*16+(lane&15),word=k/4+inner+(lane>>4)*4;
                    uint32_t address=static_cast<uint32_t>(__cvta_generic_to_shared(&raw[row][word^((row&7)*4)]));
                    uint32_t a[4];
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];":"=r"(a[0]),"=r"(a[1]),"=r"(a[2]),"=r"(a[3]):"r"(address):"memory");
                    mma_bytes(acc[m][0],a,b[0][0],b[0][1]);mma_bytes(acc[m][1],a,b[1][0],b[1][1]);
                }
            }
            __syncthreads();
        }
        #pragma unroll
        for(unsigned m=0;m<Rows/16;++m){
            #pragma unroll
            for(unsigned col=0;col<2;++col){
                #pragma unroll
                for(unsigned half=0;half<2;++half){
                    uint32_t a=acc[m][col][half*2],b=acc[m][col][half*2+1];
                    uint32_t c=__shfl_xor_sync(0xffffffffu,a,1),d=__shfl_xor_sync(0xffffffffu,b,1);
                    unsigned row=m*16+group+half*8,prime=prime_base+warp*4+col*2+member/2;
                    if((member&1)||prime>=primes)continue;
                    uint64_t sum=uint64_t(a)+(uint64_t(b)<<8)+(uint64_t(c)<<16)+(uint64_t(d)<<24);
                    Prime f=fields[prime];uint32_t high=sum>>32,other=__umulhi(uint32_t(sum)*f.inverse,f.p),value=high-other;
                    if(high<other)value+=f.p;
                    unsigned coefficient=streamed_index(row_offset+first+row,n,5);
                    work.encoded[prime-prime_base][row]=mont(value,twist[size_t(prime)*n+coefficient],f);
                }
            }
        }
        __syncthreads();
        for(unsigned at=local;at<16*Rows;at+=128){
            unsigned prime=prime_base+at/Rows,row=at%Rows;
            if(prime<primes)output[(size_t(blockIdx.z)*primes+prime)*n+streamed_index(row_offset+first+row,n,5)]=work.encoded[at/Rows][row];
        }
        __syncthreads();
    }
}

#ifndef CRT_ASYNC_TILE
#define CRT_ASYNC_TILE 128
#endif
template<int Tile=CRT_ASYNC_TILE>
__global__ void tensor_crt_async(const uint8_t* inputs,const uint8_t* weights,
                               uint32_t* out,uint64_t* dots,const Prime* fields,
                               const uint32_t* twist,uint32_t n,uint32_t count,
                               uint32_t k_bytes,uint32_t row_offset=0,uint32_t column_log=32,
                               uint32_t raw_limbs=0,const uint8_t* raw_b=nullptr,uint32_t raw_row_offset=0) {
    constexpr bool Encode=false;
    constexpr uint32_t tile_bytes=Tile, tile_words=tile_bytes/4;
    constexpr uint32_t swizzle_shift=1, swizzle_mask=tile_words/4-1;
    union Shared {
        struct Matrices { uint32_t a[2][64][tile_words],b[2][64][tile_words]; } matrices;
        uint32_t encoded[16][65];
        uint64_t reconstructed[Encode?1:64][17];
    };
    __shared__ __align__(32) Shared shared;
    uint32_t warp=threadIdx.x/32,lane=threadIdx.x%32,group=lane/4,member=lane%4;
    uint32_t row_base=row_offset+blockIdx.x*64,word_base=blockIdx.y*16+warp*4;
    uint32_t source_stride=Encode&&raw_limbs?raw_limbs*4:k_bytes;
    const uint8_t* source=Encode&&raw_limbs?(blockIdx.z?raw_b:inputs):inputs+size_t(blockIdx.z)*n*k_bytes;
    uint32_t acc[4][2][4]={};
    auto prefetch=[&](uint32_t k,uint32_t buffer){
        for(uint32_t part=threadIdx.x;part<64*(Tile/16);part+=blockDim.x){
            uint32_t row=part/(Tile/16),col=(part%(Tile/16))*4;
            uint32_t at=col^(((row>>swizzle_shift)&swizzle_mask)*4);
            auto a=static_cast<uint32_t>(__cvta_generic_to_shared(&shared.matrices.a[buffer][row][at]));
            auto b=static_cast<uint32_t>(__cvta_generic_to_shared(&shared.matrices.b[buffer][row][at]));
            const uint8_t* ap=inputs+size_t(row_base+row)*k_bytes+k+col*4;
            const uint8_t* bp=weights+size_t(blockIdx.y*64+row)*k_bytes+k+col*4;
            // Streamed rows and padded field counts make every packet valid.
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"::"r"(a),"l"(ap):"memory");
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"::"r"(b),"l"(bp):"memory");
        }
        asm volatile("cp.async.commit_group;":::"memory");
    };
    prefetch(0,0);
    for(uint32_t k=0;k<k_bytes;k+=tile_bytes){
        unsigned buffer=(k/tile_bytes)&1;
        asm volatile("cp.async.wait_group 0;":::"memory");
        __syncthreads();
        auto& shared_a=shared.matrices.a[buffer];
        auto& shared_b=shared.matrices.b[buffer];
        if(k+tile_bytes<k_bytes)prefetch(k+tile_bytes,buffer^1);
        #pragma unroll
        for(uint32_t inner=0;inner<tile_words;inner+=8) {
            if(k+inner*4>=k_bytes) continue;
            uint32_t b[2][2];
            #pragma unroll
            for(int col=0;col<2;++col) {
                uint32_t row=warp*16+col*8+(lane&7);
                uint32_t word=inner+((lane>>3)&1)*4;
                uint32_t address=static_cast<uint32_t>(__cvta_generic_to_shared(&shared_b[row][word^(((row>>swizzle_shift)&swizzle_mask)*4)]));
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                             : "=r"(b[col][0]),"=r"(b[col][1]) : "r"(address) : "memory");
            }
            #pragma unroll
            for(int m=0;m<4;++m) {
                uint32_t row=m*16+(lane&15),word=inner+(lane>>4)*4;
                uint32_t address=static_cast<uint32_t>(__cvta_generic_to_shared(&shared_a[row][word^(((row>>swizzle_shift)&swizzle_mask)*4)]));
                uint32_t a[4];
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                             : "=r"(a[0]),"=r"(a[1]),"=r"(a[2]),"=r"(a[3]) : "r"(address) : "memory");
                mma_bytes(acc[m][0],a,b[0][0],b[0][1]);
                mma_bytes(acc[m][1],a,b[1][0],b[1][1]);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int m=0;m<4;++m) {
        #pragma unroll
        for(int col=0;col<2;++col) {
            #pragma unroll
            for(int half=0;half<2;++half) {
                uint32_t first=acc[m][col][half*2],second=acc[m][col][half*2+1];
                uint32_t third=__shfl_xor_sync(0xffffffffu,first,1),fourth=__shfl_xor_sync(0xffffffffu,second,1);
                uint32_t i=row_base+m*16+group+half*8,word=word_base+col*2+member/2;
                if((member&1)||i>=n||word>=count) continue;
                uint64_t sum=uint64_t(first)+(uint64_t(second)<<8)+(uint64_t(third)<<16)+(uint64_t(fourth)<<24);
                if(Encode) {
                    Prime f=fields[word];
                    // Each byte dot has at most 32768 terms, so sum<2^55
                    // and high<2^23<p. Reduce sum directly into R^-1 form.
                    // Tensor twists carry R^3, yielding the required psi^i*R.
                    uint32_t high=sum>>32,m=uint32_t(sum)*f.inverse;
                    uint32_t other=__umulhi(m,f.p),value=high-other;
                    if(high<other)value+=f.p;
                    shared.encoded[word-blockIdx.y*16][i-row_base]=mont(value,twist[size_t(word)*n+streamed_index(i,n,column_log)],f);
                } else shared.reconstructed[i-row_base][word-blockIdx.y*16]=sum;
            }
        }
    }
    __syncthreads();
    for(uint32_t at=threadIdx.x;at<64*16;at+=blockDim.x) {
        if(Encode) {
            uint32_t col=at/64,row=at%64,i=row_base+row,word=blockIdx.y*16+col;
            if(i<n&&word<count) out[(size_t(blockIdx.z)*count+word)*n+streamed_index(i,n,column_log)]=shared.encoded[col][row];
        } else {
            uint32_t row=at/16,col=at%16,i=row_base+row,word=blockIdx.y*16+col;
            if(i<n&&word<count) dots[size_t(i)*count+word]=shared.reconstructed[row][col];
        }
    }

}
template<unsigned Rows,unsigned Words,unsigned Tile>
__global__ void tensor_crt_geometry(const uint8_t* inputs,const uint8_t* weights,
                               uint32_t* out,uint64_t* dots,const Prime* fields,
                               const uint32_t* twist,uint32_t n,uint32_t count,
                               uint32_t k_bytes,uint32_t row_offset=0,uint32_t column_log=32,
                               uint32_t raw_limbs=0,const uint8_t* raw_b=nullptr,uint32_t raw_row_offset=0) {
    constexpr bool Encode=false;
    constexpr uint32_t tile_bytes=Tile, tile_words=tile_bytes/4;
    constexpr uint32_t swizzle_shift=1, swizzle_mask=tile_words/4-1;
    union Shared {
        struct Matrices { uint32_t a[2][Rows][tile_words],b[2][Words*4][tile_words]; } matrices;
        uint32_t encoded[16][65];
        uint64_t reconstructed[Rows][Words+1];
    };
    __shared__ __align__(32) Shared shared;
    uint32_t warp=threadIdx.x/32,lane=threadIdx.x%32,group=lane/4,member=lane%4;
    uint32_t row_base=row_offset+blockIdx.x*Rows,word_base=blockIdx.y*Words+warp*4;
    uint32_t source_stride=Encode&&raw_limbs?raw_limbs*4:k_bytes;
    const uint8_t* source=Encode&&raw_limbs?(blockIdx.z?raw_b:inputs):inputs+size_t(blockIdx.z)*n*k_bytes;
    uint32_t acc[Rows/16][2][4]={};
    auto prefetch=[&](uint32_t k,uint32_t buffer){
        for(uint32_t part=threadIdx.x;part<(Rows>Words*4?Rows:Words*4)*(Tile/16);part+=blockDim.x){
            uint32_t row=part/(Tile/16),col=(part%(Tile/16))*4;
            uint32_t at=col^(((row>>swizzle_shift)&swizzle_mask)*4);
            if(row<Rows){
                auto address=static_cast<uint32_t>(__cvta_generic_to_shared(&shared.matrices.a[buffer][row][at]));
                const uint8_t* from=inputs+size_t(row_base+row)*k_bytes+k+col*4;
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"::"r"(address),"l"(from):"memory");
            }
            if(row<Words*4){
                auto address=static_cast<uint32_t>(__cvta_generic_to_shared(&shared.matrices.b[buffer][row][at]));
                const uint8_t* from=weights+size_t(blockIdx.y*Words*4+row)*k_bytes+k+col*4;
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"::"r"(address),"l"(from):"memory");
            }
        }
        asm volatile("cp.async.commit_group;":::"memory");
    };
    prefetch(0,0);
    for(uint32_t k=0;k<k_bytes;k+=tile_bytes){
        unsigned buffer=(k/tile_bytes)&1;
        asm volatile("cp.async.wait_group 0;":::"memory");
        __syncthreads();
        auto& shared_a=shared.matrices.a[buffer];
        auto& shared_b=shared.matrices.b[buffer];
        if(k+tile_bytes<k_bytes)prefetch(k+tile_bytes,buffer^1);
        #pragma unroll
        for(uint32_t inner=0;inner<tile_words;inner+=8) {
            if(k+inner*4>=k_bytes) continue;
            uint32_t b[2][2];
            #pragma unroll
            for(int col=0;col<2;++col) {
                uint32_t row=warp*16+col*8+(lane&7);
                uint32_t word=inner+((lane>>3)&1)*4;
                uint32_t address=static_cast<uint32_t>(__cvta_generic_to_shared(&shared_b[row][word^(((row>>swizzle_shift)&swizzle_mask)*4)]));
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                             : "=r"(b[col][0]),"=r"(b[col][1]) : "r"(address) : "memory");
            }
            #pragma unroll
            for(int m=0;m<Rows/16;++m) {
                uint32_t row=m*16+(lane&15),word=inner+(lane>>4)*4;
                uint32_t address=static_cast<uint32_t>(__cvta_generic_to_shared(&shared_a[row][word^(((row>>swizzle_shift)&swizzle_mask)*4)]));
                uint32_t a[4];
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                             : "=r"(a[0]),"=r"(a[1]),"=r"(a[2]),"=r"(a[3]) : "r"(address) : "memory");
                mma_bytes(acc[m][0],a,b[0][0],b[0][1]);
                mma_bytes(acc[m][1],a,b[1][0],b[1][1]);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int m=0;m<Rows/16;++m) {
        #pragma unroll
        for(int col=0;col<2;++col) {
            #pragma unroll
            for(int half=0;half<2;++half) {
                uint32_t first=acc[m][col][half*2],second=acc[m][col][half*2+1];
                uint32_t third=__shfl_xor_sync(0xffffffffu,first,1),fourth=__shfl_xor_sync(0xffffffffu,second,1);
                uint32_t i=row_base+m*16+group+half*8,word=word_base+col*2+member/2;
                if((member&1)||i>=n||word>=count) continue;
                uint64_t sum=uint64_t(first)+(uint64_t(second)<<8)+(uint64_t(third)<<16)+(uint64_t(fourth)<<24);
                if(Encode) {
                    Prime f=fields[word];
                    // Each byte dot has at most 32768 terms, so sum<2^55
                    // and high<2^23<p. Reduce sum directly into R^-1 form.
                    // Tensor twists carry R^3, yielding the required psi^i*R.
                    uint32_t high=sum>>32,m=uint32_t(sum)*f.inverse;
                    uint32_t other=__umulhi(m,f.p),value=high-other;
                    if(high<other)value+=f.p;
                    shared.encoded[word-blockIdx.y*Words][i-row_base]=mont(value,twist[size_t(word)*n+streamed_index(i,n,column_log)],f);
                } else shared.reconstructed[i-row_base][word-blockIdx.y*Words]=sum;
            }
        }
    }
    __syncthreads();
    for(uint32_t at=threadIdx.x;at<Rows*Words;at+=blockDim.x) {
        if(Encode) {
            uint32_t col=at/64,row=at%64,i=row_base+row,word=blockIdx.y*Words+col;
            if(i<n&&word<count) out[(size_t(blockIdx.z)*count+word)*n+streamed_index(i,n,column_log)]=shared.encoded[col][row];
        } else {
            uint32_t row=at/Words,col=at%Words,i=row_base+row,word=blockIdx.y*Words+col;
            if(i<n&&word<count) dots[size_t(i)*count+word]=shared.reconstructed[row][col];
        }
    }

}
template<unsigned Rows,unsigned Words,unsigned Tile>
__global__ void tensor_crt_tma(const __grid_constant__ CUtensorMap input_map,
                               const __grid_constant__ CUtensorMap weight_map,
                               const uint8_t* inputs,const uint8_t* weights,
                               uint32_t* out,uint64_t* dots,const Prime* fields,
                               const uint32_t* twist,uint32_t n,uint32_t count,
                               uint32_t k_bytes,uint32_t row_offset=0,uint32_t column_log=32,
                               uint32_t raw_limbs=0,const uint8_t* raw_b=nullptr,uint32_t raw_row_offset=0) {
#if __CUDA_ARCH__ >= 900
    constexpr bool Encode=false;
    constexpr uint32_t tile_bytes=Tile, tile_words=tile_bytes/4;
    constexpr uint32_t swizzle_shift=(Tile==128?0:1), swizzle_mask=tile_words/4-1;
    union Shared {
        struct Matrices { uint32_t a[2][Rows][tile_words],b[2][Words*4][tile_words]; } matrices;
        uint32_t encoded[16][65];
        uint64_t reconstructed[Rows][Words+1];
    };
    __shared__ __align__(1024) Shared shared;
    __shared__ uint64_t barriers[2];
    if(threadIdx.x==0){
        for(unsigned j=0;j<2;++j){
            const auto address=static_cast<uint32_t>(__cvta_generic_to_shared(barriers+j));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"::"r"(address):"memory");
        }
        asm volatile("fence.proxy.async.shared::cta;":::"memory");
    }
    __syncthreads();
    uint32_t warp=threadIdx.x/32,lane=threadIdx.x%32,group=lane/4,member=lane%4;
    uint32_t row_base=row_offset+blockIdx.x*Rows,word_base=blockIdx.y*Words+warp*4;
    uint32_t source_stride=Encode&&raw_limbs?raw_limbs*4:k_bytes;
    const uint8_t* source=Encode&&raw_limbs?(blockIdx.z?raw_b:inputs):inputs+size_t(blockIdx.z)*n*k_bytes;
    uint32_t acc[Rows/16][2][4]={};
    auto prefetch=[&](uint32_t offset,uint32_t buffer){
        if(threadIdx.x==0){
            const auto bar=static_cast<uint32_t>(__cvta_generic_to_shared(barriers+buffer));
            const auto a=static_cast<uint32_t>(__cvta_generic_to_shared(shared.matrices.a[buffer]));
            const auto b=static_cast<uint32_t>(__cvta_generic_to_shared(shared.matrices.b[buffer]));
            constexpr unsigned bytes=(Rows+Words*4)*Tile;
            asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"::"r"(bar),"r"(bytes):"memory");
            asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
                ::"r"(a),"l"(&input_map),"r"(offset),"r"(row_base),"r"(bar):"memory");
            const unsigned first_word=blockIdx.y*Words*4;
            asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
                ::"r"(b),"l"(&weight_map),"r"(offset),"r"(first_word),"r"(bar):"memory");
        }
    };
    prefetch(0,0);
    for(uint32_t k=0;k<k_bytes;k+=tile_bytes){
        unsigned buffer=(k/tile_bytes)&1;
        const auto bar=static_cast<uint32_t>(__cvta_generic_to_shared(barriers+buffer));
        const unsigned phase=(k/tile_bytes/2)&1;
        unsigned ready;
        do {
            asm volatile("{ .reg .pred done; mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 done, [%1], %2; selp.b32 %0, 1, 0, done; }"
                :"=r"(ready):"r"(bar),"r"(phase):"memory");
        }while(!ready);
        __syncthreads();
        auto& shared_a=shared.matrices.a[buffer];
        auto& shared_b=shared.matrices.b[buffer];
        if(k+tile_bytes<k_bytes)prefetch(k+tile_bytes,buffer^1);
        #pragma unroll
        for(uint32_t inner=0;inner<tile_words;inner+=8) {
            if(k+inner*4>=k_bytes) continue;
            uint32_t b[2][2];
            #pragma unroll
            for(int col=0;col<2;++col) {
                uint32_t row=warp*16+col*8+(lane&7);
                uint32_t word=inner+((lane>>3)&1)*4;
                uint32_t address=static_cast<uint32_t>(__cvta_generic_to_shared(&shared_b[row][word^(((row>>swizzle_shift)&swizzle_mask)*4)]));
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                             : "=r"(b[col][0]),"=r"(b[col][1]) : "r"(address) : "memory");
            }
            #pragma unroll
            for(int m=0;m<Rows/16;++m) {
                uint32_t row=m*16+(lane&15),word=inner+(lane>>4)*4;
                uint32_t address=static_cast<uint32_t>(__cvta_generic_to_shared(&shared_a[row][word^(((row>>swizzle_shift)&swizzle_mask)*4)]));
                uint32_t a[4];
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                             : "=r"(a[0]),"=r"(a[1]),"=r"(a[2]),"=r"(a[3]) : "r"(address) : "memory");
                mma_bytes(acc[m][0],a,b[0][0],b[0][1]);
                mma_bytes(acc[m][1],a,b[1][0],b[1][1]);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int m=0;m<Rows/16;++m) {
        #pragma unroll
        for(int col=0;col<2;++col) {
            #pragma unroll
            for(int half=0;half<2;++half) {
                uint32_t first=acc[m][col][half*2],second=acc[m][col][half*2+1];
                uint32_t third=__shfl_xor_sync(0xffffffffu,first,1),fourth=__shfl_xor_sync(0xffffffffu,second,1);
                uint32_t i=row_base+m*16+group+half*8,word=word_base+col*2+member/2;
                if((member&1)||i>=n||word>=count) continue;
                uint64_t sum=uint64_t(first)+(uint64_t(second)<<8)+(uint64_t(third)<<16)+(uint64_t(fourth)<<24);
                if(Encode) {
                    Prime f=fields[word];
                    // Each byte dot has at most 32768 terms, so sum<2^55
                    // and high<2^23<p. Reduce sum directly into R^-1 form.
                    // Tensor twists carry R^3, yielding the required psi^i*R.
                    uint32_t high=sum>>32,m=uint32_t(sum)*f.inverse;
                    uint32_t other=__umulhi(m,f.p),value=high-other;
                    if(high<other)value+=f.p;
                    shared.encoded[word-blockIdx.y*Words][i-row_base]=mont(value,twist[size_t(word)*n+streamed_index(i,n,column_log)],f);
                } else shared.reconstructed[i-row_base][word-blockIdx.y*Words]=sum;
            }
        }
    }
    __syncthreads();
    for(uint32_t at=threadIdx.x;at<Rows*Words;at+=blockDim.x) {
        if(Encode) {
            uint32_t col=at/64,row=at%64,i=row_base+row,word=blockIdx.y*Words+col;
            if(i<n&&word<count) out[(size_t(blockIdx.z)*count+word)*n+streamed_index(i,n,column_log)]=shared.encoded[col][row];
        } else {
            uint32_t row=at/Words,col=at%Words,i=row_base+row,word=blockIdx.y*Words+col;
            if(i<n&&word<count) dots[size_t(i)*count+word]=shared.reconstructed[row][col];
        }
    }

#endif
}
__global__ void forward(uint32_t* data,const Prime* fields,const uint32_t* roots,uint32_t n,uint32_t length) {
    uint32_t t=blockIdx.x*blockDim.x+threadIdx.x;
    if(t>=n/2) return;
    uint32_t prime=blockIdx.y,batch=blockIdx.z,half=length/2;
    uint32_t j=t&(half-1),i=(t/half)*length+j;
    Prime f=fields[prime]; uint32_t* row=data+(size_t(batch)*gridDim.y+prime)*n;
    uint32_t a=row[i],b=row[i+half];
    row[i]=add_mod(a,b,f.p);
    row[i+half]=mont(sub_mod(a,b,f.p),roots[size_t(prime)*(n/2)+j*(n/length)],f);
}
__global__ void pointwise(uint32_t* data,const Prime* fields,uint32_t n) {
    uint32_t i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    size_t at=size_t(blockIdx.y)*n+i;
    data[at]=mont(data[at],data[at+size_t(gridDim.y)*n],fields[blockIdx.y]);
}
__global__ void inverse(uint32_t* data,const Prime* fields,const uint32_t* roots,uint32_t n,uint32_t length) {
    uint32_t t=blockIdx.x*blockDim.x+threadIdx.x; if(t>=n/2) return;
    uint32_t prime=blockIdx.y,half=length/2,j=t&(half-1),i=(t/half)*length+j;
    Prime f=fields[prime]; uint32_t* row=data+size_t(prime)*n;
    uint32_t a=row[i],b=mont(row[i+half],roots[size_t(prime)*(n/2)+j*(n/length)],f);
    row[i]=add_mod(a,b,f.p); row[i+half]=sub_mod(a,b,f.p);
}
__global__ void finish_ntt(uint32_t* data,const Prime* fields,const uint32_t* factors,uint32_t n) {
    uint32_t i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    size_t at=size_t(blockIdx.y)*n+i; data[at]=mont(data[at],factors[at],fields[blockIdx.y]);
}

// The large transform is tiled into high-bit and low-bit stages. A tile
// retains intermediate values in shared memory throughout its stage group.
template<bool Inverse>
__global__ void high_stages(uint32_t* data,const Prime* fields,const uint32_t* roots,
                            const uint32_t* factors,uint32_t n) {
    extern __shared__ uint32_t tile[];
    uint32_t rows=n/NTT_TILE,col_base=blockIdx.x*32,prime=blockIdx.y;
    Prime f=fields[prime];
    uint32_t* source=data+(size_t(blockIdx.z)*gridDim.y+prime)*n;
    for(uint32_t t=threadIdx.x;t<rows*32;t+=blockDim.x)
        tile[t]=source[(t/32)*NTT_TILE+col_base+t%32];
    __syncthreads();
    uint32_t count=__ffs(rows)-1;
    for(uint32_t stage=0;stage<count;++stage) {
        uint32_t log_length=Inverse?stage+1:count-stage;
        uint32_t half=1u<<(log_length-1),stride=rows>>log_length;
        for(uint32_t t=threadIdx.x;t<(rows/2)*32;t+=blockDim.x) {
            uint32_t pair=t/32,col=t%32,j=pair&(half-1);
            uint32_t row=((pair>>(log_length-1))<<log_length)+j;
            uint32_t a=tile[row*32+col],b=tile[(row+half)*32+col];
            uint32_t root=roots[size_t(prime)*(n/2)+(j*NTT_TILE+col_base+col)*stride];
            if(Inverse) {
                b=mont(b,root,f); tile[row*32+col]=add_mod(a,b,f.p); tile[(row+half)*32+col]=sub_mod(a,b,f.p);
            } else {
                tile[row*32+col]=add_mod(a,b,f.p); tile[(row+half)*32+col]=mont(sub_mod(a,b,f.p),root,f);
            }
        }
        __syncthreads();
    }
    for(uint32_t t=threadIdx.x;t<rows*32;t+=blockDim.x) {
        uint32_t i=(t/32)*NTT_TILE+col_base+t%32,value=tile[t];
        if(Inverse) value=mont(value,factors[size_t(prime)*n+i],f);
        source[i]=value;
    }
}

// The row bits above the eight resident warps are local register indices.
// Keep those butterflies in registers; only the last three row bits need
// shared-memory exchanges between warps.
template<bool Inverse,int Rows>
__global__ void high_stages_register(uint32_t* data,const Prime* fields,const HighRoot* roots,
                                     const uint32_t* factors,uint32_t n,uint32_t column_offset=0) {
    constexpr int values=Rows/8;
    __shared__ uint32_t tile[Rows*32];
    uint32_t col=threadIdx.x%32,warp=threadIdx.x/32,col_base=column_offset+blockIdx.x*32,prime=blockIdx.y;
    Prime f=fields[prime];
    uint32_t* source=data+(size_t(blockIdx.z)*gridDim.y+prime)*n;
    uint32_t v[values];
    #pragma unroll
    for(int k=0;k<values;++k) v[k]=source[(warp+k*8)*NTT_TILE+col_base+col];
    if(!Inverse) {
        #pragma unroll
        for(int length=values;length>=2;length/=2) {
            constexpr uint32_t row_pitch=8;
            int half=length/2;
            #pragma unroll
            for(int pair=0;pair<values/2;++pair) {
                int j=pair%half,at=(pair/half)*length+j;
                uint2 root=load_high_root(roots,size_t(prime)*(n-NTT_TILE)+(Rows-length*8)*NTT_TILE+(j*row_pitch+warp)*NTT_TILE+col_base+col,f.p);
                uint32_t a=v[at],b=v[at+half];
                v[at]=add_mod(a,b,f.p); v[at+half]=shoup(butterfly_difference(a,b,f.p),root.x,root.y,f.p);
            }
        }
    }
    #pragma unroll
    for(int k=0;k<values;++k) tile[(warp+k*8)*32+col]=v[k];
    __syncthreads();
    #pragma unroll
    for(uint32_t stage=0;stage<3;++stage) {
        uint32_t log_length=Inverse?stage+1:3-stage;
        uint32_t half=1u<<(log_length-1),stride=Rows>>log_length;
        for(uint32_t t=threadIdx.x;t<(Rows/2)*32;t+=blockDim.x) {
            uint32_t pair=t/32,column=t%32,j=pair&(half-1),row=((pair>>(log_length-1))<<log_length)+j;
            uint32_t a=tile[row*32+column],b=tile[(row+half)*32+column];
            uint2 root=load_high_root(roots,size_t(prime)*(n-NTT_TILE)+(Rows-(1u<<log_length))*NTT_TILE+j*NTT_TILE+col_base+column,f.p);
            if(Inverse) {
                b=shoup(b,root.x,root.y,f.p); tile[row*32+column]=add_mod(a,b,f.p); tile[(row+half)*32+column]=sub_mod(a,b,f.p);
            } else {
                tile[row*32+column]=add_mod(a,b,f.p); tile[(row+half)*32+column]=shoup(butterfly_difference(a,b,f.p),root.x,root.y,f.p);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int k=0;k<values;++k) v[k]=tile[(warp+k*8)*32+col];
    if(Inverse) {
        #pragma unroll
        for(int length=2;length<=values;length*=2) {
            int half=length/2;
            #pragma unroll
            for(int pair=0;pair<values/2;++pair) {
                int j=pair%half,at=(pair/half)*length+j;
                uint2 root=load_high_root(roots,size_t(prime)*(n-NTT_TILE)+(Rows-length*8)*NTT_TILE+(j*8+warp)*NTT_TILE+col_base+col,f.p);
                uint32_t a=v[at],b=shoup(v[at+half],root.x,root.y,f.p);
                v[at]=add_mod(a,b,f.p); v[at+half]=sub_mod(a,b,f.p);
            }
        }
    }
    #pragma unroll
    for(int k=0;k<values;++k) {
        uint32_t i=(warp+k*8)*NTT_TILE+col_base+col,value=v[k];
        if(Inverse) value=mont(value,factors[size_t(prime)*n+i],f);
        source[i]=value;
    }
}

template<int Rows,int Columns,int Fields>
__global__ void inverse_transpose_cluster(uint32_t* data,const Prime* fields,const HighRoot* roots,
                                     const uint32_t* factors,uint32_t n,uint32_t column_offset,uint32_t* output,uint32_t padded_r,uint32_t primes) {
#if __CUDA_ARCH__ >= 900
    constexpr bool Inverse=true;
    constexpr int values=Rows/8;
    __shared__ uint32_t tile[Rows*Columns];
    uint32_t col=threadIdx.x%Columns,warp=threadIdx.x/Columns,col_base=column_offset+blockIdx.z*Columns,prime=blockIdx.y*Fields+blockIdx.x;
    const uint32_t field=prime<primes?prime:0;
    Prime f=fields[field];
    uint32_t* source=data+size_t(field)*n;
    uint32_t v[values];
    #pragma unroll
    for(int k=0;k<values;++k) v[k]=source[(warp+k*8)*NTT_TILE+col_base+col];
    if(!Inverse) {
        #pragma unroll
        for(int length=values;length>=2;length/=2) {
            constexpr uint32_t row_pitch=8;
            int half=length/2;
            #pragma unroll
            for(int pair=0;pair<values/2;++pair) {
                int j=pair%half,at=(pair/half)*length+j;
                uint2 root=load_high_root(roots,size_t(field)*(n-NTT_TILE)+(Rows-length*8)*NTT_TILE+(j*row_pitch+warp)*NTT_TILE+col_base+col,f.p);
                uint32_t a=v[at],b=v[at+half];
                v[at]=add_mod(a,b,f.p); v[at+half]=shoup(butterfly_difference(a,b,f.p),root.x,root.y,f.p);
            }
        }
    }
    #pragma unroll
    for(int k=0;k<values;++k) tile[(warp+k*8)*Columns+col]=v[k];
    __syncthreads();
    #pragma unroll
    for(uint32_t stage=0;stage<3;++stage) {
        uint32_t log_length=Inverse?stage+1:3-stage;
        uint32_t half=1u<<(log_length-1),stride=Rows>>log_length;
        for(uint32_t t=threadIdx.x;t<(Rows/2)*Columns;t+=blockDim.x) {
            uint32_t pair=t/Columns,column=t%Columns,j=pair&(half-1),row=((pair>>(log_length-1))<<log_length)+j;
            uint32_t a=tile[row*Columns+column],b=tile[(row+half)*Columns+column];
            uint2 root=load_high_root(roots,size_t(field)*(n-NTT_TILE)+(Rows-(1u<<log_length))*NTT_TILE+j*NTT_TILE+col_base+column,f.p);
            if(Inverse) {
                b=shoup(b,root.x,root.y,f.p); tile[row*Columns+column]=add_mod(a,b,f.p); tile[(row+half)*Columns+column]=sub_mod(a,b,f.p);
            } else {
                tile[row*Columns+column]=add_mod(a,b,f.p); tile[(row+half)*Columns+column]=shoup(butterfly_difference(a,b,f.p),root.x,root.y,f.p);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int k=0;k<values;++k) v[k]=tile[(warp+k*8)*Columns+col];
    if(Inverse) {
        #pragma unroll
        for(int length=2;length<=values;length*=2) {
            int half=length/2;
            #pragma unroll
            for(int pair=0;pair<values/2;++pair) {
                int j=pair%half,at=(pair/half)*length+j;
                uint2 root=load_high_root(roots,size_t(field)*(n-NTT_TILE)+(Rows-length*8)*NTT_TILE+(j*8+warp)*NTT_TILE+col_base+col,f.p);
                uint32_t a=v[at],b=shoup(v[at+half],root.x,root.y,f.p);
                v[at]=add_mod(a,b,f.p); v[at+half]=sub_mod(a,b,f.p);
            }
        }
    }
    #pragma unroll
    for(int k=0;k<values;++k) {
        uint32_t i=(warp+k*8)*NTT_TILE+col_base+col,value=v[k];
        if(Inverse) value=mont(value,factors[size_t(field)*n+i],f);
        tile[(warp+k*8)*Columns+col]=prime<primes?value:0;
    }
    auto cluster=cooperative_groups::this_cluster();
    cluster.sync();
    // Each warp writes adjacent fields for several output coefficients. All
    // remote shared loads complete before any block's shared storage expires.
    for(unsigned at=cluster.block_rank()*blockDim.x+threadIdx.x;at<Rows*Columns*Fields;at+=Fields*blockDim.x){
        const unsigned rank=at%Fields,coefficient=at/Fields;
        const uint32_t* peer=cluster.map_shared_rank(tile,rank);
        const unsigned packed=column_offset*Rows+(coefficient/Columns)*32+blockIdx.z*Columns+coefficient%Columns;
        output[size_t(packed)*padded_r+blockIdx.y*Fields+rank]=peer[coefficient];
    }
    cluster.sync();
#endif
}


template<int Rows,int Columns,int Fields>
__global__ void inverse_transpose_shared(uint32_t* data,const Prime* fields,const HighRoot* roots,
                                     const uint32_t* factors,uint32_t n,uint32_t column_offset,uint32_t* output,uint32_t padded_r,uint32_t primes) {
    constexpr bool Inverse=true;
    constexpr int values=Rows/8;
    __shared__ uint32_t tiles[Fields*Rows*Columns];
    const unsigned local_thread=threadIdx.x%(8*Columns),rank=threadIdx.x/(8*Columns);
    uint32_t* tile=tiles+rank*Rows*Columns;
    uint32_t col=local_thread%Columns,warp=local_thread/Columns,col_base=column_offset+blockIdx.x*Columns,prime=blockIdx.y*Fields+rank;
    const uint32_t field=prime<primes?prime:0;
    Prime f=fields[field];
    uint32_t* source=data+size_t(field)*n;
    uint32_t v[values];
    #pragma unroll
    for(int k=0;k<values;++k) v[k]=source[(warp+k*8)*NTT_TILE+col_base+col];
    if(!Inverse) {
        #pragma unroll
        for(int length=values;length>=2;length/=2) {
            constexpr uint32_t row_pitch=8;
            int half=length/2;
            #pragma unroll
            for(int pair=0;pair<values/2;++pair) {
                int j=pair%half,at=(pair/half)*length+j;
                uint2 root=load_high_root(roots,size_t(field)*(n-NTT_TILE)+(Rows-length*8)*NTT_TILE+(j*row_pitch+warp)*NTT_TILE+col_base+col,f.p);
                uint32_t a=v[at],b=v[at+half];
                v[at]=add_mod(a,b,f.p); v[at+half]=shoup(butterfly_difference(a,b,f.p),root.x,root.y,f.p);
            }
        }
    }
    constexpr unsigned local_stages=Columns==4?3:Columns==8?2:1;
    #pragma unroll
    for(unsigned stage=0;stage<local_stages;++stage){
        const unsigned half=1u<<stage,j=warp&(half-1);const bool upper=(warp&half)!=0;
        const uint2 root=load_high_root(roots,size_t(field)*(n-NTT_TILE)+(Rows-2*half)*NTT_TILE+j*NTT_TILE+col_base+col,f.p);
        #pragma unroll
        for(unsigned k=0;k<values;++k){
            uint32_t value=v[k];if(upper)value=shoup(value,root.x,root.y,f.p);
            const uint32_t peer=__shfl_xor_sync(0xffffffffu,value,half*Columns);
            v[k]=upper?sub_mod(peer,value,f.p):add_mod(value,peer,f.p);
        }
    }
    if constexpr(local_stages<3){
    #pragma unroll
    for(int k=0;k<values;++k) tile[(warp+k*8)*Columns+col]=v[k];
    __syncthreads();
    #pragma unroll
    for(uint32_t stage=local_stages;stage<3;++stage) {
        uint32_t log_length=Inverse?stage+1:3-stage;
        uint32_t half=1u<<(log_length-1),stride=Rows>>log_length;
        for(uint32_t t=local_thread;t<(Rows/2)*Columns;t+=8*Columns) {
            uint32_t pair=t/Columns,column=t%Columns,j=pair&(half-1),row=((pair>>(log_length-1))<<log_length)+j;
            uint32_t a=tile[row*Columns+column],b=tile[(row+half)*Columns+column];
            uint2 root=load_high_root(roots,size_t(field)*(n-NTT_TILE)+(Rows-(1u<<log_length))*NTT_TILE+j*NTT_TILE+col_base+column,f.p);
            if(Inverse) {
                b=shoup(b,root.x,root.y,f.p); tile[row*Columns+column]=add_mod(a,b,f.p); tile[(row+half)*Columns+column]=sub_mod(a,b,f.p);
            } else {
                tile[row*Columns+column]=add_mod(a,b,f.p); tile[(row+half)*Columns+column]=shoup(butterfly_difference(a,b,f.p),root.x,root.y,f.p);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int k=0;k<values;++k) v[k]=tile[(warp+k*8)*Columns+col];
    }
    if(Inverse) {
        #pragma unroll
        for(int length=2;length<=values;length*=2) {
            int half=length/2;
            #pragma unroll
            for(int pair=0;pair<values/2;++pair) {
                int j=pair%half,at=(pair/half)*length+j;
                uint2 root=load_high_root(roots,size_t(field)*(n-NTT_TILE)+(Rows-length*8)*NTT_TILE+(j*8+warp)*NTT_TILE+col_base+col,f.p);
                uint32_t a=v[at],b=shoup(v[at+half],root.x,root.y,f.p);
                v[at]=add_mod(a,b,f.p); v[at+half]=sub_mod(a,b,f.p);
            }
        }
    }
    #pragma unroll
    for(int k=0;k<values;++k) {
        uint32_t i=(warp+k*8)*NTT_TILE+col_base+col,value=v[k];
        if(Inverse) value=mont(value,factors[size_t(field)*n+i],f);
        tile[(warp+k*8)*Columns+col]=prime<primes?value:0;
    }
    __syncthreads();
    for(unsigned at=threadIdx.x;at<Rows*Columns*Fields;at+=blockDim.x){
        const unsigned rank=at%Fields,coefficient=at/Fields;
        const uint32_t* peer=tiles+rank*Rows*Columns;
        const unsigned packed=column_offset*Rows+(coefficient/Columns)*32+blockIdx.x*Columns+coefficient%Columns;
        output[size_t(packed)*padded_r+blockIdx.y*Fields+rank]=peer[coefficient];
    }
}



__device__ __forceinline__ uint32_t twiddle_index(uint32_t i) {return i^(i>>5);}
__global__ __launch_bounds__(128) void middle_stages_register(uint32_t* data,const Prime* fields,
                                                            const uint32_t* roots,const uint32_t* inverse_roots,const uint32_t* root_quotients,const uint32_t* inverse_quotients,uint32_t n) {
    constexpr int values=16;
    uint32_t lane=threadIdx.x%32,row=blockIdx.x*4+threadIdx.x/32,prime=blockIdx.y;
    if(row>=n/512)return;
    size_t offset=size_t(prime)*n+row*512;
    __shared__ uint32_t forward_twiddles[256],backward_twiddles[256],forward_q[256],backward_q[256];
    for(uint32_t i=threadIdx.x;i<256;i+=blockDim.x) {
        uint32_t at=i^(i>>5);
        forward_twiddles[at]=roots[size_t(prime)*256+i];
        backward_twiddles[at]=inverse_roots[size_t(prime)*256+i];
        forward_q[at]=root_quotients[size_t(prime)*256+i];backward_q[at]=inverse_quotients[size_t(prime)*256+i];
    }
    __syncthreads();
    Prime f=fields[prime];uint32_t a[values],b[values];
    #pragma unroll
    for(int k=0;k<values;++k) {a[k]=data[offset+k*32+lane];b[k]=data[offset+k*32+lane+size_t(gridDim.y)*n];}
    #pragma unroll
    for(int length=values;length>=2;length/=2) {
        int half=length/2;
        #pragma unroll
        for(int pair=0;pair<values/2;++pair) {
            int j=pair%half,at=(pair/half)*length+j;
            uint32_t root=forward_twiddles[twiddle_index((j*32+lane)*(512/(length*32)))];
            uint32_t root_q=forward_q[twiddle_index((j*32+lane)*(512/(length*32)))];
            uint32_t x=a[at],y=a[at+half];a[at]=add_mod(x,y,f.p);a[at+half]=shoup(butterfly_difference(x,y,f.p),root,root_q,f.p);
            x=b[at];y=b[at+half];b[at]=add_mod(x,y,f.p);b[at+half]=shoup(butterfly_difference(x,y,f.p),root,root_q,f.p);
        }
    }
    #pragma unroll
    for(int length=32;length>=4;length/=2) {
        int distance=length/2;bool upper=(lane&distance)!=0;
        uint32_t root=forward_twiddles[twiddle_index((lane&(distance-1))*(512/length))];
        uint32_t root_q=forward_q[twiddle_index((lane&(distance-1))*(512/length))];
        #pragma unroll
        for(int k=0;k<values;++k) {
            uint32_t peer=__shfl_xor_sync(0xffffffffu,a[k],distance);
            a[k]=upper?shoup(butterfly_difference(peer,a[k],f.p),root,root_q,f.p):add_mod(a[k],peer,f.p);
            peer=__shfl_xor_sync(0xffffffffu,b[k],distance);
            b[k]=upper?shoup(butterfly_difference(peer,b[k],f.p),root,root_q,f.p):add_mod(b[k],peer,f.p);
        }
    }
    #pragma unroll
    for(int k=0;k<values;++k) {
        // The length-two twiddle is Montgomery one. Fuse that forward
        // butterfly, the product, and the first inverse butterfly.
        bool upper=(lane&1)!=0;
        uint32_t pa=__shfl_xor_sync(0xffffffffu,a[k],1),pb=__shfl_xor_sync(0xffffffffu,b[k],1);
        uint32_t x=upper?sub_mod(pa,a[k],f.p):add_mod(a[k],pa,f.p);
        uint32_t y=WIDE_POINTWISE?(upper?pb+f.p-b[k]:b[k]+pb):(upper?sub_mod(pb,b[k],f.p):add_mod(b[k],pb,f.p));
        uint32_t product=mont(x,y,f),peer=__shfl_xor_sync(0xffffffffu,product,1);
        a[k]=upper?sub_mod(peer,product,f.p):add_mod(product,peer,f.p);
    }
    #pragma unroll
    for(int length=4;length<=32;length*=2) {
        int distance=length/2;bool upper=(lane&distance)!=0;
        uint32_t root=backward_twiddles[twiddle_index((lane&(distance-1))*(512/length))];
        uint32_t root_q=backward_q[twiddle_index((lane&(distance-1))*(512/length))];
        #pragma unroll
        for(int k=0;k<values;++k) {
            uint32_t value=upper?shoup(a[k],root,root_q,f.p):a[k];
            uint32_t peer=__shfl_xor_sync(0xffffffffu,value,distance);
            a[k]=upper?sub_mod(peer,value,f.p):add_mod(value,peer,f.p);
        }
    }
    #pragma unroll
    for(int length=2;length<=values;length*=2) {
        int half=length/2;
        #pragma unroll
        for(int pair=0;pair<values/2;++pair) {
            int j=pair%half,at=(pair/half)*length+j;
            uint32_t root=backward_twiddles[twiddle_index((j*32+lane)*(512/(length*32)))];
            uint32_t root_q=backward_q[twiddle_index((j*32+lane)*(512/(length*32)))];
            uint32_t x=a[at],y=shoup(a[at+half],root,root_q,f.p);a[at]=add_mod(x,y,f.p);a[at+half]=sub_mod(x,y,f.p);
        }
    }
    #pragma unroll
    for(int k=0;k<values;++k)data[offset+k*32+lane]=a[k];
}

template<bool First>
__global__ __launch_bounds__(128) void middle_operand_stage(uint32_t* data,const Prime* fields,
                                                            const uint32_t* roots,const uint32_t* inverse_roots,const uint32_t* root_quotients,const uint32_t* inverse_quotients,uint32_t n) {
    constexpr int values=16;
    uint32_t lane=threadIdx.x%32,row=blockIdx.x*4+threadIdx.x/32,prime=blockIdx.y;
    if(row>=n/512)return;
    size_t offset=size_t(prime)*n+row*512;
    __shared__ uint32_t forward_twiddles[256],backward_twiddles[256],forward_q[256],backward_q[256];
    for(uint32_t i=threadIdx.x;i<256;i+=blockDim.x) {
        uint32_t at=i^(i>>5);
        forward_twiddles[at]=roots[size_t(prime)*256+i];
        backward_twiddles[at]=inverse_roots[size_t(prime)*256+i];
        forward_q[at]=root_quotients[size_t(prime)*256+i];backward_q[at]=inverse_quotients[size_t(prime)*256+i];
    }
    __syncthreads();
    Prime f=fields[prime];uint32_t a[values];
    #pragma unroll
    for(int k=0;k<values;++k) {a[k]=data[offset+k*32+lane+(First?0:size_t(gridDim.y)*n)];}
    #pragma unroll
    for(int length=values;length>=2;length/=2) {
        int half=length/2;
        #pragma unroll
        for(int pair=0;pair<values/2;++pair) {
            int j=pair%half,at=(pair/half)*length+j;
            uint32_t root=forward_twiddles[twiddle_index((j*32+lane)*(512/(length*32)))];
            uint32_t root_q=forward_q[twiddle_index((j*32+lane)*(512/(length*32)))];
            uint32_t x=a[at],y=a[at+half];a[at]=add_mod(x,y,f.p);a[at+half]=shoup(OPERAND_WIDE_DIFF?x+f.p-y:sub_mod(x,y,f.p),root,root_q,f.p);

        }
    }
    #pragma unroll
    for(int length=32;length>=4;length/=2) {
        int distance=length/2;bool upper=(lane&distance)!=0;
        uint32_t root=forward_twiddles[twiddle_index((lane&(distance-1))*(512/length))];
        uint32_t root_q=forward_q[twiddle_index((lane&(distance-1))*(512/length))];
        #pragma unroll
        for(int k=0;k<values;++k) {
            uint32_t peer=__shfl_xor_sync(0xffffffffu,a[k],distance);
            a[k]=upper?shoup(OPERAND_WIDE_DIFF?peer+f.p-a[k]:sub_mod(peer,a[k],f.p),root,root_q,f.p):add_mod(a[k],peer,f.p);

        }
    }
    #pragma unroll
    for(int k=0;k<values;++k) {
        // The length-two twiddle is Montgomery one. Fuse that forward
        // butterfly, the product, and the first inverse butterfly.
        bool upper=(lane&1)!=0;
        uint32_t peer=__shfl_xor_sync(0xffffffffu,a[k],1);
        // First stores canonical A. For the product, A<p and B<2p imply
        // A*B < 2p^2 < p*2^32, so one Montgomery correction remains exact.
        uint32_t transformed=(!First&&OPERAND_WIDE_PRODUCT)?(upper?peer+f.p-a[k]:a[k]+peer)
            :(upper?sub_mod(peer,a[k],f.p):add_mod(a[k],peer,f.p));
        if constexpr(First) a[k]=transformed;
        else {
            uint32_t product=mont(data[offset+k*32+lane],transformed,f);
            peer=__shfl_xor_sync(0xffffffffu,product,1);
            a[k]=upper?sub_mod(peer,product,f.p):add_mod(product,peer,f.p);
        }
    }
    if constexpr(!First) {
    #pragma unroll
    for(int length=4;length<=32;length*=2) {
        int distance=length/2;bool upper=(lane&distance)!=0;
        uint32_t root=backward_twiddles[twiddle_index((lane&(distance-1))*(512/length))];
        uint32_t root_q=backward_q[twiddle_index((lane&(distance-1))*(512/length))];
        #pragma unroll
        for(int k=0;k<values;++k) {
            uint32_t value=upper?shoup(a[k],root,root_q,f.p):a[k];
            uint32_t peer=__shfl_xor_sync(0xffffffffu,value,distance);
            a[k]=upper?sub_mod(peer,value,f.p):add_mod(value,peer,f.p);
        }
    }
    #pragma unroll
    for(int length=2;length<=values;length*=2) {
        int half=length/2;
        #pragma unroll
        for(int pair=0;pair<values/2;++pair) {
            int j=pair%half,at=(pair/half)*length+j;
            uint32_t root=backward_twiddles[twiddle_index((j*32+lane)*(512/(length*32)))];
            uint32_t root_q=backward_q[twiddle_index((j*32+lane)*(512/(length*32)))];
            uint32_t x=a[at],y=shoup(a[at+half],root,root_q,f.p);a[at]=add_mod(x,y,f.p);a[at+half]=sub_mod(x,y,f.p);
        }
    }
    }
    #pragma unroll
    for(int k=0;k<values;++k)data[offset+k*32+lane]=a[k];
}

__global__ void middle_stages(uint32_t* data,const Prime* fields,const uint32_t* roots,
                              const uint32_t* inverse_roots,const uint32_t* factors,
                              uint32_t n,uint32_t columns) {
    __shared__ uint32_t a[NTT_TILE],b[NTT_TILE];
    uint32_t prime=blockIdx.y,start=blockIdx.x*columns,thread=threadIdx.x;
    size_t offset=size_t(prime)*n+start;
    Prime f=fields[prime];
    for(uint32_t i=thread;i<columns;i+=blockDim.x) {
        a[i]=data[offset+i]; b[i]=data[offset+i+size_t(gridDim.y)*n];
    }
    __syncthreads();
    uint32_t count=__ffs(columns)-1;
    for(uint32_t log_length=count;log_length>0;--log_length) {
        uint32_t half=1u<<(log_length-1),j=thread&(half-1),i=((thread>>(log_length-1))<<log_length)+j;
        if(thread<columns/2) {
            uint32_t root=roots[size_t(prime)*(n/2)+j*(n>>log_length)];
            uint32_t x=a[i],y=a[i+half]; a[i]=add_mod(x,y,f.p); a[i+half]=mont(sub_mod(x,y,f.p),root,f);
            x=b[i]; y=b[i+half]; b[i]=add_mod(x,y,f.p); b[i+half]=mont(sub_mod(x,y,f.p),root,f);
        }
        __syncthreads();
    }
    for(uint32_t i=thread;i<columns;i+=blockDim.x) a[i]=mont(a[i],b[i],f);
    __syncthreads();
    for(uint32_t log_length=1;log_length<=count;++log_length) {
        uint32_t half=1u<<(log_length-1),j=thread&(half-1),i=((thread>>(log_length-1))<<log_length)+j;
        if(thread<columns/2) {
            uint32_t root=inverse_roots[size_t(prime)*(n/2)+j*(n>>log_length)];
            uint32_t x=a[i],y=mont(a[i+half],root,f);
            a[i]=add_mod(x,y,f.p); a[i+half]=sub_mod(x,y,f.p);
        }
        __syncthreads();
    }
    for(uint32_t i=thread;i<columns;i+=blockDim.x) {
        uint32_t value=a[i]; if(n==columns) value=mont(value,factors[offset+i],f);
        data[offset+i]=value;
    }
}
// Adjacent coefficients stay in registers for the six smallest NTT stages.
// All communication in those stages is either within a thread or a warp.
__global__ void middle_stages_warp(uint32_t* data,const Prime* fields,const uint32_t* roots,
                                   const uint32_t* inverse_roots,const uint32_t* factors,
                                   uint32_t n,uint32_t columns) {
    __shared__ uint32_t a[NTT_TILE],b[NTT_TILE];
    uint32_t prime=blockIdx.y,start=blockIdx.x*columns,thread=threadIdx.x;
    size_t offset=size_t(prime)*n+start;
    Prime f=fields[prime];
    for(uint32_t i=thread;i<columns;i+=blockDim.x) {
        a[i]=data[offset+i]; b[i]=data[offset+i+size_t(gridDim.y)*n];
    }
    __syncthreads();
    uint32_t count=__ffs(columns)-1;
    for(uint32_t log_length=count;log_length>6;--log_length) {
        uint32_t half=1u<<(log_length-1),j=thread&(half-1),i=((thread>>(log_length-1))<<log_length)+j;
        if(thread<columns/2) {
            uint32_t root=roots[size_t(prime)*(n/2)+j*(n>>log_length)];
            uint32_t x=a[i],y=a[i+half]; a[i]=add_mod(x,y,f.p); a[i+half]=mont(sub_mod(x,y,f.p),root,f);
            x=b[i]; y=b[i+half]; b[i]=add_mod(x,y,f.p); b[i+half]=mont(sub_mod(x,y,f.p),root,f);
        }
        __syncthreads();
    }
    if(thread<columns/2) {
        uint32_t a0=a[2*thread],a1=a[2*thread+1],b0=b[2*thread],b1=b[2*thread+1];
        #pragma unroll
        for(uint32_t log_length=6;log_length>1;--log_length) {
            uint32_t half=1u<<(log_length-1),distance=half/2,j=(2*thread)&(half-1);
            bool upper=(thread&distance)!=0;
            uint32_t root0=roots[size_t(prime)*(n/2)+j*(n>>log_length)];
            uint32_t root1=roots[size_t(prime)*(n/2)+(j+1)*(n>>log_length)];
            uint32_t p0=__shfl_xor_sync(0xffffffffu,a0,distance),p1=__shfl_xor_sync(0xffffffffu,a1,distance);
            a0=upper?mont(sub_mod(p0,a0,f.p),root0,f):add_mod(a0,p0,f.p);
            a1=upper?mont(sub_mod(p1,a1,f.p),root1,f):add_mod(a1,p1,f.p);
            p0=__shfl_xor_sync(0xffffffffu,b0,distance); p1=__shfl_xor_sync(0xffffffffu,b1,distance);
            b0=upper?mont(sub_mod(p0,b0,f.p),root0,f):add_mod(b0,p0,f.p);
            b1=upper?mont(sub_mod(p1,b1,f.p),root1,f):add_mod(b1,p1,f.p);
        }
        uint32_t x=add_mod(a0,a1,f.p),y=sub_mod(a0,a1,f.p);
        a0=mont(x,add_mod(b0,b1,f.p),f); a1=mont(y,sub_mod(b0,b1,f.p),f);
        x=add_mod(a0,a1,f.p); y=sub_mod(a0,a1,f.p); a0=x; a1=y;
        #pragma unroll
        for(uint32_t log_length=2;log_length<=6;++log_length) {
            uint32_t half=1u<<(log_length-1),distance=half/2,j=(2*thread)&(half-1);
            bool upper=(thread&distance)!=0;
            uint32_t root0=inverse_roots[size_t(prime)*(n/2)+j*(n>>log_length)];
            uint32_t root1=inverse_roots[size_t(prime)*(n/2)+(j+1)*(n>>log_length)];
            uint32_t x0=upper?mont(a0,root0,f):a0,x1=upper?mont(a1,root1,f):a1;
            uint32_t p0=__shfl_xor_sync(0xffffffffu,x0,distance),p1=__shfl_xor_sync(0xffffffffu,x1,distance);
            a0=upper?sub_mod(p0,x0,f.p):add_mod(x0,p0,f.p);
            a1=upper?sub_mod(p1,x1,f.p):add_mod(x1,p1,f.p);
        }
        a[2*thread]=a0; a[2*thread+1]=a1;
    }
    __syncthreads();
    for(uint32_t log_length=7;log_length<=count;++log_length) {
        uint32_t half=1u<<(log_length-1),j=thread&(half-1),i=((thread>>(log_length-1))<<log_length)+j;
        if(thread<columns/2) {
            uint32_t root=inverse_roots[size_t(prime)*(n/2)+j*(n>>log_length)];
            uint32_t x=a[i],y=mont(a[i+half],root,f);
            a[i]=add_mod(x,y,f.p); a[i+half]=sub_mod(x,y,f.p);
        }
        __syncthreads();
    }
    for(uint32_t i=thread;i<columns;i+=blockDim.x) {
        uint32_t value=a[i]; if(n==columns) value=mont(value,factors[offset+i],f);
        data[offset+i]=value;
    }
}

// Transpose the prime-major residues into coefficient-major bytes for MMA.
__global__ void transpose_residues(const uint32_t* data,uint32_t* out,
                                   uint32_t n,uint32_t r,uint32_t padded_r) {
    __shared__ uint32_t tile[32][33];
    uint32_t x=threadIdx.x,y=threadIdx.y,coefficient=blockIdx.x*32,prime=blockIdx.y*32;
    for(uint32_t j=0;j<32;j+=8)
        tile[y+j][x]=(coefficient+x<n&&prime+y+j<r)?data[size_t(prime+y+j)*n+coefficient+x]:0;
    __syncthreads();
    for(uint32_t j=0;j<32;j+=8)
        if(coefficient+y+j<n) out[size_t(coefficient+y+j)*padded_r+prime+x]=tile[x][y+j];
}

__global__ void transpose_streamed_residues(const uint32_t* data,uint32_t* out,
                                   uint32_t n,uint32_t r,uint32_t padded_r,uint32_t row_offset,uint32_t column_log) {
    __shared__ uint32_t tile[32][33];
    uint32_t x=threadIdx.x,y=threadIdx.y,coefficient=row_offset+blockIdx.x*32,prime=blockIdx.y*32;
    for(uint32_t j=0;j<32;j+=8)
        tile[y+j][x]=(coefficient+x<n&&prime+y+j<r)?data[size_t(prime+y+j)*n+streamed_index(coefficient+x,n,column_log)]:0;
    __syncthreads();
    for(uint32_t j=0;j<32;j+=8)
        if(coefficient+y+j<n) out[size_t(coefficient+y+j)*padded_r+prime+x]=tile[x][y+j];
}

// Carry propagation is a warp-wide generate/propagate scan. The highest
// non-propagating bit determines the carry at each lane.
__device__ __forceinline__ uint32_t carry_scan(bool generate,bool propagate,uint32_t& chain) {
    constexpr uint32_t mask=0xffffffffu;
    uint32_t lane=threadIdx.x%32;
    uint32_t generators=__ballot_sync(mask,generate),propagators=__ballot_sync(mask,propagate);
    uint32_t stops=(~propagators)&(mask>>(31-lane));
    uint32_t outgoing=stops?((generators>>(31-__clz(stops)))&1):chain;
    uint32_t incoming=__shfl_up_sync(mask,outgoing,1);
    if(lane==0) incoming=chain;
    chain=__shfl_sync(mask,outgoing,31);
    return incoming;
}
template<int Chunks>
__device__ __forceinline__ void collapse_words(const uint64_t (&raw)[Chunks],uint32_t (&words)[Chunks]) {
    uint32_t lane=threadIdx.x%32,chain=0,previous_high=0;
    #pragma unroll
    for(int k=0;k<Chunks;++k) {
        uint32_t high=raw[k]>>32;
        uint32_t preceding=__shfl_up_sync(0xffffffffu,high,1);
        if(lane==0) preceding=previous_high;
        previous_high=__shfl_sync(0xffffffffu,high,31);
        uint64_t sum=uint64_t(uint32_t(raw[k]))+preceding;
        uint32_t low=uint32_t(sum);
        uint32_t incoming=carry_scan((sum>>32)!=0,low==0xffffffffu,chain);
        words[k]=low+incoming;
    }
}
template<int Chunks>
__device__ __forceinline__ uint32_t subtract_words(uint32_t (&a)[Chunks],const uint32_t (&b)[Chunks]) {
    uint32_t chain=0;
    #pragma unroll
    for(int k=0;k<Chunks;++k) {
        uint32_t incoming=carry_scan(a[k]<b[k],a[k]==b[k],chain);
        a[k]=a[k]-b[k]-incoming;
    }
    return chain;
}
template<int Chunks>
__device__ __forceinline__ uint32_t add_words(uint32_t (&a)[Chunks],const uint32_t (&b)[Chunks]) {
    uint32_t chain=0;
    #pragma unroll
    for(int k=0;k<Chunks;++k) {
        uint64_t sum=uint64_t(a[k])+b[k];
        uint32_t low=uint32_t(sum),incoming=carry_scan((sum>>32)!=0,low==0xffffffffu,chain);
        a[k]=low+incoming;
    }
    return chain;
}
template<int Chunks>
__device__ __forceinline__ int compare_words(const uint32_t (&a)[Chunks],const uint32_t (&b)[Chunks]) {
    #pragma unroll
    for(int k=Chunks-1;k>=0;--k) {
        uint32_t different=__ballot_sync(0xffffffffu,a[k]!=b[k]);
        if(different) return __shfl_sync(0xffffffffu,a[k]<b[k]?-1:1,31-__clz(different));
    }
    return 0;
}
template<int Chunks>
__device__ __forceinline__ uint32_t fetch_word(const uint32_t (&a)[Chunks],uint32_t index) {
    uint32_t value=0;
    #pragma unroll
    for(int k=0;k<Chunks;++k) {
        uint32_t v=__shfl_sync(0xffffffffu,a[k],index%32);
        if(index/32==k) value=v;
    }
    return value;
}
// With P > 4*N*(q-1)^2, the signed integer convolution x has |x/P|<1/4.
// Each fixed-point term differs from 2^32 * v/p by less than 3/2.
// For fewer than 8192 fields, rounding their sum identifies the CRT quotient
// exactly, with more than 0.249997 of an integer of safety margin.
template<int Chunks>
__global__ void reconstruct_direct(const uint32_t* residues,const uint64_t* reciprocals,
                                    const uint32_t* correction,uint32_t primes,uint32_t padded_r,const uint32_t* q,uint32_t* out,const uint64_t* dots,
                                    uint32_t n,uint32_t wide,uint32_t limbs,uint32_t width,uint32_t gap,uint32_t row_offset=0,uint32_t dot_stride=0) {
    uint32_t i=row_offset+(blockIdx.x*blockDim.x+threadIdx.x)/32,lane=threadIdx.x%32;
    if(i>=n) return;
    if(!dot_stride)dot_stride=wide;
    uint32_t quotient=0;
    if(dot_stride>wide){
        const uint64_t low=dots[size_t(i)*dot_stride+wide];
        const uint64_t high=dots[size_t(i)*dot_stride+wide+1];
        quotient=uint32_t((high+(low>>32)+(uint64_t(1)<<31))>>32);
    }else{
        uint64_t total=0;
        for(uint32_t k=lane;k<primes;k+=32)
            total+=(uint64_t(residues[size_t(i)*padded_r+k])*reciprocals[k])>>32;
        for(int delta=16;delta;delta>>=1)total+=__shfl_xor_sync(0xffffffffu,total,delta);
        quotient=uint32_t((total+(uint64_t(1)<<31))>>32);
    }
    uint32_t value[Chunks],modulus[Chunks],addition[Chunks]; uint64_t raw[Chunks],bonus[Chunks];
    #pragma unroll
    for(int k=0;k<Chunks;++k) {
        uint32_t j=k*32+lane;
        raw[k]=j<wide?dots[size_t(i)*dot_stride+j]:0;
        modulus[k]=j<wide?q[j]:0;
        bonus[k]=j<wide?uint64_t(correction[j])*quotient:0;
    }
    collapse_words(raw,value);
    collapse_words(bonus,addition);
    add_words(value,addition);
    uint32_t whole=width/32,bits=width%32;
    while(true) {
        bool high=false;
        #pragma unroll
        for(int k=0;k<Chunks;++k) {
            uint32_t j=k*32+lane,v=value[k];
            if(j==whole&&bits) v>>=bits;
            if(j>=whole) high|=v!=0;
        }
        if(!__any_sync(0xffffffffu,high)) break;
        #pragma unroll
        for(int k=0;k<Chunks;++k) {
            uint32_t j=k*32+lane,low=value[k];
            if(j>whole||(j==whole&&!bits)) low=0;
            else if(j==whole) low&=(uint32_t(1)<<bits)-1;
            uint32_t h=fetch_word(value,j+whole)>>bits;
            if(bits) h|=fetch_word(value,j+whole+1)<<(32-bits);
            raw[k]=uint64_t(h)*gap+low;
        }
        collapse_words(raw,value);
    }
    while(compare_words(value,modulus)>=0) subtract_words(value,modulus);
    #pragma unroll
    for(int k=0;k<Chunks;++k) {
        uint32_t j=k*32+lane;
        if(j<limbs) out[size_t(i)*limbs+j]=value[k];
    }
}

template<int Chunks>
__global__ void reconstruct_warp(const uint32_t* data,const Prime* fields,
                                const uint32_t* product,const uint32_t* half_product,const uint32_t* q,
                                uint32_t* out,const uint64_t* dots,uint32_t n,uint32_t r,uint32_t padded_r,
                                uint32_t wide,uint32_t limbs,uint32_t width,uint32_t gap) {
    uint32_t i=(blockIdx.x*blockDim.x+threadIdx.x)/32,lane=threadIdx.x%32;
    if(i>=n) return;
    double estimate=0;
    for(uint32_t k=lane;k<r;k+=32) estimate+=data[size_t(i)*padded_r+k]*fields[k].reciprocal;
    for(int offset=16;offset;offset>>=1) estimate+=__shfl_xor_sync(0xffffffffu,estimate,offset);
    uint32_t quotient=uint32_t(estimate);
    uint32_t value[Chunks],p[Chunks],half[Chunks],modulus[Chunks],multiple[Chunks];
    uint64_t raw[Chunks],scaled[Chunks];
    #pragma unroll
    for(int k=0;k<Chunks;++k) {
        uint32_t j=k*32+lane;
        raw[k]=j<wide?dots[size_t(i)*wide+j]:0;
        p[k]=j<wide?product[j]:0; half[k]=j<wide?half_product[j]:0;
        modulus[k]=j<wide?q[j]:0;
        scaled[k]=uint64_t(p[k])*quotient;
    }
    collapse_words(raw,value); collapse_words(scaled,multiple);
    uint32_t negative_quotient=subtract_words(value,multiple);
    while(negative_quotient) if(add_words(value,p)) negative_quotient=0;
    while(compare_words(value,p)>=0) subtract_words(value,p);
    bool negative=compare_words(value,half)>0;
    if(negative) {
        #pragma unroll
        for(int k=0;k<Chunks;++k) multiple[k]=p[k];
        subtract_words(multiple,value);
        #pragma unroll
        for(int k=0;k<Chunks;++k) value[k]=multiple[k];
    }
    uint32_t whole=width/32,bits=width%32;
    while(true) {
        bool high=false;
        #pragma unroll
        for(int k=0;k<Chunks;++k) {
            uint32_t j=k*32+lane,v=value[k];
            if(j==whole&&bits) v>>=bits;
            if(j>=whole) high|=v!=0;
        }
        if(!__any_sync(0xffffffffu,high)) break;
        #pragma unroll
        for(int k=0;k<Chunks;++k) {
            uint32_t j=k*32+lane,low=value[k];
            if(j>whole||(j==whole&&!bits)) low=0;
            else if(j==whole) low&=(uint32_t(1)<<bits)-1;
            uint32_t h=fetch_word(value,j+whole)>>bits;
            if(bits) h|=fetch_word(value,j+whole+1)<<(32-bits);
            raw[k]=uint64_t(h)*gap+low;
        }
        collapse_words(raw,value);
    }
    while(compare_words(value,modulus)>=0) subtract_words(value,modulus);
    bool nonzero=false;
    #pragma unroll
    for(int k=0;k<Chunks;++k) nonzero|=value[k]!=0;
    nonzero=__any_sync(0xffffffffu,nonzero);
    if(negative&&nonzero) {
        #pragma unroll
        for(int k=0;k<Chunks;++k) multiple[k]=modulus[k];
        subtract_words(multiple,value);
        #pragma unroll
        for(int k=0;k<Chunks;++k) value[k]=multiple[k];
    }
    #pragma unroll
    for(int k=0;k<Chunks;++k) {
        uint32_t j=k*32+lane;
        if(j<limbs) out[size_t(i)*limbs+j]=value[k];
    }
}
__device__ int compare_wide(const uint32_t* x,const uint32_t* y,uint32_t n,uint32_t words,uint32_t i) {
    for(uint32_t j=words;j-->0;) {
        uint32_t a=x[size_t(j)*n+i],b=y[j]; if(a!=b) return a<b?-1:1;
    }
    return 0;
}
__device__ void subtract_wide(uint32_t* x,const uint32_t* y,uint32_t n,uint32_t words,uint32_t i) {
    uint64_t borrow=0;
    for(uint32_t j=0;j<words;++j) {
        uint32_t v=x[size_t(j)*n+i]; uint64_t sub=uint64_t(y[j])+borrow;
        x[size_t(j)*n+i]=v-uint32_t(sub); borrow=(sub>>32)+(v<uint32_t(sub));
    }
}
template<bool Tensor>
__global__ void reconstruct(const uint32_t* data,const Prime* fields,const uint32_t* crt,
                            const uint32_t* product,const uint32_t* half_product,const uint32_t* q,
                            uint32_t* scratch,uint32_t* out,uint32_t n,uint32_t r,uint32_t wide,
                            uint32_t limbs,uint32_t width,uint32_t gap,const uint64_t* dots) {
    uint32_t i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    double estimate=0;
    for(uint32_t k=0;k<r;++k) estimate+=data[size_t(k)*n+i]*fields[k].reciprocal;
    uint32_t quotient=uint32_t(estimate); uint64_t carry=0,borrow=0;
    for(uint32_t j=0;j<wide;++j) {
        uint64_t lo=carry,hi=0;
        if(Tensor) lo+=dots[size_t(i)*wide+j];
        else for(uint32_t k=0;k<r;++k) {
            uint64_t term=uint64_t(data[size_t(k)*n+i])*crt[size_t(j)*r+k];
            uint64_t before=lo; lo+=term; hi+=lo<before;
        }
        uint32_t digit=uint32_t(lo); carry=(lo>>32)+(hi<<32);
        uint64_t sub=uint64_t(quotient)*product[j]+borrow;
        scratch[size_t(j)*n+i]=digit-uint32_t(sub); borrow=(sub>>32)+(digit<uint32_t(sub));
    }
    // Floating point estimates a quotient only. Integer corrections below
    // establish the exact representative, including near-integer boundaries.
    while(borrow) {
        uint64_t c=0;
        for(uint32_t j=0;j<wide;++j) {
            uint64_t v=uint64_t(scratch[size_t(j)*n+i])+product[j]+c;
            scratch[size_t(j)*n+i]=uint32_t(v); c=v>>32;
        }
        if(c) borrow=0;
    }
    while(compare_wide(scratch,product,n,wide,i)>=0) subtract_wide(scratch,product,n,wide,i);
    bool negative=compare_wide(scratch,half_product,n,wide,i)>0;
    if(negative) {
        uint64_t b=0;
        for(uint32_t j=0;j<wide;++j) {
            uint64_t sub=uint64_t(scratch[size_t(j)*n+i])+b;
            scratch[size_t(j)*n+i]=product[j]-uint32_t(sub); b=(sub>>32)+(product[j]<uint32_t(sub));
        }
    }
    uint32_t whole=width/32,bits=width%32;
    if(gap) {
        // Exact reduction using q=2^width-gap, derived from the supplied q.
        while(true) {
            bool high=false;
            for(uint32_t j=whole;j<wide;++j) {
                uint32_t v=scratch[size_t(j)*n+i]; if(j==whole&&bits) v>>=bits; high|=v!=0;
            }
            if(!high) break;
            uint64_t c=0;
            for(uint32_t j=0;j<wide;++j) {
                uint32_t low=scratch[size_t(j)*n+i];
                if(j>whole||(j==whole&&!bits)) low=0;
                else if(j==whole) low&=(uint32_t(1)<<bits)-1;
                uint32_t h=0;
                if(j+whole<wide) h=scratch[size_t(j+whole)*n+i]>>bits;
                if(bits&&j+whole+1<wide) h|=scratch[size_t(j+whole+1)*n+i]<<(32-bits);
                uint64_t v=uint64_t(h)*gap+c+low;
                scratch[size_t(j)*n+i]=uint32_t(v); c=v>>32;
            }
        }
    } else {
        // General-modulus fallback, without a small-gap assumption.
        for(uint32_t j=0;j<limbs;++j) out[size_t(i)*limbs+j]=0;
        for(uint64_t bit=uint64_t(wide)*32;bit-->0;) {
            uint32_t c=(scratch[(bit/32)*n+i]>>(bit%32))&1;
            for(uint32_t j=0;j<limbs;++j) {
                uint64_t v=uint64_t(out[size_t(i)*limbs+j])*2+c;
                out[size_t(i)*limbs+j]=uint32_t(v); c=v>>32;
            }
            int cmp=c?1:0;
            if(!c) for(uint32_t j=limbs;j-->0;) {
                uint32_t v=out[size_t(i)*limbs+j]; if(v!=q[j]) { cmp=v<q[j]?-1:1; break; }
            }
            if(cmp>=0) {
                uint64_t b=0;
                for(uint32_t j=0;j<limbs;++j) {
                    uint32_t v=out[size_t(i)*limbs+j]; uint64_t sub=uint64_t(q[j])+b;
                    out[size_t(i)*limbs+j]=v-uint32_t(sub); b=(sub>>32)+(v<uint32_t(sub));
                }
            }
        }
        for(uint32_t j=0;j<wide;++j) scratch[size_t(j)*n+i]=j<limbs?out[size_t(i)*limbs+j]:0;
    }
    while(compare_wide(scratch,q,n,wide,i)>=0) subtract_wide(scratch,q,n,wide,i);
    bool nonzero=false;
    for(uint32_t j=0;j<limbs;++j) nonzero|=scratch[size_t(j)*n+i]!=0;
    uint64_t b=0;
    for(uint32_t j=0;j<limbs;++j) {
        uint32_t v=scratch[size_t(j)*n+i];
        if(negative&&nonzero) { uint64_t sub=uint64_t(v)+b; v=q[j]-uint32_t(sub); b=(sub>>32)+(q[j]<uint32_t(sub)); }
        out[size_t(i)*limbs+j]=v;
    }
}
} // namespace

void initialize_graphs(State& s);

void* fherma_init(const fherma::Point& point) {
    using namespace host_big;
    auto s=std::make_unique<State>(); s->n=point.N; s->limbs=point.L; s->width=point.W;
    if(!s->n||(s->n&(s->n-1))||s->n>=(1u<<29)||s->limbs!=(s->width+31)/32)
        throw std::runtime_error("invalid point or transform exceeds available 31-bit fields");
    Big q=point.q.data; trim(q); Big maximum=subtract(q,Big{1});
    Big bound=times_word(multiply(maximum,maximum),4*s->n),product{1};
    std::vector<uint32_t> primes;
    uint64_t step=2*uint64_t(s->n),candidate=((uint64_t(1)<<31)-2)/step*step+1;
    while(compare(product,bound)<=0) {
        if(candidate<(1u<<30)) throw std::runtime_error("insufficient independent NTT fields");
        if(is_prime(uint32_t(candidate))) { primes.push_back(candidate); product=times_word(product,candidate); }
        candidate-=step;
    }
    s->primes=primes.size(); s->wide=product.size()+1;
    s->padded_bytes=(s->limbs*4+31)/32*32;
    Big half=divide_word(product,2),power_two(s->width/32+1,0); power_two[s->width/32]=1u<<(s->width%32);
    Big gap=subtract(power_two,q); s->gap=gap.size()==1?gap[0]:0;
    Big full_product=product,correction;
    s->direct_crt=s->gap&&s->limbs+2<=64&&s->n>=128&&s->primes<8192;
    if(s->direct_crt) {
        s->wide=s->limbs+2;
        correction=subtract(q,modulo(full_product,q));
        if(compare(correction,q)==0) correction={0};
        correction.resize(s->wide);
    }
    product.resize(s->wide); half.resize(s->wide); q.resize(s->wide);
    std::vector<uint64_t> fixed_reciprocals;
    std::vector<Prime> fields;
    std::vector<uint32_t> crt(size_t(s->wide)*s->primes);
    std::vector<uint32_t> twist(size_t(s->primes)*s->n),finish(twist.size());
    std::vector<uint32_t> roots(size_t(s->primes)*(s->n/2)),inverse_roots(roots.size());
    uint32_t padded_primes=(s->primes+15)/16*16;
    std::vector<uint8_t> byte_weights(size_t(padded_primes)*4*s->padded_bytes,0);
    for(uint32_t k=0;k<s->primes;++k) {
        uint32_t p=primes[k],inv=1;
        for(int j=0;j<5;++j) inv*=2-p*inv;
        uint32_t radix=(uint64_t(1)<<32)%p;
        fields.push_back({p,inv,radix,uint32_t((uint64_t(radix)<<32)/p),uint32_t((uint64_t(1)<<32)/p),1.0/p});
        uint32_t byte_power=1;
        for(uint32_t j=0;j<s->limbs*4;++j) {
            for(uint32_t digit=0;digit<4;++digit) byte_weights[(size_t(k)*4+digit)*s->padded_bytes+j]=(byte_power>>(digit*8))&255;
            byte_power=uint64_t(byte_power)*256%p;
        }
        Big partial=divide_word(full_product,p);
        uint32_t crt_inverse=power(modulo_word(partial,p),p-2,p);
        if(s->direct_crt) partial=modulo(partial,q);
        partial.resize(s->wide);
        fixed_reciprocals.push_back(UINT64_MAX/p);
        for(uint32_t j=0;j<s->wide;++j) crt[size_t(j)*s->primes+k]=partial[j];
        uint32_t psi=0;
        for(uint32_t base=2;!psi;++base) { uint32_t root=power(base,(p-1)/step,p); if(power(root,s->n,p)==p-1) psi=root; }
        uint32_t psi_inv=power(psi,p-2,p),omega=uint64_t(psi)*psi%p,omega_inv=power(omega,p-2,p);
        uint32_t f=uint64_t(radix)*radix%p,g=uint64_t(power(s->n,p-2,p))*crt_inverse%p,wr=radix,wi=radix;
        if(s->n>=128&&s->padded_bytes<=32768)f=uint64_t(f)*radix%p;
        for(uint32_t i=0;i<s->n;++i) {
            twist[size_t(k)*s->n+i]=f; finish[size_t(k)*s->n+i]=g;
            f=uint64_t(f)*psi%p; g=uint64_t(g)*psi_inv%p;
            if(i<s->n/2) {
                roots[size_t(k)*(s->n/2)+i]=wr; inverse_roots[size_t(k)*(s->n/2)+i]=wi;
                wr=uint64_t(wr)*omega%p; wi=uint64_t(wi)*omega_inv%p;
            }
        }
    }
    s->padded_r=(s->primes+31)/32*32;
    s->crt_columns=s->wide+((TENSOR_QUOTIENT&&s->direct_crt&&s->n==65536&&s->limbs==55)?2:0);
    if(CRT_ALIGNED_DOTS&&s->crt_columns>s->wide)s->crt_columns=64;
    if(s->n>=128&&s->padded_r*4<=32768) {
        uint32_t columns=(s->crt_columns+15)/16*64,k_bytes=s->padded_r*4;
        std::vector<uint8_t> weights(size_t(columns)*k_bytes,0);
        for(uint32_t col=0;col<s->wide*4;++col)
            for(uint32_t k=0;k<s->primes;++k)
                for(uint32_t digit=0;digit<4;++digit) {
                    if(col<digit) continue;
                    uint32_t at=col-digit;
                    uint32_t value=crt[size_t(at/4)*s->primes+k];
                    weights[size_t(col)*k_bytes+k*4+digit]=(value>>(8*(at%4)))&255;
                }
        if(s->crt_columns>s->wide){
            // Byte convolution yields D0 + 2^32*D1 = sum(r_k*floor(2^64/p_k)).
            // The two unused logical columns fit in the existing 64-column tile.
            for(unsigned byte=0;byte<8;++byte)for(unsigned k=0;k<s->primes;++k)
                for(unsigned digit=0;digit<4;++digit){
                    if(byte<digit)continue;
                    const unsigned position=byte-digit;
                    weights[size_t(s->wide*4+byte)*k_bytes+k*4+digit]=position<8?uint8_t(fixed_reciprocals[k]>>(position*8)):0;
                }
        }
        upload(s->crt_weights,weights);
        allocate(s->crt_inputs,size_t(s->n)*s->padded_r);
        // The padded field lanes are parameter-only zeros. Fused producers
        // overwrite every real field on every call and leave these lanes zero.
        if(FUSION_SKIP_PADDING)check(cudaMemset(s->crt_inputs,0,size_t(s->n)*s->padded_r*sizeof(uint32_t)),"initialize constant CRT padding");
        allocate(s->crt_dots,size_t(s->n)*s->crt_columns);
        int device=0,major=0;check(cudaGetDevice(&device),"query TMA device");
        check(cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,device),"query TMA architecture");
        s->crt_tma=CRT_USE_TMA&&s->n==65536&&major>=9;
        if(s->crt_tma){
            auto make_map=[&](CUtensorMap& map,void* pointer,uint64_t rows,unsigned tile_rows){
                const uint64_t dimensions[2]={uint64_t(s->padded_r)*4,rows};
                const uint64_t strides[1]={uint64_t(s->padded_r)*4};
                const uint32_t box[2]={CRT_BYTES,tile_rows},elements[2]={1,1};
                const auto swizzle=CRT_BYTES==128?CU_TENSOR_MAP_SWIZZLE_128B:CU_TENSOR_MAP_SWIZZLE_64B;
                auto result=cuTensorMapEncodeTiled(&map,CU_TENSOR_MAP_DATA_TYPE_UINT8,2,pointer,dimensions,strides,box,elements,
                    CU_TENSOR_MAP_INTERLEAVE_NONE,swizzle,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
                if(result!=CUDA_SUCCESS)throw std::runtime_error("CRT tensor map encoding failed: "+std::to_string(int(result)));
            };
            make_map(s->crt_input_map,s->crt_inputs,s->n,CRT_ROWS);
            make_map(s->crt_weight_map,s->crt_weights,columns,CRT_WORDS*4);
        }
        std::fprintf(stderr,"crt_tma:enabled=%u rows=%u words=%u bytes=%u stride=%u\n",s->crt_tma,CRT_ROWS,CRT_WORDS,CRT_BYTES,s->crt_columns);
    }
    if(s->direct_crt) upload(s->correction_mod_q,correction);
    upload(s->fixed_reciprocals,fixed_reciprocals);
    upload(s->fields,fields); upload(s->crt,crt); upload(s->product,product); upload(s->half_product,half); upload(s->q,q);
    upload(s->twist,twist); upload(s->finish,finish);
    if(!roots.empty()) { upload(s->roots,roots); upload(s->inverse_roots,inverse_roots); }
    if(s->n>=32768&&s->n<=65536) {
        std::vector<uint32_t> compact(size_t(s->primes)*256),compact_inverse(compact.size()),quotients(compact.size()),inverse_quotients(compact.size());
        for(uint32_t prime=0;prime<s->primes;++prime){
            const uint32_t p=fields[prime].p,inverse_radix=power(fields[prime].radix,p-2,p);
            for(uint32_t i=0;i<256;++i){
                const size_t at=size_t(prime)*256+i,original=size_t(prime)*(s->n/2)+i*(s->n/512);
                compact[at]=uint64_t(roots[original])*inverse_radix%p;
                compact_inverse[at]=uint64_t(inverse_roots[original])*inverse_radix%p;
                quotients[at]=(uint64_t(compact[at])<<32)/p;
                inverse_quotients[at]=(uint64_t(compact_inverse[at])<<32)/p;
            }
        }
        upload(s->middle_roots,compact);upload(s->middle_inverse_roots,compact_inverse);
        upload(s->middle_quotients,quotients);upload(s->middle_inverse_quotients,inverse_quotients);
    }

    allocate(s->a,size_t(s->n)*s->limbs*(MERGED_INPUTS?2:1)); allocate(s->b,size_t(s->n)*s->limbs); allocate(s->answer,size_t(s->n)*s->limbs);
    allocate(s->residues,size_t(s->n)*s->primes*2); allocate(s->scratch,size_t(s->n)*s->wide);
    upload(s->byte_weights,byte_weights); allocate(s->byte_inputs,size_t(s->n)*s->padded_bytes*2);
    s->pipelined=s->direct_crt&&s->n>=32768&&s->n<=65536&&s->padded_bytes<=32768;
    if(s->pipelined) {
        // Stage-major twiddles keep adjacent columns adjacent in memory.
        // These tables depend only on public parameters, never the operands.
        const uint32_t rows=s->n/NTT_TILE,table_words=s->n-NTT_TILE;
        std::vector<HighRoot> forward_pairs(size_t(s->primes)*table_words),inverse_pairs(forward_pairs.size());
        for(uint32_t prime=0;prime<s->primes;++prime){
            const uint32_t p=fields[prime].p,inverse_radix=power(fields[prime].radix,p-2,p);
            for(uint32_t length=rows;length>=2;length/=2)
                for(uint32_t j=0;j<length/2;++j)
                    for(uint32_t col=0;col<NTT_TILE;++col){
                        const size_t at=size_t(prime)*table_words+(rows-length)*NTT_TILE+j*NTT_TILE+col;
                        const size_t original=size_t(prime)*(s->n/2)+(j*NTT_TILE+col)*(rows/length);
                        uint32_t r=uint64_t(roots[original])*inverse_radix%p,ri=uint64_t(inverse_roots[original])*inverse_radix%p;
                        forward_pairs[at]=pack_high_root(r,p);
                        inverse_pairs[at]=pack_high_root(ri,p);
                    }
        }
        upload(s->high_roots,forward_pairs);upload(s->high_inverse_roots,inverse_pairs);

        check(cudaMallocHost(&s->startup_host,4096),"pin startup constants");
        std::memset(s->startup_host,0,4096);
        allocate(s->startup_device,1024);
        size_t bytes=size_t(s->n)*s->limbs*sizeof(uint32_t);
        check(cudaHostAlloc(reinterpret_cast<void**>(&s->host_inputs),bytes*2,cudaHostAllocWriteCombined|cudaHostAllocMapped),"pin write-combined input staging");
        check(cudaHostGetDevicePointer(&s->device_host_inputs,s->host_inputs,0),"map input staging");
        s->host_team=std::make_unique<StreamingTeam>(CpuPlacement::get().copy_lanes(HOST_COPY_LANES)+1);
        check(cudaHostAlloc(reinterpret_cast<void**>(&s->host_flags),48*64,cudaHostAllocDefault),"pin CPU readiness flags");
        std::memset(s->host_flags,0,48*64);
        check(cudaMallocHost(&s->host_output,bytes),"pin output staging");
        for(auto& stream:s->transfer_streams) check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"create transfer stream");
        if(SPLIT_INPUT_DMA&&s->n==65536){
            check(cudaStreamCreateWithFlags(&s->input_copy_stream,cudaStreamNonBlocking),"create ordered input DMA stream");
            for(auto& event:s->copied_inputs)check(cudaEventCreateWithFlags(&event,cudaEventDisableTiming),"create input copy event");
        }
        check(cudaStreamCreateWithFlags(&s->compute_stream,cudaStreamNonBlocking),"create compute stream");
        for(auto& event:s->inputs_ready) check(cudaEventCreateWithFlags(&event,cudaEventDisableTiming),"create input event");
        for(auto& event:s->outputs_ready) check(cudaEventCreateWithFlags(&event,cudaEventDisableTiming),"create output event");
    }
    if(s->pipelined&&TAIL_FUSION==1&&s->n==65536){
        int available=0;check(cudaDeviceGetAttribute(&available,cudaDevAttrClusterLaunch,0),"query cluster inverse support");
        s->cluster_inverse=available!=0;
    }
    if(s->pipelined) initialize_graphs(*s);
    return s.release();
}
// Capture parameter-only execution plans. Capture records operations without
// executing them; every input copy and arithmetic kernel runs in fherma_run.
template<class Emit>
cudaGraph_t capture_plan(cudaStream_t stream,Emit emit) {
    check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal),"begin execution-plan capture");
    try {emit();check(cudaGetLastError(),"captured kernels");}
    catch(...){
        cudaGraph_t abandoned=nullptr;cudaStreamEndCapture(stream,&abandoned);
        if(abandoned)cudaGraphDestroy(abandoned);
        throw;
    }
    cudaGraph_t graph=nullptr;
    check(cudaStreamEndCapture(stream,&graph),"finish execution-plan capture");
    return graph;
}
cudaError_t captured_edges(cudaGraph_t graph,cudaGraphNode_t* from,cudaGraphNode_t* to,size_t* count){
#if CUDART_VERSION >= 13000
    return cudaGraphGetEdges(graph,from,to,nullptr,count);
#else
    return cudaGraphGetEdges(graph,from,to,count);
#endif
}
// Captured plans here contain one linear chain of kernel nodes only.
// Clone the chain into the parent graph and return its terminal node.
cudaGraphNode_t append_kernel_chain(cudaGraph_t parent,cudaGraph_t plan,cudaGraphNode_t previous,cudaGraphNode_t* stage_previous=nullptr){
    size_t count=0,edges=0;
    check(cudaGraphGetNodes(plan,nullptr,&count),"count captured kernels");
    check(captured_edges(plan,nullptr,nullptr,&edges),"count captured edges");
    if(!count||edges+1!=count)throw std::runtime_error("captured plan is not a linear chain");
    std::vector<cudaGraphNode_t> nodes(count),from(edges),to(edges);
    check(cudaGraphGetNodes(plan,nodes.data(),&count),"read captured kernels");
    if(edges)check(captured_edges(plan,from.data(),to.data(),&edges),"read captured edges");
    cudaGraphNode_t current=nullptr;
    for(auto node:nodes){
        bool incoming=false;for(auto target:to)incoming|=target==node;
        if(!incoming){if(current)throw std::runtime_error("multiple captured roots");current=node;}
    }
    for(size_t i=0;i<count;++i){
        if(!current)throw std::runtime_error("broken captured chain");
        cudaGraphNodeType type;check(cudaGraphNodeGetType(current,&type),"read captured type");
        if(type!=cudaGraphNodeTypeKernel)throw std::runtime_error("unexpected captured non-kernel node");
        cudaKernelNodeParams parameters{};
        check(cudaGraphKernelNodeGetParams(current,&parameters),"read captured kernel parameters");
        cudaGraphNode_t next=nullptr;
        cudaGraphNode_t dependencies[2]{};size_t dependency_count=0;
        if(previous)dependencies[dependency_count++]=previous;
        if(stage_previous){
            if(count!=4&&count!=3)throw std::runtime_error("output plan needs four arithmetic stages");
            if((OUTPUT_STAGE_MASK&(1u<<i))&&stage_previous[i])dependencies[dependency_count++]=stage_previous[i];
        }
        check(cudaGraphAddKernelNode(&next,parent,dependencies,dependency_count,&parameters),"append pipelined kernel");
        cudaKernelNodeAttrValue cluster{};
        check(cudaGraphKernelNodeGetAttribute(current,cudaLaunchAttributeClusterDimension,&cluster),"read cluster shape");
        if(cluster.clusterDim.x>1)check(cudaGraphKernelNodeSetAttribute(next,cudaLaunchAttributeClusterDimension,&cluster),"copy cluster shape");
        if(stage_previous)stage_previous[i]=next;
        previous=next;next=nullptr;
        for(size_t edge=0;edge<edges;++edge)if(from[edge]==current){
            if(next)throw std::runtime_error("branched captured chain");next=to[edge];
        }
        current=next;
    }
    if(current)throw std::runtime_error("cyclic captured chain");
    return previous;
}

void initialize_graphs(State& s) {
    const unsigned chunks=16;
    const size_t count=size_t(s.n)*s.limbs,chunk_rows=s.n/chunks;
    const size_t chunk_words=chunk_rows*s.limbs,chunk_bytes=chunk_words*sizeof(uint32_t);
    auto stream=s.transfer_streams[0];
    for(unsigned chunk=0;chunk<(s.n==65536&&2*OPERAND_STRIPS>16?2*OPERAND_STRIPS:16);++chunk) {
        size_t offset=chunk*chunk_words;
        if(s.n==65536){
        if(chunk<2*OPERAND_STRIPS){
        const unsigned batch=chunk/OPERAND_STRIPS,part=chunk%OPERAND_STRIPS;
        const unsigned columns=operand_columns(part),column_base=operand_base(part);
        const size_t input_rows=size_t(columns)*(s.n/512),input_words=input_rows*s.limbs;
        const size_t input_offset=(size_t(batch)*s.n+size_t(column_base)*(s.n/512))*s.limbs;
        auto* input=s.a+input_offset;
        auto* residues=s.residues+size_t(batch)*s.primes*s.n;
        if(SPLIT_INPUT_DMA){
            auto copies=capture_plan(s.input_copy_stream,[&]{
                check(cudaMemcpyAsync(input,s.host_inputs+input_offset,input_words*sizeof(uint32_t),cudaMemcpyHostToDevice,s.input_copy_stream),"stage ordered input strip");
                check(cudaEventRecordWithFlags(s.copied_inputs[chunk],s.input_copy_stream,cudaEventRecordExternal),"publish captured input copy");
            });
            const auto status=cudaGraphInstantiate(&s.copy_graphs[chunk],copies,0);
            cudaGraphDestroy(copies);check(status,"instantiate input copy graph");
        }
        auto input_plan=capture_plan(stream,[&] {
            if(SPLIT_INPUT_DMA)check(cudaStreamWaitEvent(stream,s.copied_inputs[chunk],cudaEventWaitExternal),"wait captured input copy");
            else check(cudaMemcpyAsync(input,s.host_inputs+input_offset,input_words*sizeof(uint32_t),cudaMemcpyHostToDevice,stream),"stage operand strip");
            if(s.limbs==55)tensor_product<true,55><<<dim3((input_rows+63)/64,(s.primes+15)/16),128,0,stream>>>(reinterpret_cast<const uint8_t*>(input),s.byte_weights,residues,nullptr,s.fields,s.twist,s.n,s.primes,s.padded_bytes,column_base*(s.n/512),6,s.limbs,nullptr,column_base*(s.n/512));
            else tensor_product<true,28><<<dim3((input_rows+63)/64,(s.primes+15)/16),128,0,stream>>>(reinterpret_cast<const uint8_t*>(input),s.byte_weights,residues,nullptr,s.fields,s.twist,s.n,s.primes,s.padded_bytes,column_base*(s.n/512),6,s.limbs,nullptr,column_base*(s.n/512));
            if(s.n/NTT_TILE==128)high_stages_register<false,128><<<dim3(columns/32,s.primes),256,0,stream>>>(residues,s.fields,s.high_roots,s.finish,s.n,column_base);
            else high_stages_register<false,64><<<dim3(columns/32,s.primes),256,0,stream>>>(residues,s.fields,s.high_roots,s.finish,s.n,column_base);
        });
        auto status=cudaGraphInstantiate(&s.input_graphs[chunk],input_plan,0);
        cudaGraphDestroy(input_plan);check(status,"instantiate operand input plan");
        }
        }else{
        auto input_plan=capture_plan(stream,[&] {
            if(RESIDENT_DEVICE_INPUT&&(s.limbs==55||s.limbs==28))check(cudaMemcpyAsync(s.a+2*offset,s.host_inputs+2*offset,2*chunk_bytes,cudaMemcpyHostToDevice,stream),"stage resident diagnostic input");
            auto* resident_input=RESIDENT_DEVICE_INPUT?s.a:s.device_host_inputs;
            if(s.n==32768&&s.limbs==28)resident_input_encode<28><<<dim3(chunk_rows/RESIDENT_ROWS,1,2),128*8,0,stream>>>(resident_input+2*offset,resident_input+2*offset+chunk_words,s.byte_weights,s.residues,s.fields,s.twist,s.n,s.primes,chunk*chunk_rows);
            else {
            if(MERGED_INPUTS)check(cudaMemcpyAsync(s.a+2*offset,s.host_inputs+2*offset,2*chunk_bytes,cudaMemcpyHostToDevice,stream),"stage both input chunks");
            else {
            check(cudaMemcpyAsync(s.a+offset,s.host_inputs+offset,chunk_bytes,cudaMemcpyHostToDevice,stream),"stage a chunk");
            check(cudaMemcpyAsync(s.b+offset,s.host_inputs+count+offset,chunk_bytes,cudaMemcpyHostToDevice,stream),"stage b chunk");
            }
            if(s.limbs==55) tensor_product<true,55><<<dim3((chunk_rows+63)/64,(s.primes+15)/16,2),128,0,stream>>>(reinterpret_cast<const uint8_t*>(MERGED_INPUTS?s.a+2*offset:s.a),s.byte_weights,s.residues,nullptr,s.fields,s.twist,s.n,s.primes,s.padded_bytes,chunk*chunk_rows,5,s.limbs,reinterpret_cast<const uint8_t*>(MERGED_INPUTS?s.a+2*offset+chunk_words:s.b),MERGED_INPUTS?chunk*chunk_rows:0);
            else if(s.limbs==28) tensor_product<true,28><<<dim3((chunk_rows+63)/64,(s.primes+15)/16,2),128,0,stream>>>(reinterpret_cast<const uint8_t*>(MERGED_INPUTS?s.a+2*offset:s.a),s.byte_weights,s.residues,nullptr,s.fields,s.twist,s.n,s.primes,s.padded_bytes,chunk*chunk_rows,5,s.limbs,reinterpret_cast<const uint8_t*>(MERGED_INPUTS?s.a+2*offset+chunk_words:s.b),MERGED_INPUTS?chunk*chunk_rows:0);
            else tensor_product<true><<<dim3((chunk_rows+63)/64,(s.primes+15)/16,2),128,0,stream>>>(reinterpret_cast<const uint8_t*>(MERGED_INPUTS?s.a+2*offset:s.a),s.byte_weights,s.residues,nullptr,s.fields,s.twist,s.n,s.primes,s.padded_bytes,chunk*chunk_rows,5,s.limbs,reinterpret_cast<const uint8_t*>(MERGED_INPUTS?s.a+2*offset+chunk_words:s.b),MERGED_INPUTS?chunk*chunk_rows:0);
            }
            if(s.n/NTT_TILE==128) high_stages_register<false,128><<<dim3(1,s.primes,2),256,0,stream>>>(s.residues,s.fields,s.high_roots,s.finish,s.n,chunk*32);
            else high_stages_register<false,64><<<dim3(1,s.primes,2),256,0,stream>>>(s.residues,s.fields,s.high_roots,s.finish,s.n,chunk*32);
        });
        auto status=cudaGraphInstantiate(&s.input_graphs[chunk],input_plan,0);
        cudaGraphDestroy(input_plan);check(status,"instantiate input graph");
        }
        if(chunk<16){
        uint32_t first=chunk*chunk_rows;
        s.output_plans[chunk]=capture_plan(stream,[&] {
            if(s.cluster_inverse){
                cudaLaunchAttribute attr{};attr.id=cudaLaunchAttributeClusterDimension;
                attr.val.clusterDim.x=4;attr.val.clusterDim.y=1;attr.val.clusterDim.z=1;
                cudaLaunchConfig_t config{};config.gridDim=dim3(4,FUSION_SKIP_PADDING?(s.primes+3)/4:s.padded_r/4,2);
                config.blockDim=dim3(128);config.stream=stream;config.attrs=&attr;config.numAttrs=1;
                check(cudaLaunchKernelEx(&config,inverse_transpose_cluster<128,16,4>,s.residues,s.fields,s.high_inverse_roots,s.finish,s.n,chunk*32,s.crt_inputs,s.padded_r,s.primes),"capture fused inverse transpose");
            }else if(TAIL_FUSION==2&&s.n==65536){
                inverse_transpose_shared<128,16,4><<<dim3(2,FUSION_SKIP_PADDING?(s.primes+3)/4:s.padded_r/4),512,0,stream>>>(s.residues,s.fields,s.high_inverse_roots,s.finish,s.n,chunk*32,s.crt_inputs,s.padded_r,s.primes);
            }else{
            if(s.n/NTT_TILE==128) high_stages_register<true,128><<<dim3(1,s.primes),256,0,stream>>>(s.residues,s.fields,s.high_inverse_roots,s.finish,s.n,chunk*32);
            else high_stages_register<true,64><<<dim3(1,s.primes),256,0,stream>>>(s.residues,s.fields,s.high_inverse_roots,s.finish,s.n,chunk*32);
            transpose_streamed_residues<<<dim3((chunk_rows+31)/32,s.padded_r/32),dim3(32,8),0,stream>>>(s.residues,s.crt_inputs,s.n,s.primes,s.padded_r,first,5);
            }
            if(s.crt_tma)tensor_crt_tma<CRT_ROWS,CRT_WORDS,CRT_BYTES><<<dim3((chunk_rows+CRT_ROWS-1)/CRT_ROWS,(s.crt_columns+CRT_WORDS-1)/CRT_WORDS),CRT_WORDS*8,0,stream>>>(s.crt_input_map,s.crt_weight_map,reinterpret_cast<const uint8_t*>(s.crt_inputs),s.crt_weights,nullptr,s.crt_dots,nullptr,nullptr,s.n,s.crt_columns,s.padded_r*4,first);
            else if(s.n==65536)tensor_crt_geometry<CRT_ROWS,CRT_WORDS,CRT_BYTES><<<dim3((chunk_rows+CRT_ROWS-1)/CRT_ROWS,(s.crt_columns+CRT_WORDS-1)/CRT_WORDS),CRT_WORDS*8,0,stream>>>(reinterpret_cast<const uint8_t*>(s.crt_inputs),s.crt_weights,nullptr,s.crt_dots,nullptr,nullptr,s.n,s.crt_columns,s.padded_r*4,first);
            else tensor_crt_async<><<<dim3((chunk_rows+63)/64,(s.crt_columns+15)/16),128,0,stream>>>(reinterpret_cast<const uint8_t*>(s.crt_inputs),s.crt_weights,nullptr,s.crt_dots,nullptr,nullptr,s.n,s.crt_columns,s.padded_r*4,first);
            if(s.wide<=32) reconstruct_direct<1><<<(chunk_rows+3)/4,128,0,stream>>>(s.crt_inputs,s.fixed_reciprocals,s.correction_mod_q,s.primes,s.padded_r,s.q,s.answer,s.crt_dots,s.n,s.wide,s.limbs,s.width,s.gap,first,s.crt_columns);
            else reconstruct_direct<2><<<(chunk_rows+3)/4,128,0,stream>>>(s.crt_inputs,s.fixed_reciprocals,s.correction_mod_q,s.primes,s.padded_r,s.q,s.answer,s.crt_dots,s.n,s.wide,s.limbs,s.width,s.gap,first,s.crt_columns);


        });
        }
    }
    auto core=s.compute_stream;
    if(s.n==65536){
    auto first_plan=capture_plan(core,[&] {
        middle_operand_stage<true><<<dim3((s.n/NTT_TILE+3)/4,s.primes),128,0,core>>>(s.residues,s.fields,s.middle_roots,s.middle_inverse_roots,s.middle_quotients,s.middle_inverse_quotients,s.n);
    });
    auto status=cudaGraphInstantiate(&s.first_operand_graph,first_plan,0);
    cudaGraphDestroy(first_plan);check(status,"instantiate early operand transform");
    check(cudaGraphUpload(s.first_operand_graph,core),"upload early operand plan");
    }
    s.core_plan=capture_plan(core,[&] {
        if(s.n==65536)middle_operand_stage<false><<<dim3((s.n/NTT_TILE+3)/4,s.primes),128,0,core>>>(s.residues,s.fields,s.middle_roots,s.middle_inverse_roots,s.middle_quotients,s.middle_inverse_quotients,s.n);
        else middle_stages_register<<<dim3((s.n/NTT_TILE+3)/4,s.primes),128,0,core>>>(s.residues,s.fields,s.middle_roots,s.middle_inverse_roots,s.middle_quotients,s.middle_inverse_quotients,s.n);
    });
    // All tail operations have known dependencies once the input streams join.
    // Native graph event nodes expose each completed output strip to the CPU.
    cudaGraph_t graph=nullptr;check(cudaGraphCreate(&graph,0),"create tail graph");
    try{
        cudaGraphNode_t middle{},outputs[16]{},copies[16]{},events[16]{};
        middle=append_kernel_chain(graph,s.core_plan,nullptr);
        cudaGraphNode_t stage_previous[4]{};
        for(unsigned chunk=0;chunk<16;++chunk){
            // Serialize matching phases, allowing inverse NTT, CRT and final
            // reduction from different strips to overlap without synchronized bursts.
            // Main-point inverse/transpose fusion favors one arithmetic strip at
            // a time. D2H copies still overlap the next strip's arithmetic.
            // The smaller point retains its two independent compute chains.
            const unsigned chains=s.n==65536?1:OUTPUT_CHAINS;
            auto previous=(s.n==65536&&OUTPUT_STAGE_MASK)?middle:(chunk>=chains?outputs[chunk-chains]:middle);
            outputs[chunk]=append_kernel_chain(graph,s.output_plans[chunk],previous,(s.n==65536&&OUTPUT_STAGE_MASK)?stage_previous:nullptr);
            const size_t offset=chunk*chunk_words;
            cudaGraphNode_t dependencies[2]={outputs[chunk],chunk?copies[chunk-1]:outputs[chunk]};
            check(cudaGraphAddMemcpyNode1D(&copies[chunk],graph,dependencies,chunk?2:1,s.host_output+offset,s.answer+offset,chunk_bytes,cudaMemcpyDeviceToHost),"add ordered answer transfer");
            check(cudaGraphAddEventRecordNode(&events[chunk],graph,&copies[chunk],1,s.outputs_ready[chunk]),"record output event in graph");
        }
        check(cudaGraphInstantiate(&s.pipeline_graph,graph,0),"instantiate tail graph");
        check(cudaGraphUpload(s.pipeline_graph,core),"upload tail plan");
        check(cudaStreamSynchronize(core),"finish tail plan upload");
    }catch(...){cudaGraphDestroy(graph);throw;}
    cudaGraphDestroy(graph);
}

// Each input lane owns disjoint rows. A strip becomes visible
// to the GPU only after every lane has fenced its non-temporal stores.
struct StreamingCall {
    struct alignas(64) Counter {std::atomic<unsigned> left{0};};
    const State& state;
    const fherma::Inputs& input;
    fherma::Outputs& output;
    Counter input_left[32];
    std::atomic<bool> output_ready{false},cancelled{false};
    static void pause(){
#if defined(__x86_64__)
        _mm_pause();
#else
        std::this_thread::yield();
#endif
    }
    void pack_operand(unsigned lane,unsigned lanes) noexcept {
        const auto& s=state;
        const size_t rows=s.n/NTT_TILE,first=rows*lane/lanes,last=rows*(lane+1)/lanes;
        const size_t count=size_t(s.n)*s.limbs;
        for(unsigned slot=0;slot<2*OPERAND_STRIPS;++slot){
            if(cancelled.load(std::memory_order_acquire))return;
            const unsigned batch=slot/OPERAND_STRIPS,chunk=slot%OPERAND_STRIPS;
            const size_t row_words=operand_columns(chunk)*s.limbs,chunk_words=rows*row_words;
            const size_t offset=size_t(operand_base(chunk))*rows*s.limbs;
            auto* source=(batch?input.b.data.data():input.a.data.data())+operand_base(chunk)*s.limbs;
            auto* destination=s.host_inputs+batch*count+offset;
            for(size_t row=first;row<last;++row){
                auto* to=destination+row*row_words;
                auto* from=source+row*NTT_TILE*s.limbs;
#if defined(__x86_64__) && defined(__GNUC__)
                if(__builtin_cpu_supports("avx512f"))copy_512(to,from,row_words*sizeof(uint32_t),true);
                else
#endif
                    std::memcpy(to,from,row_words*sizeof(uint32_t));
            }
#if defined(__x86_64__) && defined(__GNUC__)
            _mm_sfence();
#endif
            if(input_left[slot].left.fetch_sub(1,std::memory_order_acq_rel)==1)
                __atomic_store_n(s.host_flags+slot*16,1u,__ATOMIC_RELEASE);
        }
    }
    void pack(unsigned lane,unsigned lanes) noexcept {
        if(state.n==65536){pack_operand(lane,lanes);return;}
        const auto& s=state;
        const size_t rows=s.n/NTT_TILE,first=rows*lane/lanes,last=rows*(lane+1)/lanes;
        const size_t row_words=32*s.limbs,chunk_words=(s.n/16)*s.limbs,count=size_t(s.n)*s.limbs;
        for(unsigned chunk=0;chunk<16;++chunk){
            if(cancelled.load(std::memory_order_acquire))return;
            for(unsigned batch=0;batch<2;++batch){
                auto* source=(batch?input.b.data.data():input.a.data.data())+chunk*row_words;
                auto* destination=s.host_inputs+(MERGED_INPUTS?(2*chunk+batch)*chunk_words:batch*count+chunk*chunk_words);
                for(size_t row=first;row<last;++row){
                    auto* to=destination+row*row_words;
                    auto* from=source+row*NTT_TILE*s.limbs;
#if defined(__x86_64__) && defined(__GNUC__)
                    if(__builtin_cpu_supports("avx512f"))copy_512(to,from,row_words*sizeof(uint32_t),true);
                    else
#endif
                        std::memcpy(to,from,row_words*sizeof(uint32_t));
                }
            }
#if defined(__x86_64__) && defined(__GNUC__)
            _mm_sfence();
#endif
            if(input_left[chunk].left.fetch_sub(1,std::memory_order_acq_rel)==1)
                __atomic_store_n(s.host_flags+chunk*16,1u,__ATOMIC_RELEASE);
        }
    }
    void scatter_chunk(unsigned chunk,unsigned lane,unsigned lanes) noexcept {
        const auto& s=state;
        const size_t rows=s.n/NTT_TILE,first=rows*lane/lanes,last=rows*(lane+1)/lanes;
        const size_t row_words=32*s.limbs,chunk_words=(s.n/16)*s.limbs;
        auto* source=s.host_output+chunk*chunk_words;
        auto* destination=output.c.data.data()+chunk*row_words;
        for(size_t row=first;row<last;++row)
            std::memcpy(destination+row*NTT_TILE*s.limbs,source+row*row_words,row_words*sizeof(uint32_t));
    }
    void scatter(unsigned lane,unsigned lanes) noexcept {
        const auto& s=state;
        while(!output_ready.load(std::memory_order_acquire)){
            if(cancelled.load(std::memory_order_acquire))return;
            pause();
        }
        for(unsigned chunk=0;chunk<16;++chunk){
            while(!__atomic_load_n(s.host_flags+(32+chunk)*16,__ATOMIC_ACQUIRE)){
                if(cancelled.load(std::memory_order_acquire))return;
                pause();
            }
            scatter_chunk(chunk,lane,lanes);
        }
    }
    static void worker(void* pointer,unsigned lane,unsigned lanes) noexcept {
        auto& call=*static_cast<StreamingCall*>(pointer);
        call.pack(lane-1,lanes-1);call.scatter(lane-1,lanes);

    }
};

fherma::Outputs pipeline_run(const State& s,const fherma::Inputs& in) {
    CallerPlacement placement;
    const size_t count=size_t(s.n)*s.limbs;
    fherma::Outputs out;out.c.shape={int64_t(s.n),int64_t(s.limbs)};
    std::optional<OutputWorker::Request> allocation;
    if(OUTPUT_ALLOCATE_AFTER<0)allocation.emplace(s.output_worker,out.c.data,count);
    StreamingCall call{s,in,out};
    const unsigned lanes=s.host_team->size()-1;
    for(auto& counter:call.input_left)counter.left.store(lanes,std::memory_order_relaxed);
    for(unsigned slot=0;slot<48;++slot)__atomic_store_n(s.host_flags+slot*16,0u,__ATOMIC_RELEASE);
    try {
        s.host_team->start(StreamingCall::worker,&call);
        check(cudaMemcpyAsync(s.startup_device,s.startup_host,4096,cudaMemcpyHostToDevice,s.compute_stream),"start transfer engine");
        for(unsigned chunk=0;chunk<(s.n==65536?2*OPERAND_STRIPS:16);++chunk){
            while(!__atomic_load_n(s.host_flags+chunk*16,__ATOMIC_ACQUIRE))StreamingCall::pause();
            auto stream=s.transfer_streams[chunk%INPUT_STREAMS];
            if(SPLIT_INPUT_DMA&&s.n==65536)check(cudaGraphLaunch(s.copy_graphs[chunk],s.input_copy_stream),"launch ordered input DMA graph");
            check(cudaGraphLaunch(s.input_graphs[chunk],stream),"launch input graph");
            if(int(chunk)==OUTPUT_ALLOCATE_AFTER)allocation.emplace(s.output_worker,out.c.data,count);
            if(s.n==65536&&chunk+1==OPERAND_STRIPS){
                for(unsigned i=0;i<INPUT_STREAMS;++i){
                    check(cudaEventRecord(s.inputs_ready[i],s.transfer_streams[i]),"record first operand input");
                    check(cudaStreamWaitEvent(s.compute_stream,s.inputs_ready[i],0),"join first operand input");
                }
                check(cudaGraphLaunch(s.first_operand_graph,s.compute_stream),"transform first operand early");
            }
        }
        auto core=s.compute_stream;
        for(unsigned stream=0;stream<INPUT_STREAMS;++stream){
            check(cudaEventRecord(s.inputs_ready[stream],s.transfer_streams[stream]),"record input completion");
            check(cudaStreamWaitEvent(core,s.inputs_ready[stream],0),"join inputs");
        }
        check(cudaGraphLaunch(s.pipeline_graph,core),"launch transform and output graph");
        if(!allocation)allocation.emplace(s.output_worker,out.c.data,count);
        allocation->get();call.output_ready.store(true,std::memory_order_release);
        for(unsigned chunk=0;chunk<16;++chunk){
            check(cudaEventSynchronize(s.outputs_ready[chunk]),"wait answer chunk");
            __atomic_store_n(s.host_flags+(32+chunk)*16,1u,__ATOMIC_RELEASE);
            call.scatter_chunk(chunk,lanes,lanes+1);
        }
        s.host_team->finish();
        check(cudaStreamSynchronize(core),"join output graph");
    }catch(...){
        call.cancelled.store(true,std::memory_order_release);
        s.host_team->finish();
        if(s.input_copy_stream)cudaStreamSynchronize(s.input_copy_stream);
        for(auto stream:s.transfer_streams)cudaStreamSynchronize(stream);
        cudaStreamSynchronize(s.compute_stream);throw;
    }
    return out;
}

fherma::Outputs fherma_run(void* state,const fherma::Inputs& in) {
    const auto& s=*static_cast<State*>(state); size_t count=size_t(s.n)*s.limbs,bytes=count*sizeof(uint32_t);
    if(in.a.data.size()!=count||in.b.data.size()!=count) throw std::runtime_error("wrong input size");
    if(s.pipelined) return pipeline_run(s,in);
    check(cudaMemcpy(s.a,in.a.data.data(),bytes,cudaMemcpyHostToDevice),"copy a");
    check(cudaMemcpy(s.b,in.b.data.data(),bytes,cudaMemcpyHostToDevice),"copy b");
    dim3 full((s.n+255)/256,s.primes),both(full.x,s.primes,2);
    dim3 pairs((s.n/2+255)/256,s.primes),both_pairs(pairs.x,s.primes,2);
    if(s.n>=128&&s.padded_bytes<=32768) {
        size_t words=size_t(s.n)*(s.padded_bytes/4);
        pad_inputs<<<(words+255)/256,256>>>(s.a,s.b,reinterpret_cast<uint32_t*>(s.byte_inputs),s.n,s.limbs,s.padded_bytes/4);
        dim3 tiles((s.n+63)/64,(s.primes+15)/16,2);
        tensor_product<true><<<tiles,128>>>(s.byte_inputs,s.byte_weights,s.residues,nullptr,s.fields,s.twist,s.n,s.primes,s.padded_bytes);
    } else encode<<<both,256>>>(s.a,s.b,s.residues,s.fields,s.twist,s.n,s.limbs);
    if(s.n<=65536) {
        uint32_t columns=s.n<NTT_TILE?s.n:NTT_TILE;
        if(s.n/NTT_TILE==128) high_stages_register<false,128><<<dim3(NTT_TILE/32,s.primes,2),256>>>(s.residues,s.fields,s.high_roots,s.finish,s.n);
        else if(s.n/NTT_TILE==64) high_stages_register<false,64><<<dim3(NTT_TILE/32,s.primes,2),256>>>(s.residues,s.fields,s.high_roots,s.finish,s.n);
        else if(s.n>NTT_TILE) high_stages<false><<<dim3(NTT_TILE/32,s.primes,2),256,(s.n/NTT_TILE)*32*sizeof(uint32_t)>>>(s.residues,s.fields,s.roots,s.finish,s.n);
        if(columns>=64) middle_stages_warp<<<dim3(s.n/columns,s.primes),NTT_TILE/2>>>(s.residues,s.fields,s.roots,s.inverse_roots,s.finish,s.n,columns);
        else middle_stages<<<dim3(s.n/columns,s.primes),NTT_TILE/2>>>(s.residues,s.fields,s.roots,s.inverse_roots,s.finish,s.n,columns);
        if(s.n/NTT_TILE==128) high_stages_register<true,128><<<dim3(NTT_TILE/32,s.primes),256>>>(s.residues,s.fields,s.high_inverse_roots,s.finish,s.n);
        else if(s.n/NTT_TILE==64) high_stages_register<true,64><<<dim3(NTT_TILE/32,s.primes),256>>>(s.residues,s.fields,s.high_inverse_roots,s.finish,s.n);
        else if(s.n>NTT_TILE) high_stages<true><<<dim3(NTT_TILE/32,s.primes),256,(s.n/NTT_TILE)*32*sizeof(uint32_t)>>>(s.residues,s.fields,s.inverse_roots,s.finish,s.n);
    } else {
        for(uint32_t length=s.n;length>1;length>>=1) forward<<<both_pairs,256>>>(s.residues,s.fields,s.roots,s.n,length);
        pointwise<<<full,256>>>(s.residues,s.fields,s.n);
        for(uint32_t length=2;length<=s.n;length<<=1) inverse<<<pairs,256>>>(s.residues,s.fields,s.inverse_roots,s.n,length);
        finish_ntt<<<full,256>>>(s.residues,s.fields,s.finish,s.n);
    }
    if(s.crt_weights) {
        transpose_residues<<<dim3((s.n+31)/32,s.padded_r/32),dim3(32,8)>>>(s.residues,s.crt_inputs,s.n,s.primes,s.padded_r);
        tensor_product<false><<<dim3((s.n+63)/64,(s.crt_columns+15)/16),128>>>(reinterpret_cast<const uint8_t*>(s.crt_inputs),s.crt_weights,nullptr,s.crt_dots,nullptr,nullptr,s.n,s.crt_columns,s.padded_r*4);
        if(s.direct_crt) {
            if(s.wide<=32) reconstruct_direct<1><<<(s.n+3)/4,128>>>(s.crt_inputs,s.fixed_reciprocals,s.correction_mod_q,s.primes,s.padded_r,s.q,s.answer,s.crt_dots,s.n,s.wide,s.limbs,s.width,s.gap,0,s.crt_columns);
            else reconstruct_direct<2><<<(s.n+3)/4,128>>>(s.crt_inputs,s.fixed_reciprocals,s.correction_mod_q,s.primes,s.padded_r,s.q,s.answer,s.crt_dots,s.n,s.wide,s.limbs,s.width,s.gap,0,s.crt_columns);
        } else if(s.gap&&s.wide<=128) {
            if(s.wide<=64) reconstruct_warp<2><<<(s.n+3)/4,128>>>(s.crt_inputs,s.fields,s.product,s.half_product,s.q,s.answer,s.crt_dots,s.n,s.primes,s.padded_r,s.wide,s.limbs,s.width,s.gap);
            else reconstruct_warp<4><<<(s.n+3)/4,128>>>(s.crt_inputs,s.fields,s.product,s.half_product,s.q,s.answer,s.crt_dots,s.n,s.primes,s.padded_r,s.wide,s.limbs,s.width,s.gap);
        } else
        reconstruct<true><<<(s.n+127)/128,128>>>(s.residues,s.fields,s.crt,s.product,s.half_product,s.q,s.scratch,s.answer,s.n,s.primes,s.wide,s.limbs,s.width,s.gap,s.crt_dots);
    } else reconstruct<false><<<(s.n+127)/128,128>>>(s.residues,s.fields,s.crt,s.product,s.half_product,s.q,s.scratch,s.answer,s.n,s.primes,s.wide,s.limbs,s.width,s.gap,nullptr);
    check(cudaGetLastError(),"polynomial kernels");
    fherma::Outputs out; out.c.shape={int64_t(s.n),int64_t(s.limbs)}; out.c.data.resize(count);
    check(cudaMemcpy(out.c.data.data(),s.answer,bytes,cudaMemcpyDeviceToHost),"copy answer"); return out;
}
void fherma_free(void* state) { delete static_cast<State*>(state); }
