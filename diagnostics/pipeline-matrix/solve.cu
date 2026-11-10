// Exact negacyclic multiplication using 31-bit NTT fields and integer CRT.
// All operand-dependent work, including transfers, is inside fherma_run.
#include "fherma.h"
#include "big_unsigned.h"
#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <future>
#include <cstdio>
#include <chrono>
#include "parallel_copy.h"

namespace {
constexpr uint32_t NTT_TILE=512;
struct Prime {
    uint32_t p, inverse, radix, radix_shoup, reduce32;
    double reciprocal;
};
struct State {
    uint32_t n=0,limbs=0,width=0,primes=0,wide=0,gap=0,padded_bytes=0;
    uint32_t *a=nullptr,*b=nullptr,*answer=nullptr,*residues=nullptr;
    uint32_t *twist=nullptr,*roots=nullptr,*inverse_roots=nullptr,*finish=nullptr;
    uint32_t *crt=nullptr,*product=nullptr,*half_product=nullptr,*q=nullptr,*scratch=nullptr;
    Prime* fields=nullptr;
    uint8_t *byte_inputs=nullptr,*byte_weights=nullptr,*crt_weights=nullptr;
    uint32_t* crt_inputs=nullptr;
    uint64_t* crt_dots=nullptr;
    uint32_t padded_r=0;
    bool direct_crt=false;
    uint64_t* fixed_reciprocals=nullptr;
    uint32_t* correction_mod_q=nullptr;
    bool pipelined=false;
    unsigned chunk_count=0,stream_count=0;
    cudaEvent_t trace_origin{},trace_core_start{};
    explicit State(unsigned lanes):host_copy(lanes) {}
    uint32_t *host_inputs=nullptr,*host_output=nullptr;
    mutable ParallelCopy host_copy;
    cudaStream_t transfer_streams[4]{},compute_stream{};
    cudaEvent_t inputs_ready[4]{},outputs_ready[16]{},transform_ready{};
    cudaGraphExec_t input_graphs[16]{},output_graphs[16]{},core_graph{};
    ~State() {
        if(pipelined) {
            cudaEventDestroy(trace_origin); cudaEventDestroy(trace_core_start);
            for(auto stream:transfer_streams) cudaStreamSynchronize(stream);
            cudaStreamSynchronize(compute_stream);
            for(auto event:inputs_ready) cudaEventDestroy(event);
            for(auto event:outputs_ready) cudaEventDestroy(event);
            cudaEventDestroy(transform_ready);
            for(auto stream:transfer_streams) cudaStreamDestroy(stream);
            cudaStreamDestroy(compute_stream);
        }
        for(auto graph:input_graphs) if(graph) cudaGraphExecDestroy(graph);
        for(auto graph:output_graphs) if(graph) cudaGraphExecDestroy(graph);
        if(core_graph) cudaGraphExecDestroy(core_graph);
        cudaFreeHost(host_inputs); cudaFreeHost(host_output);
        for(auto ptr:{a,b,answer,residues,twist,roots,inverse_roots,finish,crt,product,half_product,q,scratch}) cudaFree(ptr);
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
__device__ __forceinline__ uint32_t add_mod(uint32_t a,uint32_t b,uint32_t p) {
    uint32_t t=a+b; return t>=p?t-p:t;
}
__device__ __forceinline__ uint32_t sub_mod(uint32_t a,uint32_t b,uint32_t p) { return a>=b?a-b:a+p-b; }
// Subtractive Montgomery reduction: m*p and a*b have identical low words.
// For a,b<p<2^31 their high-word difference lies strictly between -p and p.
__device__ __forceinline__ uint32_t mont(uint32_t a,uint32_t b,Prime f) {
    uint32_t low=a*b,high=__umulhi(a,b),m=low*f.inverse;
    uint32_t other=__umulhi(m,f.p),reduced=high-other;
    return high<other?reduced+f.p:reduced;
}
__device__ __forceinline__ uint32_t shoup(uint32_t a,uint32_t b,uint32_t bq,uint32_t p) {
    uint32_t q=__umulhi(a,bq),r=a*b-q*p;
    return r>=p?r-p:r;
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
template<bool Encode>
__global__ void tensor_product(const uint8_t* inputs,const uint8_t* weights,
                               uint32_t* out,uint64_t* dots,const Prime* fields,
                               const uint32_t* twist,uint32_t n,uint32_t count,
                               uint32_t k_bytes,uint32_t row_offset=0) {
    constexpr uint32_t tile_bytes=Encode?32:64, tile_words=tile_bytes/4;
    constexpr uint32_t swizzle_shift=Encode?2:1, swizzle_mask=Encode?1:3;
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
    const uint8_t* source=inputs+size_t(blockIdx.z)*n*k_bytes;
    uint32_t acc[4][2][4]={};
    for(uint32_t k=0;k<k_bytes;k+=tile_bytes) {
        // XOR the shared-memory column by the row group. A warp's fragment
        // accesses then occupy distinct banks, including at a 128-byte pitch.
        for(uint32_t part=threadIdx.x;part<64*(tile_bytes/16);part+=blockDim.x) {
            uint32_t row=part/(tile_bytes/16),col=(part%(tile_bytes/16))*4;
            uint4 a=make_uint4(0,0,0,0),b=make_uint4(0,0,0,0);
            if(k+col*4<k_bytes) {
                if(row_base+row<n) a=*reinterpret_cast<const uint4*>(source+size_t(row_base+row)*k_bytes+k+col*4);
                b=*reinterpret_cast<const uint4*>(weights+size_t(blockIdx.y*64+row)*k_bytes+k+col*4);
            }
            uint32_t at=col^(((row>>swizzle_shift)&swizzle_mask)*4);
            *reinterpret_cast<uint4*>(&shared_a[row][at])=a;
            *reinterpret_cast<uint4*>(&shared_b[row][at])=b;
        }
        __syncthreads();
        #pragma unroll
        for(uint32_t inner=0;inner<tile_words;inner+=8) {
            uint32_t b[2][2];
            #pragma unroll
            for(int col=0;col<2;++col) {
                uint32_t row=warp*16+col*8+group;
                b[col][0]=shared_b[row][(inner+member)^(((row>>swizzle_shift)&swizzle_mask)*4)];
                b[col][1]=shared_b[row][(inner+member+4)^(((row>>swizzle_shift)&swizzle_mask)*4)];
            }
            #pragma unroll
            for(int m=0;m<4;++m) {
                uint32_t row=m*16+group,swizzle=((row>>swizzle_shift)&swizzle_mask)*4;
                uint32_t a[4]={shared_a[row][(inner+member)^swizzle],shared_a[row+8][(inner+member)^swizzle],
                               shared_a[row][(inner+member+4)^swizzle],shared_a[row+8][(inner+member+4)^swizzle]};
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
                    uint32_t high=sum>>32,low=uint32_t(sum);
                    high-=__umulhi(high,f.reduce32)*f.p; if(high>=f.p) high-=f.p;
                    low-=__umulhi(low,f.reduce32)*f.p; if(low>=f.p) low-=f.p;
                    uint32_t value=add_mod(shoup(high,f.radix,f.radix_shoup,f.p),low,f.p);
                    shared.encoded[word-blockIdx.y*16][i-row_base]=mont(value,twist[size_t(word)*n+i],f);
                } else shared.reconstructed[i-row_base][word-blockIdx.y*16]=sum;
            }
        }
    }
    __syncthreads();
    for(uint32_t at=threadIdx.x;at<64*16;at+=blockDim.x) {
        if(Encode) {
            uint32_t col=at/64,row=at%64,i=row_base+row,word=blockIdx.y*16+col;
            if(i<n&&word<count) out[(size_t(blockIdx.z)*count+word)*n+i]=shared.encoded[col][row];
        } else {
            uint32_t row=at/16,col=at%16,i=row_base+row,word=blockIdx.y*16+col;
            if(i<n&&word<count) dots[size_t(i)*count+word]=shared.reconstructed[row][col];
        }
    }

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
__global__ void high_stages_register(uint32_t* data,const Prime* fields,const uint32_t* roots,
                                     const uint32_t* factors,uint32_t n) {
    constexpr int values=Rows/8;
    __shared__ uint32_t tile[Rows*32];
    uint32_t col=threadIdx.x%32,warp=threadIdx.x/32,col_base=blockIdx.x*32,prime=blockIdx.y;
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
                uint32_t root=roots[size_t(prime)*(n/2)+((j*row_pitch+warp)*NTT_TILE+col_base+col)*(Rows/(length*8))];
                uint32_t a=v[at],b=v[at+half];
                v[at]=add_mod(a,b,f.p); v[at+half]=mont(sub_mod(a,b,f.p),root,f);
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
            uint32_t root=roots[size_t(prime)*(n/2)+(j*NTT_TILE+col_base+column)*stride];
            if(Inverse) {
                b=mont(b,root,f); tile[row*32+column]=add_mod(a,b,f.p); tile[(row+half)*32+column]=sub_mod(a,b,f.p);
            } else {
                tile[row*32+column]=add_mod(a,b,f.p); tile[(row+half)*32+column]=mont(sub_mod(a,b,f.p),root,f);
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
                uint32_t root=roots[size_t(prime)*(n/2)+((j*8+warp)*NTT_TILE+col_base+col)*(Rows/(length*8))];
                uint32_t a=v[at],b=mont(v[at+half],root,f);
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
                                    uint32_t n,uint32_t wide,uint32_t limbs,uint32_t width,uint32_t gap,uint32_t row_offset=0) {
    uint32_t i=row_offset+(blockIdx.x*blockDim.x+threadIdx.x)/32,lane=threadIdx.x%32;
    if(i>=n) return;
    uint64_t total=0;
    for(uint32_t k=lane;k<primes;k+=32)
        total+=(uint64_t(residues[size_t(i)*padded_r+k])*reciprocals[k])>>32;
    for(int delta=16;delta;delta>>=1) total+=__shfl_xor_sync(0xffffffffu,total,delta);
    uint32_t quotient=uint32_t((total+(uint64_t(1)<<31))>>32);
    uint32_t value[Chunks],modulus[Chunks],addition[Chunks]; uint64_t raw[Chunks],bonus[Chunks];
    #pragma unroll
    for(int k=0;k<Chunks;++k) {
        uint32_t j=k*32+lane;
        raw[k]=j<wide?dots[size_t(i)*wide+j]:0;
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

void* initialize_candidate(const fherma::Point& point,unsigned chunks,unsigned streams,unsigned lanes) {
    using namespace host_big;
    auto s=std::make_unique<State>(lanes); s->chunk_count=chunks; s->stream_count=streams; s->n=point.N; s->limbs=point.L; s->width=point.W;
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
    if(s->n>=128&&s->padded_r*4<=32768) {
        uint32_t columns=(s->wide+15)/16*64,k_bytes=s->padded_r*4;
        std::vector<uint8_t> weights(size_t(columns)*k_bytes,0);
        for(uint32_t col=0;col<s->wide*4;++col)
            for(uint32_t k=0;k<s->primes;++k)
                for(uint32_t digit=0;digit<4;++digit) {
                    if(col<digit) continue;
                    uint32_t at=col-digit;
                    uint32_t value=crt[size_t(at/4)*s->primes+k];
                    weights[size_t(col)*k_bytes+k*4+digit]=(value>>(8*(at%4)))&255;
                }
        upload(s->crt_weights,weights);
        allocate(s->crt_inputs,size_t(s->n)*s->padded_r);
        allocate(s->crt_dots,size_t(s->n)*s->wide);
    }
    if(s->direct_crt) upload(s->correction_mod_q,correction);
    upload(s->fixed_reciprocals,fixed_reciprocals);
    upload(s->fields,fields); upload(s->crt,crt); upload(s->product,product); upload(s->half_product,half); upload(s->q,q);
    upload(s->twist,twist); upload(s->finish,finish);
    if(!roots.empty()) { upload(s->roots,roots); upload(s->inverse_roots,inverse_roots); }
    allocate(s->a,size_t(s->n)*s->limbs); allocate(s->b,size_t(s->n)*s->limbs); allocate(s->answer,size_t(s->n)*s->limbs);
    allocate(s->residues,size_t(s->n)*s->primes*2); allocate(s->scratch,size_t(s->n)*s->wide);
    upload(s->byte_weights,byte_weights); allocate(s->byte_inputs,size_t(s->n)*s->padded_bytes*2);
    s->pipelined=s->direct_crt&&s->n>=32768&&s->n<=65536&&s->padded_bytes<=32768;
    if(s->pipelined) {
        check(cudaEventCreate(&s->trace_origin),"create trace origin");
        check(cudaEventCreate(&s->trace_core_start),"create core trace");
        size_t bytes=size_t(s->n)*s->limbs*sizeof(uint32_t);
        check(cudaMallocHost(&s->host_inputs,bytes*2),"pin input staging");
        check(cudaMallocHost(&s->host_output,bytes),"pin output staging");
        for(auto& stream:s->transfer_streams) check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"create transfer stream");
        check(cudaStreamCreateWithFlags(&s->compute_stream,cudaStreamNonBlocking),"create compute stream");
        for(auto& event:s->inputs_ready) check(cudaEventCreateWithFlags(&event,cudaEventDefault),"create input event");
        for(auto& event:s->outputs_ready) check(cudaEventCreateWithFlags(&event,cudaEventDefault),"create output event");
        check(cudaEventCreateWithFlags(&s->transform_ready,cudaEventDefault),"create transform event");
    }
    if(s->pipelined) initialize_graphs(*s);
    return s.release();
}
// Capture parameter-only execution plans. Capture records operations without
// executing them; every input copy and arithmetic kernel runs in fherma_run.
template<class Emit>
void capture_plan(cudaGraphExec_t& executable,cudaStream_t stream,Emit emit) {
    check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal),"begin execution-plan capture");
    try { emit(); check(cudaGetLastError(),"captured kernels"); }
    catch(...) {
        cudaGraph_t abandoned=nullptr; cudaStreamEndCapture(stream,&abandoned);
        if(abandoned) cudaGraphDestroy(abandoned);
        throw;
    }
    cudaGraph_t graph=nullptr;
    check(cudaStreamEndCapture(stream,&graph),"finish execution-plan capture");
    cudaError_t status=cudaGraphInstantiate(&executable,graph,0);
    cudaGraphDestroy(graph); check(status,"instantiate execution plan");
}
void initialize_graphs(State& s) {
    const unsigned chunks=s.chunk_count;
    const size_t count=size_t(s.n)*s.limbs,chunk_rows=s.n/chunks;
    const size_t chunk_words=chunk_rows*s.limbs,chunk_bytes=chunk_words*sizeof(uint32_t);
    auto stream=s.transfer_streams[0];
    for(unsigned chunk=0;chunk<chunks;++chunk) {
        size_t offset=chunk*chunk_words;
        capture_plan(s.input_graphs[chunk],stream,[&] {
            check(cudaMemcpyAsync(s.a+offset,s.host_inputs+offset,chunk_bytes,cudaMemcpyHostToDevice,stream),"stage a chunk");
            check(cudaMemcpyAsync(s.b+offset,s.host_inputs+count+offset,chunk_bytes,cudaMemcpyHostToDevice,stream),"stage b chunk");
            size_t words=chunk_rows*(s.padded_bytes/4);
            pad_inputs<<<(words+255)/256,256,0,stream>>>(s.a,s.b,reinterpret_cast<uint32_t*>(s.byte_inputs),s.n,s.limbs,s.padded_bytes/4,chunk*chunk_rows);
            tensor_product<true><<<dim3((chunk_rows+63)/64,(s.primes+15)/16,2),128,0,stream>>>(s.byte_inputs,s.byte_weights,s.residues,nullptr,s.fields,s.twist,s.n,s.primes,s.padded_bytes,chunk*chunk_rows);
        });
        uint32_t first=chunk*chunk_rows;
        capture_plan(s.output_graphs[chunk],stream,[&] {
            tensor_product<false><<<dim3((chunk_rows+63)/64,(s.wide+15)/16),128,0,stream>>>(reinterpret_cast<const uint8_t*>(s.crt_inputs),s.crt_weights,nullptr,s.crt_dots,nullptr,nullptr,s.n,s.wide,s.padded_r*4,first);
            if(s.wide<=32) reconstruct_direct<1><<<(chunk_rows+3)/4,128,0,stream>>>(s.crt_inputs,s.fixed_reciprocals,s.correction_mod_q,s.primes,s.padded_r,s.q,s.answer,s.crt_dots,s.n,s.wide,s.limbs,s.width,s.gap,first);
            else reconstruct_direct<2><<<(chunk_rows+3)/4,128,0,stream>>>(s.crt_inputs,s.fixed_reciprocals,s.correction_mod_q,s.primes,s.padded_r,s.q,s.answer,s.crt_dots,s.n,s.wide,s.limbs,s.width,s.gap,first);
            size_t offset=chunk*chunk_words;
            check(cudaMemcpyAsync(s.host_output+offset,s.answer+offset,chunk_bytes,cudaMemcpyDeviceToHost,stream),"return answer chunk");

        });
    }
    auto core=s.compute_stream;
    capture_plan(s.core_graph,core,[&] {
        uint32_t columns=s.n<NTT_TILE?s.n:NTT_TILE;
        if(s.n/NTT_TILE==128) high_stages_register<false,128><<<dim3(NTT_TILE/32,s.primes,2),256,0,core>>>(s.residues,s.fields,s.roots,s.finish,s.n);
        else if(s.n/NTT_TILE==64) high_stages_register<false,64><<<dim3(NTT_TILE/32,s.primes,2),256,0,core>>>(s.residues,s.fields,s.roots,s.finish,s.n);
        else if(s.n>NTT_TILE) high_stages<false><<<dim3(NTT_TILE/32,s.primes,2),256,(s.n/NTT_TILE)*32*sizeof(uint32_t),core>>>(s.residues,s.fields,s.roots,s.finish,s.n);
        if(columns>=64) middle_stages_warp<<<dim3(s.n/columns,s.primes),NTT_TILE/2,0,core>>>(s.residues,s.fields,s.roots,s.inverse_roots,s.finish,s.n,columns);
        else middle_stages<<<dim3(s.n/columns,s.primes),NTT_TILE/2,0,core>>>(s.residues,s.fields,s.roots,s.inverse_roots,s.finish,s.n,columns);
        if(s.n/NTT_TILE==128) high_stages_register<true,128><<<dim3(NTT_TILE/32,s.primes),256,0,core>>>(s.residues,s.fields,s.inverse_roots,s.finish,s.n);
        else if(s.n/NTT_TILE==64) high_stages_register<true,64><<<dim3(NTT_TILE/32,s.primes),256,0,core>>>(s.residues,s.fields,s.inverse_roots,s.finish,s.n);
        else if(s.n>NTT_TILE) high_stages<true><<<dim3(NTT_TILE/32,s.primes),256,(s.n/NTT_TILE)*32*sizeof(uint32_t),core>>>(s.residues,s.fields,s.inverse_roots,s.finish,s.n);

        transpose_residues<<<dim3((s.n+31)/32,s.padded_r/32),dim3(32,8),0,core>>>(s.residues,s.crt_inputs,s.n,s.primes,s.padded_r);

    });
}

fherma::Outputs pipeline_run(const State& s,const fherma::Inputs& in) {
    const unsigned chunks=s.chunk_count,stream_count=s.stream_count;
    const size_t count=size_t(s.n)*s.limbs,chunk_rows=s.n/chunks;
    const size_t chunk_words=chunk_rows*s.limbs,chunk_bytes=chunk_words*sizeof(uint32_t);
    fherma::Outputs out; out.c.shape={int64_t(s.n),int64_t(s.limbs)};
    // Allocation/initialization is real timed work. It runs concurrently with
    // input staging, and the future always joins before this output can die.
    auto output_ready=std::async(std::launch::async,[&out,count] { out.c.data.resize(count); });
    auto cpu_begin=std::chrono::steady_clock::now();
    check(cudaEventRecord(s.trace_origin,s.compute_stream),"trace start");
    for(unsigned j=0;j<stream_count;++j) check(cudaStreamWaitEvent(s.transfer_streams[j],s.trace_origin),"trace origin dependency");
    try {
        for(unsigned chunk=0;chunk<chunks;++chunk) {
            size_t offset=chunk*chunk_words;
            auto stream=s.transfer_streams[chunk%stream_count];
            s.host_copy.copy_pair(s.host_inputs+offset,s.host_inputs+count+offset,
                                 in.a.data.data()+offset,in.b.data.data()+offset,chunk_bytes);
            check(cudaGraphLaunch(s.input_graphs[chunk],stream),"launch input graph");
        }
        auto cpu_staged=std::chrono::steady_clock::now();
        for(unsigned j=0;j<stream_count;++j) {
            check(cudaEventRecord(s.inputs_ready[j],s.transfer_streams[j]),"record staged input");
            check(cudaStreamWaitEvent(s.compute_stream,s.inputs_ready[j]),"wait for input encoding");
        }
        auto core=s.compute_stream;
        check(cudaEventRecord(s.trace_core_start,core),"trace core start");
        check(cudaGraphLaunch(s.core_graph,core),"launch transform graph");
        check(cudaEventRecord(s.transform_ready,core),"record transform completion");
        // Queue output chunks before waiting for CPU allocation. Copies use
        // pinned staging and do not touch the output vector until it is ready.
        for(unsigned chunk=0;chunk<chunks;++chunk) {
            auto stream=s.transfer_streams[chunk%stream_count];
            uint32_t first=chunk*chunk_rows;
            check(cudaStreamWaitEvent(stream,s.transform_ready),"wait for residues");
            check(cudaGraphLaunch(s.output_graphs[chunk],stream),"launch answer graph");
            check(cudaEventRecord(s.outputs_ready[chunk],stream),"record answer chunk");
        }
        output_ready.get();
        for(unsigned chunk=0;chunk<chunks;++chunk) {
            check(cudaEventSynchronize(s.outputs_ready[chunk]),"wait for answer chunk");
            size_t offset=chunk*chunk_words;
            s.host_copy.copy_pair(out.c.data.data()+offset,nullptr,s.host_output+offset,nullptr,chunk_bytes);
        }
        check(cudaGetLastError(),"pipelined polynomial kernels");
        auto cpu_done=std::chrono::steady_clock::now();
        float input=0,core_ms=0,output=0;
        check(cudaEventElapsedTime(&input,s.trace_origin,s.trace_core_start),"input duration");
        check(cudaEventElapsedTime(&core_ms,s.trace_core_start,s.transform_ready),"core duration");
        for(unsigned chunk=0;chunk<chunks;++chunk) {
            float elapsed=0;check(cudaEventElapsedTime(&elapsed,s.transform_ready,s.outputs_ready[chunk]),"output duration");
            output=std::max(output,elapsed);
        }
        std::fprintf(stderr,"pipeline_detail_ms:input=%.6f core=%.6f output=%.6f cpu_staging=%.6f cpu_total=%.6f\n",input,core_ms,output,
            std::chrono::duration<double,std::milli>(cpu_staged-cpu_begin).count(),std::chrono::duration<double,std::milli>(cpu_done-cpu_begin).count());
    } catch(...) {
        cudaDeviceSynchronize();
        throw;
    }
    return out;
}

fherma::Outputs candidate_run(void* state,const fherma::Inputs& in) {
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
        if(s.n/NTT_TILE==128) high_stages_register<false,128><<<dim3(NTT_TILE/32,s.primes,2),256>>>(s.residues,s.fields,s.roots,s.finish,s.n);
        else if(s.n/NTT_TILE==64) high_stages_register<false,64><<<dim3(NTT_TILE/32,s.primes,2),256>>>(s.residues,s.fields,s.roots,s.finish,s.n);
        else if(s.n>NTT_TILE) high_stages<false><<<dim3(NTT_TILE/32,s.primes,2),256,(s.n/NTT_TILE)*32*sizeof(uint32_t)>>>(s.residues,s.fields,s.roots,s.finish,s.n);
        if(columns>=64) middle_stages_warp<<<dim3(s.n/columns,s.primes),NTT_TILE/2>>>(s.residues,s.fields,s.roots,s.inverse_roots,s.finish,s.n,columns);
        else middle_stages<<<dim3(s.n/columns,s.primes),NTT_TILE/2>>>(s.residues,s.fields,s.roots,s.inverse_roots,s.finish,s.n,columns);
        if(s.n/NTT_TILE==128) high_stages_register<true,128><<<dim3(NTT_TILE/32,s.primes),256>>>(s.residues,s.fields,s.inverse_roots,s.finish,s.n);
        else if(s.n/NTT_TILE==64) high_stages_register<true,64><<<dim3(NTT_TILE/32,s.primes),256>>>(s.residues,s.fields,s.inverse_roots,s.finish,s.n);
        else if(s.n>NTT_TILE) high_stages<true><<<dim3(NTT_TILE/32,s.primes),256,(s.n/NTT_TILE)*32*sizeof(uint32_t)>>>(s.residues,s.fields,s.inverse_roots,s.finish,s.n);
    } else {
        for(uint32_t length=s.n;length>1;length>>=1) forward<<<both_pairs,256>>>(s.residues,s.fields,s.roots,s.n,length);
        pointwise<<<full,256>>>(s.residues,s.fields,s.n);
        for(uint32_t length=2;length<=s.n;length<<=1) inverse<<<pairs,256>>>(s.residues,s.fields,s.inverse_roots,s.n,length);
        finish_ntt<<<full,256>>>(s.residues,s.fields,s.finish,s.n);
    }
    if(s.crt_weights) {
        transpose_residues<<<dim3((s.n+31)/32,s.padded_r/32),dim3(32,8)>>>(s.residues,s.crt_inputs,s.n,s.primes,s.padded_r);
        tensor_product<false><<<dim3((s.n+63)/64,(s.wide+15)/16),128>>>(reinterpret_cast<const uint8_t*>(s.crt_inputs),s.crt_weights,nullptr,s.crt_dots,nullptr,nullptr,s.n,s.wide,s.padded_r*4);
        if(s.direct_crt) {
            if(s.wide<=32) reconstruct_direct<1><<<(s.n+3)/4,128>>>(s.crt_inputs,s.fixed_reciprocals,s.correction_mod_q,s.primes,s.padded_r,s.q,s.answer,s.crt_dots,s.n,s.wide,s.limbs,s.width,s.gap);
            else reconstruct_direct<2><<<(s.n+3)/4,128>>>(s.crt_inputs,s.fixed_reciprocals,s.correction_mod_q,s.primes,s.padded_r,s.q,s.answer,s.crt_dots,s.n,s.wide,s.limbs,s.width,s.gap);
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

struct DiagnosticState {
    struct Mode { unsigned chunks,streams,lanes; };
    const Mode modes[9]={{16,4,4},{1,1,8},{2,2,8},{4,2,8},{8,2,8},{8,4,8},{16,4,8},{4,2,16},{8,4,16}};
    std::vector<std::unique_ptr<State>> states;
};
void* fherma_init(const fherma::Point& point) {
    auto diagnostic=std::make_unique<DiagnosticState>();
    for(const auto& m:diagnostic->modes)
        diagnostic->states.emplace_back(static_cast<State*>(initialize_candidate(point,m.chunks,m.streams,m.lanes)));
    return diagnostic.release();
}
fherma::Outputs fherma_run(void* state,const fherma::Inputs& in) {
    auto& diagnostic=*static_cast<DiagnosticState*>(state);
    fherma::Outputs first;
    for(unsigned i=0;i<diagnostic.states.size();++i) {
        const auto& mode=diagnostic.modes[i];
        auto begin=std::chrono::steady_clock::now();
        auto output=candidate_run(diagnostic.states[i].get(),in);
        double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
        std::fprintf(stderr,"pipeline_mode_ms:chunks=%u streams=%u lanes=%u time=%.6f\n",mode.chunks,mode.streams,mode.lanes,ms);
        if(i==0) first=std::move(output);
        else if(first.c.data!=output.c.data) throw std::runtime_error("pipeline mode mismatch");
    }
    // Reuse already initialized plans; only the CPU copy implementation differs.
    for(unsigned i:{3u,5u,6u}) {
        const auto& mode=diagnostic.modes[i];
        auto& candidate=*diagnostic.states[i]; candidate.host_copy.set_streaming(true);
        auto begin=std::chrono::steady_clock::now();
        auto output=candidate_run(&candidate,in);
        double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
        candidate.host_copy.set_streaming(false);
        std::fprintf(stderr,"pipeline_streaming_ms:chunks=%u streams=%u lanes=%u time=%.6f\n",mode.chunks,mode.streams,mode.lanes,ms);
        if(first.c.data!=output.c.data) throw std::runtime_error("streaming copy mismatch");
    }
    return first;
}
void fherma_free(void* state) { delete static_cast<DiagnosticState*>(state); }

