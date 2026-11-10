// Exact negacyclic multiplication using 31-bit NTT fields and integer CRT.
// All operand-dependent work, including transfers, is inside fherma_run.
#include "fherma.h"
#include "big_unsigned.h"
#include <cuda_runtime.h>
#include <memory>
#include <cstring>
#include <string>

namespace {
constexpr uint32_t NTT_TILE=512;
struct Prime {
    uint32_t p, negative_inverse, radix, radix_shoup, reduce32;
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
    uint32_t *host_inputs=nullptr,*host_output=nullptr;
    ~State() {
        for(auto ptr:{a,b,answer,residues,twist,roots,inverse_roots,finish,crt,product,half_product,q,scratch}) cudaFree(ptr);
        cudaFree(fields);
        cudaFree(byte_inputs); cudaFree(byte_weights); cudaFree(crt_weights);
        cudaFree(crt_inputs); cudaFree(crt_dots);
        cudaFreeHost(host_inputs); cudaFreeHost(host_output);
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
__device__ __forceinline__ uint32_t mont(uint32_t a,uint32_t b,Prime f) {
    uint64_t t=uint64_t(a)*b;
    uint32_t m=uint32_t(t)*f.negative_inverse;
    uint32_t r=uint32_t((t+uint64_t(m)*f.p)>>32);
    return r>=f.p?r-f.p:r;
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
                          uint32_t n,uint32_t limbs,uint32_t padded_words) {
    size_t at=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
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
                               uint32_t k_bytes) {
    __shared__ __align__(32) uint32_t shared_a[64][8],shared_b[64][8];
    uint32_t warp=threadIdx.x/32,lane=threadIdx.x%32,group=lane/4,member=lane%4;
    uint32_t row_base=blockIdx.x*64,word_base=blockIdx.y*16+warp*4;
    const uint8_t* source=inputs+size_t(blockIdx.z)*n*k_bytes;
    uint32_t acc[4][2][4]={};
    for(uint32_t k=0;k<k_bytes;k+=32) {
        uint32_t row=threadIdx.x/2,part=threadIdx.x%2;
        uint4 values=make_uint4(0,0,0,0);
        if(row_base+row<n) values=*reinterpret_cast<const uint4*>(source+size_t(row_base+row)*k_bytes+k+part*16);
        *reinterpret_cast<uint4*>(&shared_a[row][part*4])=values;
        *reinterpret_cast<uint4*>(&shared_b[row][part*4])=*reinterpret_cast<const uint4*>(weights+size_t(blockIdx.y*64+row)*k_bytes+k+part*16);
        __syncthreads();
        uint32_t b[2][2];
        #pragma unroll
        for(int col=0;col<2;++col) {
            b[col][0]=shared_b[warp*16+col*8+group][member];
            b[col][1]=shared_b[warp*16+col*8+group][member+4];
        }
        #pragma unroll
        for(int m=0;m<4;++m) {
            uint32_t a[4]={shared_a[m*16+group][member],shared_a[m*16+group+8][member],
                           shared_a[m*16+group][member+4],shared_a[m*16+group+8][member+4]};
            mma_bytes(acc[m][0],a,b[0][0],b[0][1]);
            mma_bytes(acc[m][1],a,b[1][0],b[1][1]);
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
                    out[(size_t(blockIdx.z)*count+word)*n+i]=mont(value,twist[size_t(word)*n+i],f);
                } else dots[size_t(i)*count+word]=sum;
            }
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

void* fherma_init(const fherma::Point& point) {
    using namespace host_big;
    auto s=std::make_unique<State>(); s->n=point.N; s->limbs=point.L; s->width=point.W;
    if(!s->n||(s->n&(s->n-1))||s->n>=(1u<<29)||s->limbs!=(s->width+31)/32)
        throw std::runtime_error("invalid point or transform exceeds available 31-bit fields");
    Big q=point.q.data; trim(q); Big maximum=subtract(q,Big{1});
    Big bound=times_word(multiply(maximum,maximum),2*s->n),product{1};
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
    product.resize(s->wide); half.resize(s->wide); q.resize(s->wide);
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
        fields.push_back({p,uint32_t(0-inv),radix,uint32_t((uint64_t(radix)<<32)/p),uint32_t((uint64_t(1)<<32)/p),1.0/p});
        uint32_t byte_power=1;
        for(uint32_t j=0;j<s->limbs*4;++j) {
            for(uint32_t digit=0;digit<4;++digit) byte_weights[(size_t(k)*4+digit)*s->padded_bytes+j]=(byte_power>>(digit*8))&255;
            byte_power=uint64_t(byte_power)*256%p;
        }
        Big partial=divide_word(product,p); partial.resize(s->wide);
        uint32_t crt_inverse=power(modulo_word(partial,p),p-2,p);
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
                    weights[size_t(col)*k_bytes+k*4+digit]=(crt[size_t(at/4)*s->primes+k]>>(8*(at%4)))&255;
                }
        upload(s->crt_weights,weights);
        allocate(s->crt_inputs,size_t(s->n)*s->padded_r);
        allocate(s->crt_dots,size_t(s->n)*s->wide);
    }
    upload(s->fields,fields); upload(s->crt,crt); upload(s->product,product); upload(s->half_product,half); upload(s->q,q);
    upload(s->twist,twist); upload(s->finish,finish);
    if(!roots.empty()) { upload(s->roots,roots); upload(s->inverse_roots,inverse_roots); }
    allocate(s->a,size_t(s->n)*s->limbs); allocate(s->b,size_t(s->n)*s->limbs); allocate(s->answer,size_t(s->n)*s->limbs);
    allocate(s->residues,size_t(s->n)*s->primes*2); allocate(s->scratch,size_t(s->n)*s->wide);
    upload(s->byte_weights,byte_weights); allocate(s->byte_inputs,size_t(s->n)*s->padded_bytes*2);
    check(cudaMallocHost(&s->host_inputs,size_t(s->n)*s->limbs*2*sizeof(uint32_t)),"pin input staging");
    check(cudaMallocHost(&s->host_output,size_t(s->n)*s->limbs*sizeof(uint32_t)),"pin output staging");
    return s.release();
}
fherma::Outputs fherma_run(void* state,const fherma::Inputs& in) {
    const auto& s=*static_cast<State*>(state); size_t count=size_t(s.n)*s.limbs,bytes=count*sizeof(uint32_t);
    if(in.a.data.size()!=count||in.b.data.size()!=count) throw std::runtime_error("wrong input size");
    std::memcpy(s.host_inputs,in.a.data.data(),bytes);
    check(cudaMemcpyAsync(s.a,s.host_inputs,bytes,cudaMemcpyHostToDevice),"copy a");
    std::memcpy(s.host_inputs+count,in.b.data.data(),bytes);
    check(cudaMemcpyAsync(s.b,s.host_inputs+count,bytes,cudaMemcpyHostToDevice),"copy b");
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
        if(s.n>NTT_TILE) high_stages<false><<<dim3(NTT_TILE/32,s.primes,2),256,(s.n/NTT_TILE)*32*sizeof(uint32_t)>>>(s.residues,s.fields,s.roots,s.finish,s.n);
        middle_stages<<<dim3(s.n/columns,s.primes),NTT_TILE/2>>>(s.residues,s.fields,s.roots,s.inverse_roots,s.finish,s.n,columns);
        if(s.n>NTT_TILE) high_stages<true><<<dim3(NTT_TILE/32,s.primes),256,(s.n/NTT_TILE)*32*sizeof(uint32_t)>>>(s.residues,s.fields,s.inverse_roots,s.finish,s.n);
    } else {
        for(uint32_t length=s.n;length>1;length>>=1) forward<<<both_pairs,256>>>(s.residues,s.fields,s.roots,s.n,length);
        pointwise<<<full,256>>>(s.residues,s.fields,s.n);
        for(uint32_t length=2;length<=s.n;length<<=1) inverse<<<pairs,256>>>(s.residues,s.fields,s.inverse_roots,s.n,length);
        finish_ntt<<<full,256>>>(s.residues,s.fields,s.finish,s.n);
    }
    if(s.crt_weights) {
        transpose_residues<<<dim3((s.n+31)/32,s.padded_r/32),dim3(32,8)>>>(s.residues,s.crt_inputs,s.n,s.primes,s.padded_r);
        tensor_product<false><<<dim3((s.n+63)/64,(s.wide+15)/16),128>>>(reinterpret_cast<const uint8_t*>(s.crt_inputs),s.crt_weights,nullptr,s.crt_dots,nullptr,nullptr,s.n,s.wide,s.padded_r*4);
        if(s.gap&&s.wide<=128) {
            if(s.wide<=64) reconstruct_warp<2><<<(s.n+3)/4,128>>>(s.crt_inputs,s.fields,s.product,s.half_product,s.q,s.answer,s.crt_dots,s.n,s.primes,s.padded_r,s.wide,s.limbs,s.width,s.gap);
            else reconstruct_warp<4><<<(s.n+3)/4,128>>>(s.crt_inputs,s.fields,s.product,s.half_product,s.q,s.answer,s.crt_dots,s.n,s.primes,s.padded_r,s.wide,s.limbs,s.width,s.gap);
        } else
        reconstruct<true><<<(s.n+127)/128,128>>>(s.residues,s.fields,s.crt,s.product,s.half_product,s.q,s.scratch,s.answer,s.n,s.primes,s.wide,s.limbs,s.width,s.gap,s.crt_dots);
    } else reconstruct<false><<<(s.n+127)/128,128>>>(s.residues,s.fields,s.crt,s.product,s.half_product,s.q,s.scratch,s.answer,s.n,s.primes,s.wide,s.limbs,s.width,s.gap,nullptr);
    check(cudaGetLastError(),"polynomial kernels");
    fherma::Outputs out; out.c.shape={int64_t(s.n),int64_t(s.limbs)}; out.c.data.resize(count);
    check(cudaMemcpyAsync(s.host_output,s.answer,bytes,cudaMemcpyDeviceToHost),"copy answer");
    check(cudaStreamSynchronize(nullptr),"finish answer");
    std::memcpy(out.c.data.data(),s.host_output,bytes); return out;
}
void fherma_free(void* state) { delete static_cast<State*>(state); }
