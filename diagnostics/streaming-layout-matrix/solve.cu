#include "fherma.h"
#include "big_unsigned.h"
#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <chrono>
#include <cstdio>
#include "output_worker.h"
#include "streaming_team.h"
namespace baseline {
#include "baseline.cuh"
}
#undef INPUT_STREAMS
#undef OUTPUT_CHAINS
#undef MERGED_INPUTS
#undef HOST_COL_LOG
#undef HOST_COLS
#undef HOST_CHUNKS
#undef COPY_BLOCKS
#define HOST_COL_LOG 6
namespace columns64 {
#include "columns.cuh"
}
#undef INPUT_STREAMS
#undef OUTPUT_CHAINS
#undef MERGED_INPUTS
#undef HOST_COL_LOG
#undef HOST_COLS
#undef HOST_CHUNKS
#undef COPY_BLOCKS
#define HOST_COL_LOG 7
namespace columns128 {
#include "columns.cuh"
}
#undef INPUT_STREAMS
#undef OUTPUT_CHAINS
#undef MERGED_INPUTS
#undef HOST_COL_LOG
#undef HOST_COLS
#undef HOST_CHUNKS
#undef COPY_BLOCKS
#define COPY_BLOCKS 64
namespace smcopy64 {
#include "sm_copy.cuh"
}
#undef INPUT_STREAMS
#undef OUTPUT_CHAINS
#undef MERGED_INPUTS
#undef HOST_COL_LOG
#undef HOST_COLS
#undef HOST_CHUNKS
#undef COPY_BLOCKS

struct LayoutDiagnostic { fherma::Point point; };
void* fherma_init(const fherma::Point& point){return new LayoutDiagnostic{point};}
void fherma_free(void* state){delete static_cast<LayoutDiagnostic*>(state);}
fherma::Outputs fherma_run(void* state,const fherma::Inputs& input){
    using Init=void*(*)(const fherma::Point&);
    using Run=fherma::Outputs(*)(void*,const fherma::Inputs&);
    using Free=void(*)(void*);
    struct Mode {const char* name;Init init;Run run;Free free;};
    const Mode modes[]={
      {"baseline_start",baseline::fherma_init,baseline::fherma_run,baseline::fherma_free},
      {"columns64",columns64::fherma_init,columns64::fherma_run,columns64::fherma_free},
      {"columns128",columns128::fherma_init,columns128::fherma_run,columns128::fherma_free},
      {"smcopy64",smcopy64::fherma_init,smcopy64::fherma_run,smcopy64::fherma_free},
      {"baseline_end",baseline::fherma_init,baseline::fherma_run,baseline::fherma_free}
    };
    fherma::Outputs answer;bool have_answer=false;
    for(const auto& mode:modes){
        std::unique_ptr<void,Free> candidate(mode.init(static_cast<LayoutDiagnostic*>(state)->point),mode.free);
        for(unsigned repeat=0;repeat<5;++repeat){
            const auto begin=std::chrono::steady_clock::now();
            auto output=mode.run(candidate.get(),input);
            const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
            std::fprintf(stderr,"layout_ms:mode=%s repeat=%u time=%.6f\n",mode.name,repeat,ms);
            if(!have_answer){answer=std::move(output);have_answer=true;}
            else if(answer.c.data!=output.c.data)throw std::runtime_error("layout diagnostic mismatch");
        }
    }
    return answer;
}
