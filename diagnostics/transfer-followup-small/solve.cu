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
#undef WIDE_SHOUP
#undef WIDE_POINTWISE
#undef RESIDENT_DEVICE_INPUT
#undef RESIDENT_GROUPS
#undef RESIDENT_ROWS
#undef RESIDENT_PREFETCH
#undef RESIDENT_LOCAL_BARRIERS
#define WIDE_POINTWISE 1
namespace wide {
#include "wide.cuh"
}
#undef INPUT_STREAMS
#undef OUTPUT_CHAINS
#undef MERGED_INPUTS
#undef WIDE_SHOUP
#undef WIDE_POINTWISE
#undef RESIDENT_DEVICE_INPUT
#undef RESIDENT_GROUPS
#undef RESIDENT_ROWS
#undef RESIDENT_PREFETCH
#undef RESIDENT_LOCAL_BARRIERS
namespace size_dispatch {
#include "size_dispatch.cuh"
}
#undef INPUT_STREAMS
#undef OUTPUT_CHAINS
#undef MERGED_INPUTS
#undef WIDE_SHOUP
#undef WIDE_POINTWISE
#undef RESIDENT_DEVICE_INPUT
#undef RESIDENT_GROUPS
#undef RESIDENT_ROWS
#undef RESIDENT_PREFETCH
#undef RESIDENT_LOCAL_BARRIERS
#define WIDE_POINTWISE 1
namespace flat {
#include "flat.cuh"
}
#undef INPUT_STREAMS
#undef OUTPUT_CHAINS
#undef MERGED_INPUTS
#undef WIDE_SHOUP
#undef WIDE_POINTWISE
#undef RESIDENT_DEVICE_INPUT
#undef RESIDENT_GROUPS
#undef RESIDENT_ROWS
#undef RESIDENT_PREFETCH
#undef RESIDENT_LOCAL_BARRIERS
#define RESIDENT_GROUPS 4
#define RESIDENT_ROWS 16
namespace resident_barriers {
#include "resident_barriers.cuh"
}
#undef INPUT_STREAMS
#undef OUTPUT_CHAINS
#undef MERGED_INPUTS
#undef WIDE_SHOUP
#undef WIDE_POINTWISE
#undef RESIDENT_DEVICE_INPUT
#undef RESIDENT_GROUPS
#undef RESIDENT_ROWS
#undef RESIDENT_PREFETCH
#undef RESIDENT_LOCAL_BARRIERS

struct TransferDiagnostic { fherma::Point point; };
void* fherma_init(const fherma::Point& point){return new TransferDiagnostic{point};}
void fherma_free(void* state){delete static_cast<TransferDiagnostic*>(state);}
fherma::Outputs fherma_run(void* state,const fherma::Inputs& input){
    using Init=void*(*)(const fherma::Point&);
    using Run=fherma::Outputs(*)(void*,const fherma::Inputs&);
    using Free=void(*)(void*);
    struct Mode {const char* name;Init init;Run run;Free free;};
    const Mode modes[]={
        {"baseline_start",baseline::fherma_init,baseline::fherma_run,baseline::fherma_free},
        {"wide",wide::fherma_init,wide::fherma_run,wide::fherma_free},
        {"size_dispatch",size_dispatch::fherma_init,size_dispatch::fherma_run,size_dispatch::fherma_free},
        {"resident_barriers",resident_barriers::fherma_init,resident_barriers::fherma_run,resident_barriers::fherma_free},
        {"flat",flat::fherma_init,flat::fherma_run,flat::fherma_free},
        {"baseline_end",baseline::fherma_init,baseline::fherma_run,baseline::fherma_free}
    };
    const auto& point=static_cast<TransferDiagnostic*>(state)->point;
    fherma::Outputs answer;bool have_answer=false;
    for(const auto& mode:modes){
        // Only one worker team exists at a time; idle teams could compete for CPUs.
        std::unique_ptr<void,Free> candidate(mode.init(point),mode.free);
        for(unsigned repeat=0;repeat<5;++repeat){
            auto begin=std::chrono::steady_clock::now();
            auto output=mode.run(candidate.get(),input);
            double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
            std::fprintf(stderr,"transfer_followup_ms:N=%u mode=%s repeat=%u time=%.6f\n",unsigned(point.N),mode.name,repeat,ms);
            if(!have_answer){answer=std::move(output);have_answer=true;}
            else if(answer.c.data!=output.c.data)throw std::runtime_error("transfer follow-up mismatch");
        }
    }
    return answer;
}
