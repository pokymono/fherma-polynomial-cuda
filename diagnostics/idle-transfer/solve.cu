#include "fherma.h"
#include "big_unsigned.h"
#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <chrono>
#include <thread>
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
#undef OUTPUT_STAGE_MASK
#undef CRT_ASYNC_TILE
namespace tuned {
#include "tuned.cuh"
}
#undef INPUT_STREAMS
#undef OUTPUT_CHAINS
#undef MERGED_INPUTS
#undef WIDE_SHOUP
#undef WIDE_POINTWISE
#undef RESIDENT_DEVICE_INPUT
#undef RESIDENT_GROUPS
#undef RESIDENT_ROWS
#undef OUTPUT_STAGE_MASK
#undef CRT_ASYNC_TILE

struct IdleDiagnostic {fherma::Point point;unsigned calls=0;};
void* fherma_init(const fherma::Point& p){return new IdleDiagnostic{p};}
void fherma_free(void* s){delete static_cast<IdleDiagnostic*>(s);}
fherma::Outputs fherma_run(void* pointer,const fherma::Inputs& input){
    using Init=void*(*)(const fherma::Point&);
    using Run=fherma::Outputs(*)(void*,const fherma::Inputs&);
    using Free=void(*)(void*);
    struct Mode{const char* name;Init init;Run run;Free free;};
    const Mode modes[]={
        {"baseline",baseline::fherma_init,baseline::fherma_run,baseline::fherma_free},
        {"tuned",tuned::fherma_init,tuned::fherma_run,tuned::fherma_free}
    };
    auto& diagnostic=*static_cast<IdleDiagnostic*>(pointer);
    const auto& point=diagnostic.point;unsigned call=diagnostic.calls++;
    fherma::Outputs answer;bool have_answer=false;
    auto verify=[&](const fherma::Outputs& output){
        if(!have_answer){answer=output;have_answer=true;}
        else if(output.c.data!=answer.c.data)throw std::runtime_error("idle diagnostic mismatch");
    };
    for(const auto& mode:modes){
        std::unique_ptr<void,Free> state(mode.init(point),mode.free);
        // Alternate long and short delays to expose drift rather than making
        // idle duration a proxy for how late a measurement occurs in the run.
        for(unsigned delay:{0u,50u,1u,20u,5u,0u}){
            for(unsigned repeat=0;repeat<3;++repeat){
                for(unsigned warm=0;warm<2;++warm)verify(mode.run(state.get(),input));
                std::this_thread::sleep_for(std::chrono::milliseconds(delay));
                auto start=std::chrono::steady_clock::now();
                auto output=mode.run(state.get(),input);
                auto stop=std::chrono::steady_clock::now();
                double ms=std::chrono::duration<double,std::milli>(stop-start).count();
                verify(output);
                std::fprintf(stderr,"idle_transfer_ms:call=%u N=%u mode=%s idle=%u repeat=%u time=%.6f\n",call,unsigned(point.N),mode.name,delay,repeat,ms);
            }
        }
    }
    return answer;
}
