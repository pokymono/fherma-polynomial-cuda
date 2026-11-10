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
namespace operand {
#include "operand.cuh"
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

struct WorkerDiagnostic {fherma::Point point;unsigned calls=0;};
void* fherma_init(const fherma::Point& point){return new WorkerDiagnostic{point};}
void fherma_free(void* pointer){delete static_cast<WorkerDiagnostic*>(pointer);}
fherma::Outputs fherma_run(void* pointer,const fherma::Inputs& input){
    using Init=void*(*)(const fherma::Point&);
    using Run=fherma::Outputs(*)(void*,const fherma::Inputs&);
    using Free=void(*)(void*);
    struct Mode{const char* name;Init init;Run run;Free free;bool spin_workers,spin_allocator;};
    const Mode modes[]={
        {"baseline_start",baseline::fherma_init,baseline::fherma_run,baseline::fherma_free,false,false},
        {"baseline_spin",baseline::fherma_init,baseline::fherma_run,baseline::fherma_free,true,true},
        {"tuned_spin",tuned::fherma_init,tuned::fherma_run,tuned::fherma_free,true,true},
        {"operand_wait",operand::fherma_init,operand::fherma_run,operand::fherma_free,false,false},
        {"operand_spin",operand::fherma_init,operand::fherma_run,operand::fherma_free,true,true},
        {"baseline_end",baseline::fherma_init,baseline::fherma_run,baseline::fherma_free,false,false}
    };
    auto& diagnostic=*static_cast<WorkerDiagnostic*>(pointer);
    unsigned call=diagnostic.calls++;
    fherma::Outputs answer;bool have_answer=false;
    auto verify=[&](const fherma::Outputs& output){
        if(!have_answer){answer=output;have_answer=true;}
        else if(answer.c.data!=output.c.data)throw std::runtime_error("worker diagnostic mismatch");
    };
    double timings[6][9]{};
    for(unsigned mode_index=0;mode_index<6;++mode_index){
        const auto& mode=modes[mode_index];
        spin_wait_workers.store(mode.spin_workers);spin_wait_allocator.store(mode.spin_allocator);
        std::unique_ptr<void,Free> state(mode.init(diagnostic.point),mode.free);
        for(unsigned repeat=0;repeat<3;++repeat){
            for(unsigned position=0;position<3;++position){
                for(unsigned warm=0;warm<2;++warm)verify(mode.run(state.get(),input));
                unsigned idle=position==1?20:0;
                std::this_thread::sleep_for(std::chrono::milliseconds(idle));
                auto start=std::chrono::steady_clock::now();
                auto output=mode.run(state.get(),input);
                auto stop=std::chrono::steady_clock::now();
                timings[mode_index][repeat*3+position]=std::chrono::duration<double,std::milli>(stop-start).count();
                verify(output);
            }
        }
    }
    spin_wait_workers.store(false);spin_wait_allocator.store(false);
    // Compact output after all timing avoids truncating measured samples in
    // the platform's 12,000-character log tail.
    if(call>=2)for(unsigned m=0;m<6;++m){
        std::fprintf(stderr,"worker_idle:call=%u mode=%s values=",call,modes[m].name);
        for(unsigned k=0;k<9;++k)std::fprintf(stderr,"%s%.6f",k?",":"",timings[m][k]);
        std::fprintf(stderr,"\n");
    }
    return answer;
}
