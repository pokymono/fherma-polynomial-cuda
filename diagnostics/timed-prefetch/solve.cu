#include "implementation.cuh"
#include <chrono>
#include <cstdio>

struct PrefetchDiagnostic {State* state;unsigned calls=0;};
void* fherma_init(const fherma::Point& point){
    return new PrefetchDiagnostic{static_cast<State*>(inner_fherma_init(point))};
}
void fherma_free(void* pointer){
    auto* d=static_cast<PrefetchDiagnostic*>(pointer);
    inner_fherma_free(d->state);delete d;
}
fherma::Outputs fherma_run(void* pointer,const fherma::Inputs& input){
    auto& d=*static_cast<PrefetchDiagnostic*>(pointer);
    unsigned call=d.calls++;
    constexpr unsigned sizes[]={0,65536,1048576,8388608,33554432,33554432};
    constexpr const char* modes[]={"bytes_0","bytes_65536","bytes_1048576","bytes_8388608","bytes_33554432","first_strip"};
    fherma::Outputs answer;bool have_answer=false;
    auto verify=[&](const fherma::Outputs& output){
        if(!have_answer){answer=output;have_answer=true;}
        else if(output.c.data!=answer.c.data)throw std::runtime_error("prefetch diagnostic mismatch");
    };
    double timings[6][9]{};
    for(unsigned repeat=0;repeat<3;++repeat){
        for(unsigned order=0;order<6;++order){
            unsigned mode=(order+repeat)%6;
            d.state->start_prefetch_bytes=sizes[mode];
            d.state->start_prefetch_first_strip=mode==5;
            for(unsigned position=0;position<3;++position){
                for(unsigned warm=0;warm<2;++warm)verify(inner_fherma_run(d.state,input));
                std::this_thread::sleep_for(std::chrono::milliseconds(position==1?20:0));
                auto start=std::chrono::steady_clock::now();
                auto output=inner_fherma_run(d.state,input);
                auto stop=std::chrono::steady_clock::now();
                timings[mode][repeat*3+position]=std::chrono::duration<double,std::milli>(stop-start).count();
                verify(output);
            }
        }
    }
    if(call>=2)for(unsigned mode=0;mode<6;++mode){
        std::fprintf(stderr,"timed_prefetch:call=%u mode=%s values=",call,modes[mode]);
        for(unsigned i=0;i<9;++i)std::fprintf(stderr,"%s%.6f",i?",":"",timings[mode][i]);
        std::fprintf(stderr,"\n");
    }
    return answer;
}
