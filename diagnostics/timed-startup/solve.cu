#include "implementation.cuh"
#include <chrono>
#include <cstdio>
struct TimedStartupDiagnostic{fherma::Point point;unsigned calls=0;};
void* fherma_init(const fherma::Point& p){return new TimedStartupDiagnostic{p};}
void fherma_free(void* p){delete static_cast<TimedStartupDiagnostic*>(p);}
fherma::Outputs fherma_run(void* pointer,const fherma::Inputs& input){
    auto& d=*static_cast<TimedStartupDiagnostic*>(pointer);const unsigned call=d.calls++;
    struct Mode{const char* name;unsigned id;};
    constexpr Mode modes[]={{"control",0},{"forward",1},{"twist",2},{"both",3},{"roots",4},{"compute512",10},{"compute4096",11},{"compute16384",12},{"control_repeat",0}};
    fherma::Outputs answer;bool have=false;double times[9][9]{};
    auto verify=[&](const fherma::Outputs& out){
        if(!have){answer=out;have=true;}
        else if(out.c.data!=answer.c.data)throw std::runtime_error("timed startup mismatch");
    };
    for(unsigned order=0;order<9;++order){
        const unsigned m=(order+call)%9;configured_start=modes[m].id;
        std::unique_ptr<void,void(*)(void*)> state(inner_fherma_init(d.point),inner_fherma_free);
        for(unsigned round=0;round<3;++round)for(unsigned position=0;position<3;++position){
            for(unsigned warm=0;warm<2;++warm)verify(inner_fherma_run(state.get(),input));
            if(position==1)std::this_thread::sleep_for(std::chrono::milliseconds(20));
            const auto start=std::chrono::steady_clock::now();auto out=inner_fherma_run(state.get(),input);
            const auto stop=std::chrono::steady_clock::now();
            times[m][round*3+position]=std::chrono::duration<double,std::milli>(stop-start).count();verify(out);
        }
    }
    if(call>=2)for(unsigned m=0;m<9;++m){
        std::fprintf(stderr,"timed_startup:call=%u mode=%s values=",call,modes[m].name);
        for(unsigned i=0;i<9;++i)std::fprintf(stderr,"%s%.6f",i?",":"",times[m][i]);
        std::fprintf(stderr,"\n");
    }
    return answer;
}
