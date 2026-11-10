#include "implementation.cuh"
#include <chrono>
#include <cstdio>
struct OutputScheduleDiagnostic{fherma::Point point;unsigned calls=0;};
void* fherma_init(const fherma::Point& p){return new OutputScheduleDiagnostic{p};}
void fherma_free(void* p){delete static_cast<OutputScheduleDiagnostic*>(p);}
fherma::Outputs fherma_run(void* pointer,const fherma::Inputs& input){
    auto& d=*static_cast<OutputScheduleDiagnostic*>(pointer);unsigned call=d.calls++;
    struct Mode{const char* name;unsigned mask,chains;bool tma;};
    constexpr Mode modes[]={
        {"control",7,2,false},{"serial",0,1,false},
        {"serial_tma",0,1,true},{"mask5",5,2,false},
        {"tma",7,2,true},{"control_repeat",7,2,false}
    };
    fherma::Outputs answer;bool have=false;double times[6][9]{};unsigned actual[6]{};
    auto verify=[&](const fherma::Outputs& out){
        if(!have){answer=out;have=true;}
        else if(out.c.data!=answer.c.data)throw std::runtime_error("fused output schedule mismatch");
    };
    for(unsigned order=0;order<6;++order){
        const unsigned m=(order+call)%6;
        configured_mask=modes[m].mask;configured_chains=modes[m].chains;configured_tma=modes[m].tma;
        std::unique_ptr<void,void(*)(void*)> state(inner_fherma_init(d.point),inner_fherma_free);
        actual[m]=static_cast<State*>(state.get())->crt_tma;
        for(unsigned round=0;round<3;++round)for(unsigned position=0;position<3;++position){
            for(unsigned warm=0;warm<2;++warm)verify(inner_fherma_run(state.get(),input));
            if(position==1)std::this_thread::sleep_for(std::chrono::milliseconds(20));
            const auto start=std::chrono::steady_clock::now();auto out=inner_fherma_run(state.get(),input);
            const auto stop=std::chrono::steady_clock::now();
            times[m][round*3+position]=std::chrono::duration<double,std::milli>(stop-start).count();verify(out);
        }
    }
    if(call>=2)for(unsigned m=0;m<6;++m){
        std::fprintf(stderr,"fused_schedule:call=%u mode=%s tma=%u values=",call,modes[m].name,actual[m]);
        for(unsigned i=0;i<9;++i)std::fprintf(stderr,"%s%.6f",i?",":"",times[m][i]);
        std::fprintf(stderr,"\n");
    }
    return answer;
}
