#include "implementation.cuh"
#include <chrono>
#include <cstdio>
struct InputDiagnostic{fherma::Point point;unsigned calls=0;};
void* fherma_init(const fherma::Point& p){return new InputDiagnostic{p};}
void fherma_free(void* p){delete static_cast<InputDiagnostic*>(p);}
fherma::Outputs fherma_run(void* pointer,const fherma::Inputs& input){
    auto& d=*static_cast<InputDiagnostic*>(pointer);unsigned call=d.calls++;
    struct Mode{const char* name;unsigned copy;bool gates,full,caller;};
    constexpr Mode modes[]={
        {"control",0,false,false,false},{"avx2",1,false,false,false},
        {"direct64",2,false,false,false},{"gpu_gates",0,true,false,false},
        {"full_graph",0,true,true,false},{"full_caller",0,true,true,true},
        {"control_repeat",0,false,false,false}
    };
    fherma::Outputs answer;bool have=false;double times[7][9]{};
    auto verify=[&](const fherma::Outputs& out){
        if(!have){answer=out;have=true;}
        else if(out.c.data!=answer.c.data)throw std::runtime_error("input scheduler mismatch");
    };
    for(unsigned order=0;order<7;++order){
        unsigned m=(order+call)%7;
        configured_input_copy=modes[m].copy;configured_device_readiness=modes[m].gates;
        configured_full_graph=modes[m].full;configured_caller_pack=modes[m].caller;
        std::unique_ptr<void,void(*)(void*)> state(inner_fherma_init(d.point),inner_fherma_free);
        for(unsigned round=0;round<3;++round)for(unsigned position=0;position<3;++position){
            for(unsigned warm=0;warm<2;++warm)verify(inner_fherma_run(state.get(),input));
            if(position==1)std::this_thread::sleep_for(std::chrono::milliseconds(20));
            auto start=std::chrono::steady_clock::now();auto out=inner_fherma_run(state.get(),input);
            auto stop=std::chrono::steady_clock::now();
            times[m][round*3+position]=std::chrono::duration<double,std::milli>(stop-start).count();verify(out);
        }
    }
    if(call>=2)for(unsigned m=0;m<7;++m){
        std::fprintf(stderr,"input_scheduler:call=%u mode=%s values=",call,modes[m].name);
        for(unsigned i=0;i<9;++i)std::fprintf(stderr,"%s%.6f",i?",":"",times[m][i]);
        std::fprintf(stderr,"\n");
    }
    return answer;
}
