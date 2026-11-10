#include "implementation.cuh"
#include <chrono>
#include <cstdio>
struct PrefixDiagnostic{fherma::Point point;unsigned calls=0;};
void* fherma_init(const fherma::Point& p){return new PrefixDiagnostic{p};}
void fherma_free(void* p){delete static_cast<PrefixDiagnostic*>(p);}
fherma::Outputs fherma_run(void* pointer,const fherma::Inputs& input){
    auto& d=*static_cast<PrefixDiagnostic*>(pointer);unsigned call=d.calls++;
    struct Mode{const char* name;bool prefix,vector;unsigned workers;};
    constexpr Mode modes[]={
        {"control",false,false,8},{"prefix_plain",true,false,8},
        {"prefix_vector",true,true,8},{"prefix_ten",true,true,10},
        {"prefix_six",true,true,6},{"control_repeat",false,false,8}
    };
    fherma::Outputs answer;bool have=false;double times[6][9]{};unsigned actual[6]{};
    auto verify=[&](const fherma::Outputs& out){
        if(!have){answer=out;have=true;}
        else if(out.c.data!=answer.c.data)throw std::runtime_error("prefix transform mismatch");
    };
    for(unsigned order=0;order<6;++order){
        const unsigned m=(order+call)%6;
        configured_prefix=modes[m].prefix;configured_vector_pack=modes[m].vector;configured_copy_lanes=modes[m].workers;
        std::unique_ptr<void,void(*)(void*)> state(inner_fherma_init(d.point),inner_fherma_free);
        actual[m]=static_cast<State*>(state.get())->host_team->size()-1;
        for(unsigned round=0;round<3;++round)for(unsigned position=0;position<3;++position){
            for(unsigned warm=0;warm<2;++warm)verify(inner_fherma_run(state.get(),input));
            if(position==1)std::this_thread::sleep_for(std::chrono::milliseconds(20));
            const auto start=std::chrono::steady_clock::now();auto out=inner_fherma_run(state.get(),input);
            const auto stop=std::chrono::steady_clock::now();
            times[m][round*3+position]=std::chrono::duration<double,std::milli>(stop-start).count();verify(out);
        }
    }
    if(call>=2)for(unsigned m=0;m<6;++m){
        std::fprintf(stderr,"prefix_middle:call=%u mode=%s workers=%u values=",call,modes[m].name,actual[m]);
        for(unsigned i=0;i<9;++i)std::fprintf(stderr,"%s%.6f",i?",":"",times[m][i]);
        std::fprintf(stderr,"\n");
    }
    return answer;
}
