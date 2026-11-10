#include "implementation.cuh"
#include <chrono>
#include <cstdio>
struct ClusterDiagnostic{fherma::Point point;unsigned calls=0;};
void* fherma_init(const fherma::Point& p){return new ClusterDiagnostic{p};}
void fherma_free(void* p){delete static_cast<ClusterDiagnostic*>(p);}
fherma::Outputs fherma_run(void* pointer,const fherma::Inputs& input){
    auto& d=*static_cast<ClusterDiagnostic*>(pointer);unsigned call=d.calls++;
    struct Mode{const char* name;bool enabled;unsigned fields,columns;};
    constexpr Mode modes[]={
        {"control",false,4,16},{"cluster4_columns16",true,4,16},
        {"cluster2_columns32",true,2,32},{"cluster4_columns32",true,4,32},
        {"control_repeat",false,4,16}
    };
    fherma::Outputs answer;bool have=false;double times[5][9]{};unsigned actual[5]{};
    auto verify=[&](const fherma::Outputs& out){
        if(!have){answer=out;have=true;}
        else if(out.c.data!=answer.c.data)throw std::runtime_error("cluster inverse mismatch");
    };
    for(unsigned order=0;order<5;++order){
        const unsigned m=(order+call)%5;
        configured_cluster=modes[m].enabled;configured_fields=modes[m].fields;configured_columns=modes[m].columns;
        std::unique_ptr<void,void(*)(void*)> state(inner_fherma_init(d.point),inner_fherma_free);
        actual[m]=static_cast<State*>(state.get())->cluster_inverse;
        for(unsigned round=0;round<3;++round)for(unsigned position=0;position<3;++position){
            for(unsigned warm=0;warm<2;++warm)verify(inner_fherma_run(state.get(),input));
            if(position==1)std::this_thread::sleep_for(std::chrono::milliseconds(20));
            const auto start=std::chrono::steady_clock::now();auto out=inner_fherma_run(state.get(),input);
            const auto stop=std::chrono::steady_clock::now();
            times[m][round*3+position]=std::chrono::duration<double,std::milli>(stop-start).count();verify(out);
        }
    }
    if(call>=2)for(unsigned m=0;m<5;++m){
        std::fprintf(stderr,"cluster_inverse:call=%u mode=%s executed=%u values=",call,modes[m].name,actual[m]);
        for(unsigned i=0;i<9;++i)std::fprintf(stderr,"%s%.6f",i?",":"",times[m][i]);
        std::fprintf(stderr,"\n");
    }
    return answer;
}
