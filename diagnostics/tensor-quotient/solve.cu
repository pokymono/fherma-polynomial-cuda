#include "implementation.cuh"
#include <chrono>
#include <cstdio>
struct TensorQuotientDiagnostic{fherma::Point point;unsigned calls=0;};
void* fherma_init(const fherma::Point& p){return new TensorQuotientDiagnostic{p};}
void fherma_free(void* p){delete static_cast<TensorQuotientDiagnostic*>(p);}
fherma::Outputs fherma_run(void* pointer,const fherma::Inputs& input){
    auto& d=*static_cast<TensorQuotientDiagnostic*>(pointer);unsigned call=d.calls++;
    struct Mode{const char* name;bool quotient;unsigned fusion;bool padding;};
    constexpr Mode modes[]={
        {"control",false,0,false},{"quotient",true,0,false},
        {"cluster",false,1,false},{"combined",true,1,true},
        {"shared_combined",true,2,true},{"combined_full",true,1,false},
        {"control_repeat",false,0,false}
    };
    fherma::Outputs answer;bool have=false;double times[7][9]{};unsigned actual[7]{};
    auto verify=[&](const fherma::Outputs& out){
        if(!have){answer=out;have=true;}
        else if(out.c.data!=answer.c.data)throw std::runtime_error("tensor quotient fusion mismatch");
    };
    for(unsigned order=0;order<7;++order){
        const unsigned m=(order+call)%7;
        configured_quotient=modes[m].quotient;configured_fusion=modes[m].fusion;configured_padding=modes[m].padding;
        std::unique_ptr<void,void(*)(void*)> state(inner_fherma_init(d.point),inner_fherma_free);
        actual[m]=static_cast<State*>(state.get())->cluster_inverse?1:configured_fusion==2?2:0;
        for(unsigned round=0;round<3;++round)for(unsigned position=0;position<3;++position){
            for(unsigned warm=0;warm<2;++warm)verify(inner_fherma_run(state.get(),input));
            if(position==1)std::this_thread::sleep_for(std::chrono::milliseconds(20));
            const auto start=std::chrono::steady_clock::now();auto out=inner_fherma_run(state.get(),input);
            const auto stop=std::chrono::steady_clock::now();
            times[m][round*3+position]=std::chrono::duration<double,std::milli>(stop-start).count();verify(out);
        }
    }
    if(call>=2)for(unsigned m=0;m<7;++m){
        std::fprintf(stderr,"tensor_quotient:call=%u mode=%s executed=%u values=",call,modes[m].name,actual[m]);
        for(unsigned i=0;i<9;++i)std::fprintf(stderr,"%s%.6f",i?",":"",times[m][i]);
        std::fprintf(stderr,"\n");
    }
    return answer;
}
