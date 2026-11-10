#include "implementation.cuh"
#include <chrono>
#include <cstdio>
struct OrderedDmaDiagnostic{fherma::Point point;unsigned calls=0;};
void* fherma_init(const fherma::Point& p){return new OrderedDmaDiagnostic{p};}
void fherma_free(void* p){delete static_cast<OrderedDmaDiagnostic*>(p);}
fherma::Outputs fherma_run(void* pointer,const fherma::Inputs& input){
    auto& d=*static_cast<OrderedDmaDiagnostic*>(pointer);unsigned call=d.calls++;
    struct Mode{const char* name;bool split;unsigned streams;int allocation;};
    constexpr Mode modes[]={
        {"control",false,8,-1},{"ordered8",true,8,-1},
        {"ordered2",true,2,-1},{"ordered_defer7",true,8,7},
        {"ordered_defer3",true,8,3},{"control_repeat",false,8,-1}
    };
    fherma::Outputs answer;bool have=false;double times[6][9]{};unsigned actual[6]{};
    auto verify=[&](const fherma::Outputs& out){
        if(!have){answer=out;have=true;}
        else if(out.c.data!=answer.c.data)throw std::runtime_error("ordered DMA mismatch");
    };
    for(unsigned order=0;order<6;++order){
        const unsigned m=(order+call)%6;
        configured_split=modes[m].split;configured_streams=modes[m].streams;configured_allocation=modes[m].allocation;
        std::unique_ptr<void,void(*)(void*)> state(inner_fherma_init(d.point),inner_fherma_free);
        actual[m]=configured_streams;
        for(unsigned round=0;round<3;++round)for(unsigned position=0;position<3;++position){
            for(unsigned warm=0;warm<2;++warm)verify(inner_fherma_run(state.get(),input));
            if(position==1)std::this_thread::sleep_for(std::chrono::milliseconds(20));
            const auto start=std::chrono::steady_clock::now();auto out=inner_fherma_run(state.get(),input);
            const auto stop=std::chrono::steady_clock::now();
            times[m][round*3+position]=std::chrono::duration<double,std::milli>(stop-start).count();verify(out);
        }
    }
    if(call>=2)for(unsigned m=0;m<6;++m){
        std::fprintf(stderr,"ordered_dma:call=%u mode=%s streams=%u values=",call,modes[m].name,actual[m]);
        for(unsigned i=0;i<9;++i)std::fprintf(stderr,"%s%.6f",i?",":"",times[m][i]);
        std::fprintf(stderr,"\n");
    }
    return answer;
}
