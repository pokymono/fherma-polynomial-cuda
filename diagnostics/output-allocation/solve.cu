#include "implementation.cuh"
#include <chrono>
#include <cstdio>
struct AllocationDiagnostic{fherma::Point point;unsigned calls=0;};
void* fherma_init(const fherma::Point& p){return new AllocationDiagnostic{p};}
void fherma_free(void* p){delete static_cast<AllocationDiagnostic*>(p);}
fherma::Outputs fherma_run(void* pointer,const fherma::Inputs& input){
    auto& d=*static_cast<AllocationDiagnostic*>(pointer);unsigned call=d.calls++;
    struct Mode{const char* name;unsigned huge,sibling;};
    constexpr Mode modes[]={
        {"control",0,0},{"huge",1,0},{"normal_pages",2,0},
        {"caller_sibling",0,1},{"copy_sibling",0,2},{"huge_caller",1,1},
        {"control_repeat",0,0}
    };
    fherma::Outputs answer;bool have=false;
    double times[7][9]{},allocations[7][9]{},wakes[7][9]{};
    auto verify=[&](const fherma::Outputs& out){
        if(!have){answer=out;have=true;}
        else if(out.c.data!=answer.c.data)throw std::runtime_error("output allocation mismatch");
    };
    for(unsigned order=0;order<7;++order){
        unsigned m=(order+call)%7;
        configured_output_huge=modes[m].huge;configured_allocator_sibling=modes[m].sibling;
        std::unique_ptr<void,void(*)(void*)> state(inner_fherma_init(d.point),inner_fherma_free);
        for(unsigned round=0;round<3;++round)for(unsigned position=0;position<3;++position){
            for(unsigned warm=0;warm<2;++warm)verify(inner_fherma_run(state.get(),input));
            if(position==1)std::this_thread::sleep_for(std::chrono::milliseconds(20));
            auto start=std::chrono::steady_clock::now();auto out=inner_fherma_run(state.get(),input);
            auto stop=std::chrono::steady_clock::now();
            const unsigned index=round*3+position;
            times[m][index]=std::chrono::duration<double,std::milli>(stop-start).count();
            allocations[m][index]=static_cast<State*>(state.get())->output_worker.last_allocation_ms();
            wakes[m][index]=static_cast<State*>(state.get())->output_worker.last_wake_ms();verify(out);
        }
    }
    if(call>=2)for(unsigned m=0;m<7;++m){
        std::fprintf(stderr,"output_allocation:call=%u mode=%s values=",call,modes[m].name);
        for(unsigned i=0;i<9;++i)std::fprintf(stderr,"%s%.6f",i?",":"",times[m][i]);
        std::fprintf(stderr," allocation=");
        for(unsigned i=0;i<9;++i)std::fprintf(stderr,"%s%.6f",i?",":"",allocations[m][i]);
        std::fprintf(stderr," wake=");
        for(unsigned i=0;i<9;++i)std::fprintf(stderr,"%s%.6f",i?",":"",wakes[m][i]);
        std::fprintf(stderr,"\n");
    }
    return answer;
}
