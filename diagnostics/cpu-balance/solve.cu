#include "implementation.cuh"
#include <chrono>
#include <cstdio>

struct CpuDiagnostic {fherma::Point point;unsigned calls=0;};
void* fherma_init(const fherma::Point& point){return new CpuDiagnostic{point};}
void fherma_free(void* pointer){delete static_cast<CpuDiagnostic*>(pointer);}
fherma::Outputs fherma_run(void* pointer,const fherma::Inputs& input){
    auto& d=*static_cast<CpuDiagnostic*>(pointer);unsigned call=d.calls++;
    struct Mode {const char* name;unsigned copies;bool spin;};
    constexpr Mode modes[]={{"eight_wait",8,false},{"four_wait",4,false},{"six_wait",6,false},
                            {"eight_spin",8,true},{"four_spin",4,true},{"eight_wait_end",8,false}};
    fherma::Outputs answer;bool have_answer=false;
    auto verify=[&](const fherma::Outputs& output){
        if(!have_answer){answer=output;have_answer=true;}
        else if(answer.c.data!=output.c.data)throw std::runtime_error("CPU balance diagnostic mismatch");
    };
    double timings[6][9]{};unsigned actual_lanes[6]{};
    for(unsigned order=0;order<6;++order){
        unsigned mode=(order+call)%6;
        // The previous mode's state has already joined and destroyed every
        // worker before these settings change. No job overlaps modes.
        configured_copy_lanes=modes[mode].copies;
        spin_between_jobs.store(modes[mode].spin,std::memory_order_relaxed);
        std::unique_ptr<void,void(*)(void*)> state(inner_fherma_init(d.point),inner_fherma_free);
        actual_lanes[mode]=static_cast<State*>(state.get())->host_team->size()-1;
        for(unsigned repeat=0;repeat<3;++repeat)for(unsigned position=0;position<3;++position){
            for(unsigned warm=0;warm<2;++warm)verify(inner_fherma_run(state.get(),input));
            std::this_thread::sleep_for(std::chrono::milliseconds(position==1?20:0));
            auto start=std::chrono::steady_clock::now();
            auto output=inner_fherma_run(state.get(),input);
            auto stop=std::chrono::steady_clock::now();
            timings[mode][repeat*3+position]=std::chrono::duration<double,std::milli>(stop-start).count();
            verify(output);
        }
    }
    spin_between_jobs.store(false,std::memory_order_relaxed);
    configured_copy_lanes=HOST_COPY_LANES;
    if(call>=2)for(unsigned mode=0;mode<6;++mode){
        std::fprintf(stderr,"cpu_balance:call=%u mode=%s actual=%u values=",call,modes[mode].name,actual_lanes[mode]);
        for(unsigned i=0;i<9;++i)std::fprintf(stderr,"%s%.6f",i?",":"",timings[mode][i]);
        std::fprintf(stderr,"\n");
    }
    return answer;
}
