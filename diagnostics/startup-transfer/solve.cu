#include "implementation.cuh"

struct StartupDiagnostic {
    State* state;
    unsigned calls=0;
};
void* fherma_init(const fherma::Point& point) {
    return new StartupDiagnostic{static_cast<State*>(inner_fherma_init(point))};
}
void fherma_free(void* pointer) {
    auto* diagnostic=static_cast<StartupDiagnostic*>(pointer);
    inner_fherma_free(diagnostic->state);
    delete diagnostic;
}
fherma::Outputs fherma_run(void* pointer,const fherma::Inputs& input) {
    auto& diagnostic=*static_cast<StartupDiagnostic*>(pointer);
    auto& state=*diagnostic.state;
    unsigned call=diagnostic.calls++;
    fherma::Outputs answer;bool have_answer=false;
    auto verify=[&](const fherma::Outputs& output) {
        if(!have_answer){answer=output;have_answer=true;}
        else if(output.c.data!=answer.c.data)throw std::runtime_error("startup diagnostic mismatch");
    };
    auto measure=[&](unsigned bytes,unsigned delay,unsigned repeat) {
        state.startup_bytes=bytes;state.print_timeline=false;
        for(unsigned warm=0;warm<2;++warm)verify(inner_fherma_run(&state,input));
        // Printing only after the call keeps diagnostic I/O out of the idle
        // interval. The fixed setup and every transfer remain in the harness.
        std::this_thread::sleep_for(std::chrono::milliseconds(delay));
        state.print_timeline=call>=2;
        auto start=std::chrono::steady_clock::now();
        auto output=inner_fherma_run(&state,input);
        auto stop=std::chrono::steady_clock::now();
        state.print_timeline=false;verify(output);
        if(call>=2)std::fprintf(stderr,"startup_case:call=%u bytes=%u idle=%u repeat=%u outer_inner_ms=%.6f\n",call,bytes,delay,repeat,std::chrono::duration<double,std::milli>(stop-start).count());
    };
    for(unsigned repeat=0;repeat<3;++repeat) {
        measure(4096,0,repeat);
        // Rotate the startup sizes to avoid a fixed size/time-order confound.
        constexpr unsigned sizes[]={4096,65536,262144,1048576};
        for(unsigned i=0;i<4;++i)measure(sizes[(i+repeat)%4],20,repeat);
        measure(4096,0,repeat);
    }
    return answer;
}
