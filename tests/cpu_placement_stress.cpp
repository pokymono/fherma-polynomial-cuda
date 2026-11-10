#include "cpu_placement.h"
#include <iostream>
#include <thread>
#include <stdexcept>
int main(){
#if defined(__linux__)
    cpu_set_t before,after;CPU_ZERO(&before);CPU_ZERO(&after);
    if(sched_getaffinity(0,sizeof(before),&before))throw std::runtime_error("read caller affinity");
    const auto& cores=cpu_placement::available_cores();
    if(cores.size()<2)throw std::runtime_error("need two physical cores for this test");
    bool passed=false;
    std::thread worker([&]{
        if(!cpu_placement::bind_helper(1,2))return;
        cpu_set_t actual;CPU_ZERO(&actual);
        if(sched_getaffinity(0,sizeof(actual),&actual))return;
        passed=CPU_COUNT(&actual)==1&&CPU_ISSET(cores[1],&actual);
    });
    worker.join();
    if(sched_getaffinity(0,sizeof(after),&after)||!CPU_EQUAL(&before,&after))throw std::runtime_error("caller affinity changed");
    if(!passed)throw std::runtime_error("helper placement failed");
    std::cout<<"Helper bound correctly across "<<cores.size()<<" available physical cores; caller affinity unchanged\n";
#endif
}
