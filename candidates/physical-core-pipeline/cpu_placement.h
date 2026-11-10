#pragma once
#include <vector>
#if defined(__linux__)
#include <fstream>
#include <set>
#include <string>
#include <utility>
#include <pthread.h>
#include <sched.h>
#endif
namespace cpu_placement {
// Read topology on the calling thread before any helper is bound. We change
// affinity only for our own helper threads, never for the caller or harness.
inline const std::vector<int>& available_cores() {
    static const std::vector<int> cores=[] {
        std::vector<int> result;
#if defined(__linux__)
        cpu_set_t allowed;CPU_ZERO(&allowed);
        if(sched_getaffinity(0,sizeof(allowed),&allowed)!=0)return result;
        std::set<std::pair<int,int>> seen;
        for(int cpu=0;cpu<CPU_SETSIZE;++cpu)if(CPU_ISSET(cpu,&allowed)) {
            std::string root="/sys/devices/system/cpu/cpu"+std::to_string(cpu)+"/topology/";
            int package=-1,core=-1;
            if(!(std::ifstream(root+"physical_package_id")>>package)||!(std::ifstream(root+"core_id")>>core))continue;
            if(seen.emplace(package,core).second)result.push_back(cpu);
        }
#endif
        return result;
    }();
    return cores;
}
inline bool bind_helper(unsigned index,unsigned minimum_cores) {
#if defined(__linux__)
    const auto& cores=available_cores();
    if(cores.size()<minimum_cores||index>=cores.size())return false;
    cpu_set_t mask;CPU_ZERO(&mask);CPU_SET(cores[index],&mask);
    return pthread_setaffinity_np(pthread_self(),sizeof(mask),&mask)==0;
#else
    return false;
#endif
}
}
