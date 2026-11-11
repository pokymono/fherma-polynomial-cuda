#pragma once
#ifndef HOST_COPY_LANES
#define HOST_COPY_LANES 4
#endif
#include <algorithm>
#include <array>
#include <fstream>
#include <set>
#include <string>
#include <utility>
#include <vector>
#if defined(__linux__)
#include <sched.h>
#endif
struct CpuPlacement {
    std::vector<int> physical,siblings;
    std::array<std::array<int,16>,4> copy_sets{};
    CpuPlacement(){
#if defined(__linux__)
        cpu_set_t allowed;CPU_ZERO(&allowed);
        if(sched_getaffinity(0,sizeof(allowed),&allowed))return;
        std::set<std::pair<int,int>> seen;
        for(int cpu=0;cpu<CPU_SETSIZE;++cpu)if(CPU_ISSET(cpu,&allowed)){
            int core=cpu,socket=0;
            const std::string path="/sys/devices/system/cpu/cpu"+std::to_string(cpu)+"/topology/";
            std::ifstream(path+"core_id")>>core;
            std::ifstream(path+"physical_package_id")>>socket;
            if(seen.emplace(socket,core).second)physical.push_back(cpu);
            else siblings.push_back(cpu);
        }
#endif
        // Parameter-only placement tables, restricted to the original allowed
        // affinity. The caller and dispatcher keep their existing cores.
        std::vector<int> available;
        for(int cpu:physical)if(cpu!=caller()&&cpu!=allocator())available.push_back(cpu);
        const unsigned count=copy_lanes(HOST_COPY_LANES);
        for(unsigned mode=0;mode<4;++mode)for(unsigned lane=0;lane<16;++lane){
            int selected=copy(lane);
            if(available.size()>=count&&lane<count){
                if(mode==1)selected=available[available.size()-count+lane];
                if(mode==2)selected=available[lane%2?available.size()-1-lane/2:lane/2];
                if(mode==3)selected=available[count>1?lane*(available.size()-1)/(count-1):0];
            }
            copy_sets[mode][lane]=selected;
        }
    }
    int copy_for_mode(unsigned lane,unsigned mode)const{return copy_sets[mode%4][lane%16];}
    static const CpuPlacement& get(){static const CpuPlacement placement;return placement;}
    static void pin(int cpu){
#if defined(__linux__)
        if(cpu<0)return;cpu_set_t selected;CPU_ZERO(&selected);CPU_SET(cpu,&selected);
        sched_setaffinity(0,sizeof(selected),&selected);
#endif
    }
    int copy(unsigned lane)const{return physical.empty()?-1:physical[lane%physical.size()];}
    int caller()const{
        if(physical.empty())return -1;
        return physical[std::min(copy_lanes(HOST_COPY_LANES),unsigned(physical.size()-1))];
    }
    unsigned copy_lanes(unsigned requested)const{
        return physical.empty()?requested:std::min(requested,std::max(1u,unsigned(physical.size())-1));
    }
    int allocator()const{
        unsigned copies=copy_lanes(HOST_COPY_LANES);
        if(physical.size()>copies+1)return physical[copies+1];
        return siblings.empty()?caller():siblings.back();
    }
};
class CallerPlacement {
#if defined(__linux__)
    cpu_set_t previous;bool valid=false;
#endif
public:
    CallerPlacement(){
#if defined(__linux__)
        valid=sched_getaffinity(0,sizeof(previous),&previous)==0;
        if(valid)CpuPlacement::pin(CpuPlacement::get().caller());
#endif
    }
    ~CallerPlacement(){
#if defined(__linux__)
        if(valid)sched_setaffinity(0,sizeof(previous),&previous);
#endif
    }
};
