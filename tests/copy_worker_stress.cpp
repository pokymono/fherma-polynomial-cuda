#include <stdexcept>
#include <iostream>
#include <random>
#include "parallel_copy.h"
int main() {
    ParallelCopy team;
    constexpr size_t size=1048576;
    std::vector<unsigned char> a(size+64),b(size+64),x(size+64),y(size+64);
    std::mt19937 random(14831);
    for(auto& value:a) value=random();
    for(auto& value:b) value=random();
    const size_t lengths[]={0,31,32,33,262143,262144,262145,size-19,size};
    for(unsigned run=0;run<60;++run) {
        ParallelCopy::Session session(team);
        for(size_t length:lengths) {
            std::fill(x.begin(),x.end(),0);std::fill(y.begin(),y.end(),0);
            team.copy_pair(x.data()+7,y.data()+3,a.data()+5,b.data()+1,length);
            if(std::memcmp(x.data()+7,a.data()+5,length)||std::memcmp(y.data()+3,b.data()+1,length))return 1;
            team.copy_pair(x.data()+1,nullptr,b.data()+9,nullptr,length);
            if(std::memcmp(x.data()+1,b.data()+9,length))return 2;
        }
    }
    std::cout<<"1080 copies passed across 60 joined sessions\n";
}
