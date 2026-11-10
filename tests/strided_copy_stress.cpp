#include <stdexcept>
#include <iostream>
#include <random>
#include "parallel_copy.h"
int main() {
    ParallelCopy team;std::mt19937 random(273491);
    for(size_t rows:{0u,1u,7u,8u,9u,64u,127u,128u})
        for(size_t width:{0u,1u,31u,32u,65u,7040u}) {
            size_t pitch=width+17;
            std::vector<unsigned char> a(rows*pitch+64),b(a.size()),x(rows*width+64),y(x.size()),back(a.size());
            for(auto& value:a)value=random();for(auto& value:b)value=random();
            ParallelCopy::Session session(team);
            team.copy_strided_pair(x.data()+3,y.data()+5,a.data()+1,b.data()+2,width,rows,width,pitch);
            for(size_t row=0;row<rows;++row) {
                if(std::memcmp(x.data()+3+row*width,a.data()+1+row*pitch,width))return 1;
                if(std::memcmp(y.data()+5+row*width,b.data()+2+row*pitch,width))return 2;
            }
            team.copy_strided_pair(back.data()+1,nullptr,x.data()+3,nullptr,width,rows,pitch,width);
            for(size_t row=0;row<rows;++row) {
                if(std::memcmp(back.data()+1+row*pitch,a.data()+1+row*pitch,width))return 3;
                for(size_t j=width;j<pitch;++j)if(back[1+row*pitch+j])return 4;
            }
        }
    std::cout<<"96 strided jobs passed across 48 sessions, including empty and uneven row partitions\n";
}
