#include "queued_copy.h"
#include <iostream>
#include <vector>
#include <array>
#include <stdexcept>
struct Buffer {std::vector<unsigned char> a,b,da,db,ra,rb;};
int main() {
    size_t jobs=0;
    for(unsigned lanes:{1u,2u,3u,7u,8u,16u}) {
    QueuedCopy copy(lanes);
    for(unsigned session=0;session<80;++session) {
        std::array<Buffer,32> buffers;
        std::array<uint64_t,32> tickets;
        {
            QueuedCopy::Session active(copy);
            for(unsigned j=0;j<32;++j) {
                auto& v=buffers[j];size_t rows=(session+j)%41,width=(session*197+j*11)%2048;
                size_t from=width+7,to=width+19;
                v.a.resize(rows*from+8);v.b.resize(rows*from+8);
                for(size_t i=0;i<v.a.size();++i){v.a[i]=(i*113+session)&255;v.b[i]=(i*41+j)&255;}
                v.da.assign(rows*to+8,91);v.db.assign(rows*to+8,67);v.ra=v.da;v.rb=v.db;
                bool pair=(j%2)==0;
                for(size_t row=0;row<rows;++row)for(size_t k=0;k<width;++k){v.ra[row*to+k+1]=v.a[row*from+k+3];if(pair)v.rb[row*to+k+1]=v.b[row*from+k+3];}
                tickets[j]=copy.enqueue(v.da.data()+1,pair?v.db.data()+1:nullptr,v.a.data()+3,pair?v.b.data()+3:nullptr,width,rows,to,from);
            }
            // Checking all tickets also exercises uneven completion order.
            for(unsigned j=0;j<32;++j){copy.wait(tickets[j]);auto& v=buffers[j];if(v.da!=v.ra||v.db!=v.rb)throw std::runtime_error("queued copy mismatch");++jobs;}
        }
    }
    bool rejected=false;
    try{copy.enqueue(nullptr,nullptr,nullptr,nullptr,0,0,0,0);}catch(const std::runtime_error&){rejected=true;}
    if(!rejected)throw std::runtime_error("missing session guard");
    }
    std::cout<<"Passed "<<jobs<<" queued jobs across 480 sessions and six team sizes\n";
}
