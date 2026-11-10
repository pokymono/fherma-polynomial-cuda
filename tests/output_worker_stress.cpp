#include "output_worker.h"
#include <iostream>
#include <limits>
int main() {
    OutputWorker worker;
    for(size_t i=0;i<1000;++i) {
        std::vector<uint32_t> output;
        auto request=worker.prepare(output,(i*7919)%65536);
        request.get();
        if(output.size()!=(i*7919)%65536)return 1;
        for(auto value:output)if(value)return 2;
    }
    {
        std::vector<uint32_t> output;
        {auto request=worker.prepare(output,100000);}
        if(output.size()!=100000)return 3;
    }
    {
        std::vector<uint32_t> output;bool caught=false;
        try {auto request=worker.prepare(output,std::numeric_limits<size_t>::max());request.get();}
        catch(const std::length_error&) {caught=true;}
        if(!caught)return 4;
    }
    {std::vector<uint32_t> output;auto request=worker.prepare(output,123);request.get();if(output.size()!=123)return 5;}
    std::cout<<"1003 output construction/lifetime/error-recovery checks passed\n";
}
