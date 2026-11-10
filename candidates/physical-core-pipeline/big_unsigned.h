// Unsigned host arithmetic for parameter-dependent RNS plans.
#pragma once
#include <cstdint>
#include <stdexcept>
#include <vector>
namespace host_big {
using Big = std::vector<uint32_t>;
inline void trim(Big& a) { while (a.size()>1 && !a.back()) a.pop_back(); }
inline int compare(const Big& a,const Big& b) {
    size_t na=a.size(),nb=b.size();
    while(na>1&&!a[na-1]) --na;
    while(nb>1&&!b[nb-1]) --nb;
    if(na!=nb) return na<nb?-1:1;
    for(size_t i=na;i-->0;) if(a[i]!=b[i]) return a[i]<b[i]?-1:1;
    return 0;
}
inline Big subtract(Big a,const Big& b) {
    uint64_t borrow=0;
    for(size_t i=0;i<a.size();++i) {
        uint64_t t=(i<b.size()?uint64_t(b[i]):0)+borrow;
        uint32_t before=a[i]; a[i]-=uint32_t(t);
        borrow=(t>>32)+(before<uint32_t(t));
    }
    if(borrow) throw std::runtime_error("negative host integer");
    trim(a); return a;
}
inline Big times_word(const Big& a,uint32_t b) {
    Big out(a.size()+1,0); uint64_t carry=0;
    for(size_t i=0;i<a.size();++i) {
        uint64_t t=uint64_t(a[i])*b+carry;
        out[i]=uint32_t(t); carry=t>>32;
    }
    out[a.size()]=uint32_t(carry); trim(out); return out;
}
inline Big multiply(const Big& a,const Big& b) {
    Big out(a.size()+b.size(),0);
    for(size_t i=0;i<a.size();++i) {
        uint64_t carry=0;
        for(size_t j=0;j<b.size();++j) {
            uint64_t t=uint64_t(a[i])*b[j]+out[i+j]+carry;
            out[i+j]=uint32_t(t); carry=t>>32;
        }
        out[i+b.size()]=uint32_t(carry);
    }
    trim(out); return out;
}
inline Big divide_word(const Big& a,uint32_t b) {
    Big out(a.size()); uint64_t rem=0;
    for(size_t i=a.size();i-->0;) {
        uint64_t t=(rem<<32)|a[i]; out[i]=uint32_t(t/b); rem=t%b;
    }
    trim(out); return out;
}
inline Big modulo(const Big& a,const Big& b) {
    Big rem(b.size()+1,0);
    for(size_t bit=a.size()*32;bit-->0;) {
        uint32_t carry=(a[bit/32]>>(bit%32))&1;
        for(size_t j=0;j<rem.size();++j) {
            uint32_t next=rem[j]>>31;
            rem[j]=(rem[j]<<1)|carry; carry=next;
        }
        if(compare(rem,b)>=0) {
            uint64_t borrow=0;
            for(size_t j=0;j<rem.size();++j) {
                uint64_t sub=(j<b.size()?uint64_t(b[j]):0)+borrow;
                uint32_t before=rem[j];rem[j]-=uint32_t(sub);
                borrow=(sub>>32)+(before<uint32_t(sub));
            }
        }
    }
    trim(rem); return rem;
}
inline uint32_t modulo_word(const Big& a,uint32_t b) {
    uint64_t rem=0;
    for(size_t i=a.size();i-->0;) rem=((rem<<32)|a[i])%b;
    return uint32_t(rem);
}
inline uint32_t power(uint32_t a,uint64_t b,uint32_t p) {
    uint64_t out=1;
    for(;b;b>>=1,a=uint64_t(a)*a%p) if(b&1) out=out*a%p;
    return uint32_t(out);
}
inline bool is_prime(uint32_t p) {
    if(p<2) return false;
    for(uint32_t a:{2u,3u,5u,7u,11u}) { if(p==a) return true; if(p%a==0) return false; }
    uint32_t d=p-1,shift=0;
    while(!(d&1)) { d>>=1; ++shift; }
    // Deterministic witnesses for the selected p < 2^31 range.
    for(uint32_t a:{2u,3u,5u,7u,11u}) {
        uint32_t x=power(a,d,p);
        if(x==1||x==p-1) continue;
        bool passes=false;
        for(uint32_t j=1;j<shift;++j) { x=uint64_t(x)*x%p; if(x==p-1) { passes=true; break; } }
        if(!passes) return false;
    }
    return true;
}
}
