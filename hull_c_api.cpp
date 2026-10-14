#include "hull_scalable.hpp"
#include <cfloat>
#include <cstring>
#include <vector>
#include <stdexcept>
#include <exception>
#include <memory>
#include <string>
#include <type_traits>

namespace {
thread_local std::string last_error;
template<typename F>int guard(F&& f)noexcept{
    try{last_error.clear();f();return 0;}
    catch(const std::exception&e){last_error=e.what();return -1;}
    catch(...){last_error="unknown C++ exception";return -1;}
}
template<class Real>void* create(const void*xy,size_t n,unsigned workers)noexcept{
    try{last_error.clear();auto ptr=std::make_unique<scalable::Hull<Real>>(workers);
        ptr->set(reinterpret_cast<const Real*>(xy),n);return ptr.release();}
    catch(const std::exception&e){last_error=e.what();return nullptr;}
    catch(...){last_error="native create error";return nullptr;}
}
template<class Real>void* create_text(const char*const*xy,size_t n,unsigned workers)noexcept{
    try {last_error.clear();if(!xy&&n)throw std::invalid_argument("null coordinate strings");
        std::vector<scalable::Point<Real>> pts;pts.reserve(n);
        for(size_t i=0;i<n;++i){
            if(!xy[2*i]||!xy[2*i+1])throw std::invalid_argument("null coordinate string");
            pts.emplace_back(Real(xy[2*i]),Real(xy[2*i+1]));
        }
        auto ptr=std::make_unique<scalable::Hull<Real>>(workers);
        ptr->set(std::move(pts));return ptr.release();
    }catch(const std::exception&e){last_error=e.what();return nullptr;}
     catch(...){last_error="native decimal parser error";return nullptr;}
}
template<class Real>int radix_options(void*h,unsigned bits,int detect)noexcept{
    return guard([&]{if(!h)throw std::invalid_argument("null hull");
        static_cast<scalable::Hull<Real>*>(h)->setRadixOptions({bits,detect!=0});});
}
template<class Real>int radix_stats(void*h,unsigned part,uint32_t*out)noexcept{
    return guard([&]{if(!h||!out)throw std::invalid_argument("null hull or output");
        auto v=static_cast<scalable::Hull<Real>*>(h)->radixStats();
        if(part>=v.size())throw std::out_of_range("radix worker index");
        const auto &s=v[part];
        out[0]=s.key_bits;out[1]=s.bits_per_digit;out[2]=s.planned_passes;
        out[3]=s.executed_passes;out[4]=s.skipped_passes;
    });
}
template<class Real>int run(void*h,unsigned workers)noexcept{return guard([&]{
    if(!h)throw std::invalid_argument("null hull");
    static_cast<scalable::Hull<Real>*>(h)->compute(workers);});}
template<class Real>size_t count(void*h,int kind,unsigned part)noexcept{
    try{last_error.clear();auto &H=*static_cast<scalable::Hull<Real>*>(h);
        return kind==0?H.count():kind==1?H.section(part).size():
               kind==2?H.size():H.sections();
    }catch(const std::exception&e){last_error=e.what();return SIZE_MAX;}
}
template<class Real>int copy_ids(void*h,int kind,unsigned part,uint32_t*dst,size_t cap)noexcept{
    return guard([&]{auto &H=*static_cast<scalable::Hull<Real>*>(h);
        const auto& v=kind==0?H.indices():H.section(part);
        if(cap<v.size()||(!dst&&!v.empty()))throw std::invalid_argument("buffer too small");
        std::copy(v.begin(),v.end(),dst);
    });
}
template<class Real>int copy_points(void*h,int kind,unsigned part,void*dst,size_t cap)noexcept{
    return guard([&]{auto &H=*static_cast<scalable::Hull<Real>*>(h);
        const auto& ids=kind==0?H.indices():H.section(part);
        if(cap<ids.size()||(!dst&&!ids.empty()))throw std::invalid_argument("buffer too small");
        auto*r=static_cast<Real*>(dst);
        for(size_t j=0;j<ids.size();++j){auto p=H.points()[ids[j]];r[j*2]=p.x;r[j*2+1]=p.y;}
    });
}
template<class Real>int endpoint(void*h,unsigned part,uint32_t*out,int ext)noexcept{
    return guard([&]{auto &H=*static_cast<scalable::Hull<Real>*>(h);
        if(!out)throw std::invalid_argument("null output");
        auto e=ext?H.getExtrema(part):H.getEndpoints(part);
        out[0]=e.first;out[1]=e.last;
    });
}

template<class Real>int sorted_values(const Real *values,size_t n,int desc,
    unsigned bits,int detect,uint32_t *out)noexcept{
    return guard([&]{if((!values||!out)&&n)throw std::invalid_argument("null sort array");
        std::vector<Real> src;
        if(n)src.assign(values,values+n);
        auto ids=scalable::Hull<Real>::sortValues(src,desc!=0,{bits,detect!=0});
        if(n)std::copy(ids.begin(),ids.end(),out);
    });
}
template<class Real>int sorted_text_values(const char*const*values,size_t n,int desc,
    unsigned bits,int detect,uint32_t*out)noexcept{
    return guard([&]{if((!values||!out)&&n)throw std::invalid_argument("null sort array");
        std::vector<Real> src;src.reserve(n);
        for(size_t i=0;i<n;++i){if(!values[i])throw std::invalid_argument("null radix value");src.emplace_back(values[i]);}
        auto ids=scalable::Hull<Real>::sortValues(src,desc!=0,{bits,detect!=0});
        if(n)std::copy(ids.begin(),ids.end(),out);
    });
}
}
extern "C" {
const char *sc_last_error()noexcept{return last_error.c_str();}
int sc_ld_bytes()noexcept{return int(sizeof(long double));}
int sc_ld_mant_dig()noexcept{return LDBL_MANT_DIG;}
int sc_bin128_mant_dig()noexcept{return std::numeric_limits<scalable::BinaryReal<128>>::digits;}
int sc_bin256_mant_dig()noexcept{return std::numeric_limits<scalable::BinaryReal<256>>::digits;}
void* sc_new64(const void*xy,size_t n,unsigned workers)noexcept{return create<double>(xy,n,workers);}
void* sc_newld(const void*xy,size_t n,unsigned workers)noexcept{return create<long double>(xy,n,workers);}
void* sc_newb128(const char*const*xy,size_t n,unsigned workers)noexcept{return create_text<scalable::BinaryReal<128>>(xy,n,workers);}
void sc_free64(void*h)noexcept{delete static_cast<scalable::Hull64*>(h);}
void sc_freeld(void*h)noexcept{delete static_cast<scalable::HullExtended*>(h);}
void sc_freeb128(void*h)noexcept{delete static_cast<scalable::HullBinary128*>(h);}
int sc_run64(void*h,unsigned w)noexcept{return run<double>(h,w);}
int sc_runld(void*h,unsigned w)noexcept{return run<long double>(h,w);}
int sc_runb128(void*h,unsigned w)noexcept{return run<scalable::BinaryReal<128>>(h,w);}
size_t sc_count64(void*h,int kind,unsigned part)noexcept{return count<double>(h,kind,part);}
size_t sc_countld(void*h,int kind,unsigned part)noexcept{return count<long double>(h,kind,part);}
size_t sc_countb128(void*h,int kind,unsigned part)noexcept{return count<scalable::BinaryReal<128>>(h,kind,part);}
int sc_ids64(void*h,int k,unsigned part,uint32_t*dst,size_t n)noexcept{return copy_ids<double>(h,k,part,dst,n);}
int sc_idsld(void*h,int k,unsigned part,uint32_t*dst,size_t n)noexcept{return copy_ids<long double>(h,k,part,dst,n);}
int sc_idsb128(void*h,int k,unsigned part,uint32_t*dst,size_t n)noexcept{return copy_ids<scalable::BinaryReal<128>>(h,k,part,dst,n);}
int sc_points64(void*h,int k,unsigned part,void*dst,size_t n)noexcept{return copy_points<double>(h,k,part,dst,n);}
int sc_pointsld(void*h,int k,unsigned part,void*dst,size_t n)noexcept{return copy_points<long double>(h,k,part,dst,n);}
int sc_ends64(void*h,unsigned part,uint32_t*out,int extrema)noexcept{return endpoint<double>(h,part,out,extrema);}
int sc_endsld(void*h,unsigned part,uint32_t*out,int extrema)noexcept{return endpoint<long double>(h,part,out,extrema);}
int sc_endsb128(void*h,unsigned part,uint32_t*out,int extrema)noexcept{return endpoint<scalable::BinaryReal<128>>(h,part,out,extrema);}
int sc_radix64(void*h,unsigned bits,int detect)noexcept{return radix_options<double>(h,bits,detect);}
int sc_radixstats64(void*h,unsigned part,uint32_t*out)noexcept{return radix_stats<double>(h,part,out);}
int sc_radixld(void*h,unsigned bits,int detect)noexcept{return radix_options<long double>(h,bits,detect);}
int sc_radixstatsld(void*h,unsigned part,uint32_t*out)noexcept{return radix_stats<long double>(h,part,out);}
int sc_radixb128(void*h,unsigned bits,int detect)noexcept{return radix_options<scalable::BinaryReal<128>>(h,bits,detect);}
int sc_radixstatsb128(void*h,unsigned part,uint32_t*out)noexcept{return radix_stats<scalable::BinaryReal<128>>(h,part,out);}
int sc_sort64(const void*src,size_t n,int desc,unsigned bits,int detect,uint32_t*out)noexcept{return sorted_values<double>((const double*)src,n,desc,bits,detect,out);}
int sc_sortld(const void*src,size_t n,int desc,unsigned bits,int detect,uint32_t*out)noexcept{return sorted_values<long double>((const long double*)src,n,desc,bits,detect,out);}
int sc_sortb128(const char*const*src,size_t n,int desc,unsigned bits,int detect,uint32_t*out)noexcept{
    return sorted_text_values<scalable::BinaryReal<128>>(src,n,desc,bits,detect,out);}

void* sc_newb256(const char*const*xy,size_t n,unsigned workers)noexcept{return create_text<scalable::BinaryReal<256>>(xy,n,workers);}
void sc_freeb256(void*h)noexcept{delete static_cast<scalable::HullBinary256*>(h);}
int sc_runb256(void*h,unsigned w)noexcept{return run<scalable::BinaryReal<256>>(h,w);}
size_t sc_countb256(void*h,int kind,unsigned part)noexcept{return count<scalable::BinaryReal<256>>(h,kind,part);}
int sc_idsb256(void*h,int k,unsigned part,uint32_t*dst,size_t n)noexcept{return copy_ids<scalable::BinaryReal<256>>(h,k,part,dst,n);}
int sc_endsb256(void*h,unsigned part,uint32_t*out,int extrema)noexcept{return endpoint<scalable::BinaryReal<256>>(h,part,out,extrema);}
int sc_radixb256(void*h,unsigned bits,int detect)noexcept{return radix_options<scalable::BinaryReal<256>>(h,bits,detect);}
int sc_radixstatsb256(void*h,unsigned part,uint32_t*out)noexcept{return radix_stats<scalable::BinaryReal<256>>(h,part,out);}
int sc_sortb256(const char*const*src,size_t n,int desc,unsigned bits,int detect,uint32_t*out)noexcept{
    return sorted_text_values<scalable::BinaryReal<256>>(src,n,desc,bits,detect,out);}
}
