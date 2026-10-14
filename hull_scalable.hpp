#pragma once
// A separately implemented, precision-generic, dynamic-worker version of the
// radial/behind/reflex two-pass hull with radix-free multiway chain reduction.
#include <algorithm>
#include <array>
#include <climits>
#include <cstdlib>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <coroutine>
#include <exception>
#include <iterator>
#include <limits>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace scalable {
using Index = uint32_t;

// 0 bits_per_digit chooses 8/11/16 according to the active index count.
// detect_uniform_digits skips radix passes whose digit is identical for every key.
struct RadixOptions {
    unsigned bits_per_digit = 16;
    bool detect_uniform_digits = true;
};

struct RadixStats {
    unsigned key_bits = 0, bits_per_digit = 0;
    unsigned planned_passes = 0, executed_passes = 0, skipped_passes = 0;
};

// Frexp-based key packing for binary types with a variable mantissa size.
// This also supports Boost.Multiprecision cpp_bin_float<N> (e.g. N=128,256,512).
// IEEE float/double and x87 long double use faster direct bit representations.
template<typename R> struct GenericRadixKey {
    static constexpr unsigned significand_bits = std::numeric_limits<R>::digits;
    static_assert(significand_bits > 0, "Real must be a binary floating point type");
    static constexpr unsigned mantissa_words = (significand_bits + 15) / 16;
    static constexpr unsigned total_bits = (mantissa_words + 3) * 16;
    std::array<uint16_t, mantissa_words+3> part{};
};

template<typename R> constexpr unsigned real_radix_bits() {
    if constexpr(std::is_same_v<R,float>) return 32;
    else if constexpr(std::is_same_v<R,double>) return 64;
    else if constexpr(std::is_same_v<R,long double>) return 80; // x87 on supported builds
    else return GenericRadixKey<R>::total_bits;
}

template<typename R> inline R real_abs(const R& x) {
    using std::abs;
    return abs(x);
}
template<typename R> inline bool real_finite(const R& x) {
    if constexpr(std::is_floating_point_v<R>)return std::isfinite(x);
    else return x == x && (!std::numeric_limits<R>::has_infinity ||
      (x != std::numeric_limits<R>::infinity() && x != -std::numeric_limits<R>::infinity()));
}


template<typename Real> struct Point {
    Real x{}, y{};
    Point() = default;
    Point(Real a, Real b) : x(a), y(b) {}
};

template<typename Real> inline Real cross(Point<Real> a,Point<Real> b,Point<Real> c) {
    return (b.x-a.x)*(c.y-a.y) - (b.y-a.y)*(c.x-a.x);
}
template<typename Real> inline bool less_xy(Point<Real> a,Point<Real> b) {
    return a.x<b.x || (a.x==b.x && a.y<b.y);
}

// C++20 single-item coroutine generator. Source hull must remain alive.
template<class T> class Generator {
public:
    struct promise_type {
        T current{};
        std::exception_ptr exception;
        Generator get_return_object() {return Generator(std::coroutine_handle<promise_type>::from_promise(*this));}
        std::suspend_always initial_suspend() noexcept{return {};}
        std::suspend_always final_suspend() noexcept{return {};}
        std::suspend_always yield_value(T value)noexcept{current=std::move(value);return {};}
        void return_void()noexcept{}
        void unhandled_exception()noexcept{exception=std::current_exception();}
    };
    using Handle=std::coroutine_handle<promise_type>;
    explicit Generator(Handle h):h_(h){}
    Generator(Generator&&rhs):h_(std::exchange(rhs.h_,{})){}
    Generator(const Generator&)=delete;
    Generator& operator=(const Generator&)=delete;
    ~Generator(){if(h_)h_.destroy();}
    class Iterator {
        Handle h_;
    public:
        explicit Iterator(Handle h):h_(h){}
        const T& operator*()const{return h_.promise().current;}
        Iterator& operator++(){h_.resume();if(h_.done()&&h_.promise().exception)std::rethrow_exception(h_.promise().exception);return *this;}
        bool operator==(std::default_sentinel_t)const{return !h_||h_.done();}
    };
    Iterator begin(){if(h_&&!h_.done())h_.resume();if(h_&&h_.done()&&h_.promise().exception)std::rethrow_exception(h_.promise().exception);return Iterator(h_);}
    std::default_sentinel_t end()const{return {};}
private:Handle h_{};
};

// Circular linked snapshots own their storage. Move/copy of a mutable
// source Hull does not change node values or the original-coordinate system.
template<class T> class CircularList {
public:
    struct Node {T value{};Node*prev=nullptr;Node*next=nullptr;};
private:
    std::vector<Node> nodes_;
public:
    Node*head=nullptr;
    CircularList()=default;
    template<class Sequence> explicit CircularList(const Sequence&seq){
        nodes_.resize(seq.size());const size_t n=nodes_.size();
        for(size_t i=0;i<n;++i){nodes_[i].value=seq[i];nodes_[i].prev=&nodes_[(i+n-1)%n];nodes_[i].next=&nodes_[(i+1)%n];}
        if(n)head=nodes_.data();
    }
    CircularList(const CircularList&)=delete;
    CircularList(CircularList&&other)noexcept:nodes_(std::move(other.nodes_)),head(nodes_.empty()?nullptr:nodes_.data()){other.head=nullptr;}
    size_t size()const{return nodes_.size();}
};

template<typename Real> class Hull {
    static_assert(std::numeric_limits<Real>::radix == 2, "Radix packing currently requires binary Real values");
    static_assert(std::numeric_limits<Real>::is_specialized &&
                  std::numeric_limits<Real>::digits > 0,
                  "Real requires numeric_limits and binary frexp support");
public:
    using P = Point<Real>;
    using I = Index;
    // No center-translated coordinates escape this class; all P values are original.
    struct Endpoints { I first, last; };
private:
    struct X87Key { uint64_t lo{},hi{}; };
    using Key = std::conditional_t<
        std::is_same_v<Real,float> || std::is_same_v<Real,double>, uint64_t,
        std::conditional_t<std::is_same_v<Real,long double>,X87Key,GenericRadixKey<Real>>>;
    static constexpr unsigned key_bits = real_radix_bits<Real>();
    struct Work {
        std::vector<I> ids, scratch, a, b, selected, candidate;
        std::vector<Real> angles, r2;
        std::vector<Key> keys;
        std::vector<uint32_t> counts;
        RadixOptions options{};
        RadixStats stats{};
        void ensure(size_t n) {
            ids.resize(n);scratch.resize(n);angles.resize(n);keys.resize(n);
            // r2 is scratch for selected-candidate compression, not mandatory
            // for every point in the first radial pass.
            // Allocate the histogram lazily according to configured digit size.
            // No 64Ki-entry histogram for 8-bit/11-bit runs.
        }
    };
    struct Local { std::vector<I> hull,h0; Work work; size_t exterior{}; };
    std::vector<P> points_;
    std::vector<Local> locals_;
    std::vector<I> result_;
    unsigned workers_=4;
    RadixOptions radix_options_{};
    size_t exterior_=0;

    static Key key(Real v) {
        if (!real_finite(v)) throw std::invalid_argument("nonfinite radix key");
        if (v == Real(0)) v=Real(0); // canonicalize -0
        if constexpr(std::is_same_v<Real,float>) {
            uint32_t bits=std::bit_cast<uint32_t>(v);
            return uint64_t((bits & 0x80000000u) ? ~bits : bits^0x80000000u);
        } else if constexpr(std::is_same_v<Real,double>) {
            uint64_t bits=std::bit_cast<uint64_t>(v);
            return (bits & (uint64_t(1)<<63)) ? ~bits : bits^(uint64_t(1)<<63);
        } else if constexpr(std::is_same_v<Real,long double>) {
#if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__) && defined(__SIZEOF_LONG_DOUBLE__) && (__SIZEOF_LONG_DOUBLE__ == 16) && defined(__LDBL_MANT_DIG__) && (__LDBL_MANT_DIG__ == 64)
            unsigned char b[sizeof(Real)]{};std::memcpy(b,&v,sizeof(Real));
            uint64_t sig;uint16_t ex;
            std::memcpy(&sig,b,8);std::memcpy(&ex,b+8,2);
            bool neg=(ex&0x8000u)!=0;
            return X87Key{neg?~sig:sig,uint64_t(neg?uint16_t(~ex):uint16_t(ex^0x8000u))};
#else
            static_assert(sizeof(Real)==0, "unsupported long double format: specialize key()");
#endif
        } else {
            // Generic positive binary key: biased signed exponent above fixed
            // width normalized mantissa. The ordering is lexicographic in
            // (exponent, mantissa), and the digit layout is least-first.
            Key k{};
            if(v==Real(0)){k.part.back()=1;return k;}
            bool neg=v<Real(0);
            if(neg)v=-v;
            int ex=0;
            using std::frexp;
            Real fraction=frexp(v,&ex); // ADL for Boost.Multiprecision
            constexpr unsigned M=Key::mantissa_words;
            for(unsigned i=0;i<M;++i) {
                fraction*=Real(65536);
                unsigned part=static_cast<unsigned>(fraction);
                k.part[M-1-i]=uint16_t(part);
                fraction-=Real(part);
            }
            const uint32_t ordered_exp=uint32_t(int64_t(ex) - INT32_MIN);
            k.part[M]=uint16_t(ordered_exp);
            k.part[M+1]=uint16_t(ordered_exp>>16);
            k.part[M+2]=1;
            if(neg){
                k.part.back()=0;
                for(unsigned j=0;j<M+2;++j)k.part[j]=uint16_t(~k.part[j]);
            }
            return k;
        }
    }
    static unsigned digit(const Key& k, unsigned shift, unsigned mask) {
        if constexpr(std::is_same_v<Key,uint64_t>) {
            return unsigned((k>>shift)&mask);
        } else if constexpr(std::is_same_v<Key,X87Key>) {
            if(shift<64) {
                uint64_t v=k.lo>>shift;
                if(shift>48)v|=k.hi<<(64-shift);
                return unsigned(v&mask);
            }
            return unsigned((k.hi>>(shift-64))&mask);
        } else {
            unsigned word=shift/16, rem=shift%16;
            uint32_t v=word<k.part.size()?uint32_t(k.part[word])>>rem:0;
            if(rem && word+1<k.part.size())v|=uint32_t(k.part[word+1])<<(16-rem);
            return v&mask;
        }
    }
    static bool is_before(const Key& a,const Key& b,bool descending) {
        if constexpr(std::is_same_v<Key,uint64_t>)return descending?a>b:a<b;
        else if constexpr(std::is_same_v<Key,X87Key>)return descending?(a.hi>b.hi||(a.hi==b.hi&&a.lo>b.lo)):
                            (a.hi<b.hi||(a.hi==b.hi&&a.lo<b.lo));
        else return descending?(a.part>b.part):(a.part<b.part);
    }
    static Real angle4(Real x,Real y) {
        Real den=real_abs(x)+real_abs(y);
        if(den==Real(0))return Real(0);
        Real t=y/den;
        if(t==0)return x>=0?Real(0):Real(2);
        return x>=0?(t>=0?t:Real(4)+t):Real(2)-t;
    }
    static Real adiff(Real a,Real b) { Real d=real_abs(a-b);return d>2?4-d:d; }

    static void radix(Work &w,size_t n,bool descending) {
        unsigned bits=w.options.bits_per_digit;
        if(bits==0)bits=n<4096?8:n<65536?11:16;
        if(bits<1 || bits>16)throw std::invalid_argument("radix digit width must be 0 (auto) or 1..16");
        const unsigned bucket_count=1u<<bits, mask=bucket_count-1;
        const unsigned passes=(key_bits+bits-1)/bits;
        w.stats={key_bits,bits,passes,0,0};
        if(n<2)return;
        if(w.counts.size()!=bucket_count)w.counts.resize(bucket_count);
        // A sortedness probe often saves every pass on angularly ordered input.
        bool sorted=true;
        for(size_t j=1;j<n;++j)if(is_before(w.keys[w.ids[j]],w.keys[w.ids[j-1]],descending)){
            sorted=false;break;
        }
        if(sorted){w.stats.skipped_passes=passes;return;}
        I *src=w.ids.data(),*dst=w.scratch.data();
        for(unsigned pass=0;pass<passes;++pass){
            unsigned shift=pass*bits;
            if(w.options.detect_uniform_digits){
                const unsigned first=digit(w.keys[src[0]],shift,mask);
                size_t i=1;
                for(;i<n;++i)if(digit(w.keys[src[i]],shift,mask)!=first)break;
                if(i==n){++w.stats.skipped_passes;continue;}
            }
            std::fill(w.counts.begin(),w.counts.end(),0);
            for(size_t k=0;k<n;++k){unsigned d=digit(w.keys[src[k]],shift,mask);
                if(descending)d=mask-d;++w.counts[d];}
            uint32_t sum=0;
            for(auto &c:w.counts){uint32_t num=c;c=sum;sum+=num;}
            for(size_t k=0;k<n;++k){I i=src[k];unsigned d=digit(w.keys[i],shift,mask);
                if(descending)d=mask-d;dst[w.counts[d]++]=i;}
            std::swap(src,dst);++w.stats.executed_passes;
        }
        if(src!=w.ids.data())std::memcpy(w.ids.data(),src,n*sizeof(I));
    }
    static P mean(const P *p,size_t n,const I*sel=nullptr) {
        P v{};for(size_t k=0;k<n;++k){auto &q=p[sel?sel[k]:k];v.x+=q.x;v.y+=q.y;}
        return {v.x/Real(n),v.y/Real(n)};
    }
    static std::vector<I> collinear(const P *p,size_t n) {
        if(!n)return {};
        if(n==1)return {0};
        I b=1;while(b<n&&p[b].x==p[0].x&&p[b].y==p[0].y)++b;
        if(b==n)return {0};
        for(size_t i=0;i<n;++i)if(cross(p[0],p[b],p[i])!=Real(0))return {};
        I lo=0,hi=0;
        for(I i=1;i<n;++i){if(less_xy(p[i],p[lo]))lo=i;if(less_xy(p[hi],p[i]))hi=i;}
        return lo==hi?std::vector<I>{lo}:std::vector<I>{lo,hi};
    }
    static void compress(std::vector<I>& ord,const std::vector<Real>&angle,const std::vector<Real>&r2) {
        if(ord.empty())return;
        size_t len=0;I best=ord[0];Real a=angle[best];
        for(size_t k=1;k<ord.size();++k){I i=ord[k];
            if(angle[i]==a){if(r2[i]>r2[best])best=i;}
            else{ord[len++]=best;best=i;a=angle[i];}
        }
        ord[len++]=best;
        if(len>1&&adiff(angle[ord[0]],angle[ord[len-1]])==0){
            if(r2[ord[0]]>=r2[ord[len-1]])--len;
            else{ord[0]=ord[len-1];--len;}
        }
        ord.resize(len);
    }
    static std::vector<I> convex_cycle(const P* p,const std::vector<I>&ord) {
        std::vector<I> s;s.reserve(ord.size());
        for(I i:ord){while(s.size()>=2&&cross(p[s[s.size()-2]],p[s.back()],p[i])<=Real(0))s.pop_back();s.push_back(i);}
        size_t head=0;
        while(s.size()-head>=3) {
            if(cross(p[s[s.size()-2]],p[s.back()],p[s[head]])<=Real(0))s.pop_back();
            else if(cross(p[s.back()],p[s[head]],p[s[head+1]])<=Real(0))++head;
            else break;
        }
        if(head){std::move(s.begin()+head,s.end(),s.begin());s.resize(s.size()-head);}
        if(s.size()>=3&&cross(p[s[0]],p[s[1]],p[s[2]])<0)std::reverse(s.begin(),s.end());
        return s;
    }
    static std::vector<I> hull_subset(const P *p,const std::vector<I>&subset,Work &w) {
        size_t n=subset.size();if(!n)return {};
        P c=mean(p,n,subset.data());w.ensure(n);w.r2.resize(n);
        for(size_t j=0;j<n;++j){auto q=p[subset[j]];
            Real x=q.x-c.x,y=q.y-c.y;
            w.angles[j]=angle4(x,y);w.r2[j]=x*x+y*y;
            w.keys[j]=key(w.angles[j]);w.ids[j]=I(j);
        }
        radix(w,n,false);
        std::vector<I> ord(w.ids.begin(),w.ids.begin()+n);
        compress(ord,w.angles,w.r2);
        for(auto &i:ord)i=subset[i];
        return convex_cycle(p,ord);
    }
    static std::vector<I> local_build(const P* p,size_t n,Local &L) {
        if(n<=2){std::vector<I> tmp;for(I i=0;i<n;++i)tmp.push_back(i);return tmp;}
        auto deg=collinear(p,n);if(!deg.empty())return deg;
        auto &w=L.work;w.ensure(n);
        P c=mean(p,n);
        for(I i=0;i<n;++i){Real x=p[i].x-c.x,y=p[i].y-c.y;
            w.angles[i]=angle4(x,y);
            w.keys[i]=key(x*x+y*y);w.ids[i]=i;
        }
        radix(w,n,true);
        I start=w.ids[0];Real fx=p[start].x-c.x,fy=p[start].y-c.y;
        int sxF=(fx>0)-(fx<0),syF=(fy>0)-(fy<0);
        char axis='x';
        for(size_t k=1;k<n;++k){I i=w.ids[k];Real x=p[i].x-c.x,y=p[i].y-c.y;
            int sx=(x>0)-(x<0),sy=(y>0)-(y<0);
            bool xc=sx&&sxF&&sx!=sxF,yc=sy&&syF&&sy!=syF;
            if(!(xc||yc))continue;
            axis=xc!=yc?(xc?'y':'x'):(fx>0||fy<=0?'x':'y');break;
        }
        w.a.clear();w.b.clear();w.a.reserve(n/2+1);w.b.reserve(n/2+1);
        int basis=axis=='x'?syF:sxF;
        for(size_t k=1;k<n;++k){I i=w.ids[k];Real val=(axis=='x'?p[i].y-c.y:p[i].x-c.x);
            int side=(val>0)-(val<0);
            if(side==0)continue;
            (side==basis?w.a:w.b).push_back(i);
        }
        auto run_half=[&](const std::vector<I>& owned,int side,bool reverse){
            std::vector<I> chain;chain.reserve(owned.size()+1);chain.push_back(start);
            if(owned.empty())return chain;
            // Match directional gate of the original two-half implementation.
            bool low_or_left=axis=='x'?((fy<0)==(side==1)):((fx<0)==(side==1));
            auto local_angle=[&](Real t)->Real{
                if(axis=='x')return low_or_left?t-Real(2):t;
                return low_or_left?t-Real(1):(t<Real(2)?t+Real(1):t-Real(3));
            };
            I first=owned[reverse?owned.size()-1:0];
            int direction=local_angle(w.angles[first])<Real(1)?1:-1;
            for(size_t k=0;k<owned.size();++k){I i=owned[reverse?owned.size()-1-k:k];
                while(chain.size()>=2 && direction*cross(p[chain[chain.size()-2]],p[chain.back()],p[i])<=Real(0))chain.pop_back();
                Real ti=local_angle(w.angles[i]);
                if(chain.size()==1 || direction*(ti-local_angle(w.angles[chain.back()]))>Real(0))chain.push_back(i);
            }
            return chain;
        };
        std::vector<I> A=run_half(w.a,1,false),B=run_half(w.b,-1,true);
        w.selected.clear();w.selected.reserve(A.size()+B.size());
        w.selected.insert(w.selected.end(),A.begin(),A.end());
        if(B.size()>1)w.selected.insert(w.selected.end(),B.begin()+1,B.end());
        L.h0=hull_subset(p,w.selected,w);
        P c2=L.h0.size()>=3?mean(p,L.h0.size(),L.h0.data()):mean(p,n);
        if(L.h0.size()<3) {
            w.ensure(n);w.r2.resize(n);
            for(I i=0;i<n;++i){Real x=p[i].x-c2.x,y=p[i].y-c2.y;
                w.angles[i]=angle4(x,y);w.r2[i]=x*x+y*y;
                w.keys[i]=key(w.angles[i]);w.ids[i]=i;
            }
            radix(w,n,false);
            w.candidate.clear();
            w.candidate.assign(w.ids.begin(),w.ids.begin()+n);
            compress(w.candidate,w.angles,w.r2);
            L.exterior=n;
            return convex_cycle(p,w.candidate);
        }
        // Candidate-first recovery: for a modest provisional hull, classify
        // original points BEFORE radix sorting. This eliminates the full-n
        // second radix when only a small fraction lies outside H0.
        // Robust fallback below remains available for large H0 boundaries.
        if(L.h0.size()>=3 && L.h0.size()<=64 && n>=4096) {
            auto &h=L.h0;
            std::vector<Real> edge_angles(h.size());
            size_t minpos=0;
            for(size_t j=0;j<h.size();++j){
                const auto &v=p[h[j]];
                edge_angles[j]=angle4(v.x-c2.x,v.y-c2.y);
                if(edge_angles[j]<edge_angles[minpos])minpos=j;
            }
            std::rotate(h.begin(),h.begin()+minpos,h.end());
            std::rotate(edge_angles.begin(),edge_angles.begin()+minpos,edge_angles.end());
            const Real a0=edge_angles[0];
            for(auto &a:edge_angles)if(a<a0)a+=Real(4);
            // All points on H0 must survive, whether or not classified exterior.
            std::vector<uint8_t> hull_member(n,0);
            for(I id:h)hull_member[id]=1;
            w.candidate.clear();
            w.candidate.reserve(h.size()+std::min(n/8,size_t(8192)));
            for(I id:h)w.candidate.push_back(id);
            for(I id=0;id<n;++id){
                if(hull_member[id])continue;
                Real x=p[id].x-c2.x,y=p[id].y-c2.y;
                Real a=angle4(x,y);if(a<a0)a+=Real(4);
                // Lower bound of the NEXT edge endpoint. Equality is assigned
                // to the preceding edge, as in the former angular sweep.
                size_t lo=1,hi=h.size();
                while(lo<hi){size_t mid=lo+(hi-lo)/2;
                    if(edge_angles[mid]<a)lo=mid+1;else hi=mid;
                }
                const size_t e=lo-1;
                if(cross(p[h[e]],p[h[(e+1)%h.size()]],p[id])<Real(0)){
                    w.candidate.push_back(id);++L.exterior;
                }
            }
            // Sorting the survivors never needs the full input length.
            const size_t m=w.candidate.size();
            w.ensure(m);w.r2.resize(m);
            for(size_t j=0;j<m;++j){I id=w.candidate[j];
                Real x=p[id].x-c2.x,y=p[id].y-c2.y;
                w.angles[j]=angle4(x,y);w.r2[j]=x*x+y*y;
                w.keys[j]=key(w.angles[j]);w.ids[j]=I(j);
            }
            radix(w,m,false);
            std::vector<I> order(w.ids.begin(),w.ids.begin()+m);
            compress(order,w.angles,w.r2);
            for(auto &id:order)id=w.candidate[id];
            return convex_cycle(p,order);
        }
        // Original full-n angular sweep for large or complex H0 boundaries.
        w.ensure(n);w.r2.resize(n);
        for(I i=0;i<n;++i){Real x=p[i].x-c2.x,y=p[i].y-c2.y;
            w.angles[i]=angle4(x,y);w.r2[i]=x*x+y*y;
            w.keys[i]=key(w.angles[i]);w.ids[i]=i;
        }
        radix(w,n,false);
        w.candidate.clear();
        auto &h=L.h0;
        size_t minpos=0;
        for(size_t j=1;j<h.size();++j)if(w.angles[h[j]]<w.angles[h[minpos]])minpos=j;
        std::rotate(h.begin(),h.begin()+minpos,h.end());
        Real a0=w.angles[h[0]];
        std::vector<uint8_t> marked(n,0);
        for(I i:h)marked[i]=2;
        size_t cut=0,edge=0;
        while(cut<n&&w.angles[w.ids[cut]]<a0)++cut;
        for(int sweep=0;sweep<2;++sweep){size_t begin=sweep?0:cut,end=sweep?cut:n;
            for(size_t k=begin;k<end;++k){I i=w.ids[k];Real a=w.angles[i]+(sweep?Real(4):Real(0));
                while(edge+1<h.size()){
                    Real b=w.angles[h[edge+1]];if(b<a0)b+=Real(4);
                    if(a<=b)break;++edge;
                }
                if(marked[i]&2)continue;
                I j=edge+1<h.size()?h[edge+1]:h[0];
                if(cross(p[h[edge]],p[j],p[i])<Real(0))marked[i]=1;
            }
        }
        w.candidate.reserve(n);
        for(size_t k=0;k<n;++k){I i=w.ids[k];if(marked[i]&1)++L.exterior;if(marked[i])w.candidate.push_back(i);}
        compress(w.candidate,w.angles,w.r2);
        return convex_cycle(p,w.candidate);
    }

    struct Cursor { const std::vector<I>* h{};size_t pos{},last{},step{},size{};bool done{true}; };
    struct HeapItem { I id;size_t path;};
    // Merge x-monotone chains using a min-heap: O(sum h log workers) worst case.
    // No radial/angle sort and no temporary full union point-array.
    static std::vector<I> merge_many(const P*points,const std::vector<Local>&locals,unsigned count) {
        std::vector<Cursor> cursors;cursors.reserve(2*count);
        size_t total=0;
        for(unsigned i=0;i<count;++i){const auto&h=locals[i].hull;size_t n=h.size();if(!n)continue;
            total+=n;size_t l=0,r=0;
            for(size_t k=1;k<n;++k){if(less_xy(points[h[k]],points[h[l]]))l=k;if(less_xy(points[h[r]],points[h[k]]))r=k;}
            if(n<=2)cursors.push_back({&h,l,r,1,n,false});
            else {cursors.push_back({&h,l,r,1,n,false});cursors.push_back({&h,l,r,n-1,n,false});}
        }
        if(cursors.empty())return {};
        auto gt=[&](HeapItem x,HeapItem y){return less_xy(points[y.id],points[x.id]) ||
                (!less_xy(points[x.id],points[y.id]) && x.path>y.path);};
        std::priority_queue<HeapItem,std::vector<HeapItem>,decltype(gt)> pq(gt);
        for(size_t j=0;j<cursors.size();++j)pq.push({(*cursors[j].h)[cursors[j].pos],j});
        std::vector<I> lower,upper;lower.reserve(total);upper.reserve(total);
        I previous=0;bool have_previous=false;
        while(!pq.empty()){
            auto item=pq.top();pq.pop();auto &path=cursors[item.path];I id=item.id;
            if(path.pos==path.last)path.done=true;
            else path.pos=(path.pos+path.step)%path.size;
            if(!path.done)pq.push({(*path.h)[path.pos],item.path});
            if(have_previous && points[previous].x==points[id].x && points[previous].y==points[id].y)continue;
            previous=id;have_previous=true;
            while(lower.size()>=2 && cross(points[lower[lower.size()-2]],points[lower.back()],points[id])<=Real(0))lower.pop_back();
            lower.push_back(id);
            while(upper.size()>=2 && cross(points[upper[upper.size()-2]],points[upper.back()],points[id])>=Real(0))upper.pop_back();
            upper.push_back(id);
        }
        if(lower.size()<2)return lower;
        for(size_t i=upper.size()-1;i>1;--i)lower.push_back(upper[i-1]);
        return lower;
    }
public:
    explicit Hull(unsigned workers=4, RadixOptions options={}):
        workers_(workers?workers:std::max(1u,std::thread::hardware_concurrency())), radix_options_(options){
        if(options.bits_per_digit>16)throw std::invalid_argument("bits_per_digit must be 0 or 1..16");
    }
    void setRadixOptions(RadixOptions o){
        if(o.bits_per_digit>16)throw std::invalid_argument("bits_per_digit must be 0 or 1..16");
        radix_options_=o;
    }
    RadixOptions radixOptions()const{return radix_options_;}
    static constexpr unsigned radixKeyBits(){return key_bits;}
    static std::vector<I> sortValues(const std::vector<Real>& values,
                                      bool descending=false, RadixOptions options={},
                                      RadixStats* stats=nullptr){
        if(values.size()>UINT32_MAX)throw std::length_error("uint32 index overflow");
        Work w;w.options=options;
        // Standalone value sorting needs no angle/radius scratch arrays.
        w.ids.resize(values.size());w.scratch.resize(values.size());w.keys.resize(values.size());
        for(size_t i=0;i<values.size();++i){w.ids[i]=I(i);w.keys[i]=key(values[i]);}
        radix(w,values.size(),descending);
        if(stats)*stats=w.stats;
        return std::vector<I>(w.ids.begin(),w.ids.begin()+values.size());
    }
    std::vector<RadixStats> radixStats()const{
        std::vector<RadixStats> v;v.reserve(locals_.size());
        for(const auto& L:locals_)v.push_back(L.work.stats);
        return v;
    }
    void set(const Real* xy,size_t n) {
        if(n>std::numeric_limits<I>::max())throw std::length_error("point indices exceed uint32 range");
        if(!xy && n)throw std::invalid_argument("null coordinates");
        points_.resize(n);
        for(size_t i=0;i<n;++i){Real x=xy[2*i],y=xy[2*i+1];
            if(!real_finite(x)||!real_finite(y))throw std::invalid_argument("NaN or infinity coordinate");
            points_[i]={x,y};
        }
        result_.clear();locals_.clear();
    }
    void set(std::vector<P> pts) {
        if(pts.size()>UINT32_MAX)throw std::length_error("index overflow");
        for(const auto& p:pts)if(!real_finite(p.x)||!real_finite(p.y))
            throw std::invalid_argument("NaN or infinity coordinate");
        points_=std::move(pts);result_.clear();locals_.clear();
    }
    const std::vector<P>& points()const{return points_;}
    const std::vector<I>& indices()const{return result_;}
    const std::vector<I>& section(unsigned j)const{if(j>=locals_.size())throw std::out_of_range("section index");return locals_[j].hull;}
    size_t size()const{return points_.size();}
    size_t count()const{return result_.size();}
    size_t sections()const{return locals_.size();}
    size_t exterior_count()const{return exterior_;}
    Generator<I> iter_indices()const{for(I id:result_)co_yield id;}
    Generator<P> iter_vertices()const{for(I id:result_)co_yield points_[id];}
    CircularList<I> toIndexLinkedList()const{return CircularList<I>(result_);}
    CircularList<P> toPointLinkedList()const{
        std::vector<P> vals;vals.reserve(result_.size());
        for(I id:result_)vals.push_back(points_[id]);
        return CircularList<P>(vals);
    }
    void compute(unsigned workers=0) {
        if(workers)workers_=workers;
        unsigned count=std::min<size_t>(workers_,points_.size());
        if(count>256)throw std::invalid_argument("maximum 256 worker threads");
        result_.clear();exterior_=0;
        if(!count){locals_.clear();return;}
        locals_.resize(count);
        for(auto &L:locals_)L.work.options=radix_options_;
        std::vector<std::thread> threads;threads.reserve(count>1?count-1:0);
        std::vector<std::exception_ptr> errors(count);
        for(unsigned i=1;i<count;++i){threads.emplace_back([&,i]{
            try {
                size_t lo=points_.size()*i/count,hi=points_.size()*(i+1)/count;
                auto&L=locals_[i];L.exterior=0;L.hull=local_build(points_.data()+lo,hi-lo,L);
                for(auto&idx:L.hull)idx+=I(lo);
            } catch(...) {errors[i]=std::current_exception();}
        });}
        try {size_t hi=points_.size()/count;auto&L=locals_[0];L.exterior=0;L.hull=local_build(points_.data(),hi,L);}
        catch(...){errors[0]=std::current_exception();}
        for(auto &thread:threads)thread.join();
        for(auto &e:errors)if(e)std::rethrow_exception(e);
        for(auto &L:locals_)exterior_+=L.exterior;
        if(count==1)result_=locals_[0].hull;
        else result_=merge_many(points_.data(),locals_,count);
    }
    Endpoints getEndpoints(unsigned part) const {const auto&h=section(part);if(h.empty())throw std::logic_error("empty partial hull");return {h.front(),h.back()};}
    Endpoints getExtrema(unsigned part)const {const auto&h=section(part);if(h.empty())throw std::logic_error("empty partial hull");
        I lo=h.front(),hi=lo;for(I i:h){if(less_xy(points_[i],points_[lo]))lo=i;if(less_xy(points_[hi],points_[i]))hi=i;}
        return {lo,hi};
    }
};

// Boost's digit_base_2 explicitly interprets the template argument as BINARY
// significand bits; the default cpp_bin_float<N> uses decimal digits.
template<unsigned Bits> using BinaryReal =
    boost::multiprecision::number<boost::multiprecision::cpp_bin_float<Bits,
        boost::multiprecision::digit_base_2>>;
using Hull32=Hull<float>;
using Hull64=Hull<double>;
using HullExtended=Hull<long double>;
using HullBinary128=Hull<BinaryReal<128>>;
using HullBinary256=Hull<BinaryReal<256>>;
} // scalable
