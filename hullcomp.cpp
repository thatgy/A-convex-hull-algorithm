#include <algorithm>
#include <barrier>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

using namespace std;

// One point class: two unrestricted double coordinates, no polar state.
// The radius and directional helpers are calculated on demand.
class Point {
public:
    double cartpoint[2];

    Point(double x=0, double y=0) : cartpoint{x, y} {}

    double x() const { return cartpoint[0]; }
    double y() const { return cartpoint[1]; }
    double r2() const { return x()*x() + y()*y(); }
    double r() const { return sqrt(r2()); }

    // Slope of the ray from origin to this point.  A vertical ray is
    // +/- infinity; an undefined direction (same point) yields NaN.
    // This is optional; angle4() is safer for the hull's radix keys.
    double get_slope(const Point& origin=Point()) const {
        const double dx=x()-origin.x();
        const double dy=y()-origin.y();
        if(dx==0) {
            if(dy==0) return numeric_limits<double>::quiet_NaN();
            return copysign(numeric_limits<double>::infinity(),dy);
        }
        return dy/dx;
    }

    Point operator*(double s) const { return Point(x()*s,y()*s); }
    Point& operator*=(double s) { cartpoint[0]*=s; cartpoint[1]*=s; return *this; }
    double operator*(const Point& p) const { return x()*p.y()-y()*p.x(); }
    Point operator+(const Point& p) const { return Point(x()+p.x(),y()+p.y()); }
    Point operator-(const Point& p) const { return Point(x()-p.x(),y()-p.y()); }
    operator double() const { return r(); }

    // Signed direction from edge first -> second toward a candidate point.
    // Positive: left of the directed edge; negative: right; zero: collinear.
    static double get_direction(const Point& first, const Point& second,
                                const Point& candidate) {
        return (second.x()-first.x())*(candidate.y()-first.y())
             - (second.y()-first.y())*(candidate.x()-first.x());
    }

    // Monotone pseudo-angle; no atan2 or exceptional vertical slope.
    static double angle4(double x,double y) {
        const double d = fabs(x)+fabs(y);
        if(d==0) return 0;
        const double t = y/d;
        if(t==0) return x>=0 ? 0 : 2;
        return x>=0 ? (t>=0 ? t : 4+t) : 2-t;
    }

    // All radix passes, signed floating-key encoding, and parallel scatter
    // live here, not in another sort wrapper or kernel.
    template<class ValueAt>
    static vector<uint32_t> radixSort(size_t n, ValueAt value_at, bool reverse=false,
                                unsigned digit_bits=16, unsigned workers=4,
                                bool parallel=true);

    enum class SortMode { Radius, Angle, X, Y };
    // One configurable sorter. The same signed radix implementation serves every mode.
    static vector<uint32_t> sort(const Point* points,size_t n, SortMode mode=SortMode::Radius,
                           bool reverse=false, const Point& origin=Point(),
                           double* angular=nullptr,double* squared=nullptr,
                           const uint32_t* selected=nullptr,
                           unsigned digit_bits=16, unsigned workers=4,
                           bool parallel=true);
    static vector<uint32_t> sort(const vector<Point>& points, SortMode mode=SortMode::Radius,
                           bool reverse=false, const Point& origin=Point(),
                           unsigned digit_bits=16, unsigned workers=4,
                           bool parallel=true) {
        return sort(points.data(),points.size(),mode,reverse,origin,
                    nullptr,nullptr,nullptr,digit_bits,workers,parallel);
    }
    static vector<uint32_t> sort(const vector<double>& values, bool reverse=false,
                           unsigned digit_bits=16, unsigned workers=4,
                           bool parallel=true);
    // Legacy Boolean mode selects angular vs radial SORTING, not point storage.
    static vector<uint32_t> sort(const vector<Point>& points, bool by_angle, bool reverse=false,
                           const Point& origin=Point(), unsigned digit_bits=16,
                           unsigned workers=4, bool parallel=true) {
        return sort(points, by_angle ? SortMode::Angle : SortMode::Radius, reverse,
                    origin,digit_bits,workers,parallel);
    }
};
static_assert(is_trivially_copyable_v<Point>);
static_assert(sizeof(Point)==2*sizeof(double), "Point must remain two doubles");

namespace hull_detail {
    constexpr double EPS = 1e-12, TURN = 4.0, HALF = 2.0, QUARTER = 1.0;
    thread_local bool inner_parallel = true;

    struct Result { vector<uint32_t> hull, h0; size_t outside_n = 0; };

    inline double angle_delta(double a,double b) {
        const double d=fabs(a-b);
        return d>HALF ? TURN-d : d;
    }
    inline int sign(double x) { return x>EPS ? 1 : x < -EPS ? -1 : 0; }
    Point mean(const Point* points,size_t n,const uint32_t* selection=nullptr) {
        Point c{};
        for(size_t i=0;i<n;++i) {
            const Point& p=points[selection ? selection[i] : i];
            c.cartpoint[0]+=p.cartpoint[0];
            c.cartpoint[1]+=p.cartpoint[1];
        }
        return {c.cartpoint[0]/n,c.cartpoint[1]/n};
    }
    // Returns the collinear endpoints if *all* selected points are collinear.
    // An empty result means that the full hull algorithm must run.
    vector<uint32_t> collinear(const Point* points,const uint32_t* sel,size_t n) {
        if(n==0) return {};
        uint32_t a=sel ? sel[0] : 0, b=UINT32_MAX;
        for(size_t k=1;k<n;++k) {
            uint32_t i=sel ? sel[k] : uint32_t(k);
            if(points[i].cartpoint[0]!=points[a].cartpoint[0] || points[i].cartpoint[1]!=points[a].cartpoint[1]) { b=i; break; }
        }
        if(b==UINT32_MAX) return {a};
        for(size_t k=0;k<n;++k) {
            uint32_t i=sel ? sel[k] : uint32_t(k);
            if(fabs(Point::get_direction(points[a],points[b],points[i]))>EPS) return {};
        }
        double minx=points[a].cartpoint[0],maxx=minx,miny=points[a].cartpoint[1],maxy=miny;
        for(size_t k=0;k<n;++k) {
            uint32_t i=sel ? sel[k] : uint32_t(k);
            minx=min(minx,points[i].cartpoint[0]); maxx=max(maxx,points[i].cartpoint[0]);
            miny=min(miny,points[i].cartpoint[1]); maxy=max(maxy,points[i].cartpoint[1]);
        }
        bool y=(maxy-miny)>(maxx-minx);
        uint32_t lo=a, hi=a;
        for(size_t k=0;k<n;++k) {
            uint32_t i=sel ? sel[k] : uint32_t(k);
            double ai=y ? points[i].cartpoint[1] : points[i].cartpoint[0];
            double al=y ? points[lo].cartpoint[1] : points[lo].cartpoint[0];
            double ah=y ? points[hi].cartpoint[1] : points[hi].cartpoint[0];
            double bi=y ? points[i].cartpoint[0] : points[i].cartpoint[1];
            double bl=y ? points[lo].cartpoint[0] : points[lo].cartpoint[1];
            double bh=y ? points[hi].cartpoint[0] : points[hi].cartpoint[1];
            if(ai<al || (ai==al && bi<bl)) lo=i;
            if(ai>ah || (ai==ah && bi>bh)) hi=i;
        }
        return lo==hi ? vector<uint32_t>{lo} : vector<uint32_t>{lo,hi};
    }

    // Single backtracking + insertion operation, used by BOTH passes.
    // Accept is evaluated AFTER backtracking (matching the original behind test).
    struct AlwaysAccept { bool operator()(uint32_t) const { return true; } };
    template<class Accept=AlwaysAccept>
    inline bool testandsetreflexive(vector<uint32_t>& stack,const Point* p,uint32_t next,
                                    int direction=1,Accept accept={}) {
        while(stack.size()>=2 &&
              direction*Point::get_direction(p[stack[stack.size()-2]],p[stack.back()],p[next])<=EPS) {
            stack.pop_back();
        }
        if(!accept(stack.empty() ? next : stack.back())) return false;
        stack.push_back(next);
        return true;
    }

    // After sorting by pseudo-angle, keep only the farthest point on each ray.
    // Compaction is in place; there is no duplicate vector of candidates.
    void compress(vector<uint32_t>& order,const vector<double>& ang,const vector<double>& r2) {
        if(order.empty()) return;
        size_t length=0;
        uint32_t best=order[0];
        double angle=ang[best];
        for(size_t k=1;k<order.size();++k) {
            uint32_t i=order[k];
            if(fabs(ang[i]-angle)<=EPS) {
                if(r2[i]>r2[best]) best=i;
            } else {
                order[length++]=best;
                best=i;
                angle=ang[i];
            }
        }
        order[length++]=best;
        if(length>1 && angle_delta(ang[order[0]],ang[order[length-1]])<=EPS) {
            if(r2[order[0]]>=r2[order[length-1]]) --length;
            else { order[0]=order[length-1]; --length; }
        }
        order.resize(length);
    }
    vector<uint32_t> cycle(const Point* points,const vector<uint32_t>& order) {
        vector<uint32_t> stack;
        stack.reserve(order.size());
        for(uint32_t i:order) testandsetreflexive(stack,points,i);
        size_t head=0;
        again:
        if(stack.size()-head>=3 &&
           Point::get_direction(points[stack[stack.size()-2]],points[stack.back()],points[stack[head]])<=EPS) {
            stack.pop_back();
            goto again;
        }
        if(stack.size()-head>=3 &&
           Point::get_direction(points[stack.back()],points[stack[head]],points[stack[head+1]])<=EPS) {
            ++head;
            goto again;
        }
        if(head) {
            move(stack.begin()+head,stack.end(),stack.begin());
            stack.resize(stack.size()-head);
        }
        // The monotone outside-edge sweep expects cyclic CCW order.
        // This constant-time winding probe replaces any polygon-area pass.
        if(stack.size()>=3 && Point::get_direction(points[stack[0]],points[stack[1]],points[stack[2]])<0)
            reverse(stack.begin(),stack.end());
        return stack;
    }
    vector<uint32_t> hull_subset(const Point* points,const uint32_t* selected,size_t n) {
        if(!n) return {};
        vector<uint32_t> degenerate=collinear(points,selected,n);
        if(!degenerate.empty()) return degenerate;
        Point c=mean(points,n,selected);
        vector<double> ang(n),r2(n);
        vector<uint32_t> ord=Point::sort(points,n,Point::SortMode::Angle,false,c,
                                   ang.data(),r2.data(),selected);
        compress(ord,ang,r2);
        if(selected) for(uint32_t& id:ord) id=selected[id];
        return cycle(points,ord);
    }

    // This is the ACTUAL split operation: choose the axis AND distribute
    // the radius-ordered input into the two sequences processed by workers.
    struct Split { char axis='x'; uint32_t start=0; vector<uint32_t> a,b; };
    Split determineSplitaxis(const Point* points,const vector<uint32_t>& magnitude, Point origin) {
        Split out;
        if(magnitude.empty()) return out;
        out.start=magnitude[0];
        Point F=points[out.start]-origin;
        int sxF=sign(F.cartpoint[0]),syF=sign(F.cartpoint[1]);
        for(size_t k=1;k<magnitude.size();++k) {
            Point P=points[magnitude[k]]-origin;
            int sx=sign(P.cartpoint[0]),sy=sign(P.cartpoint[1]);
            bool xc=sx && sxF && sx!=sxF, yc=sy && syF && sy!=syF;
            if(!(xc || yc)) continue;
            if(xc!=yc) out.axis=xc ? 'y' : 'x';
            else out.axis=(F.cartpoint[0]>EPS || F.cartpoint[1]<=EPS) ? 'x' : 'y';
            break;
        }
        out.a.reserve(magnitude.size()/2+1);
        out.b.reserve(magnitude.size()/2+1);
        for(size_t k=1;k<magnitude.size();++k) {
            uint32_t i=magnitude[k];
            int base=out.axis=='x' ? sign(F.cartpoint[1]) : sign(F.cartpoint[0]);
            int value=out.axis=='x' ? sign(points[i].y()-origin.y()) : sign(points[i].x()-origin.x());
            if(value==0) continue; // Preserved for Pass 2, not permanently dropped.
            (value==base ? out.a : out.b).push_back(i);
        }
        // stack.top() must yield A in descending radius and B in reverse.
        reverse(out.a.begin(),out.a.end());
        return out;
    }
    enum Half { LOWER,UPPER,LEFT,RIGHT };
    inline Half half_name(Point F,char axis,int side) {
        return axis=='x' ? ((F.cartpoint[1]<0)==(side==1) ? LOWER : UPPER)
                         : ((F.cartpoint[0]<0)==(side==1) ? LEFT : RIGHT);
    }
    inline double local_angle(double t,char axis,Half h) {
        if(axis=='x') return h==UPPER ? t : t-HALF;
        return h==LEFT ? t-QUARTER : (t<HALF ? t+QUARTER : t-3*QUARTER);
    }
    vector<uint32_t> worker(const vector<uint32_t>& owned,const Point* points,Point origin,
                      const vector<double>& ang,const Split& split,int side) {
        vector<uint32_t> hull;
        hull.reserve(owned.size()+1);
        hull.push_back(split.start);
        if(owned.empty()) return hull;
        Half h=half_name(points[split.start]-origin,split.axis,side);
        int direction=local_angle(ang[owned.back()],split.axis,h)<QUARTER ? 1 : -1;
        // LIFO traversal of already organized input without copying an input stack.
        // Identical visit order to std::stack::top/pop(), but ownership stays with Split.
        for(size_t pos=owned.size();pos>0;) {
            uint32_t i=owned[--pos];
            double t=local_angle(ang[i],split.axis,h);
            testandsetreflexive(hull,points,i,direction,[&](uint32_t last) {
                return hull.size()==1 ||
                       direction*(t-local_angle(ang[last],split.axis,h))>EPS;
            });
        }
        return hull;
    }
    vector<uint32_t> rough_pass(const Point* points,size_t n,
                                const vector<uint32_t>* initial_order=nullptr,
                                const vector<double>* initial_angles=nullptr,
                                const Split* initial_split=nullptr,
                                const Point* initial_center=nullptr) {
        const Point c=initial_center ? *initial_center : mean(points,n);
        vector<double> local_angles;
        vector<uint32_t> local_order;
        if(!initial_angles || !initial_order) {
            local_angles.resize(n);
            local_order=Point::sort(points,n,Point::SortMode::Radius,
                                    true,c,local_angles.data());
        }
        const vector<double>& ang=initial_angles ? *initial_angles : local_angles;
        const vector<uint32_t>& magnitude=initial_order ? *initial_order : local_order;
        Split local_split;
        if(!initial_split) local_split=determineSplitaxis(points,magnitude,c);
        const Split& split=initial_split ? *initial_split : local_split;
        vector<uint32_t> A,B;
        if(inner_parallel) {
            thread t([&] { A=worker(split.a,points,c,ang,split,1); });
            B=worker(split.b,points,c,ang,split,-1);
            t.join();
        } else {
            A=worker(split.a,points,c,ang,split,1);
            B=worker(split.b,points,c,ang,split,-1);
        }
        A.reserve(A.size()+B.size());
        if(B.size()>1) A.insert(A.end(),B.begin()+1,B.end());
        return hull_subset(points,A.data(),A.size());
    }

    struct Order { vector<uint32_t> sorted; vector<double> ang,r2; };
    vector<uint8_t> outside(const Point* points,vector<uint32_t>& h,const Order& order) {
        // `cycle()` has already normalized all nondegenerate H0 chains to
        // counterclockwise order.  With that invariant, a negative cross
        // product against the ray-intersected edge means strictly outside.
        // Do not re-check/flip winding in the per-point classification.
        auto smallest=min_element(h.begin(),h.end(),[&](uint32_t a,uint32_t b) {
            return order.ang[a]<order.ang[b];
        });
        rotate(h.begin(),smallest,h.end());
        double a0=order.ang[h.front()];
        vector<uint8_t> marked(order.sorted.size());
        for(uint32_t i:h) marked[i]=2;
        size_t cut=0,edge=0;
        while(cut<order.sorted.size() && order.ang[order.sorted[cut]]<a0-EPS) ++cut;
        for(int sweep=0;sweep<2;++sweep) {
            size_t begin=sweep==0 ? cut : 0;
            size_t end=sweep==0 ? order.sorted.size() : cut;
            for(size_t k=begin;k<end;++k) {
                uint32_t i=order.sorted[k];
                double a=order.ang[i]+(sweep==1 ? TURN : 0);
                while(edge+1<h.size()) {
                    double b=order.ang[h[edge+1]];
                    if(b<a0-EPS) b+=TURN;
                    if(a<=b+EPS) break;
                    ++edge;
                }
                if(marked[i]&2) continue;
                uint32_t next=edge+1<h.size() ? h[edge+1] : h[0];
                if(Point::get_direction(points[h[edge]],points[next],points[i])<-EPS) marked[i]|=1;
            }
        }
        return marked;
    }
    Result hull_compute(const Point* points,size_t n,
                        const vector<uint32_t>* prepared_order=nullptr,
                        const vector<double>* prepared_angles=nullptr,
                        const Split* prepared_split=nullptr,
                        const Point* prepared_center=nullptr) {
        Result result;
        if(!n) return result;
        vector<uint32_t> d=collinear(points,nullptr,n);
        if(!d.empty()) { result.hull=d; result.h0=move(d); return result; }
        result.h0=rough_pass(points,n,prepared_order,prepared_angles,prepared_split,prepared_center);
        Point origin=result.h0.size()>=3 ? mean(points,result.h0.size(),result.h0.data())
                                      : mean(points,n);
        Order order;
        order.ang.resize(n);
        order.r2.resize(n);
        order.sorted=Point::sort(points,n,Point::SortMode::Angle,false,origin,
                                 order.ang.data(),order.r2.data());
        if(result.h0.size()<3) {
            compress(order.sorted,order.ang,order.r2);
            result.hull=cycle(points,order.sorted);
            result.outside_n=n;
        } else {
            vector<uint8_t> mark=outside(points,result.h0,order);
            size_t len=0;
            for(uint32_t i:order.sorted) {
                if(mark[i]&1) ++result.outside_n;
                if(mark[i]) order.sorted[len++]=i;
            }
            order.sorted.resize(len);
            compress(order.sorted,order.ang,order.r2);
            result.hull=cycle(points,order.sorted);
        }
        return result;
    }
}

template<class ValueAt>
vector<uint32_t> Point::radixSort(size_t n, ValueAt value_at, bool reverse,
                            unsigned digit_bits, unsigned workers, bool parallel) {
    if(n>UINT32_MAX) throw length_error("32-bit point index overflow");
    if(digit_bits<1 || digit_bits>16) throw invalid_argument("digit_bits must be in [1,16]");
    if(workers<1 || workers>32) throw invalid_argument("workers must be in [1,32]");

    using Key=uint64_t;
    const Key sign_bit=Key(1)<<63;
    const unsigned bucket_count=1u<<digit_bits;
    const Key mask=bucket_count-1;
    const unsigned passes=(64+digit_bits-1)/digit_bits;

    vector<Key> keys(n);
    vector<uint32_t> order(n), scratch(n);
    iota(order.begin(),order.end(),uint32_t(0));

    for(size_t i=0;i<n;++i) {
        double value=value_at(i);
        if(isnan(value)) throw invalid_argument("NaN radix key");
        if(value==0) value=0; // -0 and +0 share a key
        Key bits=bit_cast<Key>(value);
        Key key=(bits&sign_bit) ? ~bits : (bits^sign_bit);
        keys[i]=reverse ? ~key : key;
    }
    if(n<2) return order;

    uint32_t* src=order.data();
    uint32_t* dst=scratch.data();
    const bool do_parallel=parallel && hull_detail::inner_parallel &&
                           n>=300000 && workers>1;

    if(!do_parallel) {
        vector<uint32_t> count(bucket_count);
        for(unsigned pass=0;pass<passes;++pass) {
            fill(count.begin(),count.end(),0);
            const unsigned shift=pass*digit_bits;
            for(size_t k=0;k<n;++k)
                ++count[(keys[src[k]]>>shift)&mask];

            uint32_t offset=0;
            for(uint32_t& value:count) {
                const uint32_t amount=value;
                value=offset;
                offset+=amount;
            }
            for(size_t k=0;k<n;++k) {
                const uint32_t id=src[k];
                dst[count[(keys[id]>>shift)&mask]++]=id;
            }
            swap(src,dst);
        }
    } else {
        vector<uint32_t> counts(size_t(workers)*bucket_count);
        barrier sync{ptrdiff_t(workers)};
        auto run=[&](unsigned tid) {
            const size_t lo=n*tid/workers;
            const size_t hi=n*(tid+1)/workers;
            uint32_t* count=counts.data()+size_t(tid)*bucket_count;

            for(unsigned pass=0;pass<passes;++pass) {
                const unsigned shift=pass*digit_bits;
                fill(count,count+bucket_count,0);
                uint32_t* s=src;
                uint32_t* d=dst;

                for(size_t k=lo;k<hi;++k)
                    ++count[(keys[s[k]]>>shift)&mask];
                sync.arrive_and_wait();

                if(tid==0) {
                    uint32_t offset=0;
                    for(unsigned bucket=0;bucket<bucket_count;++bucket) {
                        for(unsigned t=0;t<workers;++t) {
                            uint32_t& value=counts[size_t(t)*bucket_count+bucket];
                            const uint32_t amount=value;
                            value=offset;
                            offset+=amount;
                        }
                    }
                }
                sync.arrive_and_wait();

                for(size_t k=lo;k<hi;++k) {
                    const uint32_t id=s[k];
                    d[count[(keys[id]>>shift)&mask]++]=id;
                }
                sync.arrive_and_wait();
                if(tid==0) swap(src,dst);
                sync.arrive_and_wait();
            }
        };

        vector<thread> threads;
        threads.reserve(workers-1);
        for(unsigned tid=1;tid<workers;++tid) threads.emplace_back(run,tid);
        run(0);
        for(thread& t:threads) t.join();
    }

    if(src!=order.data()) copy(src,src+n,order.data());
    return order;
}

vector<uint32_t> Point::sort(const vector<double>& values, bool reverse,
                       unsigned digit_bits, unsigned workers, bool parallel) {
    return radixSort(values.size(),[&](size_t i) { return values[i]; },
                     reverse,digit_bits,workers,parallel);
}

vector<uint32_t> Point::sort(const Point* points,size_t n,SortMode mode,bool reverse,
                       const Point& origin,double* angular,double* squared,
                       const uint32_t* selected,unsigned digit_bits,unsigned workers,
                       bool parallel) {
    const double ox=origin.x(),oy=origin.y();
    if(!isfinite(ox)||!isfinite(oy)) throw invalid_argument("nonfinite origin");
    return radixSort(n,[&](size_t k) {
        const Point& p=points[selected ? selected[k] : k];
        const double x=p.x()-ox, y=p.y()-oy;
        if(!isfinite(x)||!isfinite(y)) throw invalid_argument("nonfinite point");
        const double r2=x*x+y*y;
        if(squared) squared[k]=r2;
        const double a=(mode==SortMode::Angle || angular) ? angle4(x,y) : 0;
        if(angular) angular[k]=a;
        switch(mode) {
            case SortMode::Radius: return r2;
            case SortMode::Angle: return a;
            case SortMode::X: return x;
            case SortMode::Y: return y;
        }
        return r2;
    },reverse,digit_bits,workers,parallel);
}

class hullcomp {
public:
    struct Edge { uint32_t from,to; };
    enum class Criteria { Behind,Outside };
private:
    // No custom Vec, no duplicated public-versus-private stacks.
    vector<Point> opts_;
    Point center_{};
    size_t current_=0;
    vector<uint32_t> hull_,h0_;
    vector<uint32_t> sections_[4];
    vector<uint32_t> union_ids_;
    vector<uint32_t> working_chain_;
    vector<uint32_t> magnitude_;
    vector<double> first_angles_;
    hull_detail::Split split_;
    bool prepared_=false;
    unsigned section_count_=0;

    void set_points(const Point* p,size_t n,bool prepare=true) {
        if(n>UINT32_MAX) throw length_error("32-bit point index overflow");
        if(n) opts_.assign(p,p+n);
        else opts_.clear();
        double sx=0,sy=0;
        for(const Point& q:opts_) {
            const double x=q.x(),y=q.y();
            if(!isfinite(x)||!isfinite(y)) throw invalid_argument("nonfinite point");
            sx+=x;sy+=y;
        }
        center_=n ? Point(sx/n,sy/n) : Point();
        hull_.clear();h0_.clear();
        section_count_=0;
        working_chain_.clear();
        magnitude_.clear();
        first_angles_.clear();
        split_=hull_detail::Split{};
        prepared_=false;
        if(n && prepare) prepareInput();
    }

    void prepareInput() {
        if(prepared_ || opts_.empty()) return;
        first_angles_.resize(opts_.size());
        magnitude_=Point::sort(opts_.data(),opts_.size(),Point::SortMode::Radius,
                               true,center_,first_angles_.data());
        split_=hull_detail::determineSplitaxis(opts_.data(),magnitude_,center_);
        current_=split_.start;
        prepared_=true;
    }

public:
    hullcomp()=default;
    // For outer 2/4 complete-local-hull reduction, pass prepare=false: it
    // avoids global magnitude sorting that those independent local hulls do
    // not consume. The default eagerly constructs the two-half input order.
    explicit hullcomp(const vector<Point>& pts,bool prepare=true) {
        set_points(pts.data(),pts.size(),prepare);
    }
    hullcomp(const Point* pts,size_t n,bool prepare=true) { set_points(pts,n,prepare); }
    void setPoints(const vector<Point>& pts) { set_points(pts.data(),pts.size()); }
    size_t size() const { return opts_.size(); }
    size_t hullSize() const { return hull_.size(); }
    size_t H0Size() const { return h0_.size(); }
    const Point& point(uint32_t id) const { return opts_.at(id); }
    const Point& center() const { return center_; }
    size_t get_current_point() const { return current_; }
    Point current_point() const { return current_<opts_.size() ? opts_[current_]-center_ : Point(); }
    void set_current_point(size_t id) { current_=id; }

    bool is_behind(const Point& p) const {
        return current_<opts_.size() && ((opts_[current_]-center_)*p)<-hull_detail::EPS;
    }
    bool isInside(const Point& p) const {
        const auto& h=h0_.size()>=3 ? h0_ : hull_;
        if(h.size()<3) return false;
        Point q{p.x(),p.y()};
        const auto& a=opts_[h[0]];
        if(Point::get_direction(a,opts_[h[1]],q)<-hull_detail::EPS ||
           Point::get_direction(a,opts_[h.back()],q)>hull_detail::EPS) return false;
        size_t lo=1,hi=h.size()-1;
        while(hi-lo>1) {
            size_t mid=lo+(hi-lo)/2;
            if(Point::get_direction(a,opts_[h[mid]],q)>=0) lo=mid;
            else hi=mid;
        }
        return Point::get_direction(opts_[h[lo]],opts_[h[hi]],q)>=-hull_detail::EPS;
    }
    bool isvalidentry(const Point& p,Criteria mode=Criteria::Behind) const {
        return mode==Criteria::Behind ? !is_behind(p) : !isInside(p);
    }
    void testandsetreflexive(uint32_t id=UINT32_MAX,int direction=1) {
        if(id==UINT32_MAX) id=uint32_t(current_);
        if(id>=opts_.size()) throw out_of_range("invalid point ID");
        hull_detail::testandsetreflexive(working_chain_,opts_.data(),id,direction);
        current_=id;
    }
    void clearWorkingChain() { working_chain_.clear(); }
    vector<Point> getWorkingChain() const {
        vector<Point> out;
        out.reserve(working_chain_.size());
        for(uint32_t i:working_chain_) out.push_back(opts_[i]);
        return out;
    }
    pair<Point,Point> getEndpoints(const vector<Point>& pts) const {
        return pts.empty() ? pair<Point,Point>{Point(),Point()} : make_pair(pts.front(),pts.back());
    }

    // Prepared by the constructor by default (or on demand if prepare=false).
    // Indices point into opts_; no duplicated Point array or input-stack copy.
    char determineSplitaxis() {
        prepareInput();
        return split_.axis;
    }
    const vector<uint32_t>& half(unsigned which) const {
        if(which>1) throw out_of_range("half must be 0 or 1");
        return which==0 ? split_.a : split_.b;
    }
    const vector<uint32_t>& magnitudeOrder() const { return magnitude_; }

    // The sections are COMPLETE local hulls, not geometric quadrants.
    // Merging their extreme vertices is exact in any dimension.
    void merge() {
        size_t n=0;
        for(unsigned i=0;i<section_count_;++i) n+=sections_[i].size();
        union_ids_.clear();
        union_ids_.reserve(n);
        for(unsigned i=0;i<section_count_;++i) {
            union_ids_.insert(union_ids_.end(),sections_[i].begin(),sections_[i].end());
        }
        hull_=hull_detail::hull_subset(opts_.data(),union_ids_.data(),n);
        h0_=hull_;
    }
    // Future, separate operation: merge two already complete intersecting hulls.
    // Intentionally NOT implemented by calling the main merge() routine.
    static vector<Point> mergeOrrevise(const vector<Point>& a,
                                      const vector<Point>& b,bool repair=true) {
        (void)a;(void)b;(void)repair;
        throw logic_error("mergeOrrevise: independent intersecting-hull merge not implemented");
    }

    // Single entry point: no redundant compute() wrapper.
    hullcomp& operator()(unsigned sections=4) {
        hull_.clear(); h0_.clear();
        section_count_=0;
        for(auto& s:sections_) s.clear();
        if(opts_.empty()) return *this;
        if(sections<=1) {
            prepareInput();
            hull_detail::inner_parallel=true;
            auto r=hull_detail::hull_compute(opts_.data(),opts_.size(),
                                               &magnitude_,&first_angles_,&split_,&center_);
            hull_=move(r.hull);
            h0_=move(r.h0);
            sections_[0]=hull_;
            section_count_=1;
            return *this;
        }
        section_count_=sections>=4 ? 4 : 2;
        hull_detail::Result local[4];
        size_t lo[4]{},hi[4]{};
        thread workers[4];
        for(unsigned i=0;i<section_count_;++i) {
            lo[i]=opts_.size()*i/section_count_;
            hi[i]=opts_.size()*(i+1)/section_count_;
            workers[i]=thread([&,i] {
                hull_detail::inner_parallel=false;
                local[i]=hull_detail::hull_compute(opts_.data()+lo[i],hi[i]-lo[i]);
            });
        }
        for(unsigned i=0;i<section_count_;++i) workers[i].join();
        size_t total=0;
        for(unsigned i=0;i<section_count_;++i) total+=local[i].hull.size();
        union_ids_.clear();
        union_ids_.reserve(total);
        for(unsigned i=0;i<section_count_;++i) {
            for(uint32_t id:local[i].hull) union_ids_.push_back(uint32_t(lo[i]+id));
        }
        hull_=hull_detail::hull_subset(opts_.data(),union_ids_.data(),total);
        h0_=hull_;
        for(unsigned i=0;i<section_count_;++i) {
            sections_[i]=move(local[i].hull);
            for(uint32_t& id:sections_[i]) id+=uint32_t(lo[i]);
        }
        return *this;
    }
    vector<Point> getVertexList() const {
        vector<Point> out;
        out.reserve(hull_.size());
        for(uint32_t i:hull_) out.push_back(opts_[i]);
        return out;
    }
    // Assignment output helper; faces and polygon-area APIs are deliberately
    // left for later.  This runs only when requested, never in the hull kernel.
    vector<Edge> getEdgeList() const {
        vector<Edge> out;
        out.reserve(hull_.size());
        if(hull_.size()<2) return out;
        for(size_t i=0;i<hull_.size();++i) {
            out.push_back(Edge{hull_[i],hull_[(i+1)%hull_.size()]});
        }
        return out;
    }
    // TODO (class project): getFaceList(), getArea(), graph representation.
    vector<Point> getH0() const {
        vector<Point> out;
        out.reserve(h0_.size());
        for(uint32_t i:h0_) out.push_back(opts_[i]);
        return out;
    }
    vector<Point> getSectionHull(unsigned index) const {
        if(index>=section_count_) return {};
        vector<Point> out;
        out.reserve(sections_[index].size());
        for(uint32_t i:sections_[index]) out.push_back(opts_[i]);
        return out;
    }
};

#ifdef HULL_BENCH_MAIN
int main(int argc,char**argv) {
    if(argc<3) { cerr<<"usage: benchmark sections file.bin [file2...]\n"; return 2; }
    unsigned sections=unsigned(stoul(argv[1]));
    for(int a=2;a<argc;++a) {
        ifstream in(argv[a],ios::binary|ios::ate);
        if(!in) { cerr<<"failed to open "<<argv[a]<<'\n'; return 2; }
        size_t bytes=size_t(in.tellg());
        in.seekg(0);
        if(bytes%(2*sizeof(double))) { cerr<<"invalid point file\n"; return 2; }
        size_t n=bytes/(2*sizeof(double));
        vector<double> raw(2*n);
        in.read(reinterpret_cast<char*>(raw.data()),bytes);
        vector<Point> pts;
        pts.reserve(n);
        for(size_t i=0;i<n;++i) pts.emplace_back(raw[2*i],raw[2*i+1]);
        hullcomp H(pts);
        double times[5];
        size_t count=0;
        for(double& t:times) {
            auto start=chrono::steady_clock::now();
            H(sections);
            count=H.hullSize();
            t=chrono::duration<double>(chrono::steady_clock::now()-start).count();
        }
        sort(begin(times),end(times));
        cout<<argv[a]<<','<<n<<','<<sections<<','<<times[2]<<','<<count<<'\n';
    }
}
#endif
