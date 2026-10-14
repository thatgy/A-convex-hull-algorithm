#pragma once
#include <boost/multiprecision/cpp_int.hpp>
#include <bit>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <algorithm>

// Exact for the *stored binary64 coordinates*; no floating slope/division.
namespace hull_exact {
using boost::multiprecision::cpp_int;

struct Rational {
    cpp_int num=0, den=1;
    Rational()=default;
    Rational(cpp_int n,cpp_int d=1):num(std::move(n)),den(std::move(d)) { normalize(); }
    static Rational fromDouble(double x) {
        if (!std::isfinite(x)) throw std::invalid_argument("non-finite endpoint");
        uint64_t bits=std::bit_cast<uint64_t>(x);
        bool neg=(bits>>63)!=0;
        uint64_t exp=(bits>>52)&2047, mant=bits&((uint64_t(1)<<52)-1);
        if (exp) mant|=uint64_t(1)<<52;
        if (!mant) return {};
        int shift=exp ? int(exp)-1023-52 : -1074;
        cpp_int n=mant,d=1;
        if (shift<0) d<<=-shift; else n<<=shift;
        if (neg) n=-n;
        return Rational(std::move(n),std::move(d));
    }
    void normalize() {
        if (den==0) throw std::invalid_argument("zero denominator");
        if (den<0) { num=-num;den=-den; }
        if (num==0) { den=1;return; }
        cpp_int a=num<0?-num:num,b=den;
        while (b!=0) { cpp_int t=a%b;a=b;b=t; }
        num/=a;den/=a;
    }
    std::string str() const {
        std::string n=num.convert_to<std::string>();
        if (den==1) return n;
        return n+"/"+den.convert_to<std::string>();
    }
};
inline Rational operator+(const Rational&a,const Rational&b) {return {a.num*b.den+b.num*a.den,a.den*b.den};}
inline Rational operator-(const Rational&a,const Rational&b) {return {a.num*b.den-b.num*a.den,a.den*b.den};}
inline Rational operator*(const Rational&a,const Rational&b) {return {a.num*b.num,a.den*b.den};}
inline Rational operator/(const Rational&a,const Rational&b) {return {a.num*b.den,a.den*b.num};}
inline bool operator==(const Rational&a,const Rational&b) {return a.num==b.num&&a.den==b.den;}
inline bool operator<(const Rational&a,const Rational&b) {return a.num*b.den<b.num*a.den;}
inline bool operator<=(const Rational&a,const Rational&b) {return !(b<a);}

struct ExactPoint {
    Rational x,y;
};
inline ExactPoint fromXY(double x,double y) {return {Rational::fromDouble(x),Rational::fromDouble(y)};}
inline ExactPoint operator+(const ExactPoint&a,const ExactPoint&b) {return {a.x+b.x,a.y+b.y};}
inline ExactPoint operator-(const ExactPoint&a,const ExactPoint&b) {return {a.x-b.x,a.y-b.y};}
inline ExactPoint operator*(const ExactPoint&a,const Rational&t) {return {a.x*t,a.y*t};}
inline bool operator==(const ExactPoint&a,const ExactPoint&b) {return a.x==b.x&&a.y==b.y;}
inline Rational cross(const ExactPoint&a,const ExactPoint&b) {return a.x*b.y-a.y*b.x;}

inline std::string linePointSlope(const ExactPoint&a,const ExactPoint&b) {
    if (a==b) return "point ("+a.x.str()+", "+a.y.str()+")";
    if (a.x==b.x) return "x = "+a.x.str();
    Rational slope=(b.y-a.y)/(b.x-a.x);
    return "y - ("+a.y.str()+") = ("+slope.str()+") * (x - ("+a.x.str()+"))";
}

struct Intersection {
    // 'none', 'point', or 'overlap'. Overlap has two endpoint coordinates.
    std::string kind="none";
    std::vector<ExactPoint> points;
    std::string line1,line2;
    uint32_t edge1=0,edge2=0;
};

inline Intersection segmentIntersection(double ax,double ay,double bx,double by,
                                        double cx,double cy,double dx,double dy) {
    const ExactPoint a=fromXY(ax,ay),b=fromXY(bx,by);
    const ExactPoint c=fromXY(cx,cy),d=fromXY(dx,dy);
    Intersection out;
    out.line1=linePointSlope(a,b);
    out.line2=linePointSlope(c,d);
    ExactPoint r=b-a,s=d-c,q=c-a;
    Rational z(0),one(1);
    const Rational denom=cross(r,s);
    if (denom.num!=0) {
        Rational t=cross(q,s)/denom, u=cross(q,r)/denom;
        if(z<=t && t<=one && z<=u && u<=one) {
            out.kind="point";
            out.points.push_back(a+r*t);
        }
        return out;
    }
    // A point segment is checked against the other supporting segment.
    auto onSegment=[](const ExactPoint&p,const ExactPoint&u,const ExactPoint&v) {
        if(cross(v-u,p-u).num!=0) return false;
        return std::min(u.x,v.x)<=p.x && p.x<=std::max(u.x,v.x)
            && std::min(u.y,v.y)<=p.y && p.y<=std::max(u.y,v.y);
    };
    if(a==b || c==d) {
        if(a==b && onSegment(a,c,d)) {out.kind="point";out.points.push_back(a);}
        else if(c==d && onSegment(c,a,b)) {out.kind="point";out.points.push_back(c);}
        return out;
    }
    if(cross(q,r).num!=0) return out; // distinct parallel supporting lines
    // Collinear: parameterize C and D on the first segment, intersect ranges.
    Rational t0=(r.x.num!=0 ? (c.x-a.x)/r.x : (c.y-a.y)/r.y);
    Rational t1=(r.x.num!=0 ? (d.x-a.x)/r.x : (d.y-a.y)/r.y);
    Rational lo=std::max(z,std::min(t0,t1)), hi=std::min(one,std::max(t0,t1));
    if(hi<lo) return out;
    out.points.push_back(a+r*lo);
    out.kind="point";
    if(lo<hi) {
        out.kind="overlap";
        out.points.push_back(a+r*hi);
    }
    return out;
}
}
