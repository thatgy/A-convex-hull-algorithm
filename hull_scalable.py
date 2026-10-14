"""Direct NumPy float64 / longdouble bindings for the scalable hull.

Compute uses native C++20 worker threads. NumPy longdouble is passed as raw
bytes with no conversion through ctypes.c_double or Python float.
"""
import ctypes as ct
import os
from pathlib import Path
import sys
from dataclasses import dataclass
import numpy as np
from decimal import Decimal

_libname='_hull_scalable.dll' if sys.platform=='win32' else '_hull_scalable.dylib' if sys.platform=='darwin' else '_hull_scalable.so'
lib=ct.CDLL(os.fspath(Path(__file__).resolve().with_name(_libname)))

def fn(name, result, *args):
    f=getattr(lib, name)
    f.restype=result
    f.argtypes=args
    return f

last_error=fn('sc_last_error',ct.c_char_p)
ld_bytes=fn('sc_ld_bytes',ct.c_int)
ld_mant=fn('sc_ld_mant_dig',ct.c_int)
quad_mant=fn('sc_bin128_mant_dig',ct.c_int)
q256_mant=fn('sc_bin256_mant_dig',ct.c_int)
P=ct.c_void_p; U=ct.c_uint;N=ct.c_size_t;I=ct.c_int
pi=ct.POINTER(ct.c_uint32)

def bind(suffix):
    istext = suffix in ('b128','b256')
    return dict(create=fn('sc_new'+suffix,P,ct.POINTER(ct.c_char_p) if istext else P,N,U),
                free=fn('sc_free'+suffix,None,P),
                run=fn('sc_run'+suffix,I,P,U),
                count=fn('sc_count'+suffix,N,P,I,U),
                indices=fn('sc_ids'+suffix,I,P,I,U,pi,N),
                points=None if istext else fn('sc_points'+suffix,I,P,I,U,P,N),
                endpoints=fn('sc_ends'+suffix,I,P,U,pi,I),
                radix=fn('sc_radix'+suffix,I,P,U,I),
                radixstats=fn('sc_radixstats'+suffix,I,P,U,pi),
                sort=fn('sc_sort'+suffix,I,ct.POINTER(ct.c_char_p) if istext else P,N,I,U,I,pi))

_API64=bind('64')
_APILD=bind('ld')
_APIB128=bind('b128')
_APIB256=bind('b256')

def check(code):
    if code:
        err=last_error()
        raise RuntimeError(err.decode(errors='replace') if err else 'native error')

@dataclass(frozen=True)
class Point:
    x: object
    y: object
    def __iter__(self):
        yield self.x
        yield self.y

@dataclass
class Node:
    value: object
    prev: object = None
    next: object = None

class DoublyLinkedList:
    def __init__(self, values):
        nodes=[Node(v) for v in values]
        self.head=nodes[0] if nodes else None
        self.size=len(nodes)
        for i,node in enumerate(nodes):
            node.prev=nodes[(i-1)%len(nodes)]
            node.next=nodes[(i+1)%len(nodes)]
    def __len__(self):return self.size
    def __iter__(self):
        p=self.head
        for _ in range(self.size):
            yield p.value
            p=p.next

def _worker_count(value):
    if value == 'auto':
        value = len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else (os.cpu_count() or 4)
        # Honor CFS CPU limits when exposed by Linux cgroup v2.
        try:
            quota,period=Path('/sys/fs/cgroup/cpu.max').read_text().split()[:2]
            if quota!='max':value=min(value,max(1,int(int(quota)/int(period))))
        except (OSError,ValueError,ZeroDivisionError):pass
    value=int(value)
    if not 1<=value<=256:raise ValueError('workers must be between 1 and 256')
    return value

class hullcomp:
    def __init__(self, points, workers=4, real='double',
                 radix_bits=16, detect_radix_digits=True):
        self.real=real
        if real not in ('double','extended','binary128','binary256'):
            raise ValueError("real must be 'double', 'extended', 'binary128', or 'binary256'")
        self.dtype=np.dtype(np.float64 if real=='double' else
                            np.longdouble if real=='extended' else object)
        self.api={'double':_API64,'extended':_APILD,'binary128':_APIB128,'binary256':_APIB256}[real]
        if real=='extended' and (ld_bytes()!=self.dtype.itemsize):
            raise RuntimeError('NumPy longdouble layout differs from compiler long double')
        self.workers=_worker_count(workers)
        self.radix_bits=int(radix_bits)
        self.detect_radix_digits=bool(detect_radix_digits)
        if not 0<=self.radix_bits<=16:raise ValueError('radix_bits must be 0 (auto) or 1..16')
        self._set(points,create=True)
    def _set(self,points,create=False):
        arr=np.asarray(points,dtype=self.dtype)
        if arr.size==0 and arr.ndim==1:arr=arr.reshape(0,2)
        if arr.ndim!=2 or arr.shape[1]!=2:raise ValueError('points must be an Nx2 array')
        if len(arr)>2**32-1:raise ValueError('uint32 indices exceeded')
        if self.real in ('binary128','binary256'):
            self.data=np.empty(arr.shape,dtype=object)
            values=[]
            for i,row in enumerate(arr):
                for j,value in enumerate(row):
                    d=Decimal(str(value))
                    if not d.is_finite():raise ValueError('point coordinates must be finite')
                    self.data[i,j]=d
                    values.append(str(d).encode('ascii'))
            encoded=(ct.c_char_p*len(values))(*values)
            self._ptr=self.api['create'](encoded,len(self.data),self.workers)
        else:
            self.data=np.ascontiguousarray(arr)
            if not np.isfinite(self.data).all():raise ValueError('point coordinates must be finite')
            self._ptr=self.api['create'](self.data.ctypes.data,len(self.data),self.workers)
        if not self._ptr:check(-1)
        self.setRadixOptions(self.radix_bits,self.detect_radix_digits)
    def setRadixOptions(self, bits=16, detect=True):
        self._require()
        if not 0<=int(bits)<=16:raise ValueError('radix bits must be 0 (auto) or 1..16')
        check(self.api['radix'](self._ptr,int(bits),int(bool(detect))))
        self.radix_bits=int(bits)
        self.detect_radix_digits=bool(detect)
    def radixStats(self):
        self._require()
        labels=('key_bits','bits_per_digit','planned_passes','executed_passes','skipped_passes')
        out=[]
        for j in range(self.sectionCount()):
            values=np.zeros(5,dtype=np.uint32)
            check(self.api['radixstats'](self._ptr,j,values.ctypes.data_as(pi)))
            out.append(dict(zip(labels,map(int,values))))
        return out
    def __call__(self, workers=None):
        self._require()
        if workers is not None:
            self.workers=_worker_count(workers)
        check(self.api['run'](self._ptr,self.workers))
        return self
    def close(self):
        if getattr(self,'_ptr',None):self.api['free'](self._ptr);self._ptr=None
    def __del__(self):self.close()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
    def _require(self):
        if not getattr(self,'_ptr',None):raise RuntimeError('native hull has been closed')
        return self._ptr
    def _count(self,kind=0,part=0):
        self._require()
        n=int(self.api['count'](self._ptr,kind,part))
        if n==ct.c_size_t(-1).value:check(-1)
        return n
    def vertex_indices(self):return self._ids(0,0)
    def section_indices(self,section):return self._ids(1,section)
    def _ids(self,kind,section):
        n=self._count(kind,section);out=np.empty(n,dtype=np.uint32)
        check(self.api['indices'](self._ptr,kind,int(section),out.ctypes.data_as(pi),n))
        return out
    def _coords(self,kind,section):
        if self.real in ('binary128','binary256'):
            return self.data[self._ids(kind,section)].copy()
        n=self._count(kind,section);out=np.empty((n,2),dtype=self.dtype)
        check(self.api['points'](self._ptr,kind,int(section),out.ctypes.data,n))
        return out
    def vertices(self):return self._coords(0,0)
    def getVertexList(self):return self.vertices()
    def hullSize(self):return self._count(0,0)
    def size(self):return self._count(2,0)
    def sectionCount(self):return self._count(3,0)
    def getSectionHull(self,index):return self._coords(1,index)
    def getEndpoints(self,index=0,indices=False):
        self._require()
        ids=np.zeros(2,np.uint32)
        check(self.api['endpoints'](self._ptr,index,ids.ctypes.data_as(pi),0))
        if indices:return tuple(map(int,ids))
        return tuple(Point(self.data[i,0],self.data[i,1]) for i in ids)
    def getSectionExtrema(self,index=0,indices=False):
        self._require()
        ids=np.zeros(2,np.uint32)
        check(self.api['endpoints'](self._ptr,index,ids.ctypes.data_as(pi),1))
        if indices:return tuple(map(int,ids))
        return tuple(Point(self.data[i,0],self.data[i,1]) for i in ids)
    def iter_vertices(self):
        # generator with no additional native point array; references own data
        for i in self.vertex_indices():
            yield Point(self.data[i,0],self.data[i,1])
    def __iter__(self):return self.iter_vertices()
    def iter_indices(self):
        for i in self.vertex_indices():yield int(i)
    def iter_edges(self):
        ids=self.vertex_indices()
        for k,i in enumerate(ids):yield (int(i),int(ids[(k+1)%len(ids)]))
    def getEdgeList(self):return list(self.iter_edges())
    def toPointLinkedList(self):return DoublyLinkedList(self.iter_vertices())
    def toIndexLinkedList(self):return DoublyLinkedList(self.iter_indices())


def sort_values(values, real='double', reverse=False, radix_bits=16,
                detect_radix_digits=True):
    """Stable native radix sort, returning original input indices.

    radix_bits=0 chooses a width automatically; 1..16 chooses explicitly.
    binary128 accepts Decimal or decimal string values without a float64 cast.
    """
    if real not in ('double','extended','binary128','binary256'):
        raise ValueError("real must be 'double', 'extended', 'binary128', or 'binary256'")
    if not 0<=int(radix_bits)<=16:raise ValueError('radix_bits must be 0..16')
    api={'double':_API64,'extended':_APILD,'binary128':_APIB128,'binary256':_APIB256}[real]
    if real in ('binary128','binary256'):
        val=[str(Decimal(str(v))).encode('ascii') for v in values]
        data=(ct.c_char_p*len(val))(*val)
        ptr=data
        n=len(val)
    else:
        dtype=np.float64 if real=='double' else np.longdouble
        data=np.ascontiguousarray(np.asarray(values,dtype=dtype).ravel())
        n=data.size
        ptr=data.ctypes.data
    if n>2**32-1:raise ValueError('uint32 index limit')
    out=np.empty(n,dtype=np.uint32)
    check(api['sort'](ptr,n,int(bool(reverse)),int(radix_bits),
                      int(bool(detect_radix_digits)),out.ctypes.data_as(pi)))
    return out

def convex_hull(points,workers=4,real='double',indices=False,radix_bits=16,detect_radix_digits=True):
    with hullcomp(points,workers,real,radix_bits,detect_radix_digits) as h:
        h()
        return h.vertex_indices() if indices else h.vertices()


def precision_info():
    return {'double_bits':64,'longdouble_bytes':ld_bytes(),'longdouble_mantissa_bits':ld_mant(),
            'numpy_longdouble_bytes':np.dtype(np.longdouble).itemsize,
            'binary128_mantissa_bits':quad_mant(), 'binary128_radix_key_bits':176, 'binary256_mantissa_bits':q256_mant(),
            'binary256_radix_key_bits':304}

# Exact geometry for native double or longdouble input values. Integer ratios
# are extracted from the original scalar format (never via Python float).
from fractions import Fraction
from typing import NamedTuple

class ExactPoint(NamedTuple):
    x: Fraction
    y: Fraction

@dataclass(frozen=True)
class ExactIntersection:
    kind: str
    points: tuple
    line1: str
    line2: str
    edge1: object=None
    edge2: object=None
    @property
    def point(self):return self.points[0] if self.kind=='point' else None

def _frac(x):
    if isinstance(x,(int,np.integer)):return Fraction(int(x))
    if isinstance(x,(str,Decimal)):return Fraction(x)
    if not np.isfinite(x):raise ValueError('finite coordinates required')
    numerator,denominator=x.as_integer_ratio()
    return Fraction(numerator,denominator)

def _exact_xy(a):
    x,y=a
    return (_frac(x),_frac(y))

def _cross2(a,b):return a[0]*b[1]-a[1]*b[0]

def _exact_line(a,b):
    if a[0]==b[0]:return f'x = {a[0]}'
    m=(b[1]-a[1])/(b[0]-a[0])
    return f'y - ({a[1]}) = ({m}) * (x - ({a[0]}))'

def segment_intersection(a,b,c,d):
    """Exact finite-segment intersection for integer, float64 or longdouble.

    A native longdouble is converted to its exact rational value, preserving
    all available significand bits instead of rounding to a Python float.
    """
    a,b,c,d=map(_exact_xy,(a,b,c,d))
    r=(b[0]-a[0],b[1]-a[1]);s=(d[0]-c[0],d[1]-c[1])
    e=(c[0]-a[0],c[1]-a[1]);den=_cross2(r,s)
    line1,line2=_exact_line(a,b),_exact_line(c,d)
    def none():return ExactIntersection('none',(),line1,line2)
    def result(p):return ExactIntersection('point',(ExactPoint(*p),),line1,line2)
    if den:
        t=_cross2(e,s)/den;u=_cross2(e,r)/den
        if not(0<=t<=1 and 0<=u<=1):return none()
        return result((a[0]+t*r[0],a[1]+t*r[1]))
    def on(p,x,y):
        return _cross2((p[0]-x[0],p[1]-x[1]),(y[0]-x[0],y[1]-x[1]))==0 and all(min(x[k],y[k])<=p[k]<=max(x[k],y[k]) for k in (0,1))
    if r==(0,0):return result(a) if on(a,c,d) else none()
    if s==(0,0):return result(c) if on(c,a,b) else none()
    if _cross2(e,r):return none()
    axis=0 if r[0] else 1
    t1=(c[axis]-a[axis])/r[axis];t2=(d[axis]-a[axis])/r[axis]
    lo=max(Fraction(0),min(t1,t2));hi=min(Fraction(1),max(t1,t2))
    if lo>hi:return none()
    def v(t):return ExactPoint(a[0]+t*r[0],a[1]+t*r[1])
    if lo==hi:return ExactIntersection('point',(v(lo),),line1,line2)
    return ExactIntersection('overlap',(v(lo),v(hi)),line1,line2)

def _hull_intersections(self, other, max_pairs=10_000_000):
    first=self.vertex_indices(); second=other.vertex_indices()
    if len(first)*len(second)>max_pairs:raise ValueError('too many edge pairs for O(h1*h2) intersection; raise max_pairs intentionally')
    for i,a in enumerate(first):
        a1=int(a);a2=int(first[(i+1)%len(first)])
        for j,b in enumerate(second):
            b1=int(b);b2=int(second[(j+1)%len(second)])
            it=segment_intersection(self.data[a1],self.data[a2],other.data[b1],other.data[b2])
            if it.kind!='none':
                yield ExactIntersection(it.kind,it.points,it.line1,it.line2,(a1,a2),(b1,b2))

hullcomp.intersections=_hull_intersections
