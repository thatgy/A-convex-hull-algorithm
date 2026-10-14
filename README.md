# Scalable convex hull — extensible `Real` and adaptive radix digits

This is an extension of the existing radial / behind / reflex / second-pass convex-hull implementation.  It retains independently computed partial hulls, configurable workers (default 4), the candidate-first second pass, sorting-free multiway merge, C++ coroutine iteration, and the Python hull/linked-list interfaces.

## New: compile-time flexible numerical precision

| C++ type | Real significand bits | Encoded radix-key bits | Default 16-bit digit passes (maximum) |
|---|---:|---:|---:|
| `float` | 24 | 32 | 2 |
| `double` | 53 | 64 | 4 |
| `long double` (GCC x86-64 x87) | 64 | 80 | 5 |
| `scalable::BinaryReal<128>` | 128 | 176 | 11 |
| `scalable::BinaryReal<256>` | 256 | 304 | 19 |
| `scalable::BinaryReal<512>` | 512 | 560 | 35 |

The generic `BinaryReal<Bits>` type is built on `boost::multiprecision::cpp_bin_float<Bits, digit_base_2>`. The number is **binary significand bits**, not decimal digits (plain `cpp_bin_float<Bits>` defaults to decimal-digit interpretation). Wider specializations require no sorter source edits; they instantiate more key words and passes at compile time. These are binary floating-point numbers, *not* exact rational numbers or runtime-variable-precision MPFR objects.

The key encoding uses native IEEE-754 bit transformations for `float`/`double`, 80-bit x87 exponent/significand excluding its 6 padding bytes for supported `long double`, and a sign + biased exponent + full normalized mantissa for Boost binary types. Signed numbers, positive/negative zero, equal keys, and descending order are supported. NaN/infinity radix keys are rejected. For non-x87 `long double` layouts, add an appropriate specialized key encoder; silent byte reinterpretation is explicitly avoided.

## New: variable radix digits and automatic detection

`RadixOptions{bits_per_digit, detect_uniform_digits}` determines the radix behavior:

- `bits_per_digit=16` (default): up to 65,536 buckets, usually the best general-purpose choice at large `n`.
- `bits_per_digit=1..16`: use that exact many bits per digit. The number of passes is computed from the **actual encoded key width**, not a hard-coded four or five.
- `bits_per_digit=0`: automatically choose **8** bits for fewer than 4,096 keys, **11** bits for fewer than 65,536 keys, otherwise **16** bits. This heuristic may be slower or faster than 16 depending on the data.
- `detect_uniform_digits=true`: for each possible pass, check whether every key has the same digit. If so, skip that entire counting/scatter pass. For already sorted inputs, the existing early-order check skips the entire sorting process.

The key width must include exponent, sign, and mantissa information. We do **not** infer precision from the largest decimal number (a tiny fractional difference can require hundreds of significant bits). Rather, we determine representational width from `numeric_limits<Real>::digits` and detect **which digits actually vary** over this input. This preserves stable sorting and numeric order.

For fixed precision `B`, digit size `d`, and fixed worker count, sorting costs `O(n * ceil(B/d))` and the hull retains the earlier fixed-width linear-time operation bound. If precision `B` grows with `n`, the complexity contains that factor—it is **not** uniform `O(n)` for arbitrarily growing precision.

`radixStats()` reports the **last radix call per local worker**, not every call cumulatively. Workers handling fewer than three points may report zero passes because no radix call was needed.

## C++ use

```cpp
#include "hull_binary.hpp"
#include <vector>

using Real = hull_binary::Real<256>;
using Point = hull_binary::Point<256>;

std::vector<Point> points = {
    {Real("0"), Real("0")}, {Real("1"), Real("0")},
    {Real("1"), Real("1")}, {Real("0"), Real("1")}
};

// Four workers; automatic bits per digit; detect unchanging digits.
hull_binary::Hull<256> H(4, {0, true});
H.set(std::move(points));
H.compute(8); // Optionally repartition across 8 workers.

for (const auto& p : H.iter_vertices()) {
    // p.x and p.y are original-coordinate Real<256> values.
}

auto stats = H.radixStats();
```

You can also sort arbitrary signed `Real` values stably:

```cpp
using R = scalable::BinaryReal<512>;
std::vector<R> values = {R("1"), R("1.00000000000000000000000000001"), R("-2")};
scalable::RadixStats stats;
auto indices = scalable::Hull<R>::sortValues(values, false, {0, true}, &stats);
```

For fast native doubles:

```cpp
#include "hull_double.hpp"
hull_double::Hull H(4, {16,true});
```

The compile-time type parameter adds no per-point dynamic dispatch. Wider precision itself uses more memory and software multiprecision arithmetic, however.

## Python use

Build the shared library with `python build.py`. Requirements: Python 3, NumPy, GCC/Clang with C++20, Boost.Multiprecision headers. 
```python
from decimal import Decimal
from hull_scalable import hullcomp, sort_values, precision_info

pts = [
    ["0", "0"],
    ["1.000000000000000000000000000000000000000000000000000000000001", "0"],
    ["1.000000000000000000000000000000000000000000000000000000000001", "1"],
    ["0", "1"],
]

with hullcomp(pts, workers=8, real="binary256", radix_bits=0,
              detect_radix_digits=True) as h:
    h()
    coords = h.vertices()      # NumPy object array of Decimal, original coordinates
    ids = h.vertex_indices()   # uint32 NumPy array
    print(h.radixStats())      # Per-worker last-sort pass counts
    for point in h:            # Python generator
        print(point)

order = sort_values([Decimal("2"), Decimal("-1"), Decimal("2")],
                    real="binary128", radix_bits=16)
# [1, 0, 2]  (stable equal keys)

print(precision_info())
```

Supported Python `real=` modes: `"double"`, `"extended"` (C++ `long double`, NumPy `longdouble`), `"binary128"`, and `"binary256"`. Binary modes convert **decimal strings or `Decimal`** directly in C++ without a float64 intermediate. Python returns the originally supplied decimal-coordinate values as `Decimal`; the native computation uses the nearest representable binary `Real` values, not unbounded exact decimals. Supply sufficient Python `decimal` context precision when creating fine-grained input values by arithmetic.

`radix_bits=0` automatically selects the digit width, while `detect_radix_digits=True` additionally skips uniform digits. `sort_values` returns original 32-bit index positions, not rearranged floating-point objects.

## Correctness testing

- Existing 12 Python regression tests: passed.
- Native 32/64/80/128/256/**512**-bit radix tests: passed for ascending/descending signed values, ties, stable ordering, and digit sizes `0,1,3,7,8,11,16` with detection enabled/disabled.
- Python stable-sort and hull comparisons across all four Python Real modes, digit sizes `0,8,11,16`, and SciPy reference on 8,000 random points: passed.
- A `1 + 10^-34` coordinate difference survives in the 128-bit Python path, and a `1 + 10^-60` difference survives in the 256-bit path.
- Previously reported geometric limitations remain: native orientation is floating-point rather than adaptive exact geometry. Digit selection does not resolve near-collinear roundoff or input precision already lost before parsing.

## Performance and limitations

 Higher precision is costly, particularly multiprecision types. All benchmark timings are medians of three runs from this single constrained environment, not claims about all hardware. Python-to-C++ data construction is not included in native hull comparisons.

The Python binary128/binary256 input path currently constructs a decimal-string array and converts it into the compiled multiprecision format; this is **not** a zero-copy NumPy ABI. For huge arrays, `double`/`long double` remains markedly more efficient.

## Project contents

- `hull_scalable.hpp`: all precision-specialized hull/radix logic.
- `hull_float.hpp`, `hull_double.hpp`, `hull_real.hpp`, `hull_binary.hpp`: convenience aliases.
- `hull_c_api.cpp`, `hull_scalable.py`, `build.py`: Python bindings.
- `baseline/`: unchanged earlier scalable source for comparison.
  
### Precision-scalable radial/reflex convex hull

The scalable implementation supports `float`, `double`, x87 `long double`, and Boost binary floating-point values with arbitrary **compile-time** mantissa precision, including 128, 256, and 512 bits. Radix keys include each type's full significand and exponent. Digit widths are configurable from 1 to 16 bits; automatic mode chooses an initial width by input size and may skip radix passes whose digits do not vary.

**Full-hull performance — 1 million points, four workers** (milliseconds; medians of three measurements):

| Input / precision | Earlier scalable | Extended radix (16-bit digits) | Extended radix (auto digits) |
|---|---:|---:|---:|
| Uniform / `double` | 49.5 | **32.9** | 39.3 |
| Uniform / `long double` | 113.2 | 104.3 | **97.7** |
| Circle / `double` | 54.7 | **51.7** | 59.1 |
| Circle / `long double` | 119.6 | **113.4** | 124.4 |

All configurations returned 31 vertices on uniform input and all one million vertices on circle input. These runs were performed sequentially in the same environment; timing variation means that small differences are not necessarily repeatable. The main purpose of this revision is extensibility and correctness across radix precisions, not a guaranteed performance improvement.

**Different `Real` precisions — 100,000 integer-coordinate points, four workers** (median of three, milliseconds):

| `Real` | Significand bits | Radix key bits | Time | Hull vertices |
|---|---:|---:|---:|---:|
| `float` | 24 | 32 | **2.58** | 32 |
| `double` | 53 | 64 | 2.90 | 32 |
| `long double` (x87) | 64 | 80 | 5.98 | 32 |
| Binary multiprecision | 128 | 176 | 79.37 | 32 |
| Binary multiprecision | 256 | 304 | 178.87 | 32 |

For a fixed precision `B` and radix digit width `d`, radix sorting requires no more than `ceil(B/d)` passes, and can skip uniform digits. With fixed precision and fixed worker count the operation-count upper bound remains `O(n)`. When increasing precision with problem size, include the factor `ceil(B/d)` rather than claiming an unconditional linear-time bound. Geometric predicates are still floating-point and are not exact/adaptive.


#Importing:Use Hull_normal for drop in support
## Using `hull_normal` (C++ and Python)

`hull_normal` is the unified interface to the scalable, parallel convex-hull algorithm. It supports configurable thread counts, radix-sort policies, original-coordinate hull iteration, and multiple numeric precisions. You do **not** have to import or include a different hull implementation for every precision.

### 1. Files and requirements

Keep these files together in your project:

```text
hull_normal.hpp          # Main C++ interface
hull_scalable.hpp        # Core templated C++ algorithm
hull_normal.py           # Main Python interface
hull_scalable.py         # Python-to-native wrapper
hull_c_api.cpp           # Native interface used by Python
build.py                 # Builds the Python shared library
hull_exact.hpp           # Optional exact segment-intersection utility

# Optional convenience headers (can stay in your repository):
hull_float.hpp
hull_double.hpp
hull_real.hpp
hull_binary.hpp
```

**Requirements:** C++20 compiler, pthread support, and Boost.Multiprecision headers for C++; Python 3 and NumPy for Python usage. Boost.Multiprecision is header-only for the supported binary types. The bundled tests may also use SciPy, but SciPy is not required to run the hull algorithm.

For C++, `#include "hull_normal.hpp"` automatically brings in `hull_scalable.hpp`. You do not need to include the individual `hull_double.hpp`, `hull_real.hpp`, or `hull_binary.hpp` headers as well.

For Python, build the native library **in the same directory as** `hull_scalable.py`:

```bash
python -m pip install numpy
python build.py
```

On Linux, this generates `_hull_scalable.so`; on Windows, `_hull_scalable.dll`; on macOS, `_hull_scalable.dylib`. You do not need to commit the compiled library to GitHub if users can run `build.py` themselves. The build script expects a working C++20 compiler (`g++` by default, configurable through the `CXX` environment variable).

### 2. C++ — simplest example

```cpp
#include "hull_normal.hpp"
#include <iostream>
#include <vector>
#include <utility>

int main() {
    using Point = hull::Point<double>;
    std::vector<Point> points = {
        {0,0}, {3,0}, {3,2}, {0,2}, {1,1}
    };

    hull_normal<double> H(4);       // double precision, 4 workers
    H.set(std::move(points));       // provide original coordinates
    H();                            // calculate the convex hull

    std::cout << "Hull size: " << H.count() << '\n';
    for (auto p : H.iter_vertices()) {
        std::cout << p.x << ", " << p.y << '\n';
    }
    for (auto id : H.iter_indices()) {
        std::cout << "Input index: " << id << '\n';
    }

    H(8);                           // recompute using 8 workers
    return 0;
}
```

Compile on Linux with:

```bash
g++ -std=c++20 -O3 -march=native -pthread main.cpp -o hull_demo
./hull_demo
```

`H.indices()` also returns the full hull-index vector. `H.points()` accesses the original stored point array. Coordinates returned by the hull are **not translated around its temporary center**.

### 3. C++ — precision, threads, and radix modes

```cpp
hull_normal<>                                       default_hull(4);  // double, fast
hull_normal<float>                                  float_hull(4);
hull_normal<double>                                 double_hull(8);
hull_normal<long double>                            extended_hull(8);
hull_normal<hull::binary<128>>                      high_hull(8);
hull_normal<hull::binary<256>, hull::mode::adaptive> wide_hull(16);
hull_normal<hull::binary<512>, hull::mode::full_radix> huge_hull(4);

// Override radix settings explicitly: 11 bits/digit, skip uniform digits.
hull_normal<double> custom_hull(8, {11, true});
```

| C++ radix mode | Behavior |
|---|---|
| `hull::mode::fast` (default) | 16-bit radix digits; skip uniform digits |
| `hull::mode::adaptive` | Choose an 8-, 11-, or 16-bit digit width using a size heuristic |
| `hull::mode::full_radix` | 16-bit radix digits; do not skip uniform digits |

For all C++ precision modes, call `.set(points)` and then `.compute()` or `H()`. Use `.compute(threads)` or `H(threads)` to change the worker count; the default is four. The implementation supports up to 256 workers, although using more workers than available cores can slow execution.

To preserve more than double precision when providing high-precision constants, construct values from strings, for example `hull::binary<256>("1.000000000000000000000000001")`, instead of passing a previously rounded `double`.

### 4. C++ — partial hulls, endpoints, linked lists, and radix statistics

```cpp
// After H() has computed the hull:
auto ends = H.getEndpoints(0);      // index pair for partial hull 0
const auto& points = H.points();
auto first = points[ends.first];    // original-coordinate Point
auto last  = points[ends.last];

auto extremes = H.getExtrema(0);    // leftmost/rightmost partial-hull IDs
auto part_ids = H.section(0);       // indices for partial hull 0

auto vertex_list = H.toPointLinkedList();
if (vertex_list.head) {
    auto* next = vertex_list.head->next;
    auto* prev = vertex_list.head->prev;
}

auto index_list = H.toIndexLinkedList();
auto stats = H.radixStats();        // one stats entry per worker
```

Both lists are **circular doubly linked lists** that own their returned values. C++ `getEndpoints(part)` returns the **first and last indices in that partial hull's current ordering**, not necessarily the correct tangent bridge for splicing with another hull.

### 5. Python — simplest example

```python
import numpy as np
from hull_normal import hull_normal, convex_hull

points = np.array([
    [0, 0], [3, 0], [3, 2], [0, 2], [1, 1]
], dtype=np.float64)

with hull_normal(points, threads=4, precision="double") as H:
    H.compute()                   # or H()

    vertices = H.vertices()       # Nx2 NumPy array of original coordinates
    indices = H.vertex_indices()  # uint32 indices into original points
    edges = H.getEdgeList()       # pairs of original point indices

    for point in H:               # Python generator
        print(point.x, point.y)  # fields, not x()/y() methods

    for index in H.iter_indices():
        print(index)

    H.compute(8)                  # recompute with eight workers

# Quick one-shot use:
vertices = convex_hull(points, threads=4, precision="double")
```

`hull_normal.py` imports `hull_scalable.py`, which loads the compiled native library. Run Python from the project directory (or install/add the folder to `PYTHONPATH`) so it can find these modules.

### 6. Python — precision and sorting modes

```python
from decimal import Decimal
from hull_normal import hull_normal, sort_values

# Extended precision (native C++ long double)
H = hull_normal(points, threads=8, precision="extended", mode="adaptive")
H()
H.close()

# High precision: pass decimal strings or Decimal, not float intermediates.
precise = [
    ["0", "0"],
    ["1.000000000000000000000000000001", "0"],
    ["1.000000000000000000000000000001", "1"],
    ["0", "1"],
]
with hull_normal(precise, threads=4, precision="binary256") as H:
    H()
    print(H.vertices())           # Decimal values in a NumPy object array

# Stable ascending radix sort: returns ORIGINAL positions, not sorted values.
order = sort_values([Decimal("2"), Decimal("-1"), Decimal("2")],
                    precision="binary128", mode="fast")
print(order.tolist())             # [1, 0, 2]
```

Available Python `precision=` values: `"double"`, `"extended"`, `"binary128"`, and `"binary256"`. Additional C++ precisions (such as `float` or binary512) require additional native binding exports before they can be selected from Python.

Use `mode="fast"`, `"adaptive"`, or `"full_radix"`. You can override the mode with `radix_bits=0` (automatic) or `1..16`, and `detect_radix_digits=True/False`. `threads="auto"` lets the Python wrapper estimate the available CPU allowance.

### 7. Python — partial hulls, linked lists, and exact intersections

```python
from hull_normal import segment_intersection

with hull_normal(points, threads=4) as H:
    H()
    a, b = H.getEndpoints(0)             # original-coordinate Point objects
    a_id, b_id = H.getEndpoints(0, indices=True)
    partial = H.getSectionHull(0)        # Nx2 array of section vertices
    linked = H.toPointLinkedList()       # circular doubly linked list
    linked_ids = H.toIndexLinkedList()
    if linked.head:
        print(linked.head.value, linked.head.next.value)
    print(H.radixStats())

hit = segment_intersection((0, 0), (3, 0), (1, -1), (2, 2))
print(hit.kind)       # "point"
print(hit.point.x)    # Fraction(4, 3), exactly
print(hit.line1)      # point-slope line equation
```

To find intersections between two computed hulls, use `H1.intersections(H2)` while both hull objects remain open. That checks each edge pair, so it costs `O(h1*h2)` and is not intended for two giant all-extreme hulls without spatial indexing.

For C++, the optional exact-segment utility is separate:

```cpp
#include "hull_exact.hpp"
auto hit = hull_exact::segmentIntersection(0,0,3,0,1,-1,2,2);
// hit.kind == "point", hit.points[0].x.str() == "4/3"
```

### 8. Notes and common issues

- **Python import failure / missing `_hull_scalable.so` or `.dll`:** run `python build.py` in the folder containing `hull_c_api.cpp`. The binary must match your operating system, architecture, and Python wrapper's expected exported functions.
- **Missing Boost headers during C++ compilation:** install the Boost development headers, or configure your compiler's include path to their location.
- **No need to check in binaries:** keep source, build script, documentation, and tests on GitHub; generated `.so`, `.dll`, and `.dylib` files can go in `.gitignore`.
- **Original indices:** the hull stores 32-bit unsigned input indices (up to `UINT32_MAX`); the coordinates use your selected `Real` type. These index arrays are not atomic.
- **Precision:** C++ template precision adds no dynamic type-dispatch overhead, but larger `Real` values require more storage and arithmetic. Multiprecision floating point does not automatically make orientation predicates mathematically exact.
- **Algorithmic complexity:** for fixed precision and a fixed worker count, the code uses bounded radix passes. With `k` variable workers, the multiway merge costs up to `O(H log k)` for `H` partial-hull vertices. The final-hull correctness guarantee still depends on the geometric predicates and recovery logic.

