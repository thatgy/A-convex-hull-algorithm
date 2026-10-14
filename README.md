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

Build the shared library with `python build.py`. Requirements: Python 3, NumPy, GCC/Clang with C++20, Boost.Multiprecision headers. SciPy is required **only for the supplied reference tests**, not for hull computation.

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
