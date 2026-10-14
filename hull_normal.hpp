#pragma once

// One public C++ interface for every supported precision and radix policy.
// The radial/behind/reflex hull algorithm itself lives in hull_scalable.hpp.
#include "hull_scalable.hpp"

namespace hull {

using Index = scalable::Index;
using RadixOptions = scalable::RadixOptions;
using RadixStats = scalable::RadixStats;

template<class Real> using Point = scalable::Point<Real>;
template<unsigned Bits> using binary = scalable::BinaryReal<Bits>;

// Mode classes are compile-time radix policies; no virtual methods or
// per-point mode branch is introduced. The worker count is independent.
namespace mode {
struct fast {
    static constexpr RadixOptions options() noexcept { return {16, true}; }
};
struct adaptive {
    static constexpr RadixOptions options() noexcept { return {0, true}; }
};
struct full_radix {
    static constexpr RadixOptions options() noexcept { return {16, false}; }
};
} // namespace mode

template<class Real = double, class Mode = mode::fast>
class hull_normal : public scalable::Hull<Real> {
    using Base = scalable::Hull<Real>;
public:
    using P = Point<Real>;
    using I = Index;
    using precision_type = Real;
    using operating_mode = Mode;

    explicit hull_normal(unsigned threads = 4)
        : Base(threads, Mode::options()) {}

    // Explicit override for experiments with digit widths 1..16.
    hull_normal(unsigned threads, RadixOptions radix)
        : Base(threads, radix) {}

    // H(8) reruns with eight workers; H() reuses constructor worker count.
    hull_normal& operator()(unsigned threads = 0) {
        this->compute(threads);
        return *this;
    }

    // The underlying implementation retains the original-coordinate vertices,
    // generators, linked lists, raw point indices, and per-worker statistics.
};

} // namespace hull

// Convenient global name: hull_normal<double> H(4);
template<class Real = double, class Mode = hull::mode::fast>
using hull_normal = hull::hull_normal<Real, Mode>;
