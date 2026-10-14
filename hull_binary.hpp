#pragma once
#include "hull_scalable.hpp"
// Arbitrarily chosen compile-time BINARY significand precision.
// Unlike plain cpp_bin_float<N>, this uses digit_base_2 explicitly.
namespace hull_binary {
    template<unsigned Bits> using Real = scalable::BinaryReal<Bits>;
    template<unsigned Bits> using Point = scalable::Point<Real<Bits>>;
    template<unsigned Bits> using Hull = scalable::Hull<Real<Bits>>;
    using Index = scalable::Index;
    using RadixOptions = scalable::RadixOptions;
    using RadixStats = scalable::RadixStats;
}
