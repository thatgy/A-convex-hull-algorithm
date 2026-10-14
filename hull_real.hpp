#pragma once
#include "hull_scalable.hpp"
// GCC/x86-64: usually 80-bit extended precision in 16-byte storage.
// Keep Real as long double throughout the computation: no implicit binary64 cast.
namespace hull_real {
using Real = long double;
using Point = scalable::Point<Real>;
using Hull = scalable::Hull<Real>;
using Index = scalable::Index;
}
