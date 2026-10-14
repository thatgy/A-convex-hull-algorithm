#pragma once
#include "hull_scalable.hpp"
// This lightweight interface selects IEEE-754 binary64 arithmetic.
namespace hull_double {
using Real = double;
using Point = scalable::Point<Real>;
using Hull = scalable::Hull<Real>;
using Index = scalable::Index;
}
