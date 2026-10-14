"""One Python entry point for the native scalable convex hull.

Examples:
    H = hull_normal(points, threads=4, precision='double'); H()
    H = hull_normal(points, threads=8, precision='extended', mode='adaptive'); H()
    H = hull_normal(points, threads=16, precision='binary256'); H()

The C++ template chooses precision at compile time; Python selects among the
instantiations that were exported in the compiled native library.
"""

from hull_scalable import (
    hullcomp as _Hull,
    Point, DoublyLinkedList, ExactPoint, ExactIntersection,
    sort_values as _sort_values,
    segment_intersection, convex_hull as _convex_hull,
    precision_info,
)

_MODES = {
    'fast': (16, True),
    'adaptive': (0, True),
    'full_radix': (16, False),
}

class hull_normal(_Hull):
    """Scalable native hull with unified precision, mode and thread controls.

    threads: 1..256, default 4. 'auto' selects available CPU allowance.
    precision: 'double', 'extended', 'binary128', or 'binary256'.
    mode: 'fast' (16-bit radix + digit skipping), 'adaptive' (auto digit width),
          'full_radix' (16-bit radix without uniform-digit skipping).
    Optional radix_bits/detect_radix_digits override the selected mode.
    """
    def __init__(self, points, threads=4, precision='double', mode='fast',
                 *, radix_bits=None, detect_radix_digits=None):
        if mode not in _MODES:
            raise ValueError('mode must be fast, adaptive, or full_radix')
        width, detect = _MODES[mode]
        if radix_bits is not None:
            width = radix_bits
        if detect_radix_digits is not None:
            detect = detect_radix_digits
        self.mode = mode
        super().__init__(points, workers=threads, real=precision,
                         radix_bits=width, detect_radix_digits=detect)

    def compute(self, threads=None):
        """Compute or recompute the hull, optionally repartitioning workers."""
        return self(threads)

    @property
    def threads(self):
        return self.workers

    @property
    def precision(self):
        return self.real


def convex_hull(points, threads=4, precision='double', mode='fast', *,
                indices=False, radix_bits=None, detect_radix_digits=None):
    """Return a NumPy array of hull vertices or original input indices."""
    with hull_normal(points, threads, precision, mode,
                     radix_bits=radix_bits,
                     detect_radix_digits=detect_radix_digits) as h:
        h()
        return h.vertex_indices() if indices else h.vertices()


def sort_values(values, precision='double', mode='fast', *, reverse=False,
                radix_bits=None, detect_radix_digits=None):
    if mode not in _MODES:
        raise ValueError('mode must be fast, adaptive, or full_radix')
    width, detect = _MODES[mode]
    if radix_bits is not None: width = radix_bits
    if detect_radix_digits is not None: detect = detect_radix_digits
    return _sort_values(values, real=precision, reverse=reverse,
                        radix_bits=width, detect_radix_digits=detect)

# Compatibility alias for code expecting the earlier class name.
hullcomp = hull_normal

__all__ = [
    'hull_normal', 'hullcomp', 'convex_hull', 'sort_values',
    'Point', 'DoublyLinkedList', 'ExactPoint', 'ExactIntersection',
    'segment_intersection', 'precision_info',
]
