"""End-to-end regressions for half-plane classification and scale invariance."""

import math
import unittest

from rf_searchlab.core.geometry import HalfPlane as H
from rf_searchlab.geometry import intersect_halfplanes, solve


def box(x, y, width, height, multiplier=1.0):
    return [
        H(multiplier, 0, multiplier * (x + width)),
        H(-multiplier, 0, -multiplier * x),
        H(0, multiplier, multiplier * (y + height)),
        H(0, -multiplier, -multiplier * y),
    ]


class HalfplaneRegressions(unittest.TestCase):
    def test_reported_small_squares(self):
        for side in (1.0, 3e-5, 2e-8, 1e-12):
            with self.subTest(side=side):
                result = intersect_halfplanes(box(0, 0, side, side))
                self.assertEqual(result.kind, "polygon")
                self.assertEqual(len(result.vertices), 4)
                self.assertTrue(math.isclose(result.diameter_m, math.sqrt(2) * side, rel_tol=1e-12))

    def test_thin_and_translated_rectangles(self):
        for x, y, width, height in [(0, 0, 2.0, 1e-12), (1200, -1000, 1e-8, 2e-8), (1e9, -1e9, 1.0, 2.0)]:
            with self.subTest(case=(x, y, width, height)):
                result = intersect_halfplanes(box(x, y, width, height))
                self.assertEqual(result.kind, "polygon")
                # Expected side lengths are those representable in the input bounds.
                self.assertTrue(
                    math.isclose(
                        result.diameter_m, math.hypot((x + width) - x, (y + height) - y), rel_tol=1e-12
                    )
                )

    def test_positive_constraint_rescaling(self):
        for factor in (1e-15, 1.0, 1e15):
            result = intersect_halfplanes(box(0, 0, 2, 1, factor))
            self.assertEqual(result.kind, "polygon")
            self.assertAlmostEqual(result.diameter_m, math.sqrt(5), places=12)

    def test_tiny_inconsistent_constraints_are_empty(self):
        result = intersect_halfplanes([H(1, 0, 0), H(-1, 0, -1e-12), H(0, 1, 1), H(0, -1, 0)])
        self.assertEqual(result.kind, "empty")

    def test_degenerate_and_unbounded_exact_sets(self):
        for width, height, kind in [(0.0, 0.0, "point"), (1e-12, 0.0, "segment")]:
            result = intersect_halfplanes(box(0, 0, width, height))
            self.assertEqual(result.kind, kind)
        for constraints in ([], [H(1, 0, 1)], [H(1, 0, 0), H(-1, 0, 0)]):
            self.assertEqual(intersect_halfplanes(constraints).kind, "unbounded")

    def test_bearing_pipeline_scales_and_roundoff_metadata(self):
        baseline = None
        for scale in (1e-12, 1e-8, 1.0, 1000.0):
            result = solve([[-scale, 0], [0, -scale], [scale, 0], [0, scale]], [0, 90, 180, 270])
            self.assertEqual(result["kind"], "polygon")
            self.assertGreater(result["diameter_m"], 0)
            ratio = result["diameter_m"] / scale
            if baseline is None:
                baseline = ratio
            self.assertTrue(math.isclose(ratio, baseline, rel_tol=1e-11))
            self.assertLess(result["numerical_angle_guard_rad"], 1e-12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
