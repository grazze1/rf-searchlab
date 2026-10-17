"""Independent all-pairs oracle and tests of the actual Region call chain."""

import math
import unittest
from fractions import Fraction
from unittest.mock import patch

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist

from rf_searchlab.core import geometry as frozen
from rf_searchlab.geometry import intersect_halfplanes, rotating_calipers, same_diameter_circle, solve


class RegionCalipers(unittest.TestCase):
    def check_oracle(self, p):
        d = rotating_calipers(p)
        expected = float(pdist(np.asarray(p, float)).max()) if len(p) > 1 else 0.0
        self.assertTrue(
            math.isclose(d.diameter_m, expected, rel_tol=1e-12, abs_tol=1e-25), (p, d.diameter_m, expected)
        )
        self.assertEqual(d.squared_m2, sum((a - b) ** 2 for a, b in zip(d.pD, d.qD)))
        self.assertLessEqual(d.support_advances, 2 * d.vertex_count)
        self.assertLessEqual(d.candidate_pairs, 4 * d.vertex_count)
        return d

    def test_degenerate_and_scale(self):
        rect = np.array([[0, 0], [2, 0], [2, 1], [0, 1]], float)
        cases = [
            [[4, 3]],
            [[4, 3], [4, 3]],
            [[1, 2], [5, 2]],
            [[1, 0], [0, 0], [3, 0], [2, 0]],
            rect,
            rect[::-1],
            np.repeat(rect, 3, axis=0),
            np.vstack([rect, rect[0]]),
            rect * 1e-10,
            rect * 1e8 + 1e12,
            rect * [1, 1e-12],
            [[0, 0], [1, 0], [2, 0], [2, 1], [1, 1], [0, 1]],
            [[1, 0], [2, 0], [2, 1], [0, 1], [0, 0]],
        ]
        for case in cases:
            with self.subTest(case=np.asarray(case).tolist()):
                self.check_oracle(case)
        with self.assertRaises(ValueError):
            rotating_calipers([])
        with self.assertRaises(ValueError):
            rotating_calipers([[0, 0], [math.nan, 1]])

    def test_parallel_ties(self):
        d = self.check_oracle([[0, 0], [2, 0], [2, 1], [0, 1]])
        self.assertEqual(d.parallel_ties, 4)
        self.assertEqual(d.candidate_pairs, 16)

    def test_random_hulls_and_cyclic_starts(self):
        rng = np.random.default_rng(20260913)
        for index in range(600):
            cloud = rng.normal(size=(int(rng.integers(3, 300)), 2))
            p = cloud[ConvexHull(cloud).vertices]
            for shift in (0, len(p) // 2):
                self.check_oracle(np.roll(p, shift, axis=0))

    def test_linear_operation_bound(self):
        for n in (10, 100, 1000, 10000):
            angles = np.arange(n) * 2 * np.pi / n
            d = rotating_calipers(np.c_[np.cos(angles), np.sin(angles)])
            self.assertAlmostEqual(d.diameter_m, 2.0, places=12)
            self.assertLessEqual(d.support_advances, 2 * n)
            self.assertLessEqual(d.candidate_pairs, 4 * n)

    def test_diameter_circle_criterion(self):
        self.assertFalse(same_diameter_circle([[0, 0], [36, 0], [18, 18 * math.sqrt(3)]])["cover_flag"])
        self.assertTrue(same_diameter_circle([[0, 0], [36, 0], [10, 2]])["cover_flag"])
        self.assertTrue(same_diameter_circle([[0, 0], [2, 0], [2, 1], [0, 1]])["cover_flag"])
        self.assertTrue(same_diameter_circle([[3, 4]])["cover_flag"])

    def test_exact_circle_boundary_without_padding(self):
        # A one-ULP displacement across the circle must change the result;
        # a geometric tolerance would incorrectly accept the outside point.
        for y, expected in [(1.0, True), (math.nextafter(1.0, 2.0), False), (math.nextafter(1.0, 0.0), True)]:
            p = [[-1.0, 0.0], [0.0, -0.5], [1.0, 0.0], [0.0, y]]
            c = same_diameter_circle(p)
            self.assertEqual(c["cover_flag"], expected)
            self.assertGreater(c["exact_boundary_checks"], 0)
        rng = np.random.default_rng(13)
        for _ in range(100):
            cloud = rng.normal(size=(30, 2))
            p = cloud[ConvexHull(cloud).vertices]
            d = rotating_calipers(p)
            a, b = [tuple(map(Fraction, v)) for v in (d.pD, d.qD)]
            reference = all(
                sum((2 * Fraction(v[k]) - a[k] - b[k]) ** 2 for k in (0, 1))
                <= sum((a[k] - b[k]) ** 2 for k in (0, 1))
                for v in p
            )
            self.assertEqual(same_diameter_circle(p, d)["cover_flag"], reference)

    def test_region_classification(self):
        hp = frozen.HalfPlane
        cases = [
            ("empty", [hp(1, 0, 0), hp(-1, 0, -1)]),
            ("unbounded", []),
            ("unbounded", [hp(1, 0, 1)]),
            ("point", [hp(1, 0, 0), hp(-1, 0, 0), hp(0, 1, 0), hp(0, -1, 0)]),
            ("segment", [hp(1, 0, 2), hp(-1, 0, 0), hp(0, 1, 0), hp(0, -1, 0)]),
            ("polygon", [hp(1, 0, 2), hp(-1, 0, 0), hp(0, 1, 1), hp(0, -1, 0), hp(2, 0, 4), hp(1, 0, 5)]),
        ]
        for kind, planes in cases:
            with self.subTest(kind=kind):
                r = intersect_halfplanes(planes)
                self.assertEqual(r.kind, kind)
                if r.vertices:
                    self.check_oracle([(v.x_m, v.y_m) for v in r.vertices])
                if kind == "polygon":
                    self.assertEqual(len(r.effective_edges), 4)
                    normals = [math.atan2(planes[i].ny, planes[i].nx) for i in r.effective_edges]
                    self.assertEqual(normals, sorted(normals))

    def test_real_pipeline_never_calls_frozen_brute(self):
        from rf_searchlab.geometry import bearing_halfplanes

        with (
            patch.object(frozen, "diameter", side_effect=AssertionError("Brute force called")),
            patch.object(frozen, "intersect_halfplanes", side_effect=AssertionError("Old pipeline called")),
        ):
            s = [[-100, 0], [0, -100], [100, 0], [0, 100]]
            angles = [0, 90, 180, 270]
            planes, _ = bearing_halfplanes(s, angles)
            r = intersect_halfplanes(planes)
            p = [(v.x_m, v.y_m) for v in r.vertices]
            self.assertEqual(r.kind, "polygon")
            self.check_oracle(p)
            with patch("rf_searchlab.geometry.rotating_calipers", wraps=rotating_calipers) as sweep:
                answer = solve(s, angles)
                self.assertEqual(sweep.call_count, 1)
            self.assertEqual(answer["algorithm"], "rotating_calipers")
            self.assertEqual(answer["diameter_m"], r.diameter_m)


if __name__ == "__main__":
    unittest.main(verbosity=2)
