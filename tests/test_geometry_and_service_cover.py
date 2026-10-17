"""Analytical geometry and full-cell service-coverage invariants."""

import math
from dataclasses import asdict, replace
from itertools import product

import pytest

from rf_searchlab.algorithms.directional.belief import initial_state
from rf_searchlab.algorithms.lookahead.optical import (
    commit_progress,
    cover_from_dict,
    excluded,
    make_cover,
    remaining,
)
from rf_searchlab.algorithms.lookahead.planning import Plan
from rf_searchlab.core.domain import Exclusion, GeometryError, Point
from rf_searchlab.core.geometry import HalfPlane, angle_envelope, intersect_halfplanes, mec, reference_mec
from rf_searchlab.core.planning.optical import (
    clear_position,
    completion_cost,
    optical_plan,
    route_upper,
    transform,
)


def rectangle(x0, y0, x1, y1):
    return tuple(Point(x, y) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)))


@pytest.mark.parametrize("shape", ["empty", "unbounded", "point", "segment", "polygon"])
def test_halfplane_classification_against_analytical_sets(shape):
    constraints = {
        "empty": (HalfPlane(0, 0, -1),),
        "unbounded": (HalfPlane(1, 0, 1),),
        "point": (HalfPlane(1, 0, 0), HalfPlane(-1, 0, 0), HalfPlane(0, 1, 0), HalfPlane(0, -1, 0)),
        "segment": (HalfPlane(1, 0, 1), HalfPlane(-1, 0, 0), HalfPlane(0, 1, 0), HalfPlane(0, -1, 0)),
        "polygon": (
            HalfPlane(1, 0, 2),
            HalfPlane(-1, 0, 0),
            HalfPlane(0, 1, 3),
            HalfPlane(0, -1, 0),
            HalfPlane(2, 0, 4),
        ),
    }
    result = intersect_halfplanes(constraints[shape])
    assert result.kind == shape
    expected = {"point": 0, "segment": 1, "polygon": math.sqrt(13)}
    if shape in expected:
        assert result.diameter_m == pytest.approx(expected[shape])
    if shape == "polygon":
        assert {(p.x_m, p.y_m) for p in result.vertices} == {(0, 0), (2, 0), (2, 3), (0, 3)}
        assert len(result.effective_edges) == 4
    assert intersect_halfplanes(()).kind == "unbounded"


def test_independent_circle_and_wrapped_bearing_interval():
    triangle = (Point(0, 0), Point(2, 0), Point(1, math.sqrt(3)))
    assert reference_mec(triangle).radius_m == pytest.approx(2 / math.sqrt(3))
    assert mec(triangle).radius_m == pytest.approx(reference_mec(triangle).radius_m, abs=1e-6)
    with pytest.raises(GeometryError):
        reference_mec(())
    polygon = rectangle(100, -1, 101, 1)
    lo, hi = angle_envelope(Point(0, 0), polygon)
    assert hi - lo < 0.03
    with pytest.raises(GeometryError):
        angle_envelope(Point(100, 0), polygon)


@pytest.mark.parametrize("angle", [0.0, 0.37, 1.1])
def test_rotated_grid_covers_every_sample_and_costs_actual_route(angle):
    polygon = rectangle(-30, -20, 40, 35)
    current = Point(100, 100)
    plan = optical_plan(polygon, current, angle, 3)
    centers = tuple(cell.center for cell in plan.cells)
    assert len({(cell.i, cell.j) for cell in plan.cells}) == len(plan.cells)
    for x, y in product(range(-30, 41, 5), range(-20, 36, 5)):
        assert min(math.hypot(x - p.x_m, y - p.y_m) for p in centers) <= 20
    length = sum(math.hypot(b.x_m - a.x_m, b.y_m - a.y_m) for a, b in zip((current,) + centers, centers))
    assert route_upper(plan, current) == pytest.approx(length / 5 + 3 * len(centers) + 2)
    p = Point(12.3, -91.7)
    back = transform(transform(p, Point(7, 4), angle), Point(7, 4), angle, True)
    assert back.x_m == pytest.approx(p.x_m) and back.y_m == pytest.approx(p.y_m)
    with pytest.raises(GeometryError):
        route_upper(replace(plan, index=len(plan.cells)), current)
    assert clear_position(polygon, current, 0.05) is None
    assert completion_cost(polygon, current, angle, 0.05, 3) == pytest.approx(route_upper(plan, current))
    tiny = rectangle(-1, -1, 1, 1)
    assert clear_position(tiny, Point(0, 0), 0.05) == Point(0, 0)
    assert clear_position(tiny, current, 0.05) == Point(0, 0)
    with pytest.raises(GeometryError):
        optical_plan((), current, angle, 0)


@pytest.mark.parametrize("stripes", [False, True])
def test_service_cells_require_accepted_exclusion_before_cursor_skip(stripes):
    state = initial_state()
    belief = replace(state.beliefs[0], polygon=rectangle(-60, -30, 60, 30), ever_detected=True)
    cover = make_cover(belief, Point(0, 0), stripes)
    assert cover_from_dict(asdict(cover)) == cover
    for cell in cover.cells:
        assert max(math.hypot(p.x_m - cell.center.x_m, p.y_m - cell.center.y_m) for p in cell.corners) < 19.95
    for x, y in product(range(-60, 61, 4), range(-30, 31, 4)):
        assert min(math.hypot(x - c.center.x_m, y - c.center.y_m) for c in cover.cells) < 20
    belief = replace(
        belief,
        optical=tuple(c.center for c in cover.cells),
        exclusions=(Exclusion(cover.cells[0].center, 20, "miss"),),
    )
    beliefs = (belief,) + state.beliefs[1:]
    accepted = replace(state, beliefs=beliefs, applied=("miss",))
    assert excluded(cover.cells[0], belief)
    plan = replace(Plan(), covers=(replace(cover, cursor=1),))
    updated = commit_progress(accepted, plan)
    assert updated.beliefs[0].optical_index == 1
    assert all(i >= 1 for i, _ in remaining(replace(cover, cursor=1), belief))
    with pytest.raises(GeometryError, match="accepted response"):
        commit_progress(replace(accepted, applied=()), plan)
    with pytest.raises(GeometryError, match="without real exclusion"):
        commit_progress(accepted, replace(plan, covers=(replace(cover, cursor=len(cover.cells)),)))
