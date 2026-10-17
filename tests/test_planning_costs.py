"""Independent route optima and physically priced continuation branches."""

import itertools
import math
from dataclasses import replace
from time import perf_counter

import pytest

from rf_searchlab.algorithms.directional.belief import initial_state
from rf_searchlab.algorithms.directional.candidates import Hypothesis
from rf_searchlab.algorithms.directional.policy import Decision
from rf_searchlab.algorithms.lookahead.candidates import candidates
from rf_searchlab.algorithms.lookahead.config import Config
from rf_searchlab.algorithms.lookahead.planning import Plan
from rf_searchlab.algorithms.lookahead.service import (
    Tail,
    branch_cost,
    common_tail,
    compare,
    optical_choice,
    optical_cost,
    score,
    service_decision,
    terminal,
)
from rf_searchlab.algorithms.resilient.route import optimize
from rf_searchlab.core.domain import Intent, Observation, Point


def region():
    return tuple(Point(x, y) for x, y in ((-40, -25), (40, -25), (40, 25), (-40, 25)))


def public_state():
    state = initial_state()
    belief = replace(
        state.beliefs[0],
        polygon=region(),
        ever_detected=True,
        observations=(Observation("bearing", Point(-100, 0), "direction", 0),),
    )
    return replace(state, beliefs=(belief,) + state.beliefs[1:], applied=("bearing",)), belief


def test_directed_route_matches_exhaustive_optimum_and_deadline_keeps_permutation():
    start = [3.0, 7.0, 1.0, 8.0]
    edges = [[0 if i == j else (3 * i + 7 * j) % 11 + 1 for j in range(4)] for i in range(4)]

    def cost(order):
        return start[order[0]] + sum(edges[a][b] for a, b in itertools.pairwise(order))

    best = min(cost(order) for order in itertools.permutations(range(4)))
    order, value, method = optimize(start, edges, perf_counter() + 10)
    assert value == best == cost(order) and method == "held_karp"
    order, value, method = optimize(start, edges, 0)
    assert set(order) == set(range(4)) and value == cost(order) and method == "nearest_deadline"
    n = 10
    large_edges = [[abs(i - j) for j in range(n)] for i in range(n)]
    order, value, method = optimize(list(range(n)), large_edges, perf_counter() + 1)
    assert len(set(order)) == n
    assert value == pytest.approx(order[0] + sum(abs(a - b) for a, b in itertools.pairwise(order)))


def test_continuation_prices_service_and_tail_in_same_units():
    state, belief = public_state()
    config = Config(stripes=True, scenario_limit=12)
    optical, plan, cover, rest = optical_choice(state, belief, Plan(), config)
    assert optical.intent.path == "/clear" and len(plan.covers) == 1
    assert optical.detail["optical_remaining"] == len(rest)
    world = Hypothesis(rest[0][1].center, None, 1000, 1000, False)
    first = rest[0][1].center
    travel = math.hypot(first.x_m - state.position.x_m, first.y_m - state.position.y_m) / 5
    assert optical_cost(world, state.position, rest, None) == pytest.approx(travel + 5)
    assert terminal(Point(0, 0), Tail(Point(30, 40), 2), 1) == 16
    assert terminal(Point(0, 0), Tail(Point(30, 40), 2), 2) == 15
    assert terminal(Point(0, 0), Point(30, 40)) == 10
    assert branch_cost(belief, Point(0, 0), "near", None, world, config, None) == 5
    assert score([10, 20, 30], 0) == 20 and score([10, 20, 30], 1) == 30
    assert common_tail(state, belief, Plan()) is None
    assert common_tail(state, belief, replace(Plan(), route=(Point(500, 0),))).point == Point(500, 0)


def test_measurement_comparison_is_bounded_and_never_mutates_evidence():
    state, belief = public_state()
    config = Config(stripes=True, continuation=True, candidate_families=True, scenario_limit=9)
    points = candidates(belief, state.position, config)
    assert points and len(set(points)) == len(points) <= 12
    assert all(p not in {o.position for o in belief.observations} for p in points)
    optical, plan, _, rest = optical_choice(state, belief, Plan(), config)
    world = Hypothesis(Point(0, 0), None, 1000, 1500, False)
    selected = compare(state, belief, points[:2], (world,), config, perf_counter() + 5, optical, rest, None)
    assert selected.intent.path in ("/measure", "/clear")
    assert selected.detail["scenario_count"] == 9
    assert all(
        math.isfinite(item["score_s"]) and item["score_s"] >= 0 for item in selected.detail["rf_scores"]
    )
    assert len(state.applied) == 1 and belief.optical == ()
    fallback = Decision(Intent("/clear", Point(0, 0), 1, guaranteed=True), {})
    unchanged, unchanged_plan = service_decision(state, belief, plan, config, perf_counter() + 1, fallback)
    assert unchanged is fallback and unchanged_plan is plan
    exhausted = replace(config, rf_budget=0)
    action, next_plan = service_decision(
        state,
        belief,
        Plan(),
        exhausted,
        perf_counter() + 1,
        Decision(Intent("/measure", Point(100, 0), 1), {}),
    )
    assert action.intent.path == "/clear" and next_plan.covers
