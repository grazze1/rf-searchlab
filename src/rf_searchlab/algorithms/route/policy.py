"""Tour + set-cover planner. One atomic action per decision, replanned after every reply.

Every decision solves, on public state only:
  1. which extra stops are needed so that every UNKNOWN channel's remaining fine
     cells can be excluded from stops we will visit anyway (known-source
     representatives) plus a few cheap ring/anchor stops (greedy set cover);
  2. the open tour from the current position through those stops;
  3. per UNKNOWN channel, the minimum subset of tour stops that certifies it.
The first action is then: measure here if here is in some channel's plan, else a
worthwhile RF here, else the cheapest service of the first tour node. Nothing is
predicted about future replies; a positive reply simply changes the next plan.
"""

from dataclasses import dataclass
from functools import lru_cache
from math import cos, pi, sin
from typing import Any

import numpy as np

from rf_searchlab.algorithms.route.coverage import fine_mask, fine_remaining
from rf_searchlab.algorithms.route.tour import XY, dist, open_tour
from rf_searchlab.algorithms.shared.planning import (
    CandidateAction,
    make_candidates,
    preliminary_cost,
    sampled_quality,
)
from rf_searchlab.algorithms.shared.policy import JointPolicy
from rf_searchlab.algorithms.shared.state import RESOLVED, bound
from rf_searchlab.core.belief import coverage_points, wire_point
from rf_searchlab.core.domain import Intent, Point, State, Status
from rf_searchlab.core.geometry import distance, max_distance


@dataclass(frozen=True)
class PlanConfig:
    ring_radii: tuple[float, ...] = (950.0, 1100.0, 1250.0)
    ring_count: int = 24
    detour_weight: float = 0.5
    rf_here_gain: float = 0.9
    rep_max_radius_m: float = 300.0
    info_fraction: float = 0.65
    here_strict: bool = True
    approach_fraction: float = 0.35
    approach_lateral_m: float = 100.0
    approach_min_diameter_m: float = 400.0
    max_learning: int = 400

    def __post_init__(self) -> None:
        if not (
            8 <= self.ring_count <= 48
            and all(600 <= r <= 1800 for r in self.ring_radii)
            and 0 <= self.detour_weight <= 2
            and 0 < self.rf_here_gain <= 1
            and 0 <= self.rep_max_radius_m <= 1800
            and 0 < self.info_fraction <= 2
            and 0 <= self.approach_fraction <= 1
            and 0 <= self.approach_lateral_m <= 400
            and self.approach_min_diameter_m >= 100
            and self.max_learning >= 100
        ):
            raise ValueError("Plan configuration outside reviewed bounds")


@lru_cache(maxsize=8)
def stop_candidates(config: PlanConfig) -> tuple[Point, ...]:
    points = list(coverage_points())
    for r in config.ring_radii:
        for k in range(config.ring_count):
            a = 2 * pi * k / config.ring_count
            points.append(wire_point(Point(r * cos(a), r * sin(a))))
    return tuple(dict.fromkeys(points))


def xy(p: Point) -> XY:
    return (p.x_m, p.y_m)


@lru_cache(maxsize=1)
def disk_cells() -> int:
    return int(np.count_nonzero(fine_mask(Point(0, 0))))


def count(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask))


@dataclass(frozen=True)
class Plan:
    stops: tuple[Point, ...]
    owners: dict[Point, int]
    measure_plan: dict[int, frozenset[Point]]
    tour_length_m: float
    residual_cells: int


def build_plan(state: State, config: PlanConfig) -> Plan:
    current = state.position
    known = [b for b in state.beliefs if b.ever_detected and b.status not in RESOLVED]
    owners: dict[Point, int] = {}
    reliable: list[Point] = []
    for b in known:
        circle = bound(b.polygon)
        rep = wire_point(circle.center)
        owners.setdefault(rep, b.channel)
        # Only well-localised sources are stops we will really stand at; a fresh
        # wedge's MEC centre may be hundreds of metres from where we end up.
        if circle.radius_m <= config.rep_max_radius_m and rep != current:
            reliable.append(rep)
    reps = [p for p in owners if p != current]
    remaining = {
        b.channel: fine_remaining(state, b.channel) for b in state.beliefs if b.status == Status.UNKNOWN
    }
    remaining = {c: r for c, r in remaining.items() if r.any()}
    # Here and every known-source representative are stops we occupy anyway, so
    # extra stops only need to cover what those leave. The per-channel minimum
    # cover below decides whether "here" is actually measured; if it is not, the
    # other stops already cover those cells, so nothing is lost by moving on.
    here_mask = fine_mask(current)
    covered = {}
    for c, r in remaining.items():
        m = here_mask.copy()
        for p in reliable:
            m |= fine_mask(p)
        covered[c] = m
    extras: list[Point] = []
    anchors = [current] + reps
    while True:
        need = {c: remaining[c] & ~covered[c] for c in remaining}
        need = {c: n for c, n in need.items() if n.any()}
        if not need:
            break
        best: tuple[float, Point] | None = None
        for s in stop_candidates(config):
            if s in owners or s in extras or s == current:
                continue
            m = fine_mask(s)
            gains = [count(n & m) for n in need.values()]
            total, k = sum(gains), sum(1 for g in gains if g)
            if total == 0:
                continue
            insertion = min(dist(xy(s), xy(q)) for q in anchors + extras)
            score = total / (insertion / 5 + 6 * k)
            if best is None or score > best[0]:
                best = (score, s)
        if best is None:
            break
        extras.append(best[1])
        m = fine_mask(best[1])
        for c in need:
            covered[c] |= m
    nodes = tuple(reps + extras)
    length, order = open_tour(xy(current), tuple(xy(p) for p in nodes))
    ordered = tuple(nodes[i] for i in order)
    usable = set(reliable) | set(extras)
    stops_all = [current] + [p for p in ordered if p in usable]
    masks = [fine_mask(p) for p in stops_all]

    def greedy_cover(uncovered: np.ndarray, first: int) -> tuple[list[Point], int]:
        chosen: list[Point] = []
        while uncovered.any():
            gains = [count(m & uncovered) if i >= first else 0 for i, m in enumerate(masks)]
            i = max(range(len(stops_all)), key=lambda i: (gains[i], -i))
            if gains[i] == 0:
                return chosen, count(uncovered)
            chosen.append(stops_all[i])
            uncovered = uncovered & ~masks[i]
        return chosen, 0

    measure_plan: dict[int, frozenset[Point]] = {}
    residual = 0
    for c, r in remaining.items():
        # Measure here only when that does not increase this channel's total count:
        # the realised number of measurements then stays at the offline cover size
        # instead of growing with every RF stop the robot happens to pass through.
        without, res0 = greedy_cover(r, 1)
        here_gain = count(r & here_mask)
        if here_gain:
            with_here, res1 = greedy_cover(r & ~here_mask, 1)
            with_here = [current] + with_here
            # Information rule: testing at least half a fresh reception disk is
            # worth 6 s even if a later stop could have covered it, because an
            # early detection improves every subsequent tour.
            fewer = len(with_here) < len(without) + (0 if config.here_strict else 1)
            if fewer or res1 < res0 or here_gain >= config.info_fraction * disk_cells():
                without, res0 = with_here, res1
        residual += res0
        measure_plan[c] = frozenset(without)
    used = {p for stops in measure_plan.values() for p in stops}
    kept = tuple(p for p in ordered if p in owners or p in used)
    if len(kept) != len(ordered):
        length, order = open_tour(xy(current), tuple(xy(p) for p in kept))
        kept = tuple(kept[i] for i in order)
    return Plan(kept, owners, measure_plan, length, residual)


def measure_action(state: State, channel: int, p: Point, reason: str) -> CandidateAction | None:
    b = state.beliefs[channel - 1]
    if any(o.position == p for o in b.observations):
        return None
    return CandidateAction(Intent("/measure", p, channel, reason, b.version), ("plan", "coverage"))


def approach_action(
    state: State, channel: int, nxt: Point | None, config: PlanConfig
) -> CandidateAction | None:
    """Range-finding stop for a still-elongated set: a fraction of the way along its axis.

    A single bearing leaves the source anywhere on a ~1500 m sliver; the frozen Sensing
    candidates all sit ~750 m out, which walks past sources closer than that (39%
    of local detections) and back again. Stepping a smaller fraction along the
    axis with a lateral baseline bounds the overshoot while keeping the whole set
    inside the 1000 m guaranteed-reception disk of the stop.
    """
    b = state.beliefs[channel - 1]
    poly = b.polygon
    if not poly or b.status == Status.READY or b.optical is not None:
        return None
    far_a, far_b = max(((u, v) for u in poly for v in poly), key=lambda pair: distance(pair[0], pair[1]))
    diameter = distance(far_a, far_b)
    if diameter < config.approach_min_diameter_m:
        return None
    here = state.position
    near, far = (far_a, far_b) if distance(here, far_a) <= distance(here, far_b) else (far_b, far_a)
    ux, uy = (far.x_m - near.x_m) / diameter, (far.y_m - near.y_m) / diameter
    # Guaranteed reception needs every vertex within 1000 m of the stop.
    step = max(config.approach_fraction * diameter, diameter - 990.0)
    step = min(step, diameter)
    options = []
    for side in (1, -1):
        w = side * config.approach_lateral_m
        p = wire_point(Point(near.x_m + step * ux - w * uy, near.y_m + step * uy + w * ux))
        if max_distance(p, poly) > 999.95 or any(o.position == p for o in b.observations):
            continue
        detour = distance(p, nxt) if nxt else 0.0
        options.append((distance(here, p) + detour, side, p))
    if not options:
        return None
    _, _, p = min(options)
    return CandidateAction(
        Intent("/measure", p, channel, "joint_rf", b.version),
        ("plan", "approach", "rf"),
        sampled_quality(poly, p),
        True,
    )


class PlanPolicy(JointPolicy):
    def __init__(self, config: PlanConfig = PlanConfig()):
        super().__init__("classical_joint", None, config.max_learning)
        self.config = config
        self.name = "plan_route"

    def decide(self, state: State) -> CandidateAction:
        original = super().decide(state)
        if self.detail["mode"] in ("hard_immediate_clear", "committed_fallback"):
            return original
        plan = build_plan(state, self.config)
        current = state.position
        detail: dict[str, Any] = dict(
            mode=self.name,
            model_override=False,
            tour_length_m=plan.tour_length_m,
            stops=len(plan.stops),
            residual_cells=plan.residual_cells,
            itinerary=[dict(x_m=p.x_m, y_m=p.y_m) for p in plan.stops],
            itinerary_kind="tour_and_set_cover_conditional_on_future_replies",
        )
        here = sorted(
            (c for c, stops in plan.measure_plan.items() if current in stops),
            key=lambda c: (c != state.channel, c),
        )
        for c in here:
            a = measure_action(state, c, current, "plan_here")
            if a is not None:
                self.detail = dict(detail, step="measure_here")
                return a
        candidates = make_candidates(state)
        rf_here = [
            a
            for a in candidates
            if a.position == current
            and a.operation == "measure"
            and "rf" in a.role_tags
            and a.sampled_J_loc
            <= self.config.rf_here_gain * bound(state.beliefs[a.channel - 1].polygon).radius_m
        ]
        if rf_here:
            self.detail = dict(detail, step="rf_here")
            return min(rf_here, key=lambda a: (preliminary_cost(state, a), a.channel))
        if not plan.stops:
            self.detail = dict(detail, step="fallback_reference")
            return original
        node = plan.stops[0]
        nxt = plan.stops[1] if len(plan.stops) > 1 else None
        if node in plan.owners:
            k = plan.owners[node]
            approach = approach_action(state, k, nxt, self.config)
            if approach is not None:
                self.detail = dict(detail, step="approach_source", target_channel=k)
                return approach
            group = [a for a in candidates if a.channel == k]
            if group:

                def score(a: CandidateAction) -> tuple[float, str, float, float]:
                    detour = (distance(a.position, nxt) - distance(node, nxt)) / 5 if nxt else 0.0
                    return (
                        preliminary_cost(state, a) + self.config.detour_weight * detour,
                        a.operation,
                        a.position.x_m,
                        a.position.y_m,
                    )

                self.detail = dict(detail, step="serve_source", target_channel=k)
                return min(group, key=score)
        channels = sorted(
            (c for c, stops in plan.measure_plan.items() if node in stops),
            key=lambda c: (c != state.channel, c),
        )
        for c in channels:
            a = measure_action(state, c, node, "plan_stop")
            if a is not None:
                self.detail = dict(detail, step="go_to_stop")
                return a
        self.detail = dict(detail, step="fallback_reference")
        return original
