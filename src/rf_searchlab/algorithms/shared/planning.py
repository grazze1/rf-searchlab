"""Region safe regions, sampled Sensing alternatives, atomic candidates and classical ranking.

All functions accept observed State only. Sampled future bearings rank actions;
they are explicitly not continuous worst-case certificates.
"""

from dataclasses import dataclass
from functools import lru_cache
from math import atan2, cos, pi, radians, sin

from rf_searchlab.algorithms.shared.state import CONFIG, RESOLVED, bound, covered_cells, unknown_cells
from rf_searchlab.core.belief import coverage_points, wire_point
from rf_searchlab.core.domain import Belief, GeometryError, Intent, Point, State, Status
from rf_searchlab.core.geometry import clip_wedge, distance, max_distance
from rf_searchlab.core.planning.optical import optical_plan


@dataclass(frozen=True)
class CandidateAction:
    intent: Intent
    role_tags: tuple[str, ...]
    sampled_J_loc: float = -1.0
    guaranteed_receive: bool = False
    coverage_gain: int = 0
    shared_value: float = 0.0

    @property
    def position(self) -> Point:
        assert self.intent.position is not None
        return self.intent.position

    @property
    def channel(self) -> int:
        assert self.intent.channel is not None
        return self.intent.channel

    @property
    def operation(self) -> str:
        return self.intent.path[1:]


def verify_clear_point(state: State, channel: int, position: Point) -> bool:
    b = state.beliefs[channel - 1]
    return bool(
        b.ever_detected
        and b.polygon
        and max_distance(wire_point(position), b.polygon) <= 20 - CONFIG.safety_margin_m
    )


def project_disk(p: Point, m: Point, radius: float) -> Point:
    d = distance(p, m)
    if d <= radius:
        return wire_point(p)
    t = radius / max(d, 1e-12)
    return wire_point(Point(m.x_m + t * (p.x_m - m.x_m), m.y_m + t * (p.y_m - m.y_m)))


def clear_points(b: Belief, current: Point, successors: tuple[Point, ...]) -> tuple[Point, ...]:
    circle = bound(b.polygon)
    radius = 20 - CONFIG.safety_margin_m - circle.radius_m - 2e-6
    if radius < 0:
        return ()
    m = circle.center
    points = [project_disk(current, m, radius), wire_point(m)]
    for y in successors:
        dx, dy = y.x_m - current.x_m, y.y_m - current.y_m
        d2 = dx * dx + dy * dy
        t = min(1, max(0, ((m.x_m - current.x_m) * dx + (m.y_m - current.y_m) * dy) / d2)) if d2 else 0
        on_segment = Point(current.x_m + t * dx, current.y_m + t * dy)
        points.append(project_disk(on_segment, m, radius))
    return tuple(p for p in dict.fromkeys(points) if max_distance(p, b.polygon) <= 19.95)


@lru_cache(maxsize=16384)
def sampled_quality(poly: tuple[Point, ...], p: Point) -> float:
    """Finite representative z/error quadrature; no-signal branch retains old bound.

    Representatives are hypotheses in the observable outer set, never true sources.
    A near branch has enclosing radius <=5.01, and is beneficial in Omnidirectional.
    """
    old = bound(poly).radius_m
    reps = poly[:: max(1, len(poly) // 4)] + (bound(poly).center,)
    values = []
    for z in reps:
        d = distance(p, z)
        if d <= 5:
            values.append(5.01)
        elif d > 1500:
            values.append(old)
        else:
            for error in (-1.01, 0.0, 1.01):
                angle = atan2(z.y_m - p.y_m, z.x_m - p.x_m) + radians(error)
                part = clip_wedge(poly, p, angle, CONFIG.epsilon_rad)
                if part:
                    values.append(bound(part).radius_m)
            if d > 1000:
                values.append(old)  # unknown radius can yield no_signal
    return max(values, default=old)


@lru_cache(maxsize=4096)
def active_points(
    poly: tuple[Point, ...], history: tuple[Point, ...], theta: float
) -> tuple[tuple[Point, float], ...]:
    """Multiple FIM/angle prefiltered and MEC-ranked near-optimal alternatives."""
    m = bound(poly).center
    raw = []
    anchor = history[0]
    for side in (-1, 1):
        raw.append(
            Point(
                anchor.x_m + 750 * cos(theta) - side * 100 * sin(theta),
                anchor.y_m + 750 * sin(theta) + side * 100 * cos(theta),
            )
        )
    for radius in (30, 100, 300):
        for k in range(8):
            a = theta + k * pi / 4
            raw.append(Point(m.x_m + radius * cos(a), m.y_m + radius * sin(a)))
    valid = [
        p
        for p in dict.fromkeys(wire_point(p) for p in raw)
        if p not in history and max_distance(p, poly) <= 999.95
    ]

    def fim(p: Point) -> float:
        # determinant contribution at the public MEC center; geometry proxy only
        dx, dy = p.x_m - m.x_m, p.y_m - m.y_m
        return sum(
            ((s.x_m - m.x_m) * dy - (s.y_m - m.y_m) * dx) ** 2
            / max(1.0, distance(s, m) ** 4 * distance(p, m) ** 4)
            for s in history
        )

    selected = sorted(valid, key=fim, reverse=True)[:4] + valid[:2]
    ranked = sorted(
        ((p, sampled_quality(poly, p)) for p in dict.fromkeys(selected)), key=lambda item: item[1]
    )
    return tuple(ranked[:4])


def measure_candidate(state: State, b: Belief, p: Point, tag: str) -> CandidateAction | None:
    p = wire_point(p)
    if any(o.position == p for o in b.observations):
        return None  # fixed spatial error; no repeated-angle averaging
    if b.status == Status.UNKNOWN:
        remaining = unknown_cells(state, b.channel)
        gain = len(remaining & covered_cells(p))
        if gain < min(80, len(remaining)) and p not in coverage_points():
            return None
        return CandidateAction(
            Intent("/measure", p, b.channel, "coverage", b.version),
            (tag, "coverage"),
            coverage_gain=gain,
        )
    score = sampled_quality(b.polygon, p)
    if score >= bound(b.polygon).radius_m * 0.99:
        return None
    return CandidateAction(
        Intent("/measure", p, b.channel, "joint_rf", b.version),
        (tag, "rf"),
        score,
        max_distance(p, b.polygon) <= 999.95,
    )


def make_candidates(state: State) -> list[CandidateAction]:
    active = [b for b in state.beliefs if b.status not in RESOLVED]
    successors = tuple(
        dict.fromkeys([bound(b.polygon).center for b in active if b.ever_detected] + list(coverage_points()))
    )
    groups: list[list[CandidateAction]] = []
    shared_points = [state.position]
    for b in active:
        group: list[CandidateAction] = []
        if b.status == Status.READY:
            nearby = tuple(sorted(successors, key=lambda p: distance(p, bound(b.polygon).center))[:3])
            for p in clear_points(b, state.position, nearby):
                group.append(
                    CandidateAction(
                        Intent("/clear", p, b.channel, "joint_clear", b.version, guaranteed=True),
                        ("clear",),
                    )
                )
                shared_points.append(p)
        elif b.ever_detected:
            history = tuple(o.position for o in b.observations if o.result == "direction")
            theta = radians(next((o.svd_deg for o in b.observations if o.svd_deg is not None), 0))
            if history and b.rf_used < 8:
                for p, score in active_points(b.polygon, history, theta):
                    if not any(o.position == p for o in b.observations):
                        group.append(
                            CandidateAction(
                                Intent("/measure", p, b.channel, "joint_rf", b.version),
                                ("rf", "plan_measurement"),
                                score,
                                True,
                            )
                        )
                # Short approach and the current position compete with Sensing information optima.
                m = bound(b.polygon).center
                approach = project_disk(state.position, m, min(400, bound(b.polygon).radius_m * 0.5))
                for p in (state.position, approach):
                    a = measure_candidate(state, b, p, "short")
                    if a:
                        group.append(a)
            if not group:
                group.append(optical_action(state, b))
            shared_points.extend(a.position for a in group[:2])
        else:
            unvisited = [p for i, p in enumerate(coverage_points()) if i not in b.coverage]
            for p in sorted(unvisited, key=lambda p: distance(p, state.position))[:2]:
                a = measure_candidate(state, b, p, "anchor")
                if a:
                    group.append(a)
            a = measure_candidate(state, b, state.position, "current")
            if a:
                group.insert(0, a)
        groups.append(group)
    # Offer cross-task service at a few nearest useful stop positions, replan after EACH action.
    shared_points = list(dict.fromkeys(shared_points))
    for group, b in zip(groups, active):
        for p in sorted(shared_points, key=lambda p: distance(p, state.position))[:3]:
            if b.status == Status.READY and verify_clear_point(state, b.channel, p):
                group.append(
                    CandidateAction(
                        Intent("/clear", p, b.channel, "joint_clear", b.version, guaranteed=True),
                        ("shared", "clear"),
                    )
                )
            elif b.status != Status.READY:
                a = measure_candidate(state, b, p, "shared")
                if a:
                    group.append(a)
        group.sort(key=lambda a: preliminary_cost(state, a))
    # Round-robin per-channel quotas preserve discovery, RF and clear opportunities.
    result, seen = [], set()
    for rank in range(max((len(g) for g in groups), default=0)):
        for group in groups:
            if rank >= len(group):
                continue
            a = group[rank]
            key = (a.operation, a.position, a.channel)
            if key not in seen:
                result.append(a)
                seen.add(key)
            if len(result) >= 48:
                return result
    return result


def immediate_cost(state: State, a: CandidateAction) -> float:
    return (
        distance(state.position, a.position) / 5
        + (5 if a.operation == "measure" or a.intent.guaranteed else 3)
        + int(a.operation == "measure" and state.channel != a.channel)
    )


def preliminary_cost(state: State, a: CandidateAction) -> float:
    b = state.beliefs[a.channel - 1]
    value = immediate_cost(state, a)
    if "rf" in a.role_tags:
        value += (
            distance(a.position, bound(b.polygon).center) / 5 + 2 * max(0, a.sampled_J_loc - 19.95) / 5 + 5
        )
    elif "coverage" in a.role_tags:
        value += min(120, 12 * len(unknown_cells(state, a.channel)) / max(1, a.coverage_gain))
    return value


def reference_score(state: State, a: CandidateAction) -> float:
    """Immediate + owner completion + open-path next-edge estimate, in seconds.

    Shared service and coverage gains are heuristic future route savings; no claim
    that clearing one source changes another source's independent RF physics.
    """
    others = [
        bound(b.polygon).center
        for b in state.beliefs
        if b.ever_detected and b.status not in RESOLVED and b.channel != a.channel
    ]
    if any(b.status == Status.UNKNOWN for b in state.beliefs):
        others.extend(coverage_points())
    successor = min((distance(a.position, p) for p in others if distance(a.position, p) > 1), default=0)
    current_next = min((distance(state.position, p) for p in others), default=0)
    insertion = (distance(state.position, a.position) + successor - current_next) / 5
    shared = sum(distance(a.position, p) < 150 for p in others)
    return preliminary_cost(state, a) + 0.25 * insertion - min(20, shared * 4)


def choose_reference(state: State, candidates: list[CandidateAction]) -> CandidateAction:
    if not candidates:
        return next_fallback_action(state)
    return min(
        candidates,
        key=lambda a: (
            reference_score(state, a),
            a.channel,
            a.operation,
            a.position.x_m,
            a.position.y_m,
        ),
    )


def optical_action(state: State, b: Belief) -> CandidateAction:
    angle = radians(next((o.svd_deg for o in b.observations if o.svd_deg is not None), 0))
    plan = b.optical or optical_plan(b.polygon, state.position, angle, b.version)
    if plan.index >= len(plan.cells):
        raise GeometryError("Finite optical cover exhausted without clear success")
    return CandidateAction(
        Intent(
            "/clear",
            plan.cells[plan.index].center,
            b.channel,
            "optical",
            b.version,
            optical_plan=plan,
        ),
        ("optical",),
    )


def immediate_clear(state: State) -> CandidateAction | None:
    for b in state.beliefs:
        if b.status in RESOLVED:
            continue
        if b.near_position is not None:
            return CandidateAction(
                Intent("/clear", b.near_position, b.channel, "near", b.version, guaranteed=True),
                ("near", "clear"),
            )
        if b.ever_detected and verify_clear_point(state, b.channel, state.position):
            return CandidateAction(
                Intent("/clear", state.position, b.channel, "joint_clear", b.version, guaranteed=True),
                ("current", "clear"),
            )
    return None


def next_fallback_action(state: State) -> CandidateAction:
    urgent = immediate_clear(state)
    if urgent:
        return urgent
    known = [b for b in state.beliefs if b.ever_detected and b.status not in RESOLVED]
    if known:
        # Once optical starts, finish its finite queue before changing targets.
        b = min(known, key=lambda b: (b.optical is None, b.channel))
        if b.status == Status.READY:
            points = clear_points(b, state.position, ())
            if points:
                return CandidateAction(
                    Intent("/clear", points[0], b.channel, "joint_clear", b.version, guaranteed=True),
                    ("clear",),
                )
        return optical_action(state, b)
    choices = []
    for b in state.beliefs:
        if b.status == Status.UNKNOWN:
            for i, p in enumerate(coverage_points()):
                if i not in b.coverage:
                    choices.append(
                        CandidateAction(
                            Intent("/measure", p, b.channel, "coverage", b.version),
                            ("coverage", "fallback"),
                        )
                    )
    if not choices:
        raise GeometryError("Unresolved state has no deterministic repair action")
    return min(choices, key=lambda a: (immediate_cost(state, a), a.channel))
