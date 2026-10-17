"""Persistent conditional tasks and macros are separate from accepted physical evidence."""

from dataclasses import dataclass, replace

from rf_searchlab.algorithms.directional.belief import Status
from rf_searchlab.algorithms.route.tour import open_tour
from rf_searchlab.core.domain import Point
from rf_searchlab.core.geometry import distance


@dataclass(frozen=True)
class Plan:
    initialized: bool = False
    tasks: tuple = ()
    route: tuple[Point, ...] = ()
    station: Point | None = None
    service: int | None = None
    station_due: bool = True
    service_actions: int = 0
    macro_id: int = 0
    optimization_actions: int = 0
    covers: tuple = ()
    recoveries: tuple = ()
    recovery_at: tuple = ()


def route_length(start, points):
    return sum(distance(a, b) for a, b in zip((start,) + tuple(points), points))


def segment_delta(u, segment, v=None):
    if not segment:
        return 0.0
    return route_length(u, tuple(segment)) + (
        distance(segment[-1], v) - distance(u, v) if v is not None else 0.0
    )


def reconcile(state, plan, engine):
    tasks = (
        dict(plan.tasks)
        if plan.initialized
        else {b.channel: engine.layout.stops for b in state.beliefs if b.status == Status.UNKNOWN}
    )
    for c in list(tasks):
        b = state.beliefs[c - 1]
        observed = {o.position for o in b.observations}
        if b.status != Status.UNKNOWN:
            del tasks[c]
        else:
            tasks[c] = tuple(p for p in tasks[c] if p not in observed)
    nodes = tuple(dict.fromkeys(p for ps in tasks.values() for p in ps))
    if plan.initialized:
        route = tuple(p for p in plan.route if p in nodes)
        missing = tuple(p for p in nodes if p not in route)
        route += missing
    else:
        _, order = open_tour((state.position.x_m, state.position.y_m), tuple((p.x_m, p.y_m) for p in nodes))
        route = tuple(nodes[i] for i in order)
    return replace(plan, initialized=True, tasks=tuple(sorted(tasks.items())), route=route)


def station_channels(state, plan, point):
    return tuple(sorted((c for c, ps in plan.tasks if point in ps), key=lambda c: (c != state.channel, c)))
