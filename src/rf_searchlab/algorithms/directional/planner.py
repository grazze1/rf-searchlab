"""Persistent fixed obligations and conditional shared-stop route planning."""

from dataclasses import dataclass
from time import perf_counter

from rf_searchlab.algorithms.directional.belief import RESOLVED, Status, bound
from rf_searchlab.algorithms.directional.candidates import clear_points
from rf_searchlab.algorithms.route.tour import insertion_cost, open_tour
from rf_searchlab.core.domain import Point
from rf_searchlab.core.geometry import distance


def xy(p):
    return p.x_m, p.y_m


@dataclass(frozen=True)
class Route:
    stops: tuple
    channels: dict
    owners: dict
    length: float
    conditional: bool = True
    deleted_tasks: int = 0


def fixed_tasks(state, layout):
    tasks = {}
    for b in state.beliefs:
        if b.status == Status.UNKNOWN:
            measured = {o.position for o in b.observations}
            tasks[b.channel] = tuple(p for p in layout.stops if p not in measured)
    return tasks


def build_route(state, engine, config, deadline=None):
    tasks = fixed_tasks(state, engine.layout)
    owners = {}
    deleted = 0
    if config.shared:
        for b in state.beliefs:
            if b.ever_detected and b.status not in RESOLVED and bound(b.polygon).radius_m <= 300:
                safe = clear_points(b, state.position)
                p = (
                    safe[0]
                    if safe
                    else Point(round(bound(b.polygon).center.x_m, 6), round(bound(b.polygon).center.y_m, 6))
                )
                owners.setdefault(p, []).append(b.channel)
        additions = tuple(dict.fromkeys((state.position,) + tuple(owners)))
        # Each group of channels with identical measured positions reuses the same exact
        # condition-complete result. Deletions are bounded, preserving a finite skeleton.
        for c, pending in list(tasks.items()):
            if deadline is not None and perf_counter() >= deadline:
                break
            b = state.beliefs[c - 1]
            negatives = tuple(o.position for o in b.observations if o.result == "no_signal")
            extra = tuple(p for p in additions if not any(o.position == p for o in b.observations))
            planned = tuple(dict.fromkeys(pending + extra))
            if not engine.condition_complete(negatives, planned):
                raise RuntimeError("Fixed conditional cover unexpectedly incomplete")
            # Attempt up to two distant deletions, rechecking continuous coverage each time.
            for p in sorted(pending, key=lambda p: distance(state.position, p), reverse=True)[:2]:
                if deadline is not None and perf_counter() >= deadline:
                    break
                trial = tuple(q for q in planned if q != p)
                if engine.condition_complete(negatives, trial):
                    planned = trial
                    deleted += 1
            # Keep an extra stop only if it substituted for a fixed task or is already a service.
            tasks[c] = planned if len(set(pending) - set(planned)) else pending
    nodes = tuple(dict.fromkeys(p for planned in tasks.values() for p in planned))
    if config.shared:
        nodes = tuple(dict.fromkeys(nodes + tuple(owners)))
    length, order = open_tour(xy(state.position), tuple(map(xy, nodes)))
    stops = tuple(nodes[i] for i in order)
    channels = {
        p: tuple(
            sorted(
                (c for c, planned in tasks.items() if p in planned),
                key=lambda c: (c != state.channel, c),
            )
        )
        for p in stops
    }
    return Route(stops, channels, {p: tuple(cs) for p, cs in owners.items()}, length, True, deleted)


def shared_scan_cost(state, route, p, channels):
    delta = insertion_cost(xy(state.position), tuple(map(xy, route.stops)), xy(p))
    ordered = sorted(channels, key=lambda c: (c != state.channel, c))
    current, switches = state.channel, 0
    for c in ordered:
        switches += c != current
        current = c
    return delta / 5 + 5 * len(ordered) + switches
