"""Persistent future-all-negative obligations; never changes an absence status."""

from dataclasses import replace
from time import perf_counter

from rf_searchlab.algorithms.directional.belief import Status
from rf_searchlab.core.geometry import distance


def improve(state, plan, engine, deadline):
    tasks = dict(plan.tasks)
    changed = 0
    for c, pending in list(tasks.items()):
        if perf_counter() >= deadline:
            break
        b = state.beliefs[c - 1]
        if b.status != Status.UNKNOWN:
            continue
        negatives = tuple(
            o.position for o in b.observations if o.result == "no_signal" and o.request_id in state.applied
        )
        # Empty pending requires actual certification, handled by the real-evidence engine.
        # Revalidate old substitutions before evaluating a new one.
        if not engine.condition_complete(negatives, pending):
            observed = {o.position for o in b.observations}
            pending = tuple(p for p in engine.layout.stops if p not in observed)
            if not engine.condition_complete(negatives, pending):
                raise RuntimeError("Restored fixed tasks cannot establish conditional coverage")
        if state.position in pending or any(o.position == state.position for o in b.observations):
            tasks[c] = pending
            continue
        planned = pending + (state.position,)
        removed = False
        for p in sorted(pending, key=lambda p: distance(state.position, p), reverse=True)[:2]:
            if perf_counter() >= deadline:
                break
            trial = tuple(q for q in planned if q != p)
            if engine.condition_complete(negatives, trial):
                planned = trial
                changed += 1
                removed = True
        tasks[c] = planned if removed else pending
    return replace(plan, tasks=tuple(sorted(tasks.items()))), changed
