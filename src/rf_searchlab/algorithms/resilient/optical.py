"""Only persistent long covers enter the optional stripe/recovery comparison."""

from dataclasses import replace
from time import perf_counter

from rf_searchlab.algorithms.lookahead.optical import commit_progress, cover_from_dict, make_cover
from rf_searchlab.algorithms.lookahead.policy import Decision
from rf_searchlab.algorithms.lookahead.service import optical_choice
from rf_searchlab.core.geometry import distance

__all__ = ["gated_optical", "commit_progress", "cover_from_dict"]


def gated_optical(state, decision, config, deadline):
    b = state.beliefs[decision.intent.channel - 1]
    active = next((c for c in decision.plan.covers if c.channel == b.channel), None)
    if decision.intent.guaranteed or perf_counter() >= deadline:
        return decision
    if active is None:
        if (
            decision.detail.get("category") != "optical"
            or b.optical
            or len(decision.optical_queue) < config.long_cover_threshold
            or not b.observations
            or b.observations[-1].result != "no_signal"
        ):
            return decision
        # Both schemes completely cover P, and are compared on identical worst-case work.
        covers = [make_cover(b, state.position, False), make_cover(b, state.position, True)]

        def upper(c):
            ps = tuple(z.center for z in c.cells)
            return sum(distance(a, z) / 5 for a, z in zip((state.position,) + ps, ps)) + 3 * (len(ps) - 1) + 5

        chosen = min(covers, key=lambda c: upper(c))
        plan = replace(decision.plan, covers=decision.plan.covers + (chosen,))
        scheme = "grid" if chosen is covers[0] else "stripes"
    else:
        plan = decision.plan
        scheme = "persistent"
    d, updated, cover, rest = optical_choice(state, b, plan, config)
    recovery_at = dict(plan.recovery_at)
    recoveries = dict(plan.recoveries)
    if (
        b.optical
        and config.long_cover_recovery
        and recoveries.get(b.channel, 0) < 2
        and len(b.exclusions) - recovery_at.get(b.channel, 0) >= config.long_cover_recovery
    ):
        from rf_searchlab.algorithms.resilient.continuation import compare

        recovery_at[b.channel] = len(b.exclusions)
        reviewed = replace(plan, recovery_at=tuple(sorted(recovery_at.items())))
        compared = compare(state, b, reviewed, cover, rest, d, config, deadline)
        if compared.intent.path == "/measure":
            recoveries[b.channel] = recoveries.get(b.channel, 0) + 1
            updated = replace(reviewed, recoveries=tuple(sorted(recoveries.items())))
            d = compared
            d.detail["recovery_rf_number"] = recoveries[b.channel]
        else:
            updated = replace(updated, recovery_at=tuple(sorted(recovery_at.items())))
            d = compared
    detail = dict(decision.detail, **d.detail, long_cover_gate=True, cover_scheme=scheme)
    return Decision(d.intent, updated, detail, d.optical_queue)
