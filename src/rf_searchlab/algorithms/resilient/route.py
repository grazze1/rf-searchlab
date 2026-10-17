"""Directed source-macro tour. MEC exits are ranking estimates, not observations."""

from dataclasses import replace
from time import perf_counter

from rf_searchlab.algorithms.directional.belief import bound
from rf_searchlab.algorithms.directional.policy import baseline_service
from rf_searchlab.core.geometry import distance


def optimize(start, edges, deadline):
    n = len(start)
    order = []
    remaining = set(range(n))
    while remaining:
        nxt = min(remaining, key=lambda j: (start[j] if not order else edges[order[-1]][j], j))
        order.append(nxt)
        remaining.remove(nxt)

    def length(seq):
        return start[seq[0]] + sum(edges[a][b] for a, b in zip(seq, seq[1:])) if seq else 0.0

    best = length(order)
    if n <= 9:
        dp = {(1 << i, i): (start[i], (i,)) for i in range(n)}
        for mask in range(1, 1 << n):
            if perf_counter() >= deadline:
                return tuple(order), best, "nearest_deadline"
            for last in range(n):
                if (mask, last) not in dp:
                    continue
                value, seq = dp[mask, last]
                for j in range(n):
                    if mask & (1 << j):
                        continue
                    key = (mask | (1 << j), j)
                    candidate = (value + edges[last][j], seq + (j,))
                    if key not in dp or candidate < dp[key]:
                        dp[key] = candidate
        value, seq = min(dp[(1 << n) - 1, j] for j in range(n))
        return seq, value, "held_karp"
    improved = True
    while improved and perf_counter() < deadline:
        improved = False
        trials = []
        for i in range(n - 1):
            for j in range(i + 1, n):
                trials.append(order[:i] + order[i : j + 1][::-1] + order[j + 1 :])
        for size in (1, 2, 3):
            for i in range(n - size + 1):
                piece = order[i : i + size]
                rest = order[:i] + order[i + size :]
                trials.extend(rest[:j] + piece + rest[j:] for j in range(len(rest) + 1))
        for trial in trials:
            if perf_counter() >= deadline:
                return tuple(order), best, "local_search_deadline"
            value = length(trial)
            if value < best - 1e-8:
                order, best, improved = trial, value, True
                break
    return tuple(order), best, "nearest_2opt_oropt"


def choose(state, known, config, deadline):
    offers = [baseline_service(state, b, config) for b in known]
    first = [
        distance(state.position, d.intent.position) / 5
        + 5
        + int(d.intent.path == "/measure" and state.channel != b.channel)
        for b, d in zip(known, offers)
    ]
    if config.route_mode == "nearest":
        i = min(range(len(known)), key=lambda i: (first[i], known[i].channel))
        return (known[i].channel,), dict(route_score_s=first[i], route_method="nearest_executable_entry")
    exits = [
        d.intent.position if d.intent.path == "/clear" and d.intent.guaranteed else bound(b.polygon).center
        for b, d in zip(known, offers)
    ]
    start = [v + distance(d.intent.position, q) / 5 for v, d, q in zip(first, offers, exits)]
    edges = []
    for i, p in enumerate(exits):
        if perf_counter() >= deadline:
            raise TimeoutError("source_matrix_budget")
        shadow = replace(state, position=p, channel=known[i].channel)
        row = []
        for j, b in enumerate(known):
            if perf_counter() >= deadline:
                raise TimeoutError("source_matrix_budget")
            if i == j:
                row.append(0.0)
                continue
            d = baseline_service(shadow, b, config)
            row.append(
                distance(p, d.intent.position) / 5
                + distance(d.intent.position, exits[j]) / 5
                + 5
                + int(d.intent.path == "/measure" and shadow.channel != b.channel)
            )
        edges.append(row)
    order, value, method = optimize(start, edges, deadline)
    return tuple(known[i].channel for i in order), dict(
        route_score_s=value,
        route_method=method,
        route_nodes=len(known),
        route_cost_kind="Executable entry plus estimated MEC exit, not a guaranteed service time",
        route_first_action_s=first[order[0]],
    )
