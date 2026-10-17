"""Open-path TSP from the current position: exact Held-Karp for small sets, else 2-opt/or-opt."""

from functools import lru_cache
from math import hypot

XY = tuple[float, float]
EXACT_MAX = 9


def dist(a: XY, b: XY) -> float:
    return hypot(a[0] - b[0], a[1] - b[1])


def path_length(start: XY, nodes: tuple[XY, ...]) -> float:
    total, prev = 0.0, start
    for p in nodes:
        total += dist(prev, p)
        prev = p
    return total


def _held_karp(start: XY, nodes: tuple[XY, ...]) -> tuple[float, tuple[int, ...]]:
    n = len(nodes)
    d = [[dist(a, b) for b in nodes] for a in nodes]
    dp: dict[tuple[int, int], tuple[float, tuple[int, ...]]] = {
        (1 << i, i): (dist(start, nodes[i]), (i,)) for i in range(n)
    }
    for mask in range(1, 1 << n):
        for last in range(n):
            entry = dp.get((mask, last))
            if entry is None:
                continue
            cost, order = entry
            row = d[last]
            for nxt in range(n):
                if mask & (1 << nxt):
                    continue
                key = (mask | (1 << nxt), nxt)
                value = cost + row[nxt]
                if key not in dp or value < dp[key][0]:
                    dp[key] = (value, order + (nxt,))
    full = (1 << n) - 1
    return min(dp[(full, last)] for last in range(n) if (full, last) in dp)


def _local_search(start: XY, nodes: tuple[XY, ...]) -> tuple[float, tuple[int, ...]]:
    n = len(nodes)
    remaining = set(range(n))
    order: list[int] = []
    cur = start
    while remaining:
        nxt = min(remaining, key=lambda i: (dist(cur, nodes[i]), i))
        order.append(nxt)
        remaining.remove(nxt)
        cur = nodes[nxt]

    def length(seq: list[int]) -> float:
        return path_length(start, tuple(nodes[i] for i in seq))

    best = length(order)
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                cand = order[:i] + order[i : j + 1][::-1] + order[j + 1 :]
                value = length(cand)
                if value < best - 1e-9:
                    order, best, improved = cand, value, True
        for seg in (1, 2, 3):
            for i in range(0, n - seg + 1):
                piece = order[i : i + seg]
                rest = order[:i] + order[i + seg :]
                for k in range(len(rest) + 1):
                    if k == i:
                        continue
                    cand = rest[:k] + piece + rest[k:]
                    value = length(cand)
                    if value < best - 1e-9:
                        order, best, improved = cand, value, True
    return best, tuple(order)


@lru_cache(maxsize=65536)
def open_tour(start: XY, nodes: tuple[XY, ...]) -> tuple[float, tuple[int, ...]]:
    """(length_m, visiting order as indices into nodes); deterministic for equal inputs."""
    if not nodes:
        return 0.0, ()
    if len(nodes) <= EXACT_MAX:
        return _held_karp(start, nodes)
    return _local_search(start, nodes)


def insertion_cost(start: XY, ordered: tuple[XY, ...], p: XY) -> float:
    """Cheapest extra length to insert p into an open path (appending allowed)."""
    if not ordered:
        return dist(start, p)
    best = dist(ordered[-1], p)
    prev = start
    for q in ordered:
        best = min(best, dist(prev, p) + dist(p, q) - dist(prev, q))
        prev = q
    return best
