"""Fine-resolution conservative negative coverage.

Same soundness rule as joint.state.cell_corners (a closed square is excluded only
when all four corners lie inside ONE received no-signal disk of radius 1000 m),
at 25 m resolution. Every 100 m cell excluded by the coarse rule is a union of
25 m cells that are each excluded by the same disk, so this certificate is never
weaker; it just stops losing a ~141 m rim per disk and closes on arbitrary stops.
"""

from functools import lru_cache

import numpy as np

from rf_searchlab.core.domain import Point, State, Status

FINE_M = 25
RECEIVE_M = 1000 - 1e-5


@lru_cache(maxsize=1)
def fine_corners() -> np.ndarray:
    xs = np.arange(-1800, 1800, FINE_M, dtype=float)
    gx, gy = np.meshgrid(xs, xs, indexing="ij")
    x, y = gx.ravel(), gy.ravel()
    dx = np.maximum.reduce([x, np.zeros_like(x), -x - FINE_M])
    dy = np.maximum.reduce([y, np.zeros_like(y), -y - FINE_M])
    keep = dx * dx + dy * dy <= 1800**2
    x, y = x[keep], y[keep]
    corners = np.stack(
        [
            np.stack([x, y], 1),
            np.stack([x + FINE_M, y], 1),
            np.stack([x, y + FINE_M], 1),
            np.stack([x + FINE_M, y + FINE_M], 1),
        ],
        1,
    )
    corners.flags.writeable = False
    return corners


@lru_cache(maxsize=1)
def fine_count() -> int:
    return int(len(fine_corners()))


@lru_cache(maxsize=4096)
def fine_mask(p: Point) -> np.ndarray:
    squared = np.sum((fine_corners() - (p.x_m, p.y_m)) ** 2, axis=2)
    mask = np.max(squared, axis=1) <= RECEIVE_M**2
    mask.flags.writeable = False
    return mask


@lru_cache(maxsize=4096)
def fine_remaining_for(negatives: tuple[Point, ...]) -> np.ndarray:
    remaining = np.ones(fine_count(), dtype=bool)
    for p in negatives:
        remaining &= ~fine_mask(p)
    remaining.flags.writeable = False
    return remaining


def fine_remaining(state: State, channel: int) -> np.ndarray:
    b = state.beliefs[channel - 1]
    return fine_remaining_for(tuple(o.position for o in b.observations if o.result == "no_signal"))


def fine_certified_absent(state: State, channel: int) -> bool:
    b = state.beliefs[channel - 1]
    return b.status == Status.UNKNOWN and not fine_remaining(state, channel).any()
