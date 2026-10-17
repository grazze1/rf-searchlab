"""Static closed-half-disk visibility and exact positive-stop convex hull membership."""

from math import cos, pi, sin

from rf_searchlab.algorithms.directional.certificate import hull, inside, integer_point
from rf_searchlab.core.belief import wire_point
from rf_searchlab.core.geometry import distance


def angular_visible(sensor, source, phi):
    if phi is None:
        return True
    # Only exactly cardinal input angles receive exact normals. A tiny nonzero angle
    # must not be snapped: it can change visibility on the closed-halfplane boundary.
    cardinal = {0.0: (1.0, 0.0), pi / 2: (0.0, 1.0), pi: (-1.0, 0.0), 3 * pi / 2: (0.0, -1.0)}
    nx, ny = cardinal.get(phi % (2 * pi), (cos(phi), sin(phi)))
    return nx * (sensor.x_m - source.x_m) + ny * (sensor.y_m - source.y_m) >= 0


def visible(sensor, source, radius, phi):
    return distance(sensor, source) <= radius and angular_visible(sensor, source, phi)


def positive_hull_contains(belief, candidate):
    points = tuple(
        integer_point(o.position) for o in belief.observations if o.result in ("direction", "near")
    )
    p = wire_point(candidate)
    return inside(integer_point(p), hull(points))
