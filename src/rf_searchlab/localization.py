"""Bearing localization and conservative planning of the next measurement."""

import itertools
from math import atan2, hypot, isfinite, pi, radians, sin

import numpy as np

from rf_searchlab.core.domain import Point
from rf_searchlab.core.geometry import angle_envelope, clip_wedge, disk_outer, max_distance, mec
from rf_searchlab.geometry import solve


def locate(data):
    if set(data) - {"sensors_m", "bearings_deg", "epsilon_deg"}:
        raise ValueError("Unknown Region fields")
    if len(data["sensors_m"]) > 64:
        raise ValueError("At most 64 observations")
    result = solve(**data)
    result.update(schema_version=1, units="m / degree", representation="pure_wedge_intersection")
    if result["vertices_m"]:
        circle = mec(tuple(Point(*v) for v in result["vertices_m"]))
        result["mec"] = {"center_m": [circle.center.x_m, circle.center.y_m], "radius_m": circle.radius_m}
    else:
        result["mec"] = None
    if result["diameter_m"] is not None and not isfinite(result["diameter_m"]):
        result["diameter_m"] = None
    return result


def plan_measurement(data):
    if set(data) - {"sensor_m", "bearing_deg", "precision"}:
        raise ValueError("Unknown Sensing fields")
    sensor = Point(*data.get("sensor_m", [0, 0]))
    bearing = float(data.get("bearing_deg", 25))
    precision = data.get("precision", "fast")
    if precision not in ("fast", "accurate") or not isfinite(bearing):
        raise ValueError("Invalid Sensing parameters")
    from rf_searchlab.core.geometry import clip_disk

    prior = clip_wedge(disk_outer(Point(0, 0), 1800, 64), sensor, radians(bearing), radians(1))
    prior = clip_disk(prior, sensor, 1500, 64)
    if not prior:
        raise ValueError("Empty prior")
    circle = mec(prior)
    center = circle.center
    step = 100 if precision == "fast" else 50
    bins = 16 if precision == "fast" else 64
    keep = 6 if precision == "fast" else 12
    raw = []
    for dx in range(-600, 601, step):
        for dy in range(-600, 601, step):
            p = Point(round(center.x_m + dx, 6), round(center.y_m + dy, 6))
            bound = max_distance(p, prior)
            if bound > 999.9 or hypot(p.x_m - sensor.x_m, p.y_m - sensor.y_m) < 1:
                continue
            a = atan2(center.y_m - sensor.y_m, center.x_m - sensor.x_m)
            b = atan2(center.y_m - p.y_m, center.x_m - p.x_m)
            fim = sin(a - b) ** 2 / max(1, hypot(p.x_m - center.x_m, p.y_m - center.y_m) ** 2)
            raw.append((fim, p, bound))
    candidates = []
    for fim, p, rx_bound in sorted(raw, key=lambda r: -r[0])[:keep]:
        from rf_searchlab.core.geometry import polygon_distance

        if polygon_distance(p, prior) <= 5:
            lo, hi = 0.0, 2 * pi
        else:
            lo, hi = angle_envelope(p, prior)
            lo -= radians(1)
            hi += radians(1)
        edges = np.linspace(lo, hi, bins + 1)
        worst = 0.0
        lower = 0.0
        for a, b in itertools.pairwise(edges):
            eps = radians(1) + (b - a) / 2
            outer = prior if eps >= pi / 2 else clip_wedge(prior, p, (a + b) / 2, eps)
            inner = clip_wedge(prior, p, (a + b) / 2, radians(1))
            if outer:
                m = mec(outer)
                worst = max(worst, max_distance(m.center, outer) + 1e-6)
            if inner:
                lower = max(lower, mec(inner).radius_m)
        candidates.append(
            {
                "position_m": [p.x_m, p.y_m],
                "fim_score": fim,
                "direction_upper_m": worst,
                "direction_sample_lower_m": lower,
                "worst_mec_upper_m": max(5.0, worst),
                "near_upper_m": 5.0,
                "no_signal": "impossible_under_omnidirectional_minimum_range",
                "reception_bound_m": rx_bound,
            }
        )
    candidates.sort(key=lambda c: c["worst_mec_upper_m"])
    return {
        "schema_version": 1,
        "interpretation": "discrete_candidates_with_future_interval_outer_bounds",
        "geometry": "conservative_outer_polygon",
        "assumption": "Sensing omnidirectional; fixed bounded error; FIM ranking only",
        "precision": precision,
        "grid_step_m": step,
        "bearing_intervals": bins,
        "retained": keep,
        "physical_candidates": len(raw),
        "prior_vertices_m": [[p.x_m, p.y_m] for p in prior],
        "candidates": candidates,
        "recommended": candidates[0] if candidates else None,
        "guarantee": "Bounds concern retained candidates and polygon; not continuous global optimality",
    }


def figure(result):
    import plotly.graph_objects as go

    fig = go.Figure()
    vertices = result.get("vertices_m", result.get("prior_vertices_m", []))
    if vertices:
        pts = vertices + [vertices[0]]
        fig.add_trace(
            go.Scatter(x=[p[0] for p in pts], y=[p[1] for p in pts], fill="toself", name="可行区域")
        )
    for key, name in [("circle", "直径圆"), ("mec", "最小包围圆")]:
        c = result.get(key)
        if c:
            angles = np.linspace(0, 2 * pi, 100)
            x, y = c["center_m"]
            r = c["radius_m"]
            fig.add_trace(go.Scatter(x=x + r * np.cos(angles), y=y + r * np.sin(angles), name=name))
    if result.get("candidates"):
        cs = result["candidates"]
        fig.add_trace(
            go.Scatter(
                x=[c["position_m"][0] for c in cs],
                y=[c["position_m"][1] for c in cs],
                mode="markers",
                marker={"color": [c["worst_mec_upper_m"] for c in cs], "showscale": True},
                name="候选最坏 MEC 上界 (m)",
            )
        )
    fig.update_layout(xaxis_title="东向 / m", yaxis_title="北向 / m", template="plotly_white")
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig
