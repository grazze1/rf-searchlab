import json
import math
from dataclasses import replace
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

from rf_searchlab.application import Run
from rf_searchlab.core.domain import Intent, Point, ProtocolError, RawResponse
from rf_searchlab.core.geometry import mec
from rf_searchlab.core.protocol import dumps, make_envelope, parse_response
from rf_searchlab.evaluation import ledger, replay
from rf_searchlab.localization import locate, plan_measurement
from rf_searchlab.policies import PublicPolicy
from rf_searchlab.simulator import Simulator, Source, WorldTruth, from_dict, generate
from rf_searchlab.storage import read_json, run_path


def world(sources, environment="omni"):
    return WorldTruth(tuple(sources), 123, environment, mode="custom")


def act(env, path, p=None, c=None, rid=None):
    e = make_envelope(Intent(path, p, c), "LOCAL", rid or f"r{len(env._cache)}", 0)
    return e, parse_response(path, env.execute(e))


def test_time_clear_does_not_switch_and_idempotence():
    env = Simulator(world([Source(2, 300, 400, 1000)]))
    act(env, "/enter")
    e, r = act(env, "/measure", Point(0, 0), 1)
    assert r.virtual_time_s == 5
    e, r = act(env, "/clear", Point(300, 400), 2)
    assert r.result == "success" and r.virtual_time_s == 110 and env.channel == 1
    assert parse_response(e.path, env.execute(e)) == r
    conflict = replace(e, raw=e.raw.replace(b"300", b"301"))
    assert env.execute(conflict).status == 409 and env.virtual_time == 110
    _, r = act(env, "/measure", Point(300, 400), 1)
    assert r.virtual_time_s == 115
    _, r = act(env, "/clear", Point(300, 400), 2)
    assert r.result == "no_target_in_range" and r.virtual_time_s == 118


@pytest.mark.parametrize(
    "distance,expected",
    [(4.999, "near"), (5, "near"), (5.001, "direction"), (1000, "direction"), (1000.001, "no_signal")],
)
def test_reception_boundaries(distance, expected):
    env = Simulator(world([Source(1, 0, 0, 1000)]), "zero")
    act(env, "/enter")
    _, r = act(env, "/measure", Point(distance, 0), 1)
    assert r.result == expected


@pytest.mark.parametrize(
    "distance,result", [(19.999, "success"), (20, "success"), (20.000001, "no_target_in_range")]
)
def test_clear_boundaries(distance, result):
    env = Simulator(world([Source(1, 0, 0, 1000)]))
    act(env, "/enter")
    assert act(env, "/clear", Point(distance, 0), 1)[1].result == result


@pytest.mark.parametrize(
    "point,expected", [(Point(0, 20), "direction"), (Point(-1, 0), "no_signal"), (Point(1, 0), "near")]
)
def test_directional_halfplane_and_near(point, expected):
    env = Simulator(world([Source(1, 0, 0, 1000, 0)], "mixed"), "zero")
    act(env, "/enter")
    assert act(env, "/measure", point, 1)[1].result == expected
    assert act(env, "/clear", Point(-1, 0), 1)[1].result == "success"


def test_fixed_field_and_normalized_position():
    a = Simulator(generate(1))
    b = Simulator(generate(1))
    act(a, "/enter")
    act(b, "/enter")
    p = Point(123.45678901, 17)
    assert a.error(1, p) == a.error(1, Point(123.45678902, 17)) == b.error(1, p)
    assert abs(a.error(1, p)) <= 1


@pytest.mark.parametrize("error_mode", ["positive", "negative", "spatial"])
def test_returned_bearing_bounded_and_two_decimals(error_mode):
    env = Simulator(world([Source(1, 100, 50, 1000)]), error_mode)
    act(env, "/enter")
    for k in range(30):
        p = Point(-k * 3.123456, k * 1.74325)
        _, r = act(env, "/measure", p, 1)
        true = math.degrees(math.atan2(50 - p.y_m, 100 - p.x_m)) % 360
        assert abs((r.svd_deg - true + 180) % 360 - 180) <= 1 + 1e-10
        assert round(r.svd_deg, 2) == r.svd_deg


def test_unknown_request_fields_have_no_effect():
    env = Simulator(world([Source(1, 0, 0, 1000)]))
    act(env, "/enter")
    e = make_envelope(Intent("/measure", Point(100, 0), 1), "LOCAL", "extra", 0)
    raw = json.loads(e.raw)
    raw["extra"] = 1
    response = env.execute(replace(e, raw=dumps(raw)))
    assert not json.loads(response.body)["accepted"] and env.virtual_time == 0
    assert parse_response(e.path, env.execute(e)).accepted


def test_no_movement_domain_restriction():
    env = Simulator(world([Source(1, 1799, 0, 1000)]))
    act(env, "/enter")
    assert act(env, "/measure", Point(2000, 0), 1)[1].result == "direction"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True])
def test_invalid_coordinate_rejected(value):
    with pytest.raises(ValueError):
        Point(value, 0)


def test_invalid_world_and_config():
    with pytest.raises(ValueError):
        WorldTruth((), 1)
    with pytest.raises(ValueError):
        world([Source(1, 0, 0, 1000)] * 2)
    with pytest.raises(ValueError):
        generate(1, family="typo")
    with pytest.raises(ValueError):
        Run(generate(1), "adaptive_tour")
    with pytest.raises(ValueError):
        Run(generate(1), "shared_search", max_actions=0)
    with pytest.raises(ValueError):
        from_dict({"sources": [], "bad": 1})


@pytest.mark.parametrize(
    "source",
    [
        (0, 0, 0, 1000, None),
        (1, True, 0, 1000, None),
        (1, 1800.01, 0, 1000, None),
        (1, 0, 0, 999.99, None),
        (1, 0, 0, 1000, float("inf")),
    ],
)
def test_source_rejects_out_of_domain_or_nonnumeric_values(source):
    with pytest.raises(ValueError):
        Source(*source)


def test_world_rejects_boolean_seed_and_directional_omni():
    with pytest.raises(ValueError, match="Noise seed"):
        WorldTruth((Source(1, 0, 0, 1000),), True, "omni", mode="custom")
    with pytest.raises(ValueError, match="omnidirectional"):
        world([Source(1, 0, 0, 1000, 0)])


def test_equivalent_observations_same_next_action():
    policies = [PublicPolicy("coverage_route") for _ in range(2)]
    envs = [Simulator(world([Source(2, 500, 100, 1000)])), Simulator(world([Source(3, -500, 300, 1500)]))]
    for policy, env in zip(policies, envs):
        for intent, rid in [(Intent("/enter"), "enter"), (Intent("/measure", Point(0, 0), 1), "m")]:
            e = make_envelope(intent, "LOCAL", rid, 0)
            policy.observe(e, parse_response(e.path, env.execute(e)))
    assert policies[0].next_action() == policies[1].next_action()
    source = Path(__import__("rf_searchlab.policies", fromlist=["x"]).__file__).read_text()
    assert "simulator import" not in source and "WorldTruth" not in source


def test_budget_failure_and_readonly_replay():
    run = Run(generate(9), "shared_search", max_actions=2)
    m = run.execute()
    assert m["status"] == "failed" and m["failure"] and m["seconds_per_cleared"] is None
    replayed = replay(run.id)
    assert replayed["ledger"]["total_s"] == m["total_s"]
    assert replayed["status"] == "failed"
    with (run.path / "events.jsonl").open("a") as f:
        f.write("{broken")
    assert replay(run.id)["status"] == "partial"
    with pytest.raises(RuntimeError):
        run.execute()


def test_cancel_before_first_action():
    run = Run(generate(3), "shared_search")
    run.cancel()
    assert run.execute()["status"] == "cancelled"


def test_run_path_and_input_guards(tmp_path):
    for rid in ["../secret", "a" * 31, "A" * 32, "a" * 32 + "/x"]:
        with pytest.raises(ValueError):
            run_path(rid)
    p = tmp_path / "x.json"
    p.write_text('{"a":NaN}')
    with pytest.raises(ValueError):
        read_json(p)
    p.write_text('{"a":1,"a":2}')
    with pytest.raises(ValueError):
        read_json(p)
    with pytest.raises(ValueError):
        read_json(p, 1)


def test_ledger_rejects_time_and_sequence_corruption():
    run = Run(generate(9), "shared_search", max_actions=2)
    run.execute()
    records = json.loads(json.dumps(run.events))
    records[-1]["response"]["virtual_time_s"] += 1
    with pytest.raises(ValueError):
        ledger(records)
    records = json.loads(json.dumps(run.events))
    records[0]["sequence"] = 3
    with pytest.raises(ValueError):
        ledger(records)
    records = json.loads(json.dumps(run.events))
    records[-1]["request_id"] = records[0]["request_id"]
    with pytest.raises(ValueError):
        ledger(records)


def test_locate_unbounded_and_triangle():
    r = locate({"sensors_m": [[0, 0]], "bearings_deg": [359.9]})
    assert r["kind"] == "unbounded" and r["diameter_m"] is None and r["mec"] is None
    from rf_searchlab.geometry import same_diameter_circle

    p = [(0, 0), (2, 0), (1, math.sqrt(3))]
    assert not same_diameter_circle(p)["cover_flag"]
    assert mec(tuple(Point(*x) for x in p)).radius_m == pytest.approx(2 / math.sqrt(3), abs=1e-6)


def test_mec_independent_small_point_oracle():
    rng = np.random.default_rng(71)
    for _ in range(30):
        points = rng.normal(size=(6, 2))
        circles = []
        for a, b in combinations(points, 2):
            c = (a + b) / 2
            circles.append((c, np.linalg.norm(a - b) / 2))
        for a, b, c in combinations(points, 3):
            matrix = 2 * np.array([b - a, c - a])
            if abs(np.linalg.det(matrix)) < 1e-12:
                continue
            center = np.linalg.solve(matrix, np.array([b @ b - a @ a, c @ c - a @ a]))
            circles.append((center, np.linalg.norm(center - a)))
        radius = min(r for c, r in circles if np.max(np.linalg.norm(points - c, axis=1)) <= r + 1e-9)
        assert mec(tuple(Point(*map(float, p)) for p in points)).radius_m == pytest.approx(radius, abs=2e-7)


def test_plan_measurement_branch_outer_bounds_and_no_input_mutation():
    data = {"sensor_m": [0, 0], "bearing_deg": 359, "precision": "fast"}
    before = json.dumps(data)
    r = plan_measurement(data)
    assert json.dumps(data) == before and r["physical_candidates"] > 0 and r["recommended"]
    for c in r["candidates"]:
        assert c["reception_bound_m"] <= 999.9
        assert c["direction_upper_m"] + 1e-6 >= c["direction_sample_lower_m"]
        assert c["worst_mec_upper_m"] >= c["near_upper_m"]


@pytest.mark.parametrize("strategy,environment", [("coverage_route", "omni"), ("adaptive_tour", "mixed")])
def test_end_to_end(strategy, environment):
    r = Run(generate(101, environment), strategy, wall_seconds=60)
    m = r.execute()
    assert m["status"] == "completed", m
    assert m["all_cleared"] and m["cleared"] == 10
    assert replay(r.id)["ledger"]["total_s"] == pytest.approx(m["total_s"])


def test_simulator_response_contract():
    env = Simulator(world([Source(1, 100, 0, 1000)]))
    act(env, "/enter")
    e, r = act(env, "/measure", Point(0, 0), 1)
    d = json.loads(env.execute(e).body)
    assert set(d) == {"accepted", "real_timestamp_ms", "virtual_time_s", "measure_result", "svd_deg"}
    d["hidden_count"] = 1
    with pytest.raises(ProtocolError):
        parse_response("/measure", RawResponse(200, dumps(d)))
