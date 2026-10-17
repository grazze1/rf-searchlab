from rf_searchlab.benchmark import scenario_plan


def test_benchmark_splits_are_disjoint_and_cover_requested_matrix():
    plans = {
        suite: scenario_plan(suite, n) for suite, n in (("smoke", 3), ("regression", 30), ("stress", 200))
    }
    seed_sets = {suite: {(s["environment"], s["seed"]) for s in rows} for suite, rows in plans.items()}
    assert all(len(seed_sets[suite]) == len(rows) for suite, rows in plans.items())
    assert not seed_sets["smoke"] & seed_sets["regression"]
    assert not seed_sets["smoke"] & seed_sets["stress"]
    assert not seed_sets["regression"] & seed_sets["stress"]
    for suite, rows in plans.items():
        assert {s["environment"] for s in rows} == {"omni", "mixed"}
        assert {s["count"] for s in rows} == {10, 13, 16}
        if suite != "smoke":
            assert {s["family"] for s in rows} == {"uniform", "cluster", "boundary", "radius1000", "outward"}
