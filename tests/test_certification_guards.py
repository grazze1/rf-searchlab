"""Adversarial evidence tests for public completion and Mixed absence proofs."""

import json
import shutil
from copy import deepcopy
from dataclasses import replace

import pytest

from rf_searchlab.algorithms.directional import certificate as directional_certificate
from rf_searchlab.algorithms.directional.belief import Status as MixedStatus
from rf_searchlab.algorithms.directional.belief import initial_state as directional_initial_state
from rf_searchlab.algorithms.directional.certificate import (
    Engine,
    Layout,
    check_witness,
    hull,
    inside,
    integer_point,
    load_ring,
    split,
)
from rf_searchlab.algorithms.directional.config import RESOURCE
from rf_searchlab.algorithms.shared.state import CONFIG, is_task_certified_complete, update_observation
from rf_searchlab.core.belief import initial_state
from rf_searchlab.core.domain import Certificate, GeometryError, Intent, Observation, Point, Response, Status
from rf_searchlab.core.protocol import make_envelope, parse_response
from rf_searchlab.policies import PublicPolicy
from rf_searchlab.simulator import Simulator, Source, WorldTruth


def test_sixteen_clears_require_distinct_accepted_channel_evidence():
    original = initial_state("PUBLIC", CONFIG)
    beliefs = list(original.beliefs)
    ids = tuple(f"clear-{channel}" for channel in range(1, 17))
    for channel, rid in enumerate(ids, start=1):
        beliefs[channel - 1] = replace(
            beliefs[channel - 1],
            status=Status.CLEARED,
            certificate=Certificate("clear_success", channel, (rid,), 1),
        )
    valid = replace(original, beliefs=tuple(beliefs), applied=ids)
    assert is_task_certified_complete(valid).complete
    assert is_task_certified_complete(valid).reason == "sixteen_clear_successes"

    assert not is_task_certified_complete(replace(valid, applied=ids[:-1])).complete
    duplicate = beliefs.copy()
    duplicate[1] = replace(duplicate[1], certificate=Certificate("clear_success", 2, (ids[0],), 1))
    assert not is_task_certified_complete(replace(valid, beliefs=tuple(duplicate))).complete
    for altered in (
        replace(beliefs[0], certificate=None),
        replace(beliefs[0], certificate=Certificate("other", 1, (ids[0],), 1)),
        replace(beliefs[0], certificate=Certificate("clear_success", 2, (ids[0],), 1)),
        replace(beliefs[0], certificate=Certificate("clear_success", 1, (), 1)),
    ):
        bad = beliefs.copy()
        bad[0] = altered
        assert not is_task_certified_complete(replace(valid, beliefs=tuple(bad))).complete


def test_mixed_policy_same_public_history_same_action_across_hidden_truth():
    policies = [PublicPolicy("adaptive_tour") for _ in range(2)]
    truths = [
        WorldTruth((Source(2, 800, 200, 1000, 0),), 1, "mixed", mode="custom"),
        WorldTruth((Source(3, -700, 400, 1200, 2),), 2, "mixed", mode="custom"),
    ]
    for policy, truth in zip(policies, truths):
        simulator = Simulator(truth)
        for intent, rid in ((Intent("/enter"), "enter"), (Intent("/measure", Point(0, 0), 1), "m")):
            envelope = make_envelope(intent, "LOCAL", rid, 0)
            policy.observe(envelope, parse_response(envelope.path, simulator.execute(envelope)))
    assert policies[0].next_action() == policies[1].next_action()


def test_mixed_certificate_integer_and_full_cell_guards():
    with pytest.raises(GeometryError):
        integer_point(Point(0.0000001, 0))
    with pytest.raises(GeometryError):
        split((0, 0, 3, 4))

    triangle = ((0, 0), (10, 0), (0, 10))
    boundary = hull(triangle)
    assert inside((0, 0), boundary) and inside((2, 2), boundary)
    assert not inside((7, 7), boundary)
    assert inside((5, 0), hull(((0, 0), (10, 0))))
    assert not inside((5, 1), hull(((0, 0), (10, 0))))
    assert inside((1, 1), ((1, 1),))
    with pytest.raises(GeometryError):
        check_witness((0, 0, 25_000_000, 25_000_000), (0, 1, 2), triangle)


def test_omni_inconsistent_positive_history_fails_without_certifying_absence():
    state = initial_state("PUBLIC", CONFIG)
    enter = make_envelope(Intent("/enter"), "LOCAL", "enter", 0)
    state = update_observation(state, enter, Response(True, 1, 0, remaining_s=1200))
    first = make_envelope(Intent("/measure", Point(500, 0), 1), "LOCAL", "first", 0)
    state = update_observation(state, first, Response(True, 2, 105, "direction", 0))
    assert state.beliefs[0].ever_detected
    opposite = make_envelope(Intent("/measure", Point(-500, 0), 1), "LOCAL", "opposite", 0)
    with pytest.raises(GeometryError, match="Positive observation history is inconsistent"):
        update_observation(state, opposite, Response(True, 3, 310, "direction", 180))
    assert state.beliefs[0].status != Status.ABSENT


def test_mixed_ring_certificate_rejects_edited_metadata_and_leaf(tmp_path):
    for name in ("ring_cover_certificate.json", "ring_stations_25.csv"):
        shutil.copyfile(RESOURCE / name, tmp_path / name)
    layout = load_ring(tmp_path)
    assert layout.name == "ring25" and len(layout.points) == 25
    path = tmp_path / "ring_cover_certificate.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["min_rx_radius_m"] = 999
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(GeometryError, match="metadata/CSV mismatch"):
        load_ring(tmp_path)
    data["min_rx_radius_m"] = 1000
    data["leaves"][0]["stop_indices"] = [0, 0, 0]
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(GeometryError, match="Invalid witness indices"):
        load_ring(tmp_path)


def test_mixed_actual_negative_proof_requires_accepted_same_channel_full_cell(monkeypatch):
    scale = directional_certificate.SCALE
    box = (-scale, -scale, scale, scale)
    points = ((-100 * scale, -100 * scale), (100 * scale, -100 * scale), (0, 100 * scale))
    monkeypatch.setattr(directional_certificate, "roots", lambda: (box,))
    layout = Layout("tiny", points, (box,), ((0, 1, 2),), "tiny-test")
    ids = ("m1", "m2", "m3")
    initial = directional_initial_state()
    beliefs = list(initial.beliefs)
    beliefs[0] = replace(
        beliefs[0],
        observations=tuple(
            Observation(rid, Point(x / scale, y / scale), "no_signal", None)
            for rid, (x, y) in zip(ids, points)
        ),
    )
    state = replace(initial, beliefs=tuple(beliefs), applied=ids)
    engine = Engine(layout, dynamic=True)
    assert engine.condition_complete((), tuple(Point(x / scale, y / scale) for x, y in points))
    assert state.beliefs[0].status == MixedStatus.UNKNOWN and not engine.proofs
    certified = engine.update(state, 1)
    assert certified.beliefs[0].status == MixedStatus.CERTIFIED_ABSENT
    proof = engine.proofs[1]
    directional_certificate.verify_actual_proof(proof, certified)

    forged = deepcopy(proof)
    forged["request_ids"] = ("m2", "m1", "m3")
    with pytest.raises(GeometryError, match="same-channel"):
        directional_certificate.verify_actual_proof(forged, certified)
    with pytest.raises(GeometryError, match="same-channel"):
        directional_certificate.verify_actual_proof(proof, replace(certified, applied=ids[:-1]))
    forged = deepcopy(proof)
    forged["leaves"] = []
    with pytest.raises(GeometryError, match="Missing certificate leaf"):
        directional_certificate.verify_actual_proof(forged, certified)
    forged = deepcopy(proof)
    forged["leaves"].append(forged["leaves"][0])
    with pytest.raises(GeometryError, match="Duplicate runtime certificate leaf"):
        directional_certificate.verify_actual_proof(forged, certified)
    discovered = list(certified.beliefs)
    discovered[0] = replace(discovered[0], ever_detected=True)
    with pytest.raises(GeometryError, match="Previously discovered"):
        directional_certificate.verify_actual_proof(proof, replace(certified, beliefs=tuple(discovered)))
