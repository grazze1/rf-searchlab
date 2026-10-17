"""Malformed-data boundaries of the local action protocol."""

import pytest

from rf_searchlab.core.domain import Intent, Point, ProtocolError, RawResponse
from rf_searchlab.core.protocol import dumps, loads, make_envelope, parse_response


@pytest.mark.parametrize("raw", [b'{"a":1,"a":2}', b'{"a":NaN}', b"[]", b"{broken", b"\xff"])
def test_json_parser_rejects_ambiguous_or_nonfinite_data(raw):
    with pytest.raises(ProtocolError):
        loads(raw)


@pytest.mark.parametrize("identifier", ["", "a" * 129, "a\nb", "\u200b"])
def test_request_identifier_constraints(identifier):
    with pytest.raises(ValueError):
        make_envelope(Intent("/enter"), "LOCAL", identifier, 0)


@pytest.mark.parametrize(
    "intent",
    [
        Intent("/unknown"),
        Intent("/enter", Point(0, 0)),
        Intent("/measure", Point(0, 0), 0),
        Intent("/measure", Point(2_000_001, 0), 1),
    ],
)
def test_illegal_intents_never_create_an_envelope(intent):
    with pytest.raises(ValueError):
        make_envelope(intent, "LOCAL", "m", 0)


@pytest.mark.parametrize(
    "patch",
    [
        {"accepted": 1},
        {"virtual_time_s": -1},
        {"real_timestamp_ms": True},
        {"measure_result": "unknown"},
        {"svd_deg": 360},
        {"svd_deg": None},
        {"unknown": 1},
    ],
)
def test_measure_response_is_exactly_typed(patch):
    payload = {
        "accepted": True,
        "real_timestamp_ms": 1,
        "virtual_time_s": 5,
        "measure_result": "direction",
        "svd_deg": 0,
    }
    with pytest.raises(ProtocolError):
        parse_response("/measure", RawResponse(200, dumps({**payload, **patch})))


def test_response_status_budget_and_rejection_time_are_validated():
    rejected = {"accepted": False, "real_timestamp_ms": 1, "virtual_time_s": 0}
    with pytest.raises(ProtocolError, match="HTTP"):
        parse_response("/enter", RawResponse(409, dumps(rejected)))
    with pytest.raises(ProtocolError, match="Rejected"):
        parse_response("/enter", RawResponse(200, dumps({**rejected, "virtual_time_s": 1})))
    entered = {
        "accepted": True,
        "real_timestamp_ms": 1,
        "virtual_time_s": 0,
        "remaining_real_duration_s": 1200,
        "max_virtual_duration_s": 360000,
        "max_real_duration_s": 1200,
    }
    for patch in (
        {"remaining_real_duration_s": 1.5},
        {"remaining_real_duration_s": 1201},
        {"max_real_duration_s": 0},
        {"max_virtual_duration_s": 0},
    ):
        with pytest.raises(ProtocolError):
            parse_response("/enter", RawResponse(200, dumps({**entered, **patch})))
