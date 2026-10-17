"""Strict local four-action wire contract; no I/O or request creation on import."""

import json
import math
import unicodedata
from dataclasses import replace
from typing import Any

from rf_searchlab.core.belief import wire_point
from rf_searchlab.core.domain import Envelope, Intent, Point, ProtocolError, RawResponse, Response


def dumps(value: object) -> bytes:
    """Deterministic UTF-8 JSON without BOM or non-finite values."""
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def loads(raw: bytes) -> dict[str, Any]:
    """Strict JSON object parser, rejecting duplicate keys and NaN/Infinity."""

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for k, v in items:
            if k in result:
                raise ProtocolError("Duplicate JSON key")
            result[k] = v
        return result

    def invalid(value: str) -> None:
        raise ProtocolError(f"Invalid JSON constant {value}")

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=invalid)
    except (UnicodeError, ValueError) as exc:
        raise ProtocolError("Malformed JSON") from exc
    if not isinstance(value, dict):
        raise ProtocolError("Expected JSON object")
    return value


def identifier(value: str, limit: int) -> None:
    """Validate byte lengths and Unicode control/format exclusions."""
    if (
        not isinstance(value, str)
        or not 0 < len(value.encode("utf-8")) <= limit
        or any(unicodedata.category(c).startswith("C") for c in value)
    ):
        raise ValueError("Invalid identifier")


def make_envelope(intent: Intent, robot_id: str, request_id: str, now: float) -> Envelope:
    """Freeze path, canonical position, payload bytes and ID for one new action."""
    identifier(robot_id, 64)
    identifier(request_id, 128)
    if intent.path not in ("/enter", "/measure", "/clear", "/exit"):
        raise ValueError("Only four declared POST paths exist")
    payload: dict[str, object] = {
        "arena_id": "default",
        "robot_id": robot_id,
        "request_id": request_id,
    }
    if intent.path in ("/measure", "/clear"):
        if type(intent.channel) is not int or not 1 <= intent.channel <= 20 or intent.position is None:
            raise ValueError("Invalid action channel/position")
        point = wire_point(intent.position)
        if max(abs(point.x_m), abs(point.y_m)) > 2000000:
            raise ValueError("Coordinate limit exceeded")
        intent = replace(intent, position=point)
        payload.update(position={"x": point.x_m, "y": point.y_m}, channel=intent.channel)
    elif intent.position is not None or intent.channel is not None:
        raise ValueError("Unexpected enter/exit action fields")
    return Envelope(intent.path, dumps(payload), request_id, now, intent)


def number(value: Any, name: str) -> float:
    """Read a finite nonnegative JSON number; booleans are not numbers."""
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ProtocolError(f"Invalid {name}")
    return float(value)


def parse_response(path: str, raw: RawResponse) -> Response:
    """Validate HTTP status, business status and exact result-specific fields."""
    data = loads(raw.body)
    if raw.status != 200:
        raise ProtocolError(f"HTTP {raw.status}")
    if type(data.get("accepted")) is not bool:
        raise ProtocolError("accepted must be boolean")
    timestamp = number(data.get("real_timestamp_ms"), "timestamp")
    virtual = number(data.get("virtual_time_s"), "virtual time")
    keys = {"accepted", "real_timestamp_ms", "virtual_time_s"}
    result = None
    angle = None
    remaining = None
    max_virtual = None
    max_real = None
    if data["accepted"]:
        if path == "/enter":
            keys |= {"remaining_real_duration_s", "max_virtual_duration_s", "max_real_duration_s"}
            r = number(data.get("remaining_real_duration_s"), "remaining time")
            if not r.is_integer() or r > 1200:
                raise ProtocolError("Invalid remaining time")
            remaining = int(r)
            max_virtual = number(data.get("max_virtual_duration_s"), "virtual limit")
            max_real = number(data.get("max_real_duration_s"), "real limit")
            if max_virtual <= 0 or max_real <= 0 or remaining > max_real:
                raise ProtocolError("Invalid budget limits")
        elif path == "/measure":
            keys.add("measure_result")
            result = data.get("measure_result")
            if result not in ("direction", "near", "no_signal"):
                raise ProtocolError("Invalid measure_result")
            if result == "direction":
                keys.add("svd_deg")
                angle = number(data.get("svd_deg"), "svd_deg")
                if angle >= 360:
                    raise ProtocolError("Angle must be <360")
        elif path == "/clear":
            keys.add("clear_result")
            result = data.get("clear_result")
            if result not in ("success", "no_target_in_range"):
                raise ProtocolError("Invalid clear_result")
        elif path == "/exit":
            keys.add("exit_reason")
            result = data.get("exit_reason")
            if result != "user_exit":
                raise ProtocolError("Invalid exit_reason")
        else:
            raise ProtocolError("Unknown response path")
    elif virtual != 0:
        raise ProtocolError("Rejected virtual_time_s must be zero")
    if set(data) != keys:
        raise ProtocolError("Unexpected or missing response fields")
    return Response(data["accepted"], timestamp, virtual, result, angle, remaining, max_virtual, max_real)


def intent_from_dict(data: dict[str, Any]) -> Intent:
    """Decode a local intent record for replay; no local request fields are added."""
    from rf_searchlab.core.domain import OpticalCell, OpticalPlan

    data = dict(data)
    if data.get("position") is not None:
        data["position"] = Point(**data["position"])
    plan = data.get("optical_plan")
    if plan is not None:
        data["optical_plan"] = OpticalPlan(
            tuple(OpticalCell(c["i"], c["j"], Point(**c["center"])) for c in plan["cells"]),
            plan["angle_rad"],
            Point(**plan["origin"]),
            plan["set_version"],
            plan["index"],
        )
    return Intent(**data)
