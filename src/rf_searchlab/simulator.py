"""Independent in-memory world, compatible with the retained public protocol types."""

from dataclasses import asdict, dataclass
from hashlib import sha256
from math import atan2, cos, degrees, hypot, isfinite, pi, sin, sqrt
from random import Random

from rf_searchlab.core.domain import Envelope, Point, RawResponse
from rf_searchlab.core.protocol import dumps, loads


@dataclass(frozen=True)
class Source:
    channel: int
    x: float
    y: float
    radius: float
    orientation: float | None = None

    def __post_init__(self):
        if type(self.channel) is not int or not 1 <= self.channel <= 20:
            raise ValueError("Invalid source channel")
        if not all(type(v) in (int, float) and isfinite(v) for v in (self.x, self.y, self.radius)):
            raise ValueError("Non-finite source")
        if hypot(self.x, self.y) > 1800 + 1e-8 or not 1000 <= self.radius <= 1500:
            raise ValueError("Source outside public domain")
        if self.orientation is not None and (
            type(self.orientation) not in (int, float) or not isfinite(self.orientation)
        ):
            raise ValueError("Invalid orientation")


@dataclass(frozen=True)
class WorldTruth:
    sources: tuple[Source, ...]
    noise_seed: int
    environment: str = "omni"
    family: str = "uniform"
    mode: str = "standard"

    def __post_init__(self):
        channels = [s.channel for s in self.sources]
        if type(self.noise_seed) is not int:
            raise ValueError("Noise seed must be an integer")
        if len(set(channels)) != len(channels) or len(channels) > 16:
            raise ValueError("Duplicate channels/excess sources")
        if self.mode not in ("standard", "custom") or self.environment not in ("omni", "mixed"):
            raise ValueError("Invalid mode/environment")
        if self.mode == "standard" and not 1 <= len(channels) <= 16:
            raise ValueError("Standard source count must be 1..16")
        if self.environment == "omni" and any(s.orientation is not None for s in self.sources):
            raise ValueError("Omnidirectional sources must be omnidirectional")


def generate(seed=101, environment="omni", count=10, family="uniform", mode="standard"):
    if family not in ("uniform", "cluster", "boundary", "radius1000", "outward"):
        raise ValueError("Unknown scenario family")
    rng = Random(seed)
    sources = []
    for i, c in enumerate(rng.sample(range(1, 21), count)):
        a = rng.random() * 2 * pi
        r = (
            1799.0
            if family in ("boundary", "outward")
            else 400 * sqrt(rng.random())
            if family == "cluster"
            else 1800 * sqrt(rng.random())
        )
        sources.append(
            Source(
                c,
                r * cos(a),
                r * sin(a),
                1000 if family == "radius1000" else rng.uniform(1000, 1500),
                (a if family == "outward" else rng.random() * 2 * pi)
                if environment == "mixed" and i % 2
                else None,
            )
        )
    return WorldTruth(tuple(sources), seed + 1_000_003, environment, family, mode)


def from_dict(data):
    if set(data) - {"sources", "noise_seed", "environment", "family", "mode"}:
        raise ValueError("Unknown scenario keys")
    return WorldTruth(
        tuple(Source(**s) for s in data["sources"]), **{k: v for k, v in data.items() if k != "sources"}
    )


class Simulator:
    """Never passed to a policy. Errors are fixed smooth synthetic spatial fields.

    The default synthetic model bounds the returned, two-decimal bearing error.
    The returned bearing is quantized and remains inside the configured error bound.
    """

    def __init__(self, truth: WorldTruth, error_mode="spatial", observation_model="bounded_round2"):
        if error_mode not in ("spatial", "zero", "positive", "negative"):
            raise ValueError("Unsupported error field")
        self._truth = truth
        self._sources = {s.channel: s for s in truth.sources}
        self._cleared: set[int] = set()
        self._cache: dict[str, tuple[tuple[str, bytes], RawResponse]] = {}
        self._entered = False
        self._exited = False
        self.position = Point(0, 0)
        self.channel = 1
        self.virtual_time = 0.0
        self.error_mode = error_mode
        if observation_model not in ("bounded_round2", "raw_float"):
            raise ValueError("Unsupported observation model")
        self.observation_model = observation_model

    def error(self, channel, point):
        if self.error_mode == "zero":
            return 0.0
        if self.error_mode == "positive":
            return 1.0
        if self.error_mode == "negative":
            return -1.0
        phase = (
            int.from_bytes(sha256(f"{self._truth.noise_seed}/{channel}".encode()).digest()[:8], "big")
            / 2**64
            * 2
            * pi
        )
        return sin(round(point.x_m, 6) / 113 + round(point.y_m, 6) / 197 + phase)

    def execute(self, envelope: Envelope) -> RawResponse:
        rid = envelope.request_id
        key = (envelope.path, envelope.raw)
        reject = {"accepted": False, "real_timestamp_ms": len(self._cache), "virtual_time_s": 0}
        if rid in self._cache:
            old, response = self._cache[rid]
            return response if key == old else RawResponse(409, dumps(reject))
        data = loads(envelope.raw)
        path = envelope.path
        required = {"arena_id", "robot_id", "request_id"}
        if path in ("/measure", "/clear"):
            required |= {"position", "channel"}
        if set(data) != required:
            return RawResponse(200, dumps(reject))
        if data.get("request_id") != rid or data.get("arena_id") != "default" or not data.get("robot_id"):
            return RawResponse(200, dumps(reject))
        if self._exited or (path != "/enter" and not self._entered) or (path == "/enter" and self._entered):
            return RawResponse(200, dumps(reject))
        answer: dict[str, object] = {
            "accepted": True,
            "real_timestamp_ms": len(self._cache) + 1,
            "virtual_time_s": self.virtual_time,
        }
        if path == "/enter":
            self._entered = True
            answer.update(
                remaining_real_duration_s=1200, max_virtual_duration_s=360000, max_real_duration_s=1200
            )
        elif path == "/exit":
            self._exited = True
            answer["exit_reason"] = "user_exit"
        elif path in ("/measure", "/clear"):
            c = data.get("channel")
            pos = data.get("position", {})
            if type(c) is not int or not 1 <= c <= 20 or not isinstance(pos, dict) or set(pos) != {"x", "y"}:
                return RawResponse(200, dumps(reject))
            try:
                p = Point(pos["x"], pos["y"])
            except (TypeError, ValueError):
                return RawResponse(200, dumps(reject))
            if max(abs(p.x_m), abs(p.y_m)) > 2_000_000:
                return RawResponse(200, dumps(reject))
            travel = hypot(p.x_m - self.position.x_m, p.y_m - self.position.y_m) / 5
            s = self._sources.get(c)
            d = hypot(p.x_m - s.x, p.y_m - s.y) if s else float("inf")
            if path == "/measure":
                visible = s is not None and c not in self._cleared and d <= s.radius
                if visible and s is not None and s.orientation is not None:
                    visible = (p.x_m - s.x) * cos(s.orientation) + (p.y_m - s.y) * sin(
                        s.orientation
                    ) >= -1e-10
                result = "no_signal" if not visible else "near" if d <= 5 else "direction"
                answer["measure_result"] = result
                if result == "direction":
                    assert s is not None
                    true_angle = degrees(atan2(s.y - p.y_m, s.x - p.x_m)) % 360
                    angle = (true_angle + self.error(c, p)) % 360
                    if self.observation_model == "bounded_round2":
                        angle = round(angle, 2) % 360
                        difference = (angle - true_angle + 180) % 360 - 180
                        if abs(difference) > 1:
                            angle = round(angle - math_sign(difference) * 0.01, 2) % 360
                    answer["svd_deg"] = angle
                self.virtual_time += travel + 5 + (c != self.channel)
                self.channel = c
            else:
                success = s is not None and c not in self._cleared and d <= 20
                answer["clear_result"] = "success" if success else "no_target_in_range"
                self.virtual_time += travel + 3 + 2 * success
                if success:
                    self._cleared.add(c)
            self.position = p
        else:
            return RawResponse(200, dumps(reject))
        self.virtual_time = round(self.virtual_time, 6)
        answer["virtual_time_s"] = self.virtual_time
        response = RawResponse(200, dumps(answer))
        self._cache[rid] = (key, response)
        return response

    def score(self):
        return {
            "true_count": len(self._sources),
            "cleared": len(self._cleared),
            "all_cleared": len(self._cleared) == len(self._sources),
        }

    def teaching_truth(self):
        return asdict(self._truth)


def math_sign(x):
    return 1 if x >= 0 else -1
