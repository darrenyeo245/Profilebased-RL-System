import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from config import SpaceLimits


JsonValue = str | int | float | bool | None | dict[str, Any] | list[Any]

"""Loads OSC pattern configurations from JSON files. Each pattern defines a set of observation and action signals,"""
@dataclass(frozen=True)
class SignalConfig:
    name: str
    address: str
    size: int
    low: float | list[float]
    high: float | list[float]

    @classmethod
    def from_dict(cls, data: dict[str, Any], section: str) -> "SignalConfig":
        missing = [key for key in ("name", "address", "size", "low", "high") if key not in data]
        if missing:
            raise ValueError(f"{section} signal is missing fields: {', '.join(missing)}")

        signal = cls(
            name=str(data["name"]),
            address=str(data["address"]),
            size=int(data["size"]),
            low=_number_or_number_list(data["low"], "low"),
            high=_number_or_number_list(data["high"], "high"),
        )
        signal.validate(section)
        return signal

    def validate(self, section: str) -> None:
        if not self.name:
            raise ValueError(f"{section} signal name must not be empty")
        if not self.address.startswith("/"):
            raise ValueError(f"{section} signal '{self.name}' address must start with '/'")
        if self.size <= 0:
            raise ValueError(f"{section} signal '{self.name}' size must be positive")

        low_values = self.low_values()
        high_values = self.high_values()
        if low_values.shape != high_values.shape:
            raise ValueError(f"{section} signal '{self.name}' low/high dimensions do not match")
        if np.any(low_values >= high_values):
            raise ValueError(f"{section} signal '{self.name}' low values must be smaller than high values")

    def low_values(self) -> np.ndarray:
        return _expand_bound(self.low, self.size, self.name, "low")

    def high_values(self) -> np.ndarray:
        return _expand_bound(self.high, self.size, self.name, "high")

@dataclass(frozen=True)
class PatternConfig:
    pattern_id: str
    name: str
    description: str
    observations: list[SignalConfig]
    actions: list[SignalConfig]

    @classmethod
    def from_dict(cls, data: dict[str, Any], limits: SpaceLimits) -> "PatternConfig":
        missing = [key for key in ("id", "name", "observations", "actions") if key not in data]
        if missing:
            raise ValueError(f"Pattern config is missing fields: {', '.join(missing)}")

        observations = _parse_signals(data["observations"], "observation")
        actions = _parse_signals(data["actions"], "action")

        pattern = cls(
            pattern_id=str(data["id"]),
            name=str(data["name"]),
            description=str(data.get("description", "")),
            observations=observations,
            actions=actions,
        )
        pattern.validate(limits)
        return pattern

    @property
    def observation_dim(self) -> int:
        return sum(signal.size for signal in self.observations)

    @property
    def action_dim(self) -> int:
        return sum(signal.size for signal in self.actions)

    def observation_low(self) -> np.ndarray:
        return np.concatenate([signal.low_values() for signal in self.observations]).astype(np.float32)

    def observation_high(self) -> np.ndarray:
        return np.concatenate([signal.high_values() for signal in self.observations]).astype(np.float32)

    def action_low(self) -> np.ndarray:
        return np.concatenate([signal.low_values() for signal in self.actions]).astype(np.float32)

    def action_high(self) -> np.ndarray:
        return np.concatenate([signal.high_values() for signal in self.actions]).astype(np.float32)

    def validate(self, limits: SpaceLimits) -> None:
        if not self.pattern_id:
            raise ValueError("Pattern id must not be empty")
        if not self.observations:
            raise ValueError("Pattern must define at least one observation")
        if not self.actions:
            raise ValueError("Pattern must define at least one action")

        _validate_unique_names(self.observations, "observation")
        _validate_unique_names(self.actions, "action")

        signal_count = len(self.observations) + len(self.actions)
        if signal_count > limits.max_signal_count:
            raise ValueError(
                f"Pattern uses {signal_count} signals, "
                f"but the local limit is {limits.max_signal_count}"
            )
        if self.observation_dim > limits.max_observation_dim:
            raise ValueError(
                f"Pattern observation dimension is {self.observation_dim}, "
                f"but the local limit is {limits.max_observation_dim}"
            )
        if self.action_dim > limits.max_action_dim:
            raise ValueError(
                f"Pattern action dimension is {self.action_dim}, "
                f"but the local limit is {limits.max_action_dim}"
            )


class PatternConfigLoader:
    def __init__(self, patterns_dir: str | Path, limits: SpaceLimits):
        self.patterns_dir = Path(patterns_dir)
        self.limits = limits

    def path_for(self, pattern_id: str) -> Path:
        return self.patterns_dir / f"pattern_{pattern_id}.json"

    def load(self, pattern_id: str) -> PatternConfig:
        path = self.path_for(pattern_id)
        return self.load_from_path(path)

    def load_from_path(self, path: str | Path) -> PatternConfig:
        data = _read_json_object(Path(path))
        return PatternConfig.from_dict(data, self.limits)


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a JSON object: {path}")
    return data


def _parse_signals(value: JsonValue, section: str) -> list[SignalConfig]:
    if not isinstance(value, list):
        raise ValueError(f"Pattern {section}s must be a list")
    return [SignalConfig.from_dict(_require_object(item, section), section) for item in value]


def _require_object(value: JsonValue, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} entry must be a JSON object")
    return value


def _number_or_number_list(value: JsonValue, label: str) -> float | list[float]:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a number or a number list")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list) and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value):
        return [float(item) for item in value]
    raise ValueError(f"{label} must be a number or a number list")


def _expand_bound(value: float | list[float], size: int, signal_name: str, label: str) -> np.ndarray:
    if isinstance(value, list):
        if len(value) != size:
            raise ValueError(f"Signal '{signal_name}' {label} list must contain {size} values")
        return np.asarray(value, dtype=np.float32)
    return np.full(size, value, dtype=np.float32)


def _validate_unique_names(signals: list[SignalConfig], section: str) -> None:
    names = [signal.name for signal in signals]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate {section} names: {', '.join(duplicates)}")
