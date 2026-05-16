import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SpaceLimits:
    max_observation_dim: int
    max_action_dim: int
    max_signal_count: int

    def validate(self) -> None:
        values = {
            "max_observation_dim": self.max_observation_dim,
            "max_action_dim": self.max_action_dim,
            "max_signal_count": self.max_signal_count,
        }
        for name, value in values.items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class RuntimePaths:
    models_dir: Path
    mediaip_checkout_dir: Path


@dataclass(frozen=True)
class InferenceConfig:
    step_interval_seconds: float
    deterministic: bool

    def validate(self) -> None:
        if self.step_interval_seconds <= 0:
            raise ValueError("step_interval_seconds must be positive")


@dataclass(frozen=True)
class SystemConfig:
    space_limits: SpaceLimits
    runtime_paths: RuntimePaths
    inference: InferenceConfig

    def ensure_runtime_dirs(self) -> None:
        self.runtime_paths.models_dir.mkdir(parents=True, exist_ok=True)
        self.runtime_paths.mediaip_checkout_dir.mkdir(parents=True, exist_ok=True)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_runtime_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return project_root / path


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"System config must contain a JSON object: {path}")
    return data


def load_system_config(path: str | os.PathLike[str] | None = None) -> SystemConfig:
    project_root = _project_root()
    config_path = Path(
        path
        or os.getenv("RL_SYSTEM_CONFIG")
        or project_root / "config" / "system_config.json"
    )

    data = _read_json(config_path)
    limits_data = data.get("space_limits", {})
    runtime_data = data.get("runtime_paths", {})
    inference_data = data.get("inference", {})

    limits = SpaceLimits(
        max_observation_dim=int(limits_data.get("max_observation_dim", 64)),
        max_action_dim=int(limits_data.get("max_action_dim", 64)),
        max_signal_count=int(limits_data.get("max_signal_count", 32)),
    )
    limits.validate()

    runtime_paths = RuntimePaths(
        models_dir=_resolve_runtime_path(
            project_root,
            str(runtime_data.get("models_dir", "runtime/models")),
        ),
        mediaip_checkout_dir=_resolve_runtime_path(
            project_root,
            str(runtime_data.get("mediaip_checkout_dir", "runtime/mediaip_storage")),
        ),
    )

    inference = InferenceConfig(
        step_interval_seconds=float(inference_data.get("step_interval_seconds", 0.05)),
        deterministic=bool(inference_data.get("deterministic", True)),
    )
    inference.validate()

    return SystemConfig(
        space_limits=limits,
        runtime_paths=runtime_paths,
        inference=inference,
    )
