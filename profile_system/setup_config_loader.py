import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrainingConfig:
    algorithm: str = "ppo"
    total_timesteps: int = 10000
    max_steps: int = 500

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "TrainingConfig":
        data = data or {}
        training = cls(
            algorithm=str(data.get("algorithm", "ppo")).lower(),
            total_timesteps=int(data.get("total_timesteps", 10000)),
            max_steps=int(data.get("max_steps", 500)),
        )
        training.validate()
        return training

    def validate(self) -> None:
        if self.algorithm != "ppo":
            raise ValueError(f"Unsupported training algorithm: {self.algorithm}")
        if self.total_timesteps <= 0:
            raise ValueError("training.total_timesteps must be positive")
        if self.max_steps <= 0:
            raise ValueError("training.max_steps must be positive")


@dataclass(frozen=True)
class SetupConfig:
    setup_id: str
    name: str
    description: str
    pattern_id: str
    reward_id: str
    reward_config: dict[str, Any] = field(default_factory=dict)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SetupConfig":
        missing = [key for key in ("setup_id", "pattern_id", "reward_id") if key not in data]
        if missing:
            raise ValueError(f"Setup config is missing fields: {', '.join(missing)}")

        reward_config = data.get("reward_config", {})
        if not isinstance(reward_config, dict):
            raise ValueError("setup.reward_config must be a JSON object")

        setup = cls(
            setup_id=str(data["setup_id"]),
            name=str(data.get("name", "")),
            description=str(data.get("description", "")),
            pattern_id=str(data["pattern_id"]),
            reward_id=str(data["reward_id"]),
            reward_config=reward_config,
            training=TrainingConfig.from_dict(_optional_object(data.get("training"), "training")),
        )
        setup.validate()
        return setup

    def validate(self) -> None:
        if not self.setup_id:
            raise ValueError("setup_id must not be empty")
        if not self.pattern_id:
            raise ValueError("pattern_id must not be empty")
        if not self.reward_id:
            raise ValueError("reward_id must not be empty")


class SetupConfigLoader:
    def __init__(self, setups_dir: str | Path):
        self.setups_dir = Path(setups_dir)

    def path_for(self, setup_id: str) -> Path:
        return self.setups_dir / f"setup_{setup_id}.json"

    def load(self, setup_id: str) -> SetupConfig:
        path = self.path_for(setup_id)
        return self.load_from_path(path)

    @staticmethod
    def load_from_path(path: str | Path) -> SetupConfig:
        data = _read_json_object(Path(path))
        return SetupConfig.from_dict(data)


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a JSON object: {path}")
    return data


def _optional_object(value: Any, label: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value
