import hashlib
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol, runtime_checkable

"""Loads a reward python script. 
Each reward script must define a create_reward(config) function 
that returns an object with reset() and compute(info) methods."""
@runtime_checkable
class RewardFunctionProtocol(Protocol):
    def reset(self) -> None:
        ...

    def compute(self, info: dict[str, Any]) -> float:
        ...

@dataclass(frozen=True)
class LoadedReward:
    reward_id: str
    path: Path
    module: ModuleType
    reward_function: RewardFunctionProtocol


class RewardLoader:
    def __init__(self, rewards_dir: str | Path):
        self.rewards_dir = Path(rewards_dir)

    def path_for(self, reward_id: str) -> Path:
        return self.rewards_dir / f"reward_{reward_id}.py"

    def load(self, reward_id: str, config: dict[str, Any] | None = None) -> LoadedReward:
        path = self.path_for(reward_id)
        return self.load_from_path(reward_id=reward_id, path=path, config=config)

    def load_from_path(
        self,
        reward_id: str,
        path: str | Path,
        config: dict[str, Any] | None = None,
    ) -> LoadedReward:
        reward_path = Path(path)
        module = self._load_module(reward_id=reward_id, path=reward_path)
        reward_function = self._create_reward_function(module=module, config=config)

        return LoadedReward(
            reward_id=str(reward_id),
            path=reward_path,
            module=module,
            reward_function=reward_function,
        )

    @staticmethod
    def _load_module(reward_id: str, path: Path) -> ModuleType:
        if not path.exists():
            raise FileNotFoundError(f"Reward module does not exist: {path}")
        if path.suffix != ".py":
            raise ValueError(f"Reward module must be a Python file: {path}")

        module_name = _module_name_for(reward_id=reward_id, path=path)
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not create import spec for reward module: {path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def _create_reward_function(
            module: ModuleType,
        config: dict[str, Any] | None,
    ) -> RewardFunctionProtocol:
        create_reward = getattr(module, "create_reward", None)
        if not callable(create_reward):
            raise AttributeError("Reward module must define callable create_reward(config=None)")

        reward_function = create_reward(config or {})
        if not isinstance(reward_function, RewardFunctionProtocol):
            raise TypeError("create_reward() must return an object with reset() and compute(info)")

        return reward_function


def _module_name_for(reward_id: str, path: Path) -> str:
    resolved = str(path.resolve())
    digest = hashlib.sha1(resolved.encode("utf-8")).hexdigest()[:12]
    safe_reward_id = "".join(char if char.isalnum() else "_" for char in str(reward_id))
    return f"dynamic_reward_{safe_reward_id}_{digest}"
