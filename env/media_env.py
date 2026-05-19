from typing import Any, Protocol

import gymnasium as gym
import numpy as np

from profile_system import PatternConfig, RewardFunctionProtocol, SignalConfig


class OSCInterfaceProtocol(Protocol):
    def get_signal(
        self,
        address: str,
        size: int,
        wait_for_new: bool = False,
        timeout: float | None = None,
    ) -> np.ndarray:
        ...

    def send_signal(self, address: str, values: np.ndarray) -> None:
        ...


class MediaEnv(gym.Env):

    def __init__(
        self,
        osc_interface: OSCInterfaceProtocol,
        pattern_config: PatternConfig,
        reward_function: RewardFunctionProtocol,
        max_steps: int,
        observation_timeout: float | None = 1.0,
    ):
        super().__init__()
        self.osc = osc_interface
        self.pattern_config = pattern_config
        self.reward_function = reward_function
        self.max_steps = int(max_steps)
        self.observation_timeout = observation_timeout

        self.observation_space = gym.spaces.Box(
            low=pattern_config.observation_low(),
            high=pattern_config.observation_high(),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=pattern_config.action_low(),
            high=pattern_config.action_high(),
            dtype=np.float32,
        )

        self._step_count = 0
        self._last_observation = np.zeros(self.observation_space.shape, dtype=np.float32)
        self._last_named_observations: dict[str, np.ndarray] = {}
        self._last_named_actions: dict[str, np.ndarray] = {}

    def step(self, action: np.ndarray):
        action = self._normalize_action(action)
        named_actions = self._send_action(action)
        next_observation, named_observations = self._read_observation(wait_for_new=False)
        self._step_count += 1

        terminated = False
        max_steps_reached = self._step_count >= self.max_steps
        truncated = bool(max_steps_reached)
        episode_end_reason = "max_steps" if max_steps_reached else "none"

        info = {
            "observations": _to_serializable_dict(named_observations),
            "actions": _to_serializable_dict(named_actions),
            "max_steps_reached": bool(max_steps_reached),
            "episode_end_reason": episode_end_reason,
            "pattern_id": self.pattern_config.pattern_id,
        }

        try:
            reward = self.reward_function.compute(
                self._last_observation,
                action,
                next_observation,
                info,
            )
        except TypeError as exc:
            if "unhashable type: 'list'" in str(exc):
                raise TypeError(
                    "Reward module could not handle reward_config.output as a list. "
                    "Update the reward module to iterate over multiple output signals, "
                    "or set reward_config.output to one signal name."
                ) from exc
            raise

        self._last_observation = next_observation.copy()
        self._last_named_observations = named_observations
        self._last_named_actions = named_actions

        return next_observation, float(reward), terminated, truncated, info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self._step_count = 0
        self.reward_function.reset()
        self._send_reset()

        observation, named_observations = self._read_observation(wait_for_new=True)
        self._last_observation = observation.copy()
        self._last_named_observations = named_observations
        self._last_named_actions = {}

        info = {
            "observations": _to_serializable_dict(named_observations),
            "pattern_id": self.pattern_config.pattern_id,
        }
        return observation, info

    def _read_observation(self, wait_for_new: bool) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        named_values: dict[str, np.ndarray] = {}
        values: list[np.ndarray] = []

        for signal in self.pattern_config.observations:
            signal_values = self._read_signal(signal, wait_for_new=wait_for_new)
            named_values[signal.name] = signal_values
            values.append(signal_values)

        observation = np.concatenate(values).astype(np.float32)
        return observation, named_values

    def _read_signal(self, signal: SignalConfig, wait_for_new: bool) -> np.ndarray:
        value = self.osc.get_signal(
            signal.address,
            signal.size,
            wait_for_new=wait_for_new,
            timeout=self.observation_timeout,
        )
        value = np.asarray(value, dtype=np.float32)
        if value.shape != (signal.size,):
            raise ValueError(
                f"OSC signal '{signal.name}' expected shape ({signal.size},), got {value.shape}"
            )
        return np.clip(value, signal.low_values(), signal.high_values()).astype(np.float32)

    def _normalize_action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32)
        expected_shape = self.action_space.shape
        if action.shape != expected_shape:
            raise ValueError(f"Action expected shape {expected_shape}, got {action.shape}")
        return np.clip(action, self.action_space.low, self.action_space.high).astype(np.float32)

    def _send_action(self, action: np.ndarray) -> dict[str, np.ndarray]:
        named_actions: dict[str, np.ndarray] = {}
        offset = 0

        for signal in self.pattern_config.actions:
            values = action[offset : offset + signal.size].astype(np.float32)
            offset += signal.size
            named_actions[signal.name] = values
            self.osc.send_signal(signal.address, values)

        return named_actions

    def _send_reset(self) -> None:
        send_reset = getattr(self.osc, "send_reset", None)
        if callable(send_reset):
            send_reset(self.pattern_config)

def _to_serializable_dict(values: dict[str, np.ndarray]) -> dict[str, list[float]]:
    return {name: value.astype(float).tolist() for name, value in values.items()}
