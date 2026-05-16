import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from config import SystemConfig, load_system_config
from profile_system import PatternConfig, SignalConfig


@dataclass
class InferenceManagerStatus:
    running: bool = False
    state: str = "idle"
    model_id: str | None = None
    step_interval_seconds: float | None = None
    last_action: list[float] | None = None
    last_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "state": self.state,
            "model_id": self.model_id,
            "step_interval_seconds": self.step_interval_seconds,
            "last_action": self.last_action,
            "last_error": self.last_error,
        }


class InferenceManager:
    def __init__(self, system_config: SystemConfig | None = None):
        self.system_config = system_config or load_system_config()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._status = InferenceManagerStatus(
            step_interval_seconds=self.system_config.inference.step_interval_seconds,
        )

    def start_inference(
        self,
        model: Any,
        pattern: PatternConfig,
        osc_interface: Any,
        model_id: str | None = None,
    ) -> InferenceManagerStatus:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("Inference is already running")

            self._validate_model_spaces(model, pattern)
            self._stop_event.clear()
            self._status = InferenceManagerStatus(
                running=True,
                state="starting",
                model_id=model_id,
                step_interval_seconds=self.system_config.inference.step_interval_seconds,
                last_error=None,
            )
            self._thread = threading.Thread(
                target=self._run_inference,
                args=(model, pattern, osc_interface),
                daemon=True,
            )
            self._thread.start()
            return InferenceManagerStatus(
                running=self._status.running,
                state=self._status.state,
                model_id=self._status.model_id,
                step_interval_seconds=self._status.step_interval_seconds,
                last_action=self._status.last_action,
                last_error=self._status.last_error,
            )

    def stop_inference(self, wait: bool = False, timeout: float | None = None) -> InferenceManagerStatus:
        with self._lock:
            self._stop_event.set()
            thread = self._thread
            if self._status.running:
                self._status.state = "stopping"

        if wait and thread is not None:
            thread.join(timeout=timeout)

        return self.get_status()

    def get_status(self) -> InferenceManagerStatus:
        with self._lock:
            return InferenceManagerStatus(
                running=self._status.running,
                state=self._status.state,
                model_id=self._status.model_id,
                step_interval_seconds=self._status.step_interval_seconds,
                last_action=self._status.last_action,
                last_error=self._status.last_error,
            )

    def _run_inference(
        self,
        model: Any,
        pattern: PatternConfig,
        osc_interface: Any,
    ) -> None:
        try:
            with self._lock:
                self._status.state = "running"

            interval = self.system_config.inference.step_interval_seconds
            deterministic = self.system_config.inference.deterministic
            next_tick = time.monotonic()

            while not self._stop_event.is_set():
                observation = self._read_observation(pattern, osc_interface)
                action, _ = model.predict(observation, deterministic=deterministic)
                action = self._normalize_action(action, pattern)
                self._send_action(action, pattern, osc_interface)

                with self._lock:
                    self._status.last_action = action.astype(float).tolist()

                next_tick += interval
                sleep_time = max(0.0, next_tick - time.monotonic())
                if self._stop_event.wait(timeout=sleep_time):
                    break

            self._finish(state="stopped")
        except Exception as exc:
            self._finish(state="error", error=exc)

    def _read_observation(self, pattern: PatternConfig, osc_interface: Any) -> np.ndarray:
        values = []
        for signal in pattern.observations:
            signal_values = self._read_signal(signal, osc_interface)
            values.append(signal_values)
        return np.concatenate(values).astype(np.float32)

    def _read_signal(self, signal: SignalConfig, osc_interface: Any) -> np.ndarray:
        value = osc_interface.get_signal(
            signal.address,
            signal.size,
            wait_for_new=False,
        )
        value = np.asarray(value, dtype=np.float32)
        if value.shape != (signal.size,):
            raise ValueError(
                f"OSC signal '{signal.name}' expected shape ({signal.size},), got {value.shape}"
            )
        return np.clip(value, signal.low_values(), signal.high_values()).astype(np.float32)

    def _normalize_action(self, action: np.ndarray, pattern: PatternConfig) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        expected_size = pattern.action_dim
        if action.shape != (expected_size,):
            raise ValueError(f"Inference action expected shape ({expected_size},), got {action.shape}")
        return np.clip(action, pattern.action_low(), pattern.action_high()).astype(np.float32)

    def _send_action(
        self,
        action: np.ndarray,
        pattern: PatternConfig,
        osc_interface: Any,
    ) -> None:
        offset = 0
        for signal in pattern.actions:
            values = action[offset : offset + signal.size].astype(np.float32)
            offset += signal.size
            osc_interface.send_signal(signal.address, values)

    def _validate_model_spaces(self, model: Any, pattern: PatternConfig) -> None:
        observation_shape = getattr(model.observation_space, "shape", None)
        action_shape = getattr(model.action_space, "shape", None)
        if observation_shape != (pattern.observation_dim,):
            raise ValueError(
                "Loaded model observation space does not match pattern: "
                f"{observation_shape} != ({pattern.observation_dim},)"
            )
        if action_shape != (pattern.action_dim,):
            raise ValueError(
                "Loaded model action space does not match pattern: "
                f"{action_shape} != ({pattern.action_dim},)"
            )

    def _finish(self, state: str, error: Exception | None = None) -> None:
        with self._lock:
            self._status.running = False
            self._status.state = state
            self._status.last_error = None if error is None else str(error)
