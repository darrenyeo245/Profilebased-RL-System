import threading
from dataclasses import dataclass
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from config import RuntimePaths, SystemConfig, load_system_config
from env import MediaEnv, OSCInterfaceProtocol
from profile_system import LoadedReward, PatternConfig, SetupConfig


@dataclass
class TrainingManagerStatus:
    running: bool = False
    state: str = "idle"
    algorithm: str | None = None
    total_timesteps: int | None = None
    model_path: str | None = None
    last_error: str | None = None


class StopAndSaveCallback(BaseCallback):
    def __init__(
        self,
        stop_event: threading.Event,
        model_path: Path,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.stop_event = stop_event
        self.model_path = model_path

    def _on_step(self) -> bool:
        if not self.stop_event.is_set():
            return True

        self.model.save(str(self.model_path))
        return False


class TrainingManager:
    def __init__(
        self,
        runtime_paths: RuntimePaths | None = None,
        system_config: SystemConfig | None = None,
    ):
        self.system_config = system_config or load_system_config()
        self.runtime_paths = runtime_paths or self.system_config.runtime_paths
        self.current_model_path = self.runtime_paths.models_dir / "current_model.zip"

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._model: PPO | None = None
        self._env: MediaEnv | None = None
        self._status = TrainingManagerStatus(
            model_path=str(self.current_model_path),
        )

    def start_training(
        self,
        setup: SetupConfig,
        pattern: PatternConfig,
        reward: LoadedReward,
        osc_interface: OSCInterfaceProtocol,
    ) -> TrainingManagerStatus:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("Training is already running")

            self.runtime_paths.models_dir.mkdir(parents=True, exist_ok=True)
            self._stop_event.clear()
            self._status = TrainingManagerStatus(
                running=True,
                state="starting",
                algorithm=setup.training.algorithm,
                total_timesteps=setup.training.total_timesteps,
                model_path=str(self.current_model_path),
                last_error=None,
            )

            self._thread = threading.Thread(
                target=self._run_training,
                args=(setup, pattern, reward, osc_interface),
                daemon=True,
            )
            self._thread.start()
            return self._status

    def stop_training(self, wait: bool = False, timeout: float | None = None) -> TrainingManagerStatus:
        with self._lock:
            self._stop_event.set()
            thread = self._thread
            if self._status.running:
                self._status.state = "stopping"

        if wait and thread is not None:
            thread.join(timeout=timeout)

        with self._lock:
            model = self._model
            still_running = self._status.running

        if wait and not still_running and model is not None:
            self.runtime_paths.models_dir.mkdir(parents=True, exist_ok=True)
            model.save(str(self.current_model_path))

        return self.get_status()

    def get_status(self) -> TrainingManagerStatus:
        with self._lock:
            return TrainingManagerStatus(
                running=self._status.running,
                state=self._status.state,
                algorithm=self._status.algorithm,
                total_timesteps=self._status.total_timesteps,
                model_path=self._status.model_path,
                last_error=self._status.last_error,
            )

    @property
    def model(self) -> PPO | None:
        return self._model

    @property
    def env(self) -> MediaEnv | None:
        return self._env

    def _run_training(
        self,
        setup: SetupConfig,
        pattern: PatternConfig,
        reward: LoadedReward,
        osc_interface: OSCInterfaceProtocol,
    ) -> None:
        try:
            with self._lock:
                self._status.state = "running"

            env = MediaEnv(
                osc_interface=osc_interface,
                pattern_config=pattern,
                reward_function=reward.reward_function,
                max_steps=setup.training.max_steps,
            )
            model = self._load_or_create_model(setup, env)
            self._model = model
            self._env = env
            callback = StopAndSaveCallback(
                stop_event=self._stop_event,
                model_path=self.current_model_path,
            )

            model.learn(
                total_timesteps=setup.training.total_timesteps,
                callback=callback,
            )
            model.save(str(self.current_model_path))

            self._finish(state="stopped" if self._stop_event.is_set() else "finished")
        except Exception as exc:
            self._finish(state="error", error=exc)

    def _load_or_create_model(self, setup: SetupConfig, env: MediaEnv) -> PPO:
        if setup.training.algorithm != "ppo":
            raise ValueError(f"Unsupported training algorithm: {setup.training.algorithm}")

        if self.current_model_path.exists():
            return PPO.load(str(self.current_model_path), env=env)

        return PPO(
            "MlpPolicy",
            env,
            verbose=1,
            n_steps=2048,
            batch_size=64,
        )

    def _finish(self, state: str, error: Exception | None = None) -> None:
        with self._lock:
            self._status.running = False
            self._status.state = state
            self._status.last_error = None if error is None else str(error)
