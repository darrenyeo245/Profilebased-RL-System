from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import SystemConfig, load_system_config
from storage import MediaIPStorageClient
from .pattern_config_loader import PatternConfig, PatternConfigLoader, SignalConfig
from .reward_loader import LoadedReward, RewardLoader
from .setup_config_loader import SetupConfig, SetupConfigLoader


@dataclass
class RLProfileStatus:
    setup_id: str | None = None
    pattern_id: str | None = None
    reward_id: str | None = None
    model_id: str | None = None
    training_status: str = "idle"
    inference_status: str = "idle"
    last_error: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "setup_id": self.setup_id,
            "pattern_id": self.pattern_id,
            "reward_id": self.reward_id,
            "model_id": self.model_id,
            "training_status": self.training_status,
            "inference_status": self.inference_status,
            "last_error": self.last_error,
        }


class RLProfileSystem:
    def __init__(
        self,
        mediaip_rl_root: str | Path | None = None,
        system_config: SystemConfig | None = None,
        mediaip_storage_client: MediaIPStorageClient | None = None,
        training_manager: Any | None = None,
        inference_manager: Any | None = None,
    ):
        self.system_config = system_config or load_system_config()
        self.mediaip_storage_client = mediaip_storage_client
        self.training_manager = training_manager
        self.inference_manager = inference_manager
        self.mediaip_rl_root = self._resolve_initial_mediaip_rl_root(mediaip_rl_root)

        self._configure_loaders()

        self.status = RLProfileStatus()
        self.current_setup: SetupConfig | None = None
        self.current_pattern: PatternConfig | None = None
        self.current_reward: LoadedReward | None = None
        self.inference_model: Any | None = None

    def load_setup(self, setup_id: str) -> RLProfileStatus:
        try:
            self.refresh_mediaip_storage(setup_id=setup_id)
            setup = self.setup_loader.load(setup_id)
            pattern = self.pattern_loader.load(setup.pattern_id)
            self._validate_setup_pattern_compatibility(setup, pattern)
            reward = self.reward_loader.load(setup.reward_id, setup.reward_config)

            self.current_setup = setup
            self.current_pattern = pattern
            self.current_reward = reward
            self.status = RLProfileStatus(
                setup_id=setup.setup_id,
                pattern_id=pattern.pattern_id,
                reward_id=reward.reward_id,
                model_id=self.status.model_id,
                training_status="ready",
                inference_status=self.status.inference_status,
                last_error=None,
            )
            return self.status
        except Exception as exc:
            self._set_error(exc)
            raise

    def refresh_mediaip_storage(self, setup_id: str | None = None) -> None:
        if self.mediaip_storage_client is None:
            return

        if setup_id is None:
            paths = self.mediaip_storage_client.ensure_available()
        else:
            paths = self.mediaip_storage_client.download_setup_bundle(setup_id)
        self.mediaip_rl_root = paths.rl_system_root
        self._configure_loaders()

    def require_ready_profile(self) -> tuple[SetupConfig, PatternConfig, LoadedReward]:
        if self.current_setup is None:
            raise RuntimeError("No setup is loaded")
        if self.current_pattern is None:
            raise RuntimeError("No pattern is loaded")
        if self.current_reward is None:
            raise RuntimeError("No reward module is loaded")
        return self.current_setup, self.current_pattern, self.current_reward

    def save_model(self, model_id: str, model_path: str | Path | None = None) -> RLProfileStatus:
        try:
            if self.mediaip_storage_client is None:
                raise RuntimeError("No MediaIPStorageClient is configured")

            setup, pattern, reward = self.require_ready_profile()
            current_model_path = Path(model_path or self.system_config.runtime_paths.models_dir / "current_model.zip")
            manifest = self._build_manifest(
                model_id=model_id,
                model_path=current_model_path,
                setup=setup,
                pattern=pattern,
                reward=reward,
            )

            self.mediaip_storage_client.upload_model_bundle(
                model_id=model_id,
                model_path=current_model_path,
                manifest=manifest,
                setup_id=setup.setup_id,
                pattern_id=pattern.pattern_id,
                reward_id=reward.reward_id,
            )
            self.status.model_id = model_id
            self.status.last_error = None
            return self.status
        except Exception as exc:
            self._set_error(exc)
            raise

    def start_training(self, osc_interface: Any) -> RLProfileStatus:
        try:
            setup, pattern, reward = self.require_ready_profile()
            training_manager = self._get_training_manager()
            training_status = training_manager.start_training(
                setup=setup,
                pattern=pattern,
                reward=reward,
                osc_interface=osc_interface,
            )
            self.status.training_status = training_status.state
            self.status.last_error = training_status.last_error
            return self.status
        except Exception as exc:
            self._set_error(exc)
            raise

    def stop_training(self) -> RLProfileStatus:
        try:
            training_manager = self._get_training_manager()
            training_status = training_manager.stop_training(wait=True, timeout=30.0)
            self.status.training_status = training_status.state
            self.status.last_error = training_status.last_error
            return self.status
        except Exception as exc:
            self._set_error(exc)
            raise

    def start_inference(self, osc_interface: Any) -> RLProfileStatus:
        try:
            if self.inference_model is None:
                raise RuntimeError("No model is loaded")
            if self.current_pattern is None:
                raise RuntimeError("No pattern is loaded")

            inference_manager = self._get_inference_manager()
            inference_status = inference_manager.start_inference(
                model=self.inference_model,
                pattern=self.current_pattern,
                osc_interface=osc_interface,
                model_id=self.status.model_id,
            )
            self.status.inference_status = inference_status.state
            self.status.training_status = f"inference_{inference_status.state}"
            self.status.last_error = inference_status.last_error
            return self.status
        except Exception as exc:
            self._set_error(exc)
            raise

    def stop_inference(self) -> RLProfileStatus:
        try:
            inference_manager = self._get_inference_manager()
            inference_status = inference_manager.stop_inference(wait=True, timeout=10.0)
            self.status.inference_status = inference_status.state
            self.status.training_status = f"inference_{inference_status.state}"
            self.status.last_error = inference_status.last_error
            return self.status
        except Exception as exc:
            self._set_error(exc)
            raise

    def load_model(self, model_id: str) -> RLProfileStatus:
        try:
            if self.mediaip_storage_client is None:
                raise RuntimeError("No MediaIPStorageClient is configured")

            bundle = self.mediaip_storage_client.download_model_bundle(model_id)
            manifest = bundle.manifest
            pattern_id = str(manifest["pattern_id"])
            setup_id = str(manifest.get("setup_id", ""))

            pattern_file = str(manifest.get("pattern_file", f"pattern_{pattern_id}.json"))
            pattern = self.pattern_loader.load_from_path(bundle.model_dir / pattern_file)

            setup = None
            setup_file = manifest.get("setup_file")
            if setup_file:
                setup = self.setup_loader.load_from_path(bundle.model_dir / str(setup_file))

            from stable_baselines3 import PPO

            self.inference_model = PPO.load(str(bundle.model_path))
            self.current_pattern = pattern
            if setup is not None:
                self.current_setup = setup
            self.status = RLProfileStatus(
                setup_id=setup_id or self.status.setup_id,
                pattern_id=pattern.pattern_id,
                reward_id=str(manifest.get("reward_id")) if manifest.get("reward_id") is not None else None,
                model_id=str(model_id),
                training_status="model_loaded",
                inference_status="model_loaded",
                last_error=None,
            )
            return self.status
        except Exception as exc:
            self._set_error(exc)
            raise

    def get_status(self) -> dict[str, Any]:
        if self.training_manager is not None:
            training_status = self.training_manager.get_status()
            if training_status.running or self.status.training_status in ("starting", "running", "stopping"):
                self.status.training_status = training_status.state
                self.status.last_error = training_status.last_error
        if self.inference_manager is not None:
            inference_status = self.inference_manager.get_status()
            self.status.inference_status = inference_status.state
            if inference_status.running or self.status.training_status.startswith("inference_"):
                self.status.training_status = f"inference_{inference_status.state}"
            if inference_status.last_error is not None:
                self.status.last_error = inference_status.last_error
        return self.status.to_dict()

    def _set_error(self, error: Exception) -> None:
        self.status.last_error = str(error)
        if self.status.training_status.startswith("inference_"):
            self.status.inference_status = "error"
        if self.status.training_status == "idle":
            return
        self.status.training_status = "error"

    def _resolve_initial_mediaip_rl_root(self, mediaip_rl_root: str | Path | None) -> Path:
        if mediaip_rl_root is not None:
            return Path(mediaip_rl_root)
        if self.mediaip_storage_client is not None:
            return self.mediaip_storage_client.paths.rl_system_root
        raise ValueError("mediaip_rl_root or mediaip_storage_client is required")

    def _configure_loaders(self) -> None:
        self.setup_loader = SetupConfigLoader(self.mediaip_rl_root / "setups")
        self.pattern_loader = PatternConfigLoader(
            self.mediaip_rl_root / "patterns",
            self.system_config.space_limits,
        )
        self.reward_loader = RewardLoader(self.mediaip_rl_root / "rewards")

    def _get_training_manager(self) -> Any:
        if self.training_manager is None:
            from train import TrainingManager

            self.training_manager = TrainingManager(system_config=self.system_config)
        return self.training_manager

    def _get_inference_manager(self) -> Any:
        if self.inference_manager is None:
            from inference import InferenceManager

            self.inference_manager = InferenceManager(system_config=self.system_config)
        return self.inference_manager

    def _build_manifest(
        self,
        model_id: str,
        model_path: Path,
        setup: SetupConfig,
        pattern: PatternConfig,
        reward: LoadedReward,
    ) -> dict[str, Any]:
        return {
            "model_id": str(model_id),
            "setup_id": setup.setup_id,
            "pattern_id": pattern.pattern_id,
            "reward_id": reward.reward_id,
            "model_file": f"model_{model_id}.zip",
            "setup_file": f"setup_{setup.setup_id}.json",
            "pattern_file": f"pattern_{pattern.pattern_id}.json",
            "reward_file": f"reward_{reward.reward_id}.py",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "algorithm": setup.training.algorithm,
            "total_timesteps": setup.training.total_timesteps,
            "max_steps": setup.training.max_steps,
            "reward_config": setup.reward_config,
        }

    def _validate_setup_pattern_compatibility(
        self,
        setup: SetupConfig,
        pattern: PatternConfig,
    ) -> None:
        if setup.pattern_id != pattern.pattern_id:
            raise ValueError(
                f"Setup references pattern {setup.pattern_id}, "
                f"but loaded pattern has id {pattern.pattern_id}"
            )

        reward_config = setup.reward_config
        input_name = reward_config.get("input", reward_config.get("input"))
        output_names = _reward_output_names(reward_config)

        observation_by_name = _signal_map(pattern.observations)
        action_by_name = _signal_map(pattern.actions)

        input_signal = None

        if input_name is not None:
            input_signal = _require_named_signal(
                value=input_name,
                signal_map=observation_by_name,
                config_key="reward_config.input",
                expected_section="observations",
            )

        for output_name in output_names:
            output_signal = _require_named_signal(
                value=output_name,
                signal_map=action_by_name,
                config_key="reward_config.output",
                expected_section="actions",
            )
            if input_signal is not None and input_signal.size != output_signal.size:
                raise ValueError(
                    "reward_config.input and reward_config output signals must have the same size "
                    f"for distance rewards, got {input_signal.size} and {output_signal.size}"
                )

        max_distance = reward_config.get("max_distance")
        if max_distance is not None:
            try:
                max_distance_value = float(max_distance)
            except (TypeError, ValueError) as exc:
                raise ValueError("reward_config.max_distance must be numeric") from exc
            if max_distance_value <= 0:
                raise ValueError("reward_config.max_distance must be positive")


def _signal_map(signals: list[SignalConfig]) -> dict[str, SignalConfig]:
    return {signal.name: signal for signal in signals}


def _reward_output_names(reward_config: dict[str, Any]) -> list[Any]:
    if "output" in reward_config:
        output = reward_config["output"]
        if isinstance(output, list):
            if not output:
                raise ValueError("reward_config.output must not be an empty list")
            return output
        return [output]

    return []


def _require_named_signal(
    value: Any,
    signal_map: dict[str, SignalConfig],
    config_key: str,
    expected_section: str,
) -> SignalConfig:
    if not isinstance(value, str):
        raise ValueError(f"{config_key} must be a signal name string")
    signal = signal_map.get(value)
    if signal is None:
        raise ValueError(f"{config_key}='{value}' was not found in pattern {expected_section}")
    return signal
