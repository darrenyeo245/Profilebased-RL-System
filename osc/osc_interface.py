import json
import logging
import os
import threading
import socket
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from pythonosc import osc_server, udp_client
from pythonosc.dispatcher import Dispatcher

from profile_system import PatternConfig, RLProfileSystem


load_dotenv()

RASPI_HOST = os.getenv("RASPI_HOST", "0.0.0.0")
RASPI_PORT = int(os.getenv("RASPI_PORT", 9001))
BROADCAST_IP = os.getenv("BROADCAST_IP", "255.255.255.255")
BROADCAST_PORT = int(os.getenv("BROADCAST_PORT", 9001))


class OSCInterface:
    def __init__(
        self,
        rl_system: RLProfileSystem,
        enable_logging: bool = True,
        log_path: str = "logs/agent_osc.log",
        auto_start: bool = True,
    ):
        self.rl_system = rl_system
        self.logger = self._setup_logger(enable_logging=enable_logging, log_path=log_path)
        self._ignore_self_messages = self._parse_bool_env("IGNORE_SELF_OSC", default=True)
        self._local_ip_addresses = self._collect_local_ip_addresses()

        self._lock = threading.Condition()
        self._signal_values: dict[str, np.ndarray] = {}
        self._signal_pending: dict[str, bool] = {}
        self._signal_sizes: dict[str, int] = {}
        self._mapped_observation_addresses: set[str] = set()
        self.client = udp_client.SimpleUDPClient(
            BROADCAST_IP,
            BROADCAST_PORT,
            allow_broadcast=True
        )

        self.dispatcher = Dispatcher()
        self._register_handlers()

        self.server: osc_server.ThreadingOSCUDPServer | None = None
        self.server_thread: threading.Thread | None = None

        if auto_start:
            self.start()

    @staticmethod
    def _setup_logger(enable_logging: bool, log_path: str) -> logging.Logger:
        logger = logging.getLogger("dynamic_rl_osc")
        logger.propagate = False

        if not enable_logging:
            logger.handlers.clear()
            logger.setLevel(logging.CRITICAL)
            return logger

        if logger.handlers:
            return logger

        log_file = Path(log_path)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s | %(message)s")

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        return logger

    def _register_handlers(self) -> None:
        self.dispatcher.map("/rl/status", self.status_handler)
        self.dispatcher.map("/rl/start/training", self.start_training_handler)
        self.dispatcher.map("/rl/stop/training", self.stop_training_handler)
        self.dispatcher.map("/rl/start/inference", self.start_inference_handler)
        self.dispatcher.map("/rl/stop/inference", self.stop_inference_handler)
        self.dispatcher.set_default_handler(self.default_handler, needs_reply_address=True)

    def start(self) -> None:
        if self.server is not None:
            return

        self.server = osc_server.ThreadingOSCUDPServer(
            (RASPI_HOST, RASPI_PORT),
            self.dispatcher,
        )
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()
        self._log_event("listener_started", "/", [RASPI_HOST, RASPI_PORT])

    def shutdown(self) -> None:
        if self.server is None:
            return

        self.server.shutdown()
        self.server.server_close()
        self.server = None
        self.server_thread = None
        self._log_event("listener_stopped", "/", [RASPI_HOST, RASPI_PORT])

    def default_handler(self, client_address, address, *args):
        args = self._strip_dispatcher_args(args)

        if address in self._mapped_action_addresses:
            self._log_event("recv_action_echo_ignored", address, args)
            return

        if (
                self._ignore_self_messages
                and client_address
                and client_address[0] in self._local_ip_addresses
                and not address.startswith("/rl/")
        ):
            self._log_event("recv_self_ignored", address, args)
            return

        self._log_event("recv", address, args)

        if address in {"/rl/status/response", "/rl/error"}:
            return

        try:
            setup_id = self._match_setup_address(address)
            model_id = self._match_save_model_address(address)
            load_model_id = self._match_load_model_address(address)
        except ValueError as exc:
            self._send_error(str(exc))
            return

        if setup_id is not None:
            self.setup_handler(address, setup_id)
            return
        if model_id is not None:
            self.save_model_handler(address, model_id)
            return
        if load_model_id is not None:
            self.load_model_handler(address, load_model_id)
            return

        if not address.startswith("/rl/"):
            self.unmapped_signal_handler(address, *args)
            return

        self._send_error(f"Unknown OSC command: {address}")

    def setup_handler(self, address: str, setup_id: str) -> None:
        try:
            status = self.rl_system.load_setup(setup_id)
            if self.rl_system.current_pattern is not None:
                self.configure_pattern(self.rl_system.current_pattern)
        except Exception as exc:
            self._log_event("error", address, [str(exc)])
            self._send_error(str(exc))
            return

        self._send_status(status.to_dict())

    def start_training_handler(self, address: str, *args: Any) -> None:
        args = self._strip_dispatcher_args(args)
        self._log_event("recv", address, args)
        try:
            status = self.rl_system.start_training(self)
        except Exception as exc:
            self._log_event("error", address, [str(exc)])
            self._send_error(str(exc))
            return

        self._send_status(status.to_dict())

    def stop_training_handler(self, address: str, *args: Any) -> None:
        args = self._strip_dispatcher_args(args)
        self._log_event("recv", address, args)
        try:
            status = self.rl_system.stop_training()
        except Exception as exc:
            self._log_event("error", address, [str(exc)])
            self._send_error(str(exc))
            return

        self._send_status(status.to_dict())

    def start_inference_handler(self, address: str, *args: Any) -> None:
        args = self._strip_dispatcher_args(args)
        self._log_event("recv", address, args)
        try:
            status = self.rl_system.start_inference(self)
        except Exception as exc:
            self._log_event("error", address, [str(exc)])
            self._send_error(str(exc))
            return

        self._send_status(status.to_dict())

    def stop_inference_handler(self, address: str, *args: Any) -> None:
        args = self._strip_dispatcher_args(args)
        self._log_event("recv", address, args)
        try:
            status = self.rl_system.stop_inference()
        except Exception as exc:
            self._log_event("error", address, [str(exc)])
            self._send_error(str(exc))
            return

        self._send_status(status.to_dict())

    def save_model_handler(self, address: str, model_id: str) -> None:
        try:
            status = self.rl_system.save_model(model_id)
        except Exception as exc:
            self._log_event("error", address, [str(exc)])
            self._send_error(str(exc))
            return

        self._send_status(status.to_dict())

    def load_model_handler(self, address: str, model_id: str) -> None:
        try:
            status = self.rl_system.load_model(model_id)
            if self.rl_system.current_pattern is not None:
                self.configure_pattern(self.rl_system.current_pattern)
        except Exception as exc:
            self._log_event("error", address, [str(exc)])
            self._send_error(str(exc))
            return

        self._send_status(status.to_dict())

    def status_handler(self, address: str, *args: Any) -> None:
        args = self._strip_dispatcher_args(args)
        self._log_event("recv", address, args)
        self._send_status(self.rl_system.get_status())

    def configure_pattern(self, pattern_config: PatternConfig) -> None:
        for signal in pattern_config.observations:
            with self._lock:
                self._signal_values.setdefault(signal.address, np.zeros(signal.size, dtype=np.float32))
                self._signal_pending.setdefault(signal.address, False)
                self._signal_sizes[signal.address] = int(signal.size)

            if signal.address not in self._mapped_observation_addresses:
                self.dispatcher.map(signal.address, self.signal_handler)
                self._mapped_observation_addresses.add(signal.address)
                self._log_event("map", signal.address, [signal.name, signal.size])

    def signal_handler(self, address: str, *args: Any) -> None:
        args = self._strip_dispatcher_args(args)
        address = str(address)

        with self._lock:
            expected_size = self._signal_sizes.get(address, len(args))
        expected_size = int(expected_size)

        if len(args) < expected_size:
            self._send_error(
                f"OSC signal {address} expected {expected_size} values, got {len(args)}"
            )
            return

        values = np.asarray(args[:expected_size], dtype=np.float32)
        with self._lock:
            self._signal_values[address] = values
            self._signal_pending[address] = True
            self._lock.notify_all()
        self._log_event("recv", address, values.tolist())

    def unmapped_signal_handler(self, address: str, *args: Any) -> None:
        args = self._strip_dispatcher_args(args)
        self._log_event("recv_unmapped_ignored", address, args)
        return


    def get_signal(
        self,
        address: str,
        size: int,
        wait_for_new: bool = False,
        timeout: float | None = None,
    ) -> np.ndarray:
        address = str(address)
        size = int(size)
        with self._lock:
            if address not in self._signal_values:
                self._signal_values[address] = np.zeros(size, dtype=np.float32)
                self._signal_pending[address] = False
                self._signal_sizes[address] = size

            if wait_for_new and not self._signal_pending.get(address, False):
                self._lock.wait(timeout=timeout)

            values = self._signal_values[address].copy()
            self._signal_pending[address] = False

        if values.shape != (size,):
            resized = np.zeros(size, dtype=np.float32)
            resized[: min(size, values.size)] = values[: min(size, values.size)]
            return resized
        return values.astype(np.float32)

    def send_signal(self, address: str, values: np.ndarray) -> None:
        payload = np.asarray(values, dtype=np.float32).tolist()
        self.client.send_message(address, payload)
        self._log_event("send", address, payload)

    def send_reset(self, pattern_config: PatternConfig) -> None:
        first_observation = pattern_config.observations[0]
        payload = np.zeros(first_observation.size, dtype=np.float32).tolist()
        self.client.send_message("/episode/reset", payload)
        self._log_event("send", "/episode/reset", payload)

    @staticmethod
    def _match_setup_address(address: str) -> str | None:
        prefix = "/rl/set/config/setup/"
        return OSCInterface._match_id_address(address=address, prefix=prefix, label="setup")

    @staticmethod
    def _match_save_model_address(address: str) -> str | None:
        prefix = "/rl/savemodel/"
        return OSCInterface._match_id_address(address=address, prefix=prefix, label="model")

    @staticmethod
    def _match_load_model_address(address: str) -> str | None:
        prefix = "/rl/loadmodel/"
        return OSCInterface._match_id_address(address=address, prefix=prefix, label="model")

    @staticmethod
    def _match_id_address(address: str, prefix: str, label: str) -> str | None:
        if not address.startswith(prefix):
            return None

        item_id = address[len(prefix) :].strip("/")
        if not item_id or "/" in item_id:
            raise ValueError(f"Invalid {label} OSC address: {address}")
        return item_id

    def _send_status(self, status: dict[str, Any]) -> None:
        payload = json.dumps(status)
        self.client.send_message("/rl/status/response", payload)
        self._log_event("send", "/rl/status/response", [payload])

    def _send_error(self, message: str) -> None:
        payload = json.dumps({"error": message})
        self.client.send_message("/rl/error", payload)
        self._log_event("send", "/rl/error", [payload])

    @staticmethod
    def _strip_dispatcher_args(args: tuple[Any, ...]) -> tuple[Any, ...]:
        if args and isinstance(args[0], list):
            return tuple(args[1:])
        return args

    def _log_event(self, direction: str, address: str, payload: Any) -> None:
        if self.logger.isEnabledFor(logging.INFO):
            self.logger.info("%s | %s | %s", direction, address, list(payload))

    @staticmethod
    def _parse_bool_env(name: str, default: bool) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return default
        return raw.strip().lower() not in {"0", "false", "no", "off"}

    @staticmethod
    def _collect_local_ip_addresses() -> set[str]:
        addresses = {"127.0.0.1"}
        try:
            hostname = socket.gethostname()
            addresses.update(socket.gethostbyname_ex(hostname)[2])
        except socket.gaierror:
            pass

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.connect(("8.8.8.8", 80))
                addresses.add(sock.getsockname()[0])
        except OSError:
            pass

        return addresses
