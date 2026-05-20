import base64
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from config import SystemConfig, load_system_config


@dataclass(frozen=True)
class MediaIPStoragePaths:
    root: Path
    rl_system_root: Path
    patterns_dir: Path
    rewards_dir: Path
    setups_dir: Path
    models_dir: Path


@dataclass(frozen=True)
class CommitFile:
    file_path: str
    content: str | bytes
    encoding: str = "text"
    action: str | None = None


@dataclass(frozen=True)
class MediaIPModelBundle:
    model_id: str
    model_dir: Path
    model_path: Path
    manifest_path: Path
    manifest: dict[str, Any]


class MediaIPStorageClient:
    def __init__(
        self,
        gitlab_url: str,
        project_id: str,
        cache_dir: str | Path,
        branch: str = "main",
        root_dir: str = "local_rl-system",
        token_env_var: str = "MEDIAIP_STORAGE_TOKEN",
    ):
        if not gitlab_url:
            raise ValueError("MediaIPStorage gitlab_url must not be empty")
        if not project_id:
            raise ValueError("MediaIPStorage project_id must not be empty")

        self.gitlab_url = gitlab_url.rstrip("/")
        self.project_id = str(project_id)
        self.cache_dir = Path(cache_dir)
        self.branch = branch
        self.root_dir = root_dir.strip("/")
        self.token_env_var = token_env_var

    @classmethod
    def from_env(cls, system_config: SystemConfig | None = None) -> "MediaIPStorageClient":
        system_config = system_config or load_system_config()
        return cls(
            gitlab_url=os.getenv("MEDIAIP_STORAGE_GITLAB_URL", "https://gitlab.com"),
            project_id=os.getenv("MEDIAIP_STORAGE_PROJECT_ID", ""),
            cache_dir=system_config.runtime_paths.mediaip_checkout_dir,
            branch=os.getenv("MEDIAIP_STORAGE_BRANCH", "main"),
            root_dir=os.getenv("MEDIAIP_STORAGE_ROOT", "local_rl-system"),
            token_env_var="MEDIAIP_STORAGE_TOKEN",
        )

    @property
    def paths(self) -> MediaIPStoragePaths:
        rl_root = self.cache_dir / self.root_dir
        return MediaIPStoragePaths(
            root=self.cache_dir,
            rl_system_root=rl_root,
            patterns_dir=rl_root / "patterns",
            rewards_dir=rl_root / "rewards",
            setups_dir=rl_root / "setups",
            models_dir=rl_root / "models",
        )

    def ensure_available(self) -> MediaIPStoragePaths:
        self.ensure_layout()
        return self.paths

    def ensure_layout(self) -> None:
        paths = self.paths
        paths.patterns_dir.mkdir(parents=True, exist_ok=True)
        paths.rewards_dir.mkdir(parents=True, exist_ok=True)
        paths.setups_dir.mkdir(parents=True, exist_ok=True)
        paths.models_dir.mkdir(parents=True, exist_ok=True)

    def download_setup_bundle(self, setup_id: str) -> MediaIPStoragePaths:
        self.ensure_layout()

        setup_file = f"setup_{setup_id}.json"
        setup_repo_path = self._repo_path("setups", setup_file)
        setup_text = self.download_text_file(setup_repo_path)
        setup_data = json.loads(setup_text)

        pattern_id = str(setup_data["pattern_id"])
        reward_id = str(setup_data["reward_id"])
        pattern_file = f"pattern_{pattern_id}.json"
        reward_file = f"reward_{reward_id}.py"

        pattern_text = self.download_text_file(self._repo_path("patterns", pattern_file))
        reward_text = self.download_text_file(self._repo_path("rewards", reward_file))

        paths = self.paths
        (paths.setups_dir / setup_file).write_text(setup_text, encoding="utf-8")
        (paths.patterns_dir / pattern_file).write_text(pattern_text, encoding="utf-8")
        (paths.rewards_dir / reward_file).write_text(reward_text, encoding="utf-8")
        return paths

    def download_model_bundle(self, model_id: str) -> MediaIPModelBundle:
        self.ensure_layout()

        model_dir_name = f"model_{model_id}"
        model_repo_dir = f"models/{model_dir_name}"
        manifest_file = "manifest.json"
        manifest_text = self.download_text_file(self._repo_path(model_repo_dir, manifest_file))
        manifest = json.loads(manifest_text)

        model_file = str(manifest.get("model_file", f"model_{model_id}.zip"))
        setup_file = str(manifest.get("setup_file", f"setup_{manifest['setup_id']}.json"))
        pattern_file = str(manifest.get("pattern_file", f"pattern_{manifest['pattern_id']}.json"))

        model_bytes = self.download_binary_file(self._repo_path(model_repo_dir, model_file))
        setup_text = self.download_text_file(self._repo_path(model_repo_dir, setup_file))
        pattern_text = self.download_text_file(self._repo_path(model_repo_dir, pattern_file))

        paths = self.paths
        model_dir = paths.models_dir / model_dir_name
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / model_file
        manifest_path = model_dir / manifest_file

        model_path.write_bytes(model_bytes)
        manifest_path.write_text(manifest_text, encoding="utf-8")
        (model_dir / setup_file).write_text(setup_text, encoding="utf-8")
        (model_dir / pattern_file).write_text(pattern_text, encoding="utf-8")

        return MediaIPModelBundle(
            model_id=str(model_id),
            model_dir=model_dir,
            model_path=model_path,
            manifest_path=manifest_path,
            manifest=manifest,
        )

    def download_text_file(self, repo_file_path: str) -> str:
        data = self.download_binary_file(repo_file_path)
        return data.decode("utf-8")

    def download_binary_file(self, repo_file_path: str) -> bytes:
        query = urlencode({"ref": self.branch})
        url = f"{self._files_url(repo_file_path)}/raw?{query}"
        return self._request("GET", url)

    def upload_model_bundle(
        self,
        model_id: str,
        model_path: str | Path,
        manifest: dict[str, Any],
        setup_id: str,
        pattern_id: str,
        reward_id: str,
        commit_message: str | None = None,
    ) -> None:
        paths = self.paths
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file does not exist: {model_path}")

        model_dir = f"models/model_{model_id}"
        files = [
            CommitFile(
                file_path=self._repo_path(model_dir, f"model_{model_id}.zip"),
                content=model_path.read_bytes(),
                encoding="base64",
            ),
            CommitFile(
                file_path=self._repo_path(model_dir, "manifest.json"),
                content=json.dumps(manifest, indent=2),
            ),
            CommitFile(
                file_path=self._repo_path(model_dir, f"setup_{setup_id}.json"),
                content=(paths.setups_dir / f"setup_{setup_id}.json").read_text(encoding="utf-8"),
            ),
            CommitFile(
                file_path=self._repo_path(model_dir, f"pattern_{pattern_id}.json"),
                content=(paths.patterns_dir / f"pattern_{pattern_id}.json").read_text(encoding="utf-8"),
            ),
            CommitFile(
                file_path=self._repo_path(model_dir, f"reward_{reward_id}.py"),
                content=(paths.rewards_dir / f"reward_{reward_id}.py").read_text(encoding="utf-8"),
            ),
        ]

        self.commit_files(
            message=commit_message or f"Save RL model {model_id}",
            files=files,
        )

    def commit_files(self, message: str, files: list[CommitFile]) -> None:
        if not message:
            raise ValueError("Commit message must not be empty")
        if not files:
            raise ValueError("At least one file is required for a GitLab commit")

        actions = []
        for file in files:
            content = file.content
            encoding = file.encoding
            if isinstance(content, bytes):
                content = base64.b64encode(content).decode("ascii")
                encoding = "base64"

            actions.append(
                {
                    "action": file.action or self._create_or_update_action(file.file_path),
                    "file_path": file.file_path,
                    "content": content,
                    "encoding": encoding,
                }
            )

        payload = {
            "branch": self.branch,
            "commit_message": message,
            "actions": actions,
        }
        self._request_json("POST", self._commits_url(), payload)

    def file_exists(self, repo_file_path: str) -> bool:
        query = urlencode({"ref": self.branch})
        url = f"{self._files_url(repo_file_path)}?{query}"
        try:
            self._request("GET", url)
            return True
        except FileNotFoundError:
            return False

    def _create_or_update_action(self, repo_file_path: str) -> str:
        return "update" if self.file_exists(repo_file_path) else "create"

    def _repo_path(self, *parts: str) -> str:
        clean_parts = [str(part).strip("/") for part in parts if str(part).strip("/")]
        return "/".join([self.root_dir, *clean_parts])

    def _files_url(self, repo_file_path: str) -> str:
        project = quote(self.project_id, safe="")
        file_path = quote(repo_file_path, safe="")
        return f"{self._api_base()}/projects/{project}/repository/files/{file_path}"

    def _commits_url(self) -> str:
        project = quote(self.project_id, safe="")
        return f"{self._api_base()}/projects/{project}/repository/commits"

    def _api_base(self) -> str:
        return f"{self.gitlab_url}/api/v4"

    def _request_json(self, method: str, url: str, payload: dict[str, Any]) -> bytes:
        body = json.dumps(payload).encode("utf-8")
        return self._request(method, url, body=body, content_type="application/json")

    def _request(
        self,
        method: str,
        url: str,
        body: bytes | None = None,
        content_type: str | None = None,
    ) -> bytes:
        headers = {"PRIVATE-TOKEN": self._token()}
        if content_type:
            headers["Content-Type"] = content_type

        request = Request(url=url, data=body, headers=headers, method=method)
        try:
            with urlopen(request, timeout=30) as response:
                return response.read()
        except HTTPError as exc:
            message = self._read_error(exc)
            if exc.code == 404:
                raise FileNotFoundError(message) from exc
            raise RuntimeError(message) from exc
        except URLError as exc:
            raise RuntimeError(f"GitLab API request failed: {exc.reason}") from exc

    def _token(self) -> str:
        token = os.getenv(self.token_env_var)
        if not token:
            raise RuntimeError(f"Missing GitLab token environment variable: {self.token_env_var}")
        return token

    def _read_error(self, error: HTTPError) -> str:
        try:
            body = error.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        body = self._mask_secret(body)
        return f"GitLab API request failed with HTTP {error.code}: {body or error.reason}"

    def _mask_secret(self, value: str) -> str:
        token = os.getenv(self.token_env_var)
        if token:
            value = value.replace(token, "***")
        return value
