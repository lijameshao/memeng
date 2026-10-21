import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

BUNDLED = Path(__file__).parent / "prompts"


@dataclass(frozen=True)
class Config:
    user_id: str = "local"
    gate_model: str = "sonnet"
    extract_model: str = "sonnet"
    gate_prompt_path: Path = BUNDLED / "gate.md"
    extract_prompt_path: Path = BUNDLED / "extract.md"
    embed_model: str = "embeddinggemma-2:270m"
    ollama_url: str = "http://localhost:11434"
    db_path: Path = Path("~/.memeng/store.db")
    top_k: int = 10
    overfetch: int = 4
    semantic_threshold: float = 0.74
    entity_boost_cap: float = 0.5
    conflict_threshold: float = 0.6
    window_token_cap: int = 8000
    first_capture_turns: int = 20
    gate_retire_after: int = 3
    gated_prefix_share: float = 0.3
    record_decisions: bool = True

    @classmethod
    def load(cls, path: Path | None = None, *, env: Mapping[str, str] = os.environ) -> "Config":
        named = env.get("MEMENG_CONFIG") if path is None else None
        path = Path(path or named or "~/.memeng/config.toml").expanduser()
        if not path.exists():
            if named:
                raise FileNotFoundError(f"MEMENG_CONFIG names no file: {path}")
            return cls()

        raw = tomllib.loads(path.read_text())
        unknown = set(raw) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"unknown config key: {', '.join(sorted(unknown))}")

        overrides = {
            key: Path(value).expanduser() if key.endswith("_path") else value
            for key, value in raw.items()
        }
        return replace(cls(), **overrides)


class _BoundConfig:
    """Reads raise until something binds a Config, so a missing DB.open(config)
    names itself instead of surfacing as a default value nobody chose.
    """

    _config: Config | None = None

    def bind(self, config: Config) -> None:
        self._config = config

    def __getattr__(self, name: str) -> Any:
        if self._config is None:
            raise RuntimeError("no bound config; call DB.open(config)")
        return getattr(self._config, name)


CONFIG = _BoundConfig()
