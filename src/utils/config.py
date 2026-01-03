# src/utils/config.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively update dict dst with src (src wins).
    """
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def load_config(path: str | Path) -> Dict[str, Any]:
    """
    Load YAML config with optional inheritance:

    child.yaml:
      inherits: configs/base.yaml
      ...

    Rules:
    - child values override base values
    - nested dicts are merged recursively
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    base_path = cfg.pop("inherits", None)
    if base_path:
        base_path = (path.parent / base_path).resolve() if not Path(base_path).is_absolute() else Path(base_path)
        base_cfg = load_config(base_path)
        return _deep_update(base_cfg, cfg)

    return cfg


def get(cfg: Dict[str, Any], key: str, default: Optional[Any] = None) -> Any:
    """
    Dot-key getter:
      get(cfg, "train.lr") -> cfg["train"]["lr"]
    """
    cur: Any = cfg
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur
