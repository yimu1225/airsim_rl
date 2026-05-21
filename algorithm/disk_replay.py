from __future__ import annotations

import atexit
import os
import shutil
import signal
import tempfile
from pathlib import Path

import numpy as np


_RUN_DIRS: list[Path] = []
_CLEANUP_REGISTERED = False
_ORIGINAL_SIGNAL_HANDLERS = {}
_STALE_CLEANED_ROOTS: set[tuple[Path, str]] = set()


def _default_replay_root() -> Path:
    if Path("/mnt/d").exists():
        return Path("/mnt/d/airsim_rl_replay")
    return Path(tempfile.gettempdir()) / "airsim_rl_replay"


def _cleanup_run_dirs() -> None:
    while _RUN_DIRS:
        run_dir = _RUN_DIRS.pop()
        try:
            shutil.rmtree(run_dir, ignore_errors=True)
        except Exception:
            pass


def _cleanup_stale_run_dirs(root: Path, prefix: str) -> None:
    """Remove replay temp dirs left by a previous interrupted run."""
    key = (root.resolve(), prefix)
    if key in _STALE_CLEANED_ROOTS:
        return
    _STALE_CLEANED_ROOTS.add(key)

    if not root.exists():
        return
    for child in root.iterdir():
        if not child.is_dir() or not child.name.startswith(f"{prefix}_"):
            continue
        if child in _RUN_DIRS:
            continue
        try:
            shutil.rmtree(child, ignore_errors=True)
        except Exception:
            pass


def _handle_signal(signum, frame):
    _cleanup_run_dirs()
    original = _ORIGINAL_SIGNAL_HANDLERS.get(signum)
    if callable(original):
        original(signum, frame)
    elif original == signal.SIG_IGN:
        return
    else:
        raise SystemExit(128 + int(signum))


def _register_cleanup_once() -> None:
    global _CLEANUP_REGISTERED
    if _CLEANUP_REGISTERED:
        return
    atexit.register(_cleanup_run_dirs)
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            _ORIGINAL_SIGNAL_HANDLERS[signum] = signal.getsignal(signum)
            signal.signal(signum, _handle_signal)
        except Exception:
            pass
    _CLEANUP_REGISTERED = True


def make_run_dir(root=None, prefix: str = "run") -> Path:
    _register_cleanup_once()
    base = Path(root).expanduser() if root else _default_replay_root()
    base.mkdir(parents=True, exist_ok=True)
    _cleanup_stale_run_dirs(base, prefix)
    run_dir = Path(tempfile.mkdtemp(prefix=f"{prefix}_", dir=str(base)))
    _RUN_DIRS.append(run_dir)
    return run_dir


class DiskArrayFactory:
    """Small owner for replay arrays that should live on disk for this run."""

    def __init__(self, root=None, prefix: str = "pl_replay"):
        self.run_dir = None
        self._counter = 0
        self.run_dir = make_run_dir(root=root, prefix=prefix)

    def zeros(self, shape, dtype=np.float32, name: str = "array"):
        if self.run_dir is None:
            raise RuntimeError("Disk replay directory was not initialized.")
        self._counter += 1
        safe_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(name))
        path = self.run_dir / f"{self._counter:03d}_{safe_name}.dat"
        array = np.memmap(path, mode="w+", dtype=dtype, shape=tuple(shape))
        return array

    def cleanup(self) -> None:
        if self.run_dir is None:
            return
        try:
            shutil.rmtree(self.run_dir, ignore_errors=True)
        except Exception:
            pass
        try:
            _RUN_DIRS.remove(self.run_dir)
        except ValueError:
            pass
        self.run_dir = None

    def __del__(self):
        self.cleanup()


def zeros(shape, dtype=np.float32, factory: DiskArrayFactory | None = None, name: str = "array"):
    if factory is None:
        return np.zeros(shape, dtype=dtype)
    return factory.zeros(shape, dtype=dtype, name=name)
