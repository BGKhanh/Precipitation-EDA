"""Reproducibility, worker seeding, and artifact run lifecycle utilities.

Strictly aligned with Protocol v1.0.20:
- §14.1 & §14.2: Seed control & DataLoader worker seeding
- §5, §13.2.B, §14.2: Directory collision prevention & requirements-lock export
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from typing import Dict, Optional, Union
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class RunDirectoryExistsError(FileExistsError):
    """Raised when Fresh Run mode encounters an existing benchmark run directory."""


class RunDirectoryNotFoundError(FileNotFoundError):
    """Raised when Resume mode encounters a non-existent benchmark run directory."""


def set_global_seed(seed: int = 42) -> None:
    """Set global seeds across random, numpy, and torch (CPU & CUDA).

    Per §14.2 of Protocol v1.0.20. Note: PYTHONHASHSEED must be set
    in the shell prior to Python launch per §14.1.
    """
    random.seed(seed)
    np.random.seed(seed)
    if HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


def seed_worker(worker_id: int) -> None:
    """Worker initialization function for PyTorch DataLoader (Item 8).

    Ensures that each DataLoader worker has a distinct, reproducible seed
    derived from the parent process seed.
    """
    if HAS_TORCH:
        worker_seed = torch.initial_seed() % 2**32
    else:
        worker_seed = (42 + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def export_requirements_lock(output_path: Union[str, Path]) -> Path:
    """Export frozen runtime package versions to requirements-lock.txt (Item 7).

    Captures actual environment distributions without network access.
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    try:
        import importlib.metadata
        dists = sorted(importlib.metadata.distributions(), key=lambda d: d.metadata['Name'].lower())
        for d in dists:
            name = d.metadata['Name']
            version = d.version
            lines.append(f"{name}=={version}")
    except Exception as e:
        lines.append(f"# Error capturing metadata: {e}")

    out_p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_p


def check_directory_collision(
    run_dir: Union[str, Path],
    mode: str = "fresh",
) -> Path:
    """Enforce Directory Collision Prevention Gate per §5 & §13.2.B.

    Args:
        run_dir: Path to `artifacts/runs/<run_id>/`.
        mode: Operating mode ('fresh' or 'resume').

    Returns:
        Validated Path.

    Raises:
        RunDirectoryExistsError: If mode=='fresh' and directory already exists.
        RunDirectoryNotFoundError: If mode=='resume' and directory does not exist.
    """
    r_path = Path(run_dir)
    mode_clean = mode.strip().lower()

    if mode_clean == "fresh":
        if r_path.exists():
            raise RunDirectoryExistsError(
                f"Directory Collision Prevention Gate: Run directory '{r_path}' already exists! "
                "Fresh Run mode requires a completely new run_id to guarantee artifact isolation."
            )
    elif mode_clean == "resume":
        if not r_path.exists():
            raise RunDirectoryNotFoundError(
                f"Resume Preflight Gate: Run directory '{r_path}' does not exist! "
                "Resume mode requires an existing benchmark run space to recover."
            )
    else:
        raise ValueError(f"Unknown runner mode '{mode}'. Must be 'fresh' or 'resume'.")

    return r_path


def init_run_directory_structure(
    run_dir: Union[str, Path],
    mode: str = "fresh",
) -> Dict[str, Path]:
    """Initialize canonical isolated artifact directories for a benchmark run.

    Structure:
    artifacts/runs/<run_id>/
      ├── manifests/
      │     └── requirements-lock.txt
      ├── checkpoints/
      └── predictions/
    """
    r_path = check_directory_collision(run_dir, mode=mode)

    manifests_dir = r_path / "manifests"
    checkpoints_dir = r_path / "checkpoints"
    predictions_dir = r_path / "predictions"

    if mode.strip().lower() == "fresh":
        manifests_dir.mkdir(parents=True, exist_ok=True)
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        predictions_dir.mkdir(parents=True, exist_ok=True)

        # Automatically export requirements-lock.txt right after directory creation
        lock_file = manifests_dir / "requirements-lock.txt"
        export_requirements_lock(lock_file)

    return {
        "run_dir": r_path,
        "manifests_dir": manifests_dir,
        "checkpoints_dir": checkpoints_dir,
        "predictions_dir": predictions_dir,
    }

