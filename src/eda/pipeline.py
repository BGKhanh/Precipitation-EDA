# -*- coding: utf-8 -*-
"""
EDA Synthesis & Comprehensive Report Pipeline Module (Component 01.7).

Locks all architectural invariants, serialization schemas, atomic staging, fail-closed rollback,
cross-process locking, provenance binding, and downstream consumer handshakes.
"""

import copy
import errno
import hashlib
import json
import logging
import math
import os
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# ==============================================================================
# 1. Canonical System Paths & Path Accessors
# ==============================================================================

CANONICAL_REPO_ROOT: Path = Path(__file__).resolve().parent.parent.parent
CANONICAL_SOURCE_ROOT: Path = (CANONICAL_REPO_ROOT / "reports/eda").resolve()
CANONICAL_EXPORT_LOCK_PATH: Path = (CANONICAL_SOURCE_ROOT / ".export_lock").resolve()

CANONICAL_PRODUCTION_DESTINATIONS: frozenset[Path] = frozenset({
    (CANONICAL_REPO_ROOT / "reports/eda/eda_report.json").resolve(),
    (CANONICAL_REPO_ROOT / "data/processed/eda_report.json").resolve(),
    (CANONICAL_REPO_ROOT / "notebooks/eda/outputs/eda_report.json").resolve(),
})

REQUIRED_LINEAGE_DOMAINS: frozenset[str] = frozenset({
    "threshold", "distribution", "correlation",
    "temporal", "stationarity", "extreme"
})

CANONICAL_DOMAIN_FILENAMES: Dict[str, str] = {
    "threshold": "01.1_threshold_report.json",
    "distribution": "01.2_distribution_report.json",
    "correlation": "01.3_correlation_report.json",
    "temporal": "01.4_temporal_structure_report.json",
    "stationarity": "01.5_stationarity_report.json",
    "extreme": "01.6_extreme_events_report.json",
}

BRIDGE_DOMAIN_FILENAMES: Dict[str, str] = {
    "threshold": "threshold_report.json",
    "distribution": "distribution_report.json",
    "correlation": "correlation_report.json",
    "temporal": "temporal_structure_report.json",
    "stationarity": "stationarity_report.json",
    "extreme": "extreme_events_report.json",
}

CANONICAL_SOURCE_PATHS: Dict[str, Path] = {
    domain: (CANONICAL_SOURCE_ROOT / filename).resolve()
    for domain, filename in CANONICAL_DOMAIN_FILENAMES.items()
}

BRIDGE_SOURCE_PATHS: Dict[str, Path] = {
    domain: (CANONICAL_REPO_ROOT / "notebooks/eda/outputs" / filename).resolve()
    for domain, filename in BRIDGE_DOMAIN_FILENAMES.items()
}

PROTECTED_INTERNAL_PATHS: frozenset[Path] = frozenset({CANONICAL_EXPORT_LOCK_PATH})
PROTECTED_SOURCE_PATHS: frozenset[Path] = frozenset(
    list(CANONICAL_SOURCE_PATHS.values()) + list(BRIDGE_SOURCE_PATHS.values())
)
FORBIDDEN_WRITE_DESTINATIONS: frozenset[Path] = frozenset(
    PROTECTED_INTERNAL_PATHS | PROTECTED_SOURCE_PATHS
)


def get_protected_internal_paths() -> frozenset[Path]:
    """Return immutable set of protected internal pipeline lock paths (R127, R131)."""
    return frozenset({CANONICAL_EXPORT_LOCK_PATH})


def get_protected_source_paths() -> frozenset[Path]:
    """Return immutable set of protected canonical and bridge source artifacts (R128, R131)."""
    return frozenset(list(CANONICAL_SOURCE_PATHS.values()) + list(BRIDGE_SOURCE_PATHS.values()))


def get_forbidden_write_destinations() -> frozenset[Path]:
    """Return unified union of all protected paths forbidden as write targets (R127, R128, R131)."""
    return frozenset(get_protected_internal_paths() | get_protected_source_paths())


def sync_derived_path_constants() -> None:
    """Synchronize module-level derived sets when monkeypatching path constants (R131)."""
    global PROTECTED_INTERNAL_PATHS, PROTECTED_SOURCE_PATHS, FORBIDDEN_WRITE_DESTINATIONS
    PROTECTED_INTERNAL_PATHS = get_protected_internal_paths()
    PROTECTED_SOURCE_PATHS = get_protected_source_paths()
    FORBIDDEN_WRITE_DESTINATIONS = get_forbidden_write_destinations()


def get_repo_root() -> Path:
    return CANONICAL_REPO_ROOT


def get_production_destinations() -> frozenset[Path]:
    return CANONICAL_PRODUCTION_DESTINATIONS


def normalize_export_path(p: Union[str, Path]) -> Path:
    target = Path(p)
    if not target.is_absolute():
        target = (CANONICAL_REPO_ROOT / target).resolve()
    else:
        target = target.resolve()
    return target


def normalize_source_path(p: Union[str, Path]) -> Path:
    target = Path(p)
    if not target.is_absolute():
        target = (CANONICAL_REPO_ROOT / target).resolve()
    else:
        target = target.resolve()
    return target


# ==============================================================================
# 2. Lineage Schema Specifications & Allowlist Governance
# ==============================================================================

DOMAIN_LINEAGE_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "threshold": {
        "allowed_n_fields": frozenset({"n_observations"}),
        "allowed_split_fields": frozenset({"data_split"}),
        "allowed_date_fields": frozenset({"train_start", "train_end"}),
        "date_style": "flat_train",
    },
    "distribution": {
        "allowed_n_fields": frozenset({"n_observations"}),
        "allowed_split_fields": frozenset({"data_split"}),
        "allowed_date_fields": frozenset({"train_start", "train_end"}),
        "date_style": "flat_train",
    },
    "correlation": {
        "allowed_n_fields": frozenset({"n_observations"}),
        "allowed_split_fields": frozenset({"data_split"}),
        "allowed_date_fields": frozenset({"train_start", "train_end"}),
        "date_style": "flat_train",
    },
    "temporal": {
        "allowed_n_fields": frozenset({"sample_size"}),
        "allowed_split_fields": frozenset({"analysis_population"}),
        "allowed_date_fields": frozenset({"date_range"}),
        "date_style": "nested_dict",
    },
    "stationarity": {
        "allowed_n_fields": frozenset({"n_observations"}),
        "allowed_split_fields": frozenset({"population"}),
        "allowed_date_fields": frozenset({"population"}),
        "date_style": "population_string",
    },
    "extreme": {
        "allowed_n_fields": frozenset({"n_observations"}),
        "allowed_split_fields": frozenset({"split"}),
        "allowed_date_fields": frozenset({"start_date", "end_date"}),
        "date_style": "flat_start_end",
    },
}

ALL_KNOWN_LINEAGE_KEYS: frozenset[str] = frozenset({
    "n_observations", "sample_size",
    "data_split", "analysis_population", "population", "split",
    "train_start", "train_end", "date_range", "start_date", "end_date",
})


# ==============================================================================
# 3. Interprocess Advisory Lock
# ==============================================================================

class InterprocessExportLock:
    """Cross-platform, OS-backed interprocess lock with re-acquisition guard, single-owner cleanup and error discrimination (R115, R120, R122, R127, R142, R143)."""

    def __init__(self, lock_path: Union[str, Path], owner_token: Optional[str] = None):
        self.lock_path = Path(lock_path).resolve()
        self.owner_token = owner_token or f"{os.getpid()}_{time.time_ns()}_{uuid.uuid4().hex[:8]}"
        self._fd: Optional[int] = None
        self._is_locked: bool = False

    def acquire(self) -> None:
        if self._is_locked or self._fd is not None:
            raise RuntimeError(
                f"Lock instance is already acquired on {self.lock_path} (fd={self._fd}). "
                f"Re-acquisition on the same instance is strictly forbidden."
            )

        fd: Optional[int] = None
        lock_acquired: bool = False
        try:
            self.lock_path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(str(self.lock_path), os.O_CREAT | os.O_RDWR, 0o644)

            if sys.platform == "win32":
                import msvcrt
                os.lseek(fd, 0, os.SEEK_SET)
                try:
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                    lock_acquired = True
                except OSError as err:
                    if err.errno in (errno.EDEADLK, errno.EACCES):
                        holder_info = self._read_lock_metadata_safe()
                        raise BlockingIOError(
                            f"Export transaction is locked by another active process on {self.lock_path}. "
                            f"{holder_info}"
                        ) from err
                    raise
            else:
                import fcntl
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    lock_acquired = True
                except (BlockingIOError, OSError) as err:
                    is_contention = isinstance(err, BlockingIOError) or (
                        isinstance(err, OSError) and err.errno in (errno.EAGAIN, errno.EWOULDBLOCK, errno.EACCES)
                    )
                    if is_contention:
                        holder_info = self._read_lock_metadata_safe()
                        raise BlockingIOError(
                            f"Export transaction is locked by another active process on {self.lock_path}. "
                            f"{holder_info}"
                        ) from err
                    raise

            self._fd = fd
            self._is_locked = True
            self._write_metadata()

        except BaseException as orig_err:
            cleanup_err: Optional[BaseException] = None
            try:
                if lock_acquired and fd is not None:
                    try:
                        if sys.platform == "win32":
                            import msvcrt
                            os.lseek(fd, 0, os.SEEK_SET)
                            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                        else:
                            import fcntl
                            fcntl.flock(fd, fcntl.LOCK_UN)
                    except BaseException as unlock_err:
                        cleanup_err = unlock_err

                if fd is not None:
                    try:
                        os.close(fd)
                    except BaseException as close_err:
                        if cleanup_err is None:
                            cleanup_err = close_err
            finally:
                self._fd = None
                self._is_locked = False

            if cleanup_err is not None:
                logger.critical(
                    "Cleanup error (%s: %s) while handling failure (%s: %s) during lock acquisition on %s",
                    type(cleanup_err).__name__, cleanup_err,
                    type(orig_err).__name__, orig_err, self.lock_path
                )
                if _is_interruption(cleanup_err) and not _is_interruption(orig_err):
                    if isinstance(orig_err, Exception):
                        try:
                            raise cleanup_err from orig_err
                        except BaseException as chained:
                            raise chained
                    raise cleanup_err

            raise orig_err

    def release(self) -> None:
        if not self._is_locked or self._fd is None:
            return

        try:
            if sys.platform == "win32":
                import msvcrt
                try:
                    os.lseek(self._fd, 0, os.SEEK_SET)
                    msvcrt.locking(self._fd, msvcrt.LK_UNLCK, 1)
                except Exception:
                    pass
            else:
                import fcntl
                try:
                    fcntl.flock(self._fd, fcntl.LOCK_UN)
                except Exception:
                    pass
        finally:
            try:
                os.close(self._fd)
            except Exception:
                pass
            self._fd = None
            self._is_locked = False

    def __enter__(self) -> "InterprocessExportLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    def _write_metadata(self) -> None:
        if self._fd is None:
            return
        try:
            meta = {
                "owner_token": self.owner_token,
                "pid": os.getpid(),
                "time_utc": datetime.now(timezone.utc).isoformat(),
            }
            raw = json.dumps(meta).encode("utf-8")
            os.lseek(self._fd, 0, os.SEEK_SET)
            os.write(self._fd, raw)
            os.fsync(self._fd)
        except Exception as e:
            logger.debug("Failed writing lock metadata: %s", e)

    def _read_lock_metadata_safe(self) -> str:
        try:
            if self.lock_path.exists():
                content = self.lock_path.read_text(encoding="utf-8").strip()
                if content:
                    return f"Lock holder metadata: {content}"
        except Exception:
            pass
        return "Lock holder metadata unavailable."


# ==============================================================================
# 4. Atomic Staging & Exception Helpers
# ==============================================================================

def _is_interruption(exc: BaseException) -> bool:
    """Distinguish non-Exception BaseExceptions (KeyboardInterrupt, SystemExit) from regular Exceptions."""
    return isinstance(exc, (KeyboardInterrupt, SystemExit)) or not isinstance(exc, Exception)


def _create_empty_staged_temp(dest_path: Path) -> Tuple[int, Path]:
    """Create empty temporary file exclusively using O_CREAT | O_EXCL and retry on collision (R121)."""
    parent = dest_path.parent
    parent.mkdir(parents=True, exist_ok=True)
    prefix = f".tmp_{dest_path.name}_{os.getpid()}_"
    suffix = ".tmp"

    for attempt in range(100):
        unique_token = f"{time.time_ns()}_{uuid.uuid4().hex[:6]}"
        candidate_path = parent / f"{prefix}{unique_token}{suffix}"
        try:
            fd = os.open(
                str(candidate_path),
                os.O_CREAT | os.O_EXCL | os.O_RDWR,
                0o600
            )
            return fd, candidate_path
        except FileExistsError:
            continue
        except OSError as e:
            raise IOError(f"Failed to exclusively allocate staged temporary file for {dest_path}: {e}") from e

    raise IOError(f"Exhausted 100 attempts allocating exclusive staged temporary file for {dest_path}")


def _synthesize_and_raise_staging_errors(
    body_err: Optional[BaseException],
    wrapper_close_err: Optional[BaseException],
    raw_close_err: Optional[BaseException],
    boundary_errs: List[BaseException],
    dest_path: Path,
) -> None:
    """Deterministic multi-stage error synthesizer preserving original operational errors and interruption semantics (R133, R135, R136)."""
    all_cleanup_errs: List[BaseException] = []
    if wrapper_close_err is not None:
        all_cleanup_errs.append(wrapper_close_err)
    if raw_close_err is not None:
        all_cleanup_errs.append(raw_close_err)

    all_interrupts = (
        ([body_err] if body_err is not None and _is_interruption(body_err) else []) +
        [e for e in boundary_errs if _is_interruption(e)] +
        [e for e in all_cleanup_errs if _is_interruption(e)]
    )

    if all_interrupts:
        primary_interrupt = all_interrupts[0]
        if body_err is not None and isinstance(body_err, Exception):
            if all_cleanup_errs:
                cleanup_err_msgs = [f"({type(e).__name__}) {e}" for e in all_cleanup_errs]
                logger.critical(
                    "Process interruption (%s) during cleanup for %s after body error (%s). Cleanup failures: %s",
                    type(primary_interrupt).__name__, dest_path, body_err, cleanup_err_msgs
                )
            try:
                raise primary_interrupt from body_err
            except BaseException as chained_exc:
                raise chained_exc
        raise primary_interrupt

    if body_err is not None:
        if all_cleanup_errs:
            cleanup_err_msgs = [f"({type(e).__name__}) {e}" for e in all_cleanup_errs]
            raise IOError(
                f"Failed writing staged temp content for {dest_path}: {body_err} "
                f"AND cleanup failures occurred: {cleanup_err_msgs}"
            ) from body_err
        raise body_err

    if boundary_errs:
        raise boundary_errs[0]

    if wrapper_close_err is not None and raw_close_err is not None:
        raise IOError(
            f"Dual close failure for {dest_path}: wrapper close failed ({wrapper_close_err}) "
            f"AND raw descriptor close failed: {raw_close_err}"
        ) from wrapper_close_err

    if wrapper_close_err is not None:
        raise IOError(f"Failed to close wrapper for {dest_path}: {wrapper_close_err}") from wrapper_close_err

    if raw_close_err is not None:
        raise IOError(f"Failed to close raw descriptor for {dest_path}: {raw_close_err}") from raw_close_err


def _populate_staged_temp(fd: int, tmp_path: Path, content_bytes: bytes) -> None:
    """Populate staged temp file with single-point sequential cleanup in outer finally (R130, R134, R136, R138, R140, R144)."""
    f = None
    body_err: Optional[BaseException] = None
    wrapper_close_err: Optional[BaseException] = None
    raw_close_err: Optional[BaseException] = None
    boundary_errs: List[BaseException] = []

    try:
        try:
            f = open(fd, "wb", closefd=False)
            f.write(content_bytes)
            f.flush()
            os.fsync(f.fileno())
        except BaseException as e:
            body_err = e
    except BaseException as unhandled_boundary_err:
        boundary_errs.append(unhandled_boundary_err)
    finally:
        try:
            if f is not None:
                try:
                    f.close()
                except BaseException as we:
                    wrapper_close_err = we
        finally:
            try:
                os.close(fd)
            except BaseException as re:
                raw_close_err = re

    if body_err is not None or wrapper_close_err is not None or raw_close_err is not None or boundary_errs:
        _synthesize_and_raise_staging_errors(body_err, wrapper_close_err, raw_close_err, boundary_errs, tmp_path)


def _allocate_atomic_staged_temp(dest_path: Path, content_bytes: bytes) -> Path:
    """Allocate and populate staged temp file with direct unlinking on failure (R121, R139)."""
    fd, tmp_p = _create_empty_staged_temp(dest_path)
    try:
        _populate_staged_temp(fd, tmp_p, content_bytes)
        return tmp_p
    except BaseException as e:
        try:
            tmp_p.unlink(missing_ok=True)
        except BaseException as ce:
            logger.critical("Failed to unlink temporary file '%s': %s", tmp_p, ce)
        raise e


def _allocate_atomic_backup_path(dest_path: Path) -> Tuple[Path, int]:
    """Exclusively reserve an atomic backup placeholder for safe transactional rollback (R126)."""
    parent = dest_path.parent
    parent.mkdir(parents=True, exist_ok=True)
    prefix = f".bak_{dest_path.name}_{os.getpid()}_"
    suffix = ".bak"

    for attempt in range(100):
        unique_token = f"{time.time_ns()}_{uuid.uuid4().hex[:6]}"
        candidate_path = parent / f"{prefix}{unique_token}{suffix}"
        try:
            fd = os.open(
                str(candidate_path),
                os.O_CREAT | os.O_EXCL | os.O_RDWR,
                0o600
            )
            return candidate_path, fd
        except FileExistsError:
            continue
        except OSError as e:
            raise IOError(f"Failed to allocate exclusive backup placeholder for {dest_path}: {e}") from e

    raise IOError(f"Exhausted 100 attempts allocating backup placeholder for {dest_path}")


# ==============================================================================
# 5. Strict Type Validators & Value Assertions
# ==============================================================================

def _is_threshold_match(candidate_th: float, target_th: float, tol: float = 1e-4) -> bool:
    c_val = _validate_scalar_threshold(candidate_th, "candidate_threshold")
    t_val = _validate_scalar_threshold(target_th, "target_threshold")
    return abs(c_val - t_val) <= (tol + 1e-9)


def _validate_scalar_threshold(val: Any, field_name: str) -> float:
    if isinstance(val, bool):
        raise ValueError(f"[{field_name}] Boolean is strictly forbidden as a numeric threshold (boolean is strictly forbidden), got {val}")
    if isinstance(val, (int, float, Decimal)):
        f_val = float(val)
        if math.isnan(f_val) or math.isinf(f_val):
            raise ValueError(f"[{field_name}] Threshold cannot be NaN or Infinite")
        if f_val < 0.0:
            raise ValueError(f"[{field_name}] Threshold must be non-negative, got {f_val}")
        return f_val
    raise ValueError(f"[{field_name}] {field_name} must be a real float or int, got {type(val).__name__}: {val}")


def _validate_nonnegative_scalar(val: Any, field_name: str) -> float:
    if isinstance(val, bool):
        raise ValueError(f"[{field_name}] Boolean is forbidden as numeric scalar")
    if isinstance(val, (int, float, Decimal)):
        f_val = float(val)
        if math.isnan(f_val) or math.isinf(f_val):
            raise ValueError(f"[{field_name}] Scalar cannot be NaN or Infinite")
        if f_val < 0.0:
            raise ValueError(f"[{field_name}] Scalar must be non-negative, got {f_val}")
        return f_val
    raise ValueError(f"[{field_name}] Expected non-negative numeric scalar, got {type(val).__name__}: {val}")


def _validate_optional_nonnegative_scalar(val: Any, field_name: str) -> Optional[float]:
    if val is None:
        return None
    return _validate_nonnegative_scalar(val, field_name)


def _validate_sarima_order(val: Any, source_name: str) -> Tuple[int, int, int]:
    if isinstance(val, (str, dict, set)):
        raise ValueError(f"[{source_name}] suggested_sarima_order must be a sequence, got {type(val).__name__}")
    if not isinstance(val, (list, tuple)) or len(val) != 3:
        raise ValueError(f"[{source_name}] suggested_sarima_order must be a 3-element tuple (p, d, q), got {val}")
    res = []
    for idx, elem in enumerate(val):
        if isinstance(elem, bool) or not isinstance(elem, int) or elem < 0:
            raise ValueError(f"[{source_name}] SARIMA order[{idx}] must be a non-negative int, got {elem}")
        res.append(int(elem))
    return (res[0], res[1], res[2])


def _validate_seasonal_order(val: Any, source_name: str) -> Tuple[int, int, int, int]:
    if isinstance(val, (str, dict, set)):
        raise ValueError(f"[{source_name}] suggested_seasonal_order must be a sequence, got {type(val).__name__}")
    if not isinstance(val, (list, tuple)) or len(val) != 4:
        raise ValueError(f"[{source_name}] suggested_seasonal_order must be a 4-element tuple (P, D, Q, s), got {val}")
    res = []
    for idx, elem in enumerate(val):
        if isinstance(elem, bool) or not isinstance(elem, int) or elem < 0:
            raise ValueError(f"[{source_name}] Seasonal order[{idx}] must be a non-negative int, got {elem}")
        res.append(int(elem))
    return (res[0], res[1], res[2], res[3])


def _validate_month_list(val: Any, source_name: str) -> List[int]:
    if not isinstance(val, list) or len(val) == 0:
        raise ValueError(f"[{source_name}] wet_season_months must be a non-empty list")
    res = []
    for elem in val:
        if isinstance(elem, bool) or not isinstance(elem, int) or not (1 <= elem <= 12):
            raise ValueError(f"[{source_name}] Month must be an int in [1, 12], got {elem}")
        res.append(int(elem))
    return res


def _validate_period_list(val: Any, source_name: str) -> List[int]:
    if not isinstance(val, list) or len(val) == 0:
        raise ValueError(f"[{source_name}] representative_periods must be a non-empty list")
    res = []
    for elem in val:
        if isinstance(elem, bool) or not isinstance(elem, int) or elem <= 0:
            raise ValueError(f"[{source_name}] Period must be a positive int, got {elem}")
        res.append(int(elem))
    return res


def _validate_feature_list(val: Any, source_name: str, field_name: str) -> List[str]:
    if not isinstance(val, list):
        raise ValueError(f"[{source_name}] {field_name} must be a list, got {type(val).__name__}")
    res = []
    for elem in val:
        if not isinstance(elem, str) or len(elem.strip()) == 0:
            raise ValueError(f"[{source_name}] {field_name} elements must be non-empty strings, got {elem}")
        res.append(elem.strip())
    return res


def _validate_str_dict(val: Any, field_name: str) -> Dict[str, str]:
    if not isinstance(val, dict):
        raise ValueError(f"[{field_name}] must be a dict, got {type(val).__name__}")
    res = {}
    for k, v in val.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise ValueError(f"[{field_name}] key and value must be strings, got ({k!r}: {v!r})")
        res[str(k)] = str(v)
    return res


def _validate_vif_results_records(records: Any, source_name: str) -> List[str]:
    if not isinstance(records, list):
        raise ValueError(f"[{source_name}] vif_results must be a list, got {type(records).__name__}")
    out = []
    for idx, rec in enumerate(records):
        if not isinstance(rec, dict):
            raise ValueError(f"[{source_name}] vif_results[{idx}] must be a dict: {rec}")
        if "feature" not in rec or not isinstance(rec["feature"], str):
            raise ValueError(f"[{source_name}] vif_results[{idx}] missing valid 'feature' string: {rec}")
        if "vif_score" not in rec:
            raise ValueError(f"[{source_name}] vif_results[{idx}] missing required 'vif_score': {rec}")
        score = _validate_nonnegative_scalar(rec["vif_score"], f"{source_name}.vif_results[{idx}].vif_score")
        if score < 10.0:
            out.append(rec["feature"])
    return out


# ==============================================================================
# 6. Lineage Validation & Provenance Verification
# ==============================================================================

def _project_and_validate_container_lineage(
    container: Dict[str, Any],
    domain_name: str,
    container_name: str,
) -> Tuple[int, str, Optional[str], Optional[str]]:
    """Schema-governed lineage validation for a single metadata container (R96, R99)."""
    if not isinstance(container, dict):
        raise ValueError(f"[{domain_name}.{container_name}] Container must be a dict, got {type(container).__name__}")

    if domain_name not in DOMAIN_LINEAGE_SCHEMAS:
        raise ValueError(f"Unknown domain name '{domain_name}' not in DOMAIN_LINEAGE_SCHEMAS")

    schema = DOMAIN_LINEAGE_SCHEMAS[domain_name]
    allowed_n = schema["allowed_n_fields"]
    allowed_split = schema["allowed_split_fields"]
    allowed_date = schema["allowed_date_fields"]
    date_style = schema["date_style"]

    for k in container.keys():
        if k in ALL_KNOWN_LINEAGE_KEYS:
            if k not in allowed_n and k not in allowed_split and k not in allowed_date:
                raise ValueError(
                    f"[{domain_name}.{container_name}] Disallowed lineage key '{k}' for domain '{domain_name}'. "
                    f"Allowed for this domain: N={allowed_n}, split={allowed_split}, date={allowed_date}"
                )

    n_val: Optional[int] = None
    for k in allowed_n:
        if k in container:
            raw = container[k]
            if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
                raise ValueError(f"[{domain_name}.{container_name}] '{k}' must be positive int, got {raw!r}")
            n_val = int(raw)
            break

    if n_val is None:
        raise ValueError(f"[{domain_name}.{container_name}] Missing sample size field. Expected one of {allowed_n}")

    split_val: Optional[str] = None
    for k in allowed_split:
        if k in container:
            raw = container[k]
            if not isinstance(raw, str) or len(raw.strip()) == 0:
                raise ValueError(f"[{domain_name}.{container_name}] '{k}' must be non-empty string, got {raw!r}")
            split_val = raw.strip()
            break

    if split_val is None:
        raise ValueError(f"[{domain_name}.{container_name}] Missing split field. Expected one of {allowed_split}")

    start_date: Optional[str] = None
    end_date: Optional[str] = None

    if date_style == "flat_train":
        s = container.get("train_start")
        e = container.get("train_end")
        if (s is None) ^ (e is None):
            raise ValueError(f"[{domain_name}.{container_name}] Incomplete date boundary: train_start={s!r}, train_end={e!r}")
        if s is not None and e is not None:
            if not isinstance(s, str) or not isinstance(e, str):
                raise ValueError(f"[{domain_name}.{container_name}] train_start and train_end must be strings")
            start_date, end_date = s.strip(), e.strip()

    elif date_style == "nested_dict":
        dr = container.get("date_range")
        if dr is not None:
            if not isinstance(dr, dict) or "start" not in dr or "end" not in dr:
                raise ValueError(f"[{domain_name}.{container_name}] date_range must be a dict with 'start' and 'end'")
            s, e = dr["start"], dr["end"]
            if not isinstance(s, str) or not isinstance(e, str):
                raise ValueError(f"[{domain_name}.{container_name}] date_range start/end must be strings")
            start_date, end_date = s.strip(), e.strip()

    elif date_style == "flat_start_end":
        s = container.get("start_date")
        e = container.get("end_date")
        if (s is None) ^ (e is None):
            raise ValueError(f"[{domain_name}.{container_name}] Incomplete date boundary: start_date={s!r}, end_date={e!r}")
        if s is not None and e is not None:
            if not isinstance(s, str) or not isinstance(e, str):
                raise ValueError(f"[{domain_name}.{container_name}] start_date and end_date must be strings")
            start_date, end_date = s.strip(), e.strip()

    elif date_style == "population_string":
        if "2001-01-01" in split_val and "2018-04-30" in split_val:
            start_date = "2001-01-01"
            end_date = "2018-04-30"

    return n_val, split_val, start_date, end_date


def _validate_domain_lineage_metadata(
    container1: Optional[Dict[str, Any]],
    container2: Optional[Dict[str, Any]],
    domain_name: str,
    allow_unverified: bool = False,
) -> Tuple[str, Optional[int], Optional[str], Optional[str], Optional[str]]:
    """Validate dual metadata containers independently and enforce cross-container invariants."""
    c1_valid = container1 is not None and isinstance(container1, dict) and len(container1) > 0
    c2_valid = container2 is not None and isinstance(container2, dict) and len(container2) > 0

    if not c1_valid and not c2_valid:
        if allow_unverified:
            return "LEGACY_UNVERIFIED_PERMITTED", None, None, None, None
        raise ValueError(f"[{domain_name}] Missing required lineage metadata containers ('metadata' / 'contract_metadata')")

    res1 = _project_and_validate_container_lineage(container1, domain_name, "metadata") if c1_valid else None
    res2 = _project_and_validate_container_lineage(container2, domain_name, "contract_metadata") if c2_valid else None

    if res1 and res2:
        if res1[0] != res2[0]:
            raise ValueError(f"[{domain_name}] Conflicting sample sizes between containers: {res1[0]} vs {res2[0]}")
        if res1[1] != res2[1]:
            raise ValueError(f"[{domain_name}] Conflicting split definitions between containers: {res1[1]!r} vs {res2[1]!r}")
        if (res1[2], res1[3]) != (res2[2], res2[3]):
            raise ValueError(f"[{domain_name}] Conflicting date ranges between containers: ({res1[2]}, {res1[3]}) vs ({res2[2]}, {res2[3]})")

    active_res = res1 or res2
    assert active_res is not None
    n_obs, split_str, s_dt, e_dt = active_res

    # Enforce canonical Dev Train invariant (N=6329, 2001-01-01 -> 2018-04-30)
    is_canonical = (
        n_obs == 6329 and
        split_str in ("dev_train", "Dev Train (2001-01-01 -> 2018-04-30)") and
        s_dt == "2001-01-01" and e_dt == "2018-04-30"
    )

    if not is_canonical:
        if allow_unverified:
            return "LEGACY_UNVERIFIED_PERMITTED", n_obs, split_str, s_dt, e_dt
        raise ValueError(
            f"[{domain_name}] Lineage audit mismatch: expected canonical Dev Train (N=6329, 2001-01-01 -> 2018-04-30), "
            f"got N={n_obs}, split={split_str!r}, dates=({s_dt}, {e_dt})"
        )

    return "VERIFIED_CANONICAL_DEV_TRAIN", n_obs, split_str, s_dt, e_dt


def _validate_provenance_binding(report: "EDAReport") -> bool:
    """Fail-closed provenance verification for production export authorization (R108, R111, R112)."""
    ctx = report._provenance_context
    if ctx is None or not isinstance(ctx, dict):
        return False

    if ctx.get("verified_domains_count") != 6 or ctx.get("all_verified") is not True:
        return False

    if ctx.get("canonical_source_root") != str(CANONICAL_SOURCE_ROOT):
        return False

    manifest = ctx.get("source_files_manifest")
    if not isinstance(manifest, dict) or set(manifest.keys()) != set(REQUIRED_LINEAGE_DOMAINS):
        return False

    for dom, expected_p in CANONICAL_SOURCE_PATHS.items():
        if manifest.get(dom) != str(expected_p):
            return False

    sha_map = ctx.get("source_files_sha256")
    if not isinstance(sha_map, dict) or set(sha_map.keys()) != set(REQUIRED_LINEAGE_DOMAINS):
        return False

    for dom, stored_hash in sha_map.items():
        src_p = CANONICAL_SOURCE_PATHS[dom]
        if not src_p.exists():
            raise PermissionError(f"Canonical source file '{src_p}' does not exist on disk!")
        curr_hash = hashlib.sha256(src_p.read_bytes()).hexdigest()
        if curr_hash != stored_hash:
            raise PermissionError(f"Source file '{src_p}' content has changed on disk since factory ingestion!")

    raw_results_sha = ctx.get("raw_results_sha256")
    if not raw_results_sha or not isinstance(raw_results_sha, str):
        return False

    actual_raw_sha = hashlib.sha256(
        json.dumps(report.raw_results, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if actual_raw_sha != raw_results_sha:
        raise PermissionError("Report raw_results payload has been tampered with in memory!")

    baseline_conf_sha = ctx.get("baseline_config_sha256")
    if not baseline_conf_sha or not isinstance(baseline_conf_sha, str):
        return False

    actual_conf_sha = hashlib.sha256(
        json.dumps(report.to_dict(include_raw=False, include_diagnostics=True), sort_keys=True).encode("utf-8")
    ).hexdigest()
    if actual_conf_sha != baseline_conf_sha:
        raise PermissionError("Report configuration parameters have been altered since factory instantiation!")

    audit = ctx.get("lineage_audit")
    if not isinstance(audit, dict) or set(audit.keys()) != set(REQUIRED_LINEAGE_DOMAINS):
        return False

    for dom, rec in audit.items():
        if not isinstance(rec, dict) or rec.get("status") != "VERIFIED_CANONICAL_DEV_TRAIN":
            return False

    return True


def _restore_and_verify(
    dest: Path,
    backup_p: Optional[Path],
    orig_sha: Optional[str],
    rollback_errors: List[str]
) -> None:
    """Helper performing bi-directional verification and non-existent destination cleanup on rollback (R105, R117)."""
    if orig_sha is None:
        try:
            dest.unlink(missing_ok=True)
            if backup_p is not None:
                backup_p.unlink(missing_ok=True)
            if dest.exists():
                rollback_errors.append(f"Failed to remove newly created destination '{dest}' during rollback")
        except Exception as e:
            rollback_errors.append(f"Error removing non-existent destination '{dest}': {e}")
        return

    if backup_p is None or not backup_p.exists():
        rollback_errors.append(f"Missing backup file '{backup_p}' for destination '{dest}'")
        return

    try:
        backup_bytes = backup_p.read_bytes()
        backup_sha = hashlib.sha256(backup_bytes).hexdigest()
        if backup_sha != orig_sha:
            rollback_errors.append(
                f"Backup SHA-256 mismatch for '{backup_p}': expected {orig_sha}, got {backup_sha}"
            )
            return

        os.replace(str(backup_p), str(dest))
        dest_bytes = dest.read_bytes()
        dest_sha = hashlib.sha256(dest_bytes).hexdigest()
        if dest_sha != orig_sha:
            rollback_errors.append(
                f"Post-restore SHA-256 mismatch for '{dest}': expected {orig_sha}, got {dest_sha}"
            )
    except Exception as e:
        rollback_errors.append(f"Rollback restore failed for '{dest}': {e}")


def _extract_intermittency_from_candidates(
    candidates: List[Dict[str, Any]],
    val_th: float,
    source_name: str,
) -> Tuple[Optional[float], Optional[float], str]:
    if not isinstance(candidates, list) or len(candidates) == 0:
        raise ValueError(f"[{source_name}] Candidate evaluations must be a non-empty list")

    matched = []
    for rec in candidates:
        if not isinstance(rec, dict) or "threshold_mm" not in rec:
            raise ValueError(f"[{source_name}] Malformed candidate record: {rec}")
        th_mm = _validate_scalar_threshold(rec["threshold_mm"], f"{source_name}.threshold_mm")
        if _is_threshold_match(th_mm, val_th):
            matched.append(rec)

    if len(matched) > 1:
        raise ValueError(f"[{source_name}] Ambiguous intermittency match: multiple candidates match threshold {val_th}: {matched}")

    if len(matched) == 1:
        m_rec = matched[0]
        adi_raw = m_rec.get("adi") if "adi" in m_rec else m_rec.get("adi_demand_ratio")
        if adi_raw is None:
            adi_raw = m_rec.get("calendar_mean_gap_days")

        cv2_raw = m_rec.get("cv2") if "cv2" in m_rec else m_rec.get("cv_squared")

        if adi_raw is None or cv2_raw is None:
            raise ValueError(f"[{source_name}] Candidate record missing 'adi' or 'cv2': {m_rec}")
        adi_val = _validate_nonnegative_scalar(adi_raw, "candidate.adi")
        cv2_val = _validate_nonnegative_scalar(cv2_raw, "candidate.cv2")
        analogy = m_rec.get("syntetos_boylan_analogy")
        if not isinstance(analogy, str) or len(analogy.strip()) == 0:
            raise ValueError(f"[{source_name}] Candidate missing valid 'syntetos_boylan_analogy': {m_rec}")
        return adi_val, cv2_val, analogy.strip()

    return None, None, "UNMATCHED_THRESHOLD"


def _extract_domain_metadata(
    dist_res: Dict[str, Any],
    source_name: str,
    val_th: float,
    diag_labels: Dict[str, str],
) -> Tuple[Optional[float], Optional[float], str]:
    candidates = None
    if "intermittency_diagnostics_candidate_dependent" in dist_res and isinstance(dist_res["intermittency_diagnostics_candidate_dependent"], dict):
        candidates = dist_res["intermittency_diagnostics_candidate_dependent"].get("candidate_evaluations")
    elif "candidate_evaluations" in dist_res:
        candidates = dist_res["candidate_evaluations"]

    if candidates is not None and isinstance(candidates, list):
        adi, cv2, lbl = _extract_intermittency_from_candidates(
            candidates, val_th, source_name
        )
        diag_labels["intermittency_classification"] = lbl
        return adi, cv2, lbl

    if "intermittency" in dist_res:
        inter = dist_res["intermittency"]
        if not isinstance(inter, dict):
            raise ValueError(f"[{source_name}] 'intermittency' must be a dict")
        bridge_th = inter.get("rainfall_threshold_used_mm")
        if bridge_th is None:
            lbl = "UNVERIFIED_BRIDGE_THRESHOLD"
            diag_labels["intermittency_classification"] = lbl
            return None, None, lbl

        b_th = _validate_scalar_threshold(bridge_th, "intermittency.rainfall_threshold_used_mm")
        if _is_threshold_match(b_th, val_th):
            adi = _validate_nonnegative_scalar(inter["adi"], "intermittency.adi")
            cv2 = _validate_nonnegative_scalar(inter["cv2"], "intermittency.cv2")
            cls_name = inter.get("classification")
            if not isinstance(cls_name, str) or len(cls_name.strip()) == 0:
                raise ValueError(f"[{source_name}] Missing classification string")
            lbl = cls_name.strip()
            diag_labels["intermittency_classification"] = lbl
            return adi, cv2, lbl

        lbl = "UNMATCHED_THRESHOLD"
        diag_labels["intermittency_classification"] = lbl
        return None, None, lbl

    lbl = "UNMATCHED_THRESHOLD"
    diag_labels["intermittency_classification"] = lbl
    return None, None, lbl


# ==============================================================================
# 7. Unified EDAReport Dataclass & Serialization APIs
# ==============================================================================

@dataclass
class EDAReport:
    """Structured, immutable-by-contract synthesis of all 5 EDA analysis components."""

    suggested_sarima_order: Tuple[int, int, int] = (3, 0, 3)
    suggested_seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 7)
    wet_season_months: List[int] = field(default_factory=lambda: [5, 6, 7, 8, 9, 10, 11])
    representative_periods: List[int] = field(default_factory=lambda: [7, 30, 122, 365])
    vif_filtered_features: List[str] = field(default_factory=list)
    high_skew_features: List[str] = field(default_factory=lambda: ["Lượng mưa"])
    intermittency_adi: Optional[float] = None
    intermittency_cv2: Optional[float] = None
    diagnostic_labels_optional: Dict[str, str] = field(default_factory=dict)
    extreme_event_thresholds: Dict[str, float] = field(default_factory=lambda: {"p95": 17.37, "p99": 34.18, "max_daily_mm": 107.9})
    validated_rain_threshold: float = 0.1
    raw_results: Dict[str, Any] = field(default_factory=dict, repr=False)
    _provenance_context: Optional[Dict[str, Any]] = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.suggested_sarima_order = _validate_sarima_order(self.suggested_sarima_order, "EDAReport")
        self.suggested_seasonal_order = _validate_seasonal_order(self.suggested_seasonal_order, "EDAReport")
        self.wet_season_months = _validate_month_list(self.wet_season_months, "EDAReport")
        self.representative_periods = _validate_period_list(self.representative_periods, "EDAReport")
        self.vif_filtered_features = _validate_feature_list(self.vif_filtered_features, "EDAReport", "vif_filtered_features")
        self.high_skew_features = _validate_feature_list(self.high_skew_features, "EDAReport", "high_skew_features")
        self.validated_rain_threshold = _validate_scalar_threshold(self.validated_rain_threshold, "EDAReport.validated_rain_threshold")
        self.diagnostic_labels_optional = _validate_str_dict(self.diagnostic_labels_optional, "EDAReport.diagnostic_labels_optional")

        for k in ("p95", "p99", "max_daily_mm"):
            if k not in self.extreme_event_thresholds:
                raise ValueError(f"[EDAReport] extreme_event_thresholds missing required key '{k}'")
            self.extreme_event_thresholds[k] = _validate_nonnegative_scalar(
                self.extreme_event_thresholds[k], f"extreme_event_thresholds.{k}"
            )

        if self.extreme_event_thresholds["p95"] > self.extreme_event_thresholds["p99"]:
            raise ValueError(f"Extreme event threshold p95 ({self.extreme_event_thresholds['p95']}) must be <= p99 ({self.extreme_event_thresholds['p99']})")

        has_adi = self.intermittency_adi is not None
        has_cv2 = self.intermittency_cv2 is not None
        if has_adi ^ has_cv2:
            raise ValueError("intermittency_adi and intermittency_cv2 must both be None or both be float")

        if not has_adi:
            lbl = self.diagnostic_labels_optional.get("intermittency_classification", "")
            if lbl in ("LUMPY-like (demand taxonomy analogy)", "LUMPY", "LUMPY-like", "SMOOTH", "ERRATIC"):
                raise ValueError(f"Invalid state: intermittency metrics are None but label claims valid taxonomy: '{lbl}'")
            if "intermittency_classification" not in self.diagnostic_labels_optional:
                self.diagnostic_labels_optional["intermittency_classification"] = "UNMATCHED_THRESHOLD"

    @property
    def lineage_audit(self) -> Dict[str, Any]:
        if self._provenance_context is not None and "lineage_audit" in self._provenance_context:
            return copy.deepcopy(self._provenance_context["lineage_audit"])
        return {}

    @property
    def intermittency_classification(self) -> str:
        return self.diagnostic_labels_optional.get("intermittency_classification", "UNKNOWN")

    def to_dict(self, include_raw: bool = False, include_diagnostics: Optional[bool] = None) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "suggested_sarima_order": list(self.suggested_sarima_order),
            "suggested_seasonal_order": list(self.suggested_seasonal_order),
            "wet_season_months": list(self.wet_season_months),
            "representative_periods": list(self.representative_periods),
            "validated_rain_threshold": float(self.validated_rain_threshold),
            "vif_filtered_features": list(self.vif_filtered_features),
            "high_skew_features": list(self.high_skew_features),
            "extreme_event_thresholds": {k: float(v) for k, v in self.extreme_event_thresholds.items()},
            "intermittency_adi": float(self.intermittency_adi) if self.intermittency_adi is not None else None,
            "intermittency_cv2": float(self.intermittency_cv2) if self.intermittency_cv2 is not None else None,
        }
        should_include_diag = (
            include_diagnostics is True
            or (
                include_diagnostics is None
                and self.intermittency_adi is None
                and self.diagnostic_labels_optional.get("intermittency_classification") == "UNMATCHED_THRESHOLD"
            )
        )
        if should_include_diag:
            d["diagnostic_labels_optional"] = dict(self.diagnostic_labels_optional)
        if include_raw and self.raw_results:
            d["raw_results"] = dict(self.raw_results)
        return d

    def to_json(self, path: Optional[Union[str, Path]] = None, indent: int = 2) -> Path:
        if path is None:
            raise PermissionError("Direct export to production destination via to_json() is strictly forbidden. Use export_triple_artifacts().")

        target_path = normalize_export_path(path)
        if target_path in get_production_destinations():
            raise PermissionError("Direct export to production destination via to_json() is strictly forbidden. Use export_triple_artifacts().")

        if target_path in get_protected_internal_paths():
            raise PermissionError(f"Target destination '{target_path}' is a protected internal pipeline lock path ({target_path.name}).")

        if target_path in get_protected_source_paths():
            raise PermissionError(f"Target destination '{target_path}' is a protected EDA source artifact ({target_path.name}).")

        lock = InterprocessExportLock(CANONICAL_EXPORT_LOCK_PATH)
        lock.acquire()
        try:
            content_bytes = json.dumps(
                self.to_dict(include_raw=False, include_diagnostics=True),
                indent=indent,
                ensure_ascii=False,
            ).encode("utf-8")

            fd, tmp_p = _create_empty_staged_temp(target_path)
            try:
                _populate_staged_temp(fd, tmp_p, content_bytes)
                try:
                    os.replace(str(tmp_p), str(target_path))
                except Exception as replace_err:
                    try:
                        tmp_p.unlink(missing_ok=True)
                    except Exception as clean_err:
                        raise IOError(
                            f"Failed to atomic-replace '{tmp_p}' -> '{target_path}': {replace_err} "
                            f"AND failed to cleanup temporary file '{tmp_p}': {clean_err}"
                        ) from replace_err
                    raise IOError(f"Failed to atomic-replace '{tmp_p}' -> '{target_path}': {replace_err}") from replace_err
            except BaseException as e:
                tmp_p.unlink(missing_ok=True)
                raise e

            return target_path.resolve()
        finally:
            lock.release()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EDAReport":
        if not isinstance(data, dict):
            raise ValueError(f"EDAReport.from_dict expects a dict, got {type(data).__name__}")

        raw_val_th = data.get("validated_rain_threshold")
        if raw_val_th is None:
            raw_val_th = data.get("chosen_threshold_mm", 0.1)
        val_th = _validate_scalar_threshold(raw_val_th, "validated_rain_threshold")

        sarima_order = _validate_sarima_order(data.get("suggested_sarima_order", (3, 0, 3)), "suggested_sarima_order")
        seasonal_order = _validate_seasonal_order(data.get("suggested_seasonal_order", (1, 1, 1, 7)), "suggested_seasonal_order")
        wet_months = _validate_month_list(data.get("wet_season_months", [5, 6, 7, 8, 9, 10, 11]), "wet_season_months")
        periods = _validate_period_list(data.get("representative_periods", [7, 30, 122, 365]), "representative_periods")
        vif_feats = _validate_feature_list(data.get("vif_filtered_features", []), "EDAReport", "vif_filtered_features")
        skew_feats = _validate_feature_list(data.get("high_skew_features", ["Lượng mưa"]), "EDAReport", "high_skew_features")

        ext = data.get("extreme_event_thresholds")
        if ext is None:
            ext = {"p95": 17.37, "p99": 34.18, "max_daily_mm": 107.9}
        elif not isinstance(ext, dict):
            raise ValueError("extreme_event_thresholds must be a dict")
        ext_dict = {
            k: _validate_nonnegative_scalar(ext.get(k, 0.0), f"extreme_event_thresholds.{k}")
            for k in ("p95", "p99", "max_daily_mm")
        }

        adi = _validate_optional_nonnegative_scalar(data.get("intermittency_adi"), "intermittency_adi")
        cv2 = _validate_optional_nonnegative_scalar(data.get("intermittency_cv2"), "intermittency_cv2")
        diag = _validate_str_dict(data.get("diagnostic_labels_optional", {}), "diagnostic_labels_optional")

        return cls(
            suggested_sarima_order=sarima_order,
            suggested_seasonal_order=seasonal_order,
            wet_season_months=wet_months,
            representative_periods=periods,
            vif_filtered_features=vif_feats,
            high_skew_features=skew_feats,
            intermittency_adi=adi,
            intermittency_cv2=cv2,
            diagnostic_labels_optional=diag,
            extreme_event_thresholds=ext_dict,
            validated_rain_threshold=val_th,
            raw_results={},
            _provenance_context=None,
        )

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "EDAReport":
        p = Path(path)
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_raw(cls, *args, **kwargs) -> "EDAReport":
        """
        Versatile factory supporting both raw domain result dictionaries and explicit field initialization.
        """
        is_raw_domain = (
            "threshold_res" in kwargs or "distribution_res" in kwargs or
            "temporal_res" in kwargs or "stationarity_res" in kwargs or
            "distribution" in kwargs or "correlation" in kwargs or
            "temporal" in kwargs or "stationarity" in kwargs or "extreme" in kwargs or "threshold" in kwargs
        )
        if is_raw_domain:
            th_res = kwargs.get("threshold_res") or kwargs.get("threshold") or {}
            dist_res = kwargs.get("distribution_res") or kwargs.get("distribution") or {}
            corr_res = kwargs.get("correlation_res") or kwargs.get("correlation") or {}
            temp_res = kwargs.get("temporal_res") or kwargs.get("temporal") or {}
            stat_res = kwargs.get("stationarity_res") or kwargs.get("stationarity") or {}
            ext_res = kwargs.get("extreme_res") or kwargs.get("extreme") or {}

            v_th = kwargs.get("validated_threshold")
            if v_th is None:
                v_th = kwargs.get("validated_rain_threshold")
            if v_th is None:
                v_th = th_res.get("chosen_threshold_mm", 0.1)
            val_th = _validate_scalar_threshold(v_th, "from_raw.validated_threshold")

            diag_labels: Dict[str, str] = {}

            # Extract SARIMA order from stationarity or temporal
            if "exploratory_sarima_heuristic" in stat_res:
                esh = stat_res["exploratory_sarima_heuristic"]
                p = int(esh.get("p_recommendation", 3))
                d = int(esh.get("d_recommendation", 0))
                q = int(esh.get("q_recommendation", 3))
                sarima_order = _validate_sarima_order((p, d, q), "stationarity")
                diag_labels["sarima_order_status"] = str(esh.get("status", "EDA-suggested baseline heuristic (NOT optimal model search)"))
                c_orders = esh.get("candidate_seasonal_orders", [])
                if c_orders and isinstance(c_orders, list) and isinstance(c_orders[0], dict):
                    cand0 = c_orders[0]
                    seasonal_order = _validate_seasonal_order((cand0["P"], cand0["D"], cand0["Q"], cand0["s"]), "stationarity")
                else:
                    seasonal_order = (1, 1, 1, 7)
            elif "suggested_sarima_order" in stat_res:
                sarima_order = _validate_sarima_order(stat_res["suggested_sarima_order"], "stationarity")
                seasonal_order = _validate_seasonal_order(stat_res.get("suggested_seasonal_order", (1, 1, 1, 7)), "stationarity")
            elif "suggested_sarima_order" in temp_res:
                sarima_order = _validate_sarima_order(temp_res["suggested_sarima_order"], "temporal")
                seasonal_order = _validate_seasonal_order(temp_res.get("suggested_seasonal_order", (1, 1, 1, 7)), "temporal")
            else:
                sarima_order = (3, 0, 3)
                seasonal_order = (1, 1, 1, 7)

            # Wavelet status
            if "wavelet_analysis_status" in temp_res:
                ws = temp_res["wavelet_analysis_status"]
                if not isinstance(ws, dict) or "rationale" not in ws or not isinstance(ws["rationale"], str) or len(ws["rationale"].strip()) == 0:
                    raise ValueError("[temporal] wavelet_analysis_status must be a dict with non-empty 'rationale'")
                diag_labels["wavelet_status"] = "excluded_from_primary_pipeline"

            # Representative periods
            if "period_candidates_for_downstream_modeling" in temp_res:
                pc = temp_res["period_candidates_for_downstream_modeling"]
                periods = _validate_period_list(pc.get("candidate_periods_days"), "temporal")
            elif "representative_periods" in temp_res:
                periods = _validate_period_list(temp_res["representative_periods"], "temporal")
            else:
                periods = [7, 30, 122, 365]

            # Wet season months
            if "climate_seasons" in temp_res and "wet_season_months" in temp_res["climate_seasons"]:
                wet_months = _validate_month_list(temp_res["climate_seasons"].get("wet_season_months"), "temporal")
            elif "wet_season_months" in temp_res:
                wet_months = _validate_month_list(temp_res["wet_season_months"], "temporal")
            elif "wet_season_months" in th_res:
                wet_months = _validate_month_list(th_res["wet_season_months"], "threshold")
            else:
                wet_months = [5, 6, 7, 8, 9, 10, 11]

            # VIF features
            if "vif_results" in corr_res:
                vif_feats = _validate_vif_results_records(corr_res["vif_results"], "correlation")
            elif "vif_filtered_features" in corr_res:
                vif_feats = _validate_feature_list(corr_res["vif_filtered_features"], "correlation", "vif_filtered_features")
            elif "multicollinearity_results" in corr_res:
                mc = corr_res["multicollinearity_results"]
                if hasattr(mc, "to_dict"):
                    mc_df = mc
                    vif_feats = mc_df[mc_df["VIF_Score"] < 10.0]["Feature"].tolist()
                elif isinstance(mc, list):
                    vif_feats = [r["Feature"] for r in mc if r.get("VIF_Score", 0) < 10.0]
                else:
                    vif_feats = []
            else:
                vif_feats = []

            # High skew features
            if "high_skew_features" in dist_res:
                skew_feats = _validate_feature_list(dist_res["high_skew_features"], "distribution", "high_skew_features")
            elif "target_variable" in dist_res and isinstance(dist_res["target_variable"], str):
                skew_feats = [dist_res["target_variable"].strip()]
            elif "descriptive_stats" in dist_res:
                ds = dist_res["descriptive_stats"]
                if hasattr(ds, "index"):
                    skew_feats = [str(x) for x in ds.index.tolist()]
                else:
                    skew_feats = ["Lượng mưa"]
            else:
                skew_feats = ["Lượng mưa"]

            # Intermittency
            adi_val, cv2_val, _ = _extract_domain_metadata(dist_res, "distribution", val_th, diag_labels)
            if adi_val is None and "intermittency" in dist_res and isinstance(dist_res["intermittency"], dict):
                inter = dist_res["intermittency"]
                if "adi" in inter and "cv2" in inter:
                    adi_val = float(inter["adi"])
                    cv2_val = float(inter["cv2"])
                    diag_labels["intermittency_classification"] = str(inter.get("classification", "LUMPY"))

            # Stationarity ADF
            if "stationarity_tests" in stat_res:
                st = stat_res["stationarity_tests"]
                if not isinstance(st, dict) or "adf" not in st or not isinstance(st["adf"], dict):
                    raise ValueError("[stationarity] stationarity_tests missing 'adf' dict")
                is_stat = st["adf"].get("is_stationary")
                if not isinstance(is_stat, bool):
                    raise ValueError(f"[stationarity] adf.is_stationary must be bool, got {type(is_stat).__name__}")
                diag_labels["stationarity_adf"] = "STATIONARY" if is_stat else "NON_STATIONARY"
            elif "adf_test" in stat_res:
                at = stat_res["adf_test"]
                if not isinstance(at, dict):
                    raise ValueError("[stationarity] adf_test must be a dict")
                is_stat = at.get("is_stationary")
                if not isinstance(is_stat, bool):
                    raise ValueError(f"[stationarity] adf_test.is_stationary must be bool, got {type(is_stat).__name__}")
                diag_labels["stationarity_adf"] = "STATIONARY" if is_stat else "NON_STATIONARY"

            # Extreme thresholds
            ext_dict: Dict[str, float] = {}
            if ext_res is not None:
                et = ext_res.get("thresholds") or ext_res.get("extreme_thresholds") or ext_res.get("percentiles")
                if isinstance(et, dict):
                    for k in ("p95", "p99", "max_daily_mm"):
                        if k in et:
                            ext_dict[k] = _validate_nonnegative_scalar(et[k], f"extreme_thresholds.{k}")
                elif isinstance(ext_res, dict):
                    for k in ("p95", "p99", "max_daily_mm"):
                        if k in ext_res:
                            ext_dict[k] = _validate_nonnegative_scalar(ext_res[k], f"extreme_thresholds.{k}")
            if not ext_dict or len(ext_dict) < 3:
                ext_dict = {"p95": 17.37, "p99": 34.18, "max_daily_mm": 107.9}

            return cls(
                suggested_sarima_order=sarima_order,
                suggested_seasonal_order=seasonal_order,
                wet_season_months=wet_months,
                representative_periods=periods,
                validated_rain_threshold=val_th,
                vif_filtered_features=vif_feats,
                high_skew_features=skew_feats,
                extreme_event_thresholds=ext_dict,
                intermittency_adi=adi_val,
                intermittency_cv2=cv2_val,
                diagnostic_labels_optional=diag_labels,
                raw_results={},
            )

        # Explicit fields instantiation
        if "validated_threshold" not in kwargs and "validated_rain_threshold" not in kwargs:
            raise TypeError("EDAReport.from_raw requires explicit 'validated_threshold'")

        v_raw = kwargs.get("validated_threshold")
        if v_raw is None:
            v_raw = kwargs.get("validated_rain_threshold")
        v_th = _validate_scalar_threshold(v_raw, "validated_threshold")

        sarima = _validate_sarima_order(kwargs["suggested_sarima_order"], "from_raw")
        seasonal = _validate_seasonal_order(kwargs["suggested_seasonal_order"], "from_raw")
        wet_m = _validate_month_list(kwargs["wet_season_months"], "from_raw")
        periods = _validate_period_list(kwargs["representative_periods"], "from_raw")
        vif_f = _validate_feature_list(kwargs["vif_filtered_features"], "from_raw", "vif_filtered_features")
        skew_f = _validate_feature_list(kwargs["high_skew_features"], "from_raw", "high_skew_features")
        diag = _validate_str_dict(kwargs.get("diagnostic_labels_optional", {}), "from_raw")

        ext = kwargs["extreme_event_thresholds"]
        if not isinstance(ext, dict):
            raise ValueError("extreme_event_thresholds must be a dict")
        ext_dict = {
            k: _validate_nonnegative_scalar(ext[k], f"extreme_event_thresholds.{k}")
            for k in ("p95", "p99", "max_daily_mm")
        }

        adi = _validate_optional_nonnegative_scalar(kwargs.get("intermittency_adi"), "intermittency_adi")
        cv2 = _validate_optional_nonnegative_scalar(kwargs.get("intermittency_cv2"), "intermittency_cv2")

        return cls(
            suggested_sarima_order=sarima,
            suggested_seasonal_order=seasonal,
            wet_season_months=wet_m,
            representative_periods=periods,
            vif_filtered_features=vif_f,
            high_skew_features=skew_f,
            intermittency_adi=adi,
            intermittency_cv2=cv2,
            diagnostic_labels_optional=diag,
            extreme_event_thresholds=ext_dict,
            validated_rain_threshold=v_th,
            raw_results={},
            _provenance_context=None,
        )

    @classmethod
    def from_domain_reports(
        cls,
        source_root: Union[str, Path] = "reports/eda",
        validated_threshold: Optional[float] = None,
        override_rain_threshold: Optional[float] = None,
        allow_legacy_unverified_domains: Optional[Set[str]] = None,
        allow_unverified_lineage: bool = False,
    ) -> "EDAReport":
        """Factory ingesting 6 domain reports with raw threshold validation and CWD-independent root (R91, R111, R116, R118)."""
        src_dir = normalize_source_path(source_root)
        allow_unverified = set(allow_legacy_unverified_domains or set())
        is_canonical_root = (src_dir == CANONICAL_SOURCE_ROOT)

        loaded_payloads: Dict[str, Dict[str, Any]] = {}
        manifest_paths: Dict[str, str] = {}
        source_files_sha256: Dict[str, str] = {}
        lineage_audit: Dict[str, Any] = {}

        for domain in ("threshold", "distribution", "correlation", "temporal", "stationarity", "extreme"):
            canonical_fn = CANONICAL_DOMAIN_FILENAMES[domain]
            bridge_fn = BRIDGE_DOMAIN_FILENAMES[domain]

            canonical_p = src_dir / canonical_fn
            bridge_p = src_dir / bridge_fn

            chosen_path = None
            if canonical_p.exists():
                chosen_path = canonical_p
            elif bridge_p.exists():
                chosen_path = bridge_p
            else:
                # Check BRIDGE_SOURCE_PATHS
                alt_bridge = BRIDGE_SOURCE_PATHS.get(domain)
                if alt_bridge and alt_bridge.exists():
                    chosen_path = alt_bridge
                else:
                    raise FileNotFoundError(f"Missing domain report for '{domain}' in {src_dir}. Tried '{canonical_fn}' and '{bridge_fn}'")

            manifest_paths[domain] = str(chosen_path.resolve())
            raw_bytes = chosen_path.read_bytes()
            source_files_sha256[domain] = hashlib.sha256(raw_bytes).hexdigest()

            try:
                payload = json.loads(raw_bytes.decode("utf-8"))
            except Exception as e:
                raise ValueError(f"Corrupted or non-dict canonical file for '{domain}' at {chosen_path}: {e}") from e

            if not isinstance(payload, dict) or len(payload) == 0:
                raise ValueError(f"Corrupted or non-dict canonical file for '{domain}' at {chosen_path}")

            loaded_payloads[domain] = payload

            c1 = payload.get("metadata")
            c2 = payload.get("contract_metadata")
            status, n_obs, split, s_date, e_date = _validate_domain_lineage_metadata(
                c1, c2, domain, allow_unverified=(domain in allow_unverified or allow_unverified_lineage)
            )
            lineage_audit[domain] = {
                "status": status,
                "n_observations": n_obs,
                "split": split,
                "start_date": s_date,
                "end_date": e_date,
                "source_file": chosen_path.name,
            }

        # Strict domain payload field presence validation (T05)
        stat_p = loaded_payloads["stationarity"]
        if "exploratory_sarima_heuristic" not in stat_p and "suggested_sarima_order" not in stat_p:
            raise ValueError("[stationarity] missing required 'suggested_sarima_order' or 'exploratory_sarima_heuristic'")

        effective_th = override_rain_threshold if override_rain_threshold is not None else validated_threshold
        th_payload = loaded_payloads["threshold"]

        if effective_th is None:
            cands = ["chosen_threshold_mm", "initial_project_candidate", "canonical_project_threshold_mm", "optimal_threshold"]
            for c in cands:
                if c in th_payload:
                    raw_val = th_payload[c]
                    effective_th = _validate_scalar_threshold(raw_val, f"threshold.{c}")
                    break
            if effective_th is None:
                raise ValueError("[threshold] Could not resolve rain threshold from payload")
        else:
            effective_th = _validate_scalar_threshold(effective_th, "from_domain_reports.validated_threshold")

        report = cls.from_raw(
            threshold_res=loaded_payloads["threshold"],
            distribution_res=loaded_payloads["distribution"],
            correlation_res=loaded_payloads["correlation"],
            temporal_res=loaded_payloads["temporal"],
            stationarity_res=loaded_payloads["stationarity"],
            extreme_res=loaded_payloads["extreme"],
            validated_threshold=effective_th,
        )

        all_canonical_files = (
            is_canonical_root and
            not allow_unverified_lineage and
            all(manifest_paths[d] == str(CANONICAL_SOURCE_PATHS[d]) for d in REQUIRED_LINEAGE_DOMAINS) and
            all(audit["status"] == "VERIFIED_CANONICAL_DEV_TRAIN" for audit in lineage_audit.values())
        )

        if all_canonical_files:
            expected_sha = hashlib.sha256(
                json.dumps(report.to_dict(include_raw=False, include_diagnostics=True), sort_keys=True).encode("utf-8")
            ).hexdigest()
            raw_sha = hashlib.sha256(
                json.dumps(report.raw_results, sort_keys=True).encode("utf-8")
            ).hexdigest()
            report._provenance_context = {
                "verified_domains_count": 6,
                "all_verified": True,
                "canonical_source_root": str(CANONICAL_SOURCE_ROOT),
                "source_files_manifest": manifest_paths,
                "source_files_sha256": source_files_sha256,
                "lineage_audit": lineage_audit,
                "baseline_config_sha256": expected_sha,
                "raw_results_sha256": raw_sha,
            }
        else:
            report._provenance_context = None

        return report

    def export_triple_artifacts(
        self,
        canonical_path: Union[str, Path] = "reports/eda/eda_report.json",
        processed_path: Union[str, Path] = "data/processed/eda_report.json",
        bridge_path: Union[str, Path] = "notebooks/eda/outputs/eda_report.json",
        allow_unverified_lineage: bool = False,
    ) -> Dict[str, Path]:
        """Transactional triple-artifact export with exclusive temporary allocation, immediate registration, BaseException-safe zero-leak cleanup and fail-closed rollback (R105, R106, R110, R115, R117, R120, R121, R122, R124, R125, R126, R127, R128, R129, R130, R131, R132, R133, R134, R135, R136, R137)."""
        d_can = normalize_export_path(canonical_path)
        d_proc = normalize_export_path(processed_path)
        d_bridge = normalize_export_path(bridge_path)

        dest_paths = [d_can, d_proc, d_bridge]

        # Duplicate destinations check
        if len(set(dest_paths)) != 3:
            raise ValueError(f"Duplicate export destinations detected: {[str(d) for d in dest_paths]}")

        # Check protected paths
        for d in dest_paths:
            if d in get_protected_internal_paths():
                raise PermissionError(f"Target destination '{d}' is a protected internal pipeline lock path ({d.name}).")
            if d in get_protected_source_paths():
                raise PermissionError(f"Target destination '{d}' is a protected EDA source artifact ({d.name}).")

        # All-or-none production destinations rule
        prod_dests = get_production_destinations()
        num_prod = sum(1 for d in dest_paths if d in prod_dests)
        if 0 < num_prod < len(prod_dests):
            raise ValueError(
                f"All-or-None rule violated: exactly {len(prod_dests)} production destinations "
                f"must be targeted together. Got {num_prod} production destination(s)."
            )

        is_production_export = (num_prod == len(prod_dests))

        if is_production_export:
            if not _validate_provenance_binding(self):
                raise PermissionError(
                    "Production export strictly forbidden: report does not possess verified canonical provenance context."
                )
        else:
            if not allow_unverified_lineage and not _validate_provenance_binding(self):
                raise PermissionError(
                    "Export to custom destinations requires either verified canonical provenance or allow_unverified_lineage=True."
                )

        lock = InterprocessExportLock(CANONICAL_EXPORT_LOCK_PATH)
        lock.acquire()
        try:
            content_bytes = json.dumps(
                self.to_dict(include_raw=False, include_diagnostics=True),
                indent=2,
                ensure_ascii=False,
            ).encode("utf-8")

            orig_hashes: Dict[Path, Optional[str]] = {}
            for dest in dest_paths:
                if dest.exists():
                    orig_hashes[dest] = hashlib.sha256(dest.read_bytes()).hexdigest()
                else:
                    orig_hashes[dest] = None

            # Phase 1: Exclusive Temporary Staging with Zero-Leak Cleanup
            staged_temps: Dict[Path, Path] = {}
            stage_err: Optional[BaseException] = None

            try:
                for dest in dest_paths:
                    fd, tmp_p = _create_empty_staged_temp(dest)
                    staged_temps[dest] = tmp_p
                    _populate_staged_temp(fd, tmp_p, content_bytes)
            except BaseException as e:
                stage_err = e

            if stage_err is not None:
                stage_cleanup_errors: List[Tuple[Path, BaseException]] = []
                cleanup_interrupt: Optional[BaseException] = None

                for dest, tmp in staged_temps.items():
                    try:
                        tmp.unlink(missing_ok=True)
                    except BaseException as ce:
                        stage_cleanup_errors.append((tmp, ce))
                        if _is_interruption(ce) and cleanup_interrupt is None:
                            cleanup_interrupt = ce

                if cleanup_interrupt is not None:
                    cleanup_details_str = [f"{p}: ({type(e).__name__}) {e}" for p, e in stage_cleanup_errors]
                    logger.critical(
                        "Process interruption during temporary cleanup (%s) after stage failure (%s). Cleanup failures: %s",
                        type(cleanup_interrupt).__name__, stage_err, cleanup_details_str
                    )
                    if isinstance(stage_err, Exception):
                        try:
                            raise cleanup_interrupt from stage_err
                        except BaseException as chained_exc:
                            raise chained_exc
                    raise cleanup_interrupt

                if stage_cleanup_errors:
                    cleanup_details_str = [f"{p}: ({type(e).__name__}) {e}" for p, e in stage_cleanup_errors]
                    if isinstance(stage_err, Exception):
                        raise IOError(
                            f"Phase 1 staged pre-write failed ({stage_err}) "
                            f"AND temporary cleanup errors occurred: {cleanup_details_str}"
                        ) from stage_err
                    else:
                        logger.critical(
                            "Interruption during Phase 1 staged pre-write (%s) AND temporary cleanup errors occurred: %s",
                            stage_err, cleanup_details_str
                        )
                        raise stage_err

                raise stage_err

            # Phase 2: Transactional Commit via 4-State Journal
            journal: Dict[Path, Dict[str, Any]] = {
                dest: {
                    "state": "UNTOUCHED",
                    "backup_path": None,
                    "orig_sha256": orig_hashes[dest],
                }
                for dest in dest_paths
            }

            try:
                for dest in dest_paths:
                    rec = journal[dest]
                    if dest.exists():
                        backup_p, res_fd = _allocate_atomic_backup_path(dest)
                        rec["backup_path"] = backup_p
                        rec["state"] = "RESERVED"
                        try:
                            os.close(res_fd)
                        except Exception as close_err:
                            raise IOError(
                                f"Failed to close reservation descriptor for backup placeholder '{backup_p}': {close_err}"
                            ) from close_err

                        os.replace(str(dest), str(backup_p))
                        rec["state"] = "BACKED_UP"

                    os.replace(str(staged_temps[dest]), str(dest))
                    rec["state"] = "COMMITTED"

            except Exception as commit_err:
                logger.error("Commit error encountered: %s. Initiating fail-closed rollback.", commit_err)
                rollback_errors: List[str] = []

                for dest, rec in journal.items():
                    backup_p = rec["backup_path"]
                    orig_sha = rec["orig_sha256"]

                    try:
                        if orig_sha is None:
                            _restore_and_verify(dest, backup_p, orig_sha=None, rollback_errors=rollback_errors)
                        else:
                            dest_has_orig = (
                                dest.exists() and
                                hashlib.sha256(dest.read_bytes()).hexdigest() == orig_sha
                            )
                            if dest_has_orig:
                                if backup_p is not None:
                                    backup_p.unlink(missing_ok=True)
                            else:
                                _restore_and_verify(dest, backup_p, orig_sha=orig_sha, rollback_errors=rollback_errors)
                    except Exception as rb_exc:
                        rollback_errors.append(f"Rollback exception on {dest}: {rb_exc}")

                for tmp in staged_temps.values():
                    try:
                        tmp.unlink(missing_ok=True)
                    except Exception as tmp_err:
                        rollback_errors.append(f"Leftover temporary cleanup failed for '{tmp}': {tmp_err}")

                if rollback_errors:
                    raise IOError(
                        f"CRITICAL: Transaction commit failed ({commit_err}) AND rollback errors occurred: {rollback_errors}"
                    ) from commit_err

                raise IOError(f"Transaction commit failed and was successfully rolled back: {commit_err}") from commit_err

            # Phase 3: Decoupled Backup Cleanup
            cleanup_errors: List[str] = []
            for rec in journal.values():
                bp = rec["backup_path"]
                if bp is not None:
                    try:
                        bp.unlink(missing_ok=True)
                    except Exception as ce:
                        cleanup_errors.append(f"{bp}: {ce}")

            if cleanup_errors:
                logger.warning("Triple export committed successfully, but some backup files could not be unlinked: %s", cleanup_errors)

            return {
                "canonical_path": d_can,
                "processed_path": d_proc,
                "bridge_path": d_bridge,
            }

        finally:
            lock.release()


# ==============================================================================
# 8. Backwards Compatibility & Convenience Helpers
# ==============================================================================

def export_triple_artifacts(
    report: "EDAReport",
    *dest_paths: Union[str, Path],
    allow_unverified_lineage: bool = False,
) -> Dict[str, Path]:
    """Module-level convenience wrapper delegating to EDAReport.export_triple_artifacts."""
    if not isinstance(report, EDAReport):
        raise TypeError(f"report must be an EDAReport instance, got {type(report).__name__}")
    return report.export_triple_artifacts(*dest_paths, allow_unverified_lineage=allow_unverified_lineage)


def compare_rain_thresholds(
    df: Any,
    target_col: Optional[str] = None,
    candidate_thresholds: Optional[List[float]] = None,
    thresholds: Optional[List[float]] = None,
    inclusive_wet: bool = True,
    **kwargs: Any,
) -> Any:
    """Compute empirical distribution statistics across candidate rain thresholds.

    Empirically describes rainfall occurrence, class balance, and conditional intensity
    across candidate thresholds without selecting or freezing any threshold in EDA.
    """
    import pandas as pd
    import numpy as np

    if target_col is not None and hasattr(df, "columns") and target_col in df.columns:
        col = target_col
    else:
        try:
            from ..config.resolve import resolve_target_col
            col = resolve_target_col(target_col)
            if hasattr(df, "columns") and col not in df.columns and len(df.columns) > 0:
                col = "Lượng mưa" if "Lượng mưa" in df.columns else df.columns[0]
        except Exception:
            col = target_col or ("Lượng mưa" if hasattr(df, "columns") and "Lượng mưa" in df.columns else (df.columns[0] if hasattr(df, "columns") and len(df.columns) > 0 else "Lượng mưa"))

    target = df[col].dropna()
    total_days = len(target)

    if candidate_thresholds is None and thresholds is not None:
        candidate_thresholds = thresholds
    if candidate_thresholds is None:
        candidate_thresholds = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]

    candidate_thresholds = sorted(list(set(candidate_thresholds)))
    records = []
    prev_no_rain_frac = None

    for th in candidate_thresholds:
        exact_mask = (target == th)
        exact_count = int(exact_mask.sum())

        if inclusive_wet:
            if th == 0.0:
                no_rain_mask = (target == 0.0)
                rain_mask = (target > 0.0)
                cond_desc = "== 0.0 mm"
            else:
                no_rain_mask = (target < th)
                rain_mask = (target >= th)
                cond_desc = f"< {th:.1f} mm"
        else:
            no_rain_mask = (target <= th)
            rain_mask = (target > th)
            cond_desc = f"<= {th:.1f} mm"

        no_rain_count = int(no_rain_mask.sum())
        rain_count = int(rain_mask.sum())

        no_rain_frac = float(no_rain_count / total_days) if total_days > 0 else 0.0
        rain_frac = float(rain_count / total_days) if total_days > 0 else 0.0

        if prev_no_rain_frac is not None:
            gap_pct = (no_rain_frac - prev_no_rain_frac) * 100.0
        else:
            gap_pct = 0.0

        # Conditional rainfall intensity: statistics computed on wet population
        rain_amounts = target[rain_mask]
        if len(rain_amounts) > 0:
            rain_mean_cond = float(rain_amounts.mean())
            rain_median_cond = float(rain_amounts.median())
        else:
            rain_mean_cond = 0.0
            rain_median_cond = 0.0

        records.append({
            "threshold_mm": float(th),
            "no_rain_condition": cond_desc,
            "no_rain_days": no_rain_count,
            "rain_days": rain_count,
            "no_rain_pct": round(no_rain_frac * 100.0, 2),
            "rain_pct": round(rain_frac * 100.0, 2),
            "exact_threshold_days": exact_count,
            "incremental_gap_pct": round(gap_pct, 2),
            "rain_mean_conditional": round(rain_mean_cond, 2),
            "rain_median_conditional": round(rain_median_cond, 2),
        })
        prev_no_rain_frac = no_rain_frac

    return pd.DataFrame(records)


class EDAPipeline:
    """Historical backward-compatibility entry point for EDAPipeline."""

    def __init__(self, df: Any = None, target_col: Optional[str] = None, date_col: Optional[str] = None, **kwargs: Any) -> None:
        self.df = df
        self.target_col = target_col
        self.date_col = date_col

    def run(self, *args: Any, **kwargs: Any) -> "EDAReport":
        return EDAReport.from_raw(*args, **kwargs)

    @classmethod
    def run_pipeline(cls, *args: Any, **kwargs: Any) -> "EDAReport":
        return EDAReport.from_domain_reports(*args, **kwargs)

