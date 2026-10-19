"""
Utility helpers for the Sarcasm & Irony Detection Inference Service.
"""

import hashlib
import json
import os
import platform
import time
from typing import Dict, Any, Tuple
from .exceptions import ArtifactValidationError


def compute_sha256(filepath: str) -> str:
    """Computes SHA-256 checksum for a given file path."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found for checksum: {filepath}")
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def verify_artifact_checksums(model_dir: str) -> Tuple[bool, Dict[str, Any]]:
    """
    Verifies all artifacts listed in checksums.json against disk.

    Raises:
        ArtifactValidationError: If any file is missing or SHA-256 hash does not match.
    """
    checksum_file = os.path.join(model_dir, "checksums.json")
    if not os.path.exists(checksum_file):
        raise ArtifactValidationError(f"Checksum manifest missing at {checksum_file}")

    with open(checksum_file, "r", encoding="utf-8") as f:
        records = json.load(f)

    results = {}
    for rel_path, meta in records.items():
        abs_path = os.path.join(model_dir, rel_path)
        if not os.path.exists(abs_path):
            raise ArtifactValidationError(f"Required artifact missing: {rel_path} ({abs_path})")

        computed = compute_sha256(abs_path)
        expected = meta["sha256"]
        if computed != expected:
            raise ArtifactValidationError(
                f"Checksum mismatch for {rel_path}! Expected: {expected}, Computed: {computed}"
            )
        results[rel_path] = {"status": "verified", "sha256": computed}

    return True, results


def get_execution_device() -> Dict[str, Any]:
    """Returns runtime execution device information."""
    info = {
        "device": "cpu",
        "processor": platform.processor() or "x86_64",
        "python_version": platform.python_version(),
        "framework": "scikit-learn (CPU optimized)",
    }
    return info


def get_memory_info() -> Dict[str, Any]:
    """Returns approximate process memory usage in megabytes."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem_bytes = process.memory_info().rss
        mem_mb = round(mem_bytes / (1024 * 1024), 2)
        return {"memory_rss_mb": mem_mb, "status": "ok"}
    except Exception:
        return {"memory_rss_mb": "N/A", "status": "psutil_not_installed"}
