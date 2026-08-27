"""
BNNSGraph acceleration backend.

This module provides the Python-side boundary for native Apple
BNNSGraph execution. The native implementation will be added
without changing the existing Core ML/ANE path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BNNSGraphModel:
    """Reference to a compiled Core ML model artifact."""

    path: Path

    def __post_init__(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(
                f"BNNSGraph model artifact does not exist: {self.path}"
            )


class BNNSGraphAccelerator:
    """
    BNNSGraph execution boundary.

    The existing ml-ane-transformers Core ML/ANE implementation
    remains the reference path. Native BNNSGraph execution will
    be connected here once the SDK interface is verified.
    """

    backend_name = "apple-bnnsgraph"

    def __init__(self, model_path: str | Path) -> None:
        self.model = BNNSGraphModel(Path(model_path))

    @property
    def path(self) -> Path:
        return self.model.path

    def describe(self) -> dict[str, str]:
        return {
            "backend": self.backend_name,
            "model": str(self.path),
            "status": "artifact-ready",
        }

    def execute(self, *args, **kwargs):
        raise NotImplementedError(
            "Native BNNSGraph execution has not been connected yet."
        )
