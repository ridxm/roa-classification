"""Quadrotor 2D system: R^2 x S1 x R^3 manifold."""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .base import DynamicalSystem, ManifoldComponent


class Quadrotor2DSystem(DynamicalSystem):
    """2D Quadrotor with state [x, z, theta, x_dot, z_dot, theta_dot].

    Manifold: R^2 x S^1 x R^3 (position, pitch angle on circle, velocities).
    Embedded representation:
        [x_norm, z_norm, sin(theta), cos(theta), x_dot_norm, z_dot_norm, theta_dot_norm] (7D).

    Note: z has asymmetric bounds (0.1 to 1.5), so we normalize using center and range.
    """

    def __init__(self, dataset_dir: str) -> None:
        self.dataset_dir = Path(dataset_dir)
        self._load_bounds()
        super().__init__()

    def _load_bounds(self) -> None:
        desc_path = self.dataset_dir / "dataset_description.json"
        with open(desc_path) as f:
            info = json.load(f)
        bounds = info["achieved_bounds"]

        # Symmetric bounds (use max absolute value)
        self.x_limit = max(abs(bounds["x"]["min"]), abs(bounds["x"]["max"]))
        self.x_dot_limit = max(
            abs(bounds["x_dot"]["min"]), abs(bounds["x_dot"]["max"])
        )
        self.z_dot_limit = max(
            abs(bounds["z_dot"]["min"]), abs(bounds["z_dot"]["max"])
        )
        self.theta_dot_limit = max(
            abs(bounds["theta_dot"]["min"]), abs(bounds["theta_dot"]["max"])
        )

        # Asymmetric z bounds: normalize to [-1, 1] via center and half-range
        z_min, z_max = bounds["z"]["min"], bounds["z"]["max"]
        self.z_center = (z_min + z_max) / 2
        self.z_range = (z_max - z_min) / 2

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    def define_manifold_structure(self) -> List[ManifoldComponent]:
        return [
            ManifoldComponent(name="x", manifold_type="Real"),
            ManifoldComponent(name="z", manifold_type="Real"),
            ManifoldComponent(name="theta", manifold_type="SO2"),
            ManifoldComponent(name="x_dot", manifold_type="Real"),
            ManifoldComponent(name="z_dot", manifold_type="Real"),
            ManifoldComponent(name="theta_dot", manifold_type="Real"),
        ]

    def define_state_bounds(self) -> Dict[str, Tuple[float, float]]:
        return {
            "x": (-self.x_limit, self.x_limit),
            "z": (self.z_center - self.z_range, self.z_center + self.z_range),
            "theta": (-np.pi, np.pi),
            "x_dot": (-self.x_dot_limit, self.x_dot_limit),
            "z_dot": (-self.z_dot_limit, self.z_dot_limit),
            "theta_dot": (-self.theta_dot_limit, self.theta_dot_limit),
        }

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def embed_state(self, state: np.ndarray) -> np.ndarray:
        """[x, z, theta, x_dot, z_dot, theta_dot] ->
        [x_norm, z_norm, sin(theta), cos(theta), x_dot_norm, z_dot_norm, theta_dot_norm]."""
        x, z, theta = state[0], state[1], state[2]
        x_dot, z_dot, theta_dot = state[3], state[4], state[5]
        return np.array(
            [
                x / self.x_limit,
                (z - self.z_center) / self.z_range,
                np.sin(theta),
                np.cos(theta),
                x_dot / self.x_dot_limit,
                z_dot / self.z_dot_limit,
                theta_dot / self.theta_dot_limit,
            ],
            dtype=np.float32,
        )
