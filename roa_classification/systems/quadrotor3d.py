"""Quadrotor 3D system: R^3 x SO(3) x R^6 manifold."""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .base import DynamicalSystem, ManifoldComponent


class Quadrotor3DSystem(DynamicalSystem):
    """3D Quadrotor with state [x, y, z, qw, qx, qy, qz, x_dot, y_dot, z_dot, p, q, r].

    Manifold: R^3 x SO(3) x R^6 (position, attitude via unit quaternion, velocities).
    Embedded representation (13D):
        [x_norm, y_norm, z_norm, qw, qx, qy, qz, x_dot_norm, y_dot_norm, z_dot_norm, p_norm, q_norm, r_norm]

    Quaternions are already unit norm and canonicalized (qw >= 0), so no embedding needed.
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

        # Position bounds (symmetric via max absolute value)
        self.x_limit = max(abs(bounds["x"]["min"]), abs(bounds["x"]["max"]))
        self.y_limit = max(abs(bounds["y"]["min"]), abs(bounds["y"]["max"]))
        self.z_limit = max(abs(bounds["z"]["min"]), abs(bounds["z"]["max"]))

        # Linear velocity bounds
        self.x_dot_limit = max(abs(bounds["x_dot"]["min"]), abs(bounds["x_dot"]["max"]))
        self.y_dot_limit = max(abs(bounds["y_dot"]["min"]), abs(bounds["y_dot"]["max"]))
        self.z_dot_limit = max(abs(bounds["z_dot"]["min"]), abs(bounds["z_dot"]["max"]))

        # Angular velocity bounds
        self.p_limit = max(abs(bounds["p"]["min"]), abs(bounds["p"]["max"]))
        self.q_limit = max(abs(bounds["q"]["min"]), abs(bounds["q"]["max"]))
        self.r_limit = max(abs(bounds["r"]["min"]), abs(bounds["r"]["max"]))

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    def define_manifold_structure(self) -> List[ManifoldComponent]:
        return [
            ManifoldComponent(name="x", manifold_type="Real"),
            ManifoldComponent(name="y", manifold_type="Real"),
            ManifoldComponent(name="z", manifold_type="Real"),
            ManifoldComponent(name="qw", manifold_type="Real"),
            ManifoldComponent(name="qx", manifold_type="Real"),
            ManifoldComponent(name="qy", manifold_type="Real"),
            ManifoldComponent(name="qz", manifold_type="Real"),
            ManifoldComponent(name="x_dot", manifold_type="Real"),
            ManifoldComponent(name="y_dot", manifold_type="Real"),
            ManifoldComponent(name="z_dot", manifold_type="Real"),
            ManifoldComponent(name="p", manifold_type="Real"),
            ManifoldComponent(name="q", manifold_type="Real"),
            ManifoldComponent(name="r", manifold_type="Real"),
        ]

    def define_state_bounds(self) -> Dict[str, Tuple[float, float]]:
        return {
            "x": (-self.x_limit, self.x_limit),
            "y": (-self.y_limit, self.y_limit),
            "z": (-self.z_limit, self.z_limit),
            "qw": (0.0, 1.0),
            "qx": (-1.0, 1.0),
            "qy": (-1.0, 1.0),
            "qz": (-1.0, 1.0),
            "x_dot": (-self.x_dot_limit, self.x_dot_limit),
            "y_dot": (-self.y_dot_limit, self.y_dot_limit),
            "z_dot": (-self.z_dot_limit, self.z_dot_limit),
            "p": (-self.p_limit, self.p_limit),
            "q": (-self.q_limit, self.q_limit),
            "r": (-self.r_limit, self.r_limit),
        }

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def embed_state(self, state: np.ndarray) -> np.ndarray:
        """[x, y, z, qw, qx, qy, qz, x_dot, y_dot, z_dot, p, q, r] ->
        [x_norm, y_norm, z_norm, qw, qx, qy, qz, x_dot_norm, y_dot_norm, z_dot_norm, p_norm, q_norm, r_norm]."""
        x, y, z = state[0], state[1], state[2]
        qw, qx, qy, qz = state[3], state[4], state[5], state[6]
        x_dot, y_dot, z_dot = state[7], state[8], state[9]
        p, q, r = state[10], state[11], state[12]

        return np.array(
            [
                x / self.x_limit,
                y / self.y_limit,
                z / self.z_limit,
                qw,
                qx,
                qy,
                qz,
                x_dot / self.x_dot_limit,
                y_dot / self.y_dot_limit,
                z_dot / self.z_dot_limit,
                p / self.p_limit,
                q / self.q_limit,
                r / self.r_limit,
            ],
            dtype=np.float32,
        )
