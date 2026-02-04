"""Pendulum system: S1 x R manifold."""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .base import DynamicalSystem, ManifoldComponent


class PendulumSystem(DynamicalSystem):
    """Simple pendulum with state [theta, theta_dot].

    Manifold: S1 x R  (angle on circle, angular velocity on real line).
    Embedded representation: [sin(theta), cos(theta), theta_dot_norm] (3D).
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
        self.angular_velocity_limit = max(
            abs(bounds["theta_dot"]["min"]),
            abs(bounds["theta_dot"]["max"]),
        )

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    def define_manifold_structure(self) -> List[ManifoldComponent]:
        return [
            ManifoldComponent(name="theta", manifold_type="SO2"),
            ManifoldComponent(name="theta_dot", manifold_type="Real"),
        ]

    def define_state_bounds(self) -> Dict[str, Tuple[float, float]]:
        return {
            "theta": (-np.pi, np.pi),
            "theta_dot": (-self.angular_velocity_limit, self.angular_velocity_limit),
        }

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def embed_state(self, state: np.ndarray) -> np.ndarray:
        """[theta, theta_dot] -> [sin(theta), cos(theta), theta_dot_norm]."""
        theta, theta_dot = state[0], state[1]
        return np.array(
            [
                np.sin(theta),
                np.cos(theta),
                theta_dot / self.angular_velocity_limit,
            ],
            dtype=np.float32,
        )
