"""CartPole PyBullet system: R2 x S1 x R manifold."""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .base import DynamicalSystem, ManifoldComponent


class CartPoleSystem(DynamicalSystem):
    """CartPole with state [x, theta, x_dot, theta_dot].

    Manifold: R x S1 x R^2  (position, angle on circle, velocities).
    Embedded representation:
        [x_norm, sin(theta), cos(theta), x_dot_norm, theta_dot_norm] (5D).
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
        self.x_limit = max(abs(bounds["x"]["min"]), abs(bounds["x"]["max"]))
        self.x_dot_limit = max(
            abs(bounds["x_dot"]["min"]), abs(bounds["x_dot"]["max"])
        )
        self.theta_dot_limit = max(
            abs(bounds["theta_dot"]["min"]), abs(bounds["theta_dot"]["max"])
        )

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    def define_manifold_structure(self) -> List[ManifoldComponent]:
        return [
            ManifoldComponent(name="x", manifold_type="Real"),
            ManifoldComponent(name="theta", manifold_type="SO2"),
            ManifoldComponent(name="x_dot", manifold_type="Real"),
            ManifoldComponent(name="theta_dot", manifold_type="Real"),
        ]

    def define_state_bounds(self) -> Dict[str, Tuple[float, float]]:
        return {
            "x": (-self.x_limit, self.x_limit),
            "theta": (-np.pi, np.pi),
            "x_dot": (-self.x_dot_limit, self.x_dot_limit),
            "theta_dot": (-self.theta_dot_limit, self.theta_dot_limit),
        }

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def embed_state(self, state: np.ndarray) -> np.ndarray:
        """[x, theta, x_dot, theta_dot] ->
        [x_norm, sin(theta), cos(theta), x_dot_norm, theta_dot_norm]."""
        x, theta, x_dot, theta_dot = state[0], state[1], state[2], state[3]
        return np.array(
            [
                x / self.x_limit,
                np.sin(theta),
                np.cos(theta),
                x_dot / self.x_dot_limit,
                theta_dot / self.theta_dot_limit,
            ],
            dtype=np.float32,
        )
