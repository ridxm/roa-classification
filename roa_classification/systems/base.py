"""Base classes for dynamical systems with manifold-aware state representations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class ManifoldComponent:
    """A single component of the state space manifold.

    Attributes:
        name: Human-readable name (e.g. "pole_angle").
        manifold_type: One of "SO2", "Real".
        dim: Dimension of this component (always 1 for SO2/Real scalars).
    """

    name: str
    manifold_type: str  # "SO2" or "Real"
    dim: int = 1


class DynamicalSystem(ABC):
    """Abstract base for a dynamical system with manifold structure.

    Subclasses define their manifold components and state bounds.
    The base class provides embedding (SO2 -> sin/cos) and
    normalization helpers.
    """

    def __init__(self) -> None:
        self._manifold_components = self.define_manifold_structure()
        self._state_bounds = self.define_state_bounds()

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def define_manifold_structure(self) -> List[ManifoldComponent]:
        ...

    @abstractmethod
    def define_state_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Return {component_name: (lower, upper)} for every component."""
        ...

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def manifold_components(self) -> List[ManifoldComponent]:
        return self._manifold_components

    @property
    def state_dim(self) -> int:
        return sum(c.dim for c in self._manifold_components)

    @property
    def embedded_dim(self) -> int:
        """Dimension after SO2 components are expanded to (sin, cos)."""
        d = 0
        for c in self._manifold_components:
            d += 2 if c.manifold_type == "SO2" else c.dim
        return d

    @property
    def state_bounds(self) -> Dict[str, Tuple[float, float]]:
        return self._state_bounds

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_circular_indices(self) -> List[int]:
        """Return raw-state indices that live on SO2."""
        indices, idx = [], 0
        for c in self._manifold_components:
            if c.manifold_type == "SO2":
                indices.append(idx)
            idx += c.dim
        return indices

    def normalization_limit(self, name: str) -> float:
        """Symmetric normalization limit for a Real component."""
        lo, hi = self._state_bounds[name]
        return max(abs(lo), abs(hi))
