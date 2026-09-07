"""Pluggable network-delay simulators for the frame stream.

Each delay module buffers ground-truth frames and releases them after a
configured delay, simulating the effect of network latency.  New delay
profiles (jitter, variable RTT, packet loss, …) can be added by subclassing
:class:`DelayModel`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class DelayModel(ABC):
    """Abstract delay simulator for the frame stream.

    Subclass this to implement different delay profiles.
    """

    @abstractmethod
    def submit(self, frame: np.ndarray, proprio: np.ndarray,
               timestep: int) -> None:
        """Buffer a ground-truth frame (and its proprio) produced at
        *timestep*."""
        ...

    @abstractmethod
    def receive(self, current_timestep: int) -> tuple[np.ndarray, np.ndarray, int] | None:
        """Check for a delayed frame whose transit time has elapsed.

        Returns ``(frame, proprio, original_timestep)`` for the oldest
        qualifying frame, or ``None`` if nothing is ready.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Clear all buffered frames (called on env reset)."""
        ...


class NoDelay(DelayModel):
    """Passthrough — frames are available immediately (delay = 0)."""

    def submit(self, frame: np.ndarray, proprio: np.ndarray,
               timestep: int) -> None:
        self._latest = (frame, proprio, timestep)

    def receive(self, current_timestep: int) -> tuple[np.ndarray, np.ndarray, int] | None:
        if hasattr(self, "_latest"):
            result = self._latest
            del self._latest
            return result
        return None

    def reset(self) -> None:
        if hasattr(self, "_latest"):
            del self._latest


class ConstantDelay(DelayModel):
    """Delays every frame by a fixed number of timesteps.

    Parameters
    ----------
    delay_steps:
        Number of timesteps a frame is held before it becomes available.
        0 = no delay (frames available immediately).
    """

    def __init__(self, delay_steps: int) -> None:
        self._delay = delay_steps
        self._buffer: list[tuple[np.ndarray, np.ndarray, int]] = []

    def submit(self, frame: np.ndarray, proprio: np.ndarray,
               timestep: int) -> None:
        self._buffer.append((frame.copy(), proprio.copy(), timestep))

    def receive(self, current_timestep: int) -> tuple[np.ndarray, np.ndarray, int] | None:
        result = None
        remaining: list[tuple[np.ndarray, np.ndarray, int]] = []
        for frame, proprio, ts in self._buffer:
            if current_timestep - ts >= self._delay:
                result = (frame, proprio, ts)  # keep the most-recent qualifying frame
            else:
                remaining.append((frame, proprio, ts))
        if result is not None:
            self._buffer = remaining
        return result

    def reset(self) -> None:
        self._buffer.clear()
