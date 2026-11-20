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


class JitterDelay(DelayModel):
    """Base delay plus an independent per-tick uniform jitter draw.

    ``d_t = max(1, D + U(-J, +J))``, drawn ONCE per ``receive()`` call.
    The study drivers call ``receive()`` exactly once per tick, so a fixed
    seed reproduces the entire per-tick sequence (each run builds a fresh
    instance seeded from the schedule's ``seed`` column).  Release
    semantics mirror :class:`ConstantDelay`: the most-recent frame with
    ``current_timestep - ts >= d_t`` wins; older released frames are
    dropped.

    Parameters
    ----------
    delay_steps:
        Base delay D (the schedule's ``delay_n``).
    jitter_steps:
        Jitter amplitude J in frames.
    seed:
        RNG seed for the per-tick draws.
    """

    def __init__(self, delay_steps: int, jitter_steps: int, seed: int) -> None:
        self._delay = int(delay_steps)
        self._jitter = int(jitter_steps)
        self._rng = np.random.default_rng(seed)
        self._buffer: list[tuple[np.ndarray, np.ndarray, int]] = []
        self.realized: list[int] = []   # one d_t per receive() call

    def submit(self, frame: np.ndarray, proprio: np.ndarray,
               timestep: int) -> None:
        self._buffer.append((frame.copy(), proprio.copy(), timestep))

    def receive(self, current_timestep: int) -> tuple[np.ndarray, np.ndarray, int] | None:
        d_t = max(1, self._delay
                  + int(self._rng.integers(-self._jitter, self._jitter + 1)))
        self.realized.append(d_t)
        result = None
        remaining: list[tuple[np.ndarray, np.ndarray, int]] = []
        for frame, proprio, ts in self._buffer:
            if current_timestep - ts >= d_t:
                result = (frame, proprio, ts)  # most-recent qualifying wins
            else:
                remaining.append((frame, proprio, ts))
        if result is not None:
            self._buffer = remaining
        return result

    def reset(self) -> None:
        # Buffer only: the RNG (and realized history) deliberately continue —
        # but the study builds a fresh instance per run anyway.
        self._buffer.clear()
