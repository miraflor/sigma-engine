"""Small live-progress reporter used by the CLI pipeline.

The engine intentionally avoids a logging-framework dependency.  Progress messages are
human-facing diagnostics, written immediately to stderr so long-running network operations
never look frozen in a terminal.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime
from time import perf_counter
from typing import Callable

ProgressCallback = Callable[[str], None]


def _format_elapsed(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


@dataclass
class ProgressReporter:
    """Timestamped, immediately flushed progress output."""

    enabled: bool = True
    _started: float = field(default_factory=perf_counter, init=False, repr=False)

    def __call__(self, message: str) -> None:
        if not self.enabled:
            return
        now = datetime.now().strftime("%H:%M:%S")
        elapsed = _format_elapsed(perf_counter() - self._started)
        print(f"[{now} +{elapsed}] {message}", file=sys.stderr, flush=True)
