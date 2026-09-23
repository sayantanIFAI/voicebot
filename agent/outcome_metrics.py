"""Per-intent counters for outcomes that need their own rate tracked,
distinct from a plain log line -- KCD-442 ("its own metric... rate is
reported per intent because a rise means a data or integration problem")
and KCD-457 ("every tier records its abstention reason and the
distribution is reported per intent and language").

Process-local, exposed through main.py's existing /api/stats the same
way agent/fast_path.py's FastPath.stats and agent/semantic_cache.py's
SemanticCache already are -- not a new metrics system, the same one.
"""
from __future__ import annotations

import collections
import threading


class OutcomeCounter:
    """(outcome_or_reason, intent, lang) -> count, thread-safe (dispatch
    runs turns from an asyncio event loop, but tests and future worker
    pools may not)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._counts: collections.Counter = collections.Counter()

    def record(self, key: str, intent: str, lang: str = "") -> None:
        with self._lock:
            self._counts[(key, intent, lang)] += 1

    def snapshot(self) -> dict:
        with self._lock:
            by_key: dict[str, dict[str, int]] = collections.defaultdict(dict)
            for (key, intent, lang), count in self._counts.items():
                label = f"{intent}:{lang}" if lang else intent
                by_key[key][label] = count
            return {key: dict(sorted(v.items())) for key, v in sorted(by_key.items())}


# Two independent instruments -- KCD-442's insufficient-information rate
# and KCD-457's abstention-reason distribution measure different tiers of
# the pipeline (post-ASR confidence gate vs. fast_path's own abstains)
# and must not be conflated into one counter's key space.
insufficient_information = OutcomeCounter()
abstentions = OutcomeCounter()
