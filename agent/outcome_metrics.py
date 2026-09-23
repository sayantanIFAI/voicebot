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

# KCD-065: how often each script-mixture bucket ("bn", "bn+en", ...)
# actually occurs on real calls -- the per-bucket breakdown Appendix F
# asks code-switch accuracy to be reported against, not only in
# aggregate. `intent` here is reused as the mixture-bucket key itself
# (OutcomeCounter's shape is generic enough), `lang` left blank.
code_switch_buckets = OutcomeCounter()

# KCD-075: how often live traffic falls into each channel-quality class,
# so it can be compared against the evaluation population's own
# distribution (this module's whole reason to exist).
channel_quality_buckets = OutcomeCounter()

# KCD-460: fast_path's serve rate, published PER INTENT (not just the
# aggregate served/abstained fast_path.py's own stats already had) --
# "a routine question is answered instantly, and their serve rate is
# published per intent and language" is this story's own wording.
fast_path_served = OutcomeCounter()

# KCD-464: how often a caller uses the manual interrupt control (main.py's
# _handle_control, type "interrupt") while the agent is mid-reply. This is
# NOT acoustic barge-in (main.py's own module docstring: the mic stays
# muted client-side during playback, and there is no AEC reference signal
# to detect talk-over on) -- it is a visible "stop" affordance the caller
# can tap instead. Tracking how often it fires is what tells a human
# whether callers are actually reaching for it, which is the signal that
# would justify the larger AEC-based barge-in project later. `intent` is
# reused as a fixed literal key ("manual"); `lang` is the call's language.
barge_in_interrupts = OutcomeCounter()
