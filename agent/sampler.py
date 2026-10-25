"""A tiny in-process sampling profiler, for finding what a busy voice-agent process is doing (off unless PROFILE_SAMPLER=1).

Why it exists: py-spy cannot attach inside the pod's container (no ptrace), and the 24-call load test showed the agent burning 2-4 CPU cores while the GPU sat idle. A thread that looks at every
other thread's current stack every few milliseconds, and counts the innermost frames, is enough to say where the time goes. GET /debug/profile returns the counts (and resets them).
"""

from __future__ import annotations

import collections
import os
import sys
import threading
import time

ENABLED = os.environ.get("PROFILE_SAMPLER", "0") == "1"
INTERVAL_S = float(os.environ.get("PROFILE_INTERVAL_S", "0.02"))

_lock = threading.Lock()
_self_counts: collections.Counter = collections.Counter()
_stack_counts: collections.Counter = collections.Counter()
_samples = 0
_started = False


def _label(frame) -> str:
    code = frame.f_code
    path = code.co_filename.replace("\\", "/")
    for marker in ("site-packages/", "workspace/kolkata-care-voice-agent/", "lib/python3.12/"):
        if marker in path:
            path = path.split(marker, 1)[1]
            break
    return f"{path}:{code.co_name}:{frame.f_lineno}"


def _run() -> None:
    global _samples
    me = threading.get_ident()
    while True:
        time.sleep(INTERVAL_S)
        frames = sys._current_frames()
        with _lock:
            for tid, frame in frames.items():
                if tid == me:
                    continue
                _self_counts[_label(frame)] += 1
                chain, f, depth = [], frame, 0
                while f is not None and depth < 6:
                    chain.append(f"{f.f_code.co_name}")
                    f, depth = f.f_back, depth + 1
                _stack_counts[" < ".join(chain)] += 1
            _samples += 1


def start(force: bool = False) -> None:
    global _started
    if (ENABLED or force) and not _started:
        _started = True
        threading.Thread(target=_run, name="profile-sampler", daemon=True).start()


def report(top: int = 40, reset: bool = True) -> dict:
    global _samples
    with _lock:
        out = {
            "samples": _samples,
            "interval_s": INTERVAL_S,
            "innermost": _self_counts.most_common(top),
            "stacks": _stack_counts.most_common(top),
        }
        if reset:
            _self_counts.clear()
            _stack_counts.clear()
            _samples = 0
    return out
