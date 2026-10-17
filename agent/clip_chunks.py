"""Cut a long utterance into recogniser-sized pieces at its quietest points.

Real call 2026-10-17 (saved audio, replayed on the pod): one 17-second five-sentence question was read by the Bengali recogniser's main
(RNNT) decoder as only its LAST 27 words -- the first 19 were simply not written -- while its backup (CTC) decoder had all 46 words. The
same speech read in growing pieces was complete up to about 12 seconds (35 of 35 words) and lost its beginning beyond that: the
model was trained on shorter clips. So a clip longer than MAX_CLIP_S is cut into pieces, each cut placed at the quietest 0.2 s
between 60% and 100% of the limit after the previous cut (a pause between sentences, not the middle of a word), each piece is read on its
own, and the texts are joined in order. A clip at or under the limit is untouched.
"""

from __future__ import annotations

import os
import tempfile
import wave

import numpy as np

MAX_CLIP_S = float(os.environ.get("ASR_MAX_CLIP_S", "11"))
_SEARCH_FROM = 0.6  # a cut is searched for between 60% and 100% of MAX_CLIP_S after the previous one
_QUIET_WINDOW_S = 0.2


def split_points(x: np.ndarray, sr: int, max_s: float = MAX_CLIP_S) -> list[int]:
    """Sample indices to cut at (empty when the clip is short enough)."""
    if len(x) <= int(max_s * sr):
        return []
    win = max(1, int(_QUIET_WINDOW_S * sr))
    energy = np.convolve(np.abs(x.astype(np.float32)), np.ones(win, dtype=np.float32) / win, mode="same")
    cuts: list[int] = []
    start = 0
    while len(x) - start > int(max_s * sr):
        lo, hi = start + int(_SEARCH_FROM * max_s * sr), start + int(max_s * sr)
        cut = lo + int(np.argmin(energy[lo:hi]))
        cuts.append(cut)
        start = cut
    return cuts


def chunk_wav(wav_path: str, max_s: float = MAX_CLIP_S) -> list[str]:
    """[wav_path] for a short clip; otherwise paths of temporary piece files (the caller removes them). Any problem reading the file
    returns [wav_path]: the clip is then read whole, exactly as before."""
    try:
        with wave.open(wav_path) as w:
            if w.getsampwidth() != 2 or w.getnchannels() != 1:
                return [wav_path]
            sr, frames = w.getframerate(), w.readframes(w.getnframes())
        x = np.frombuffer(frames, dtype=np.int16)
        cuts = split_points(x, sr, max_s)
        if not cuts:
            return [wav_path]
        paths = []
        for a, b in zip([0, *cuts], [*cuts, len(x)]):
            fd, path = tempfile.mkstemp(suffix=".wav", prefix="asr_chunk_")
            os.close(fd)
            with wave.open(path, "wb") as o:
                o.setnchannels(1)
                o.setsampwidth(2)
                o.setframerate(sr)
                o.writeframes(x[a:b].tobytes())
            paths.append(path)
        return paths
    except Exception:  # noqa: BLE001 - chunking is an aid; failing to chunk means reading the clip whole
        return [wav_path]
