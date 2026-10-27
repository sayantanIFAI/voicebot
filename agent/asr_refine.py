"""ASR refinement: choose between the texts the recogniser ALREADY HEARD, never write one (docs/adr/0004-asr-refine-shadow.md).

agent/asr.py runs two decoders (CTC and RNNT) on every clip and used to keep only one text. Sometimes the discarded one names a doctor or test the kept one garbled ("thyroid" heard by CTC, a
non-word by RNNT). This module looks at both against the clinic catalogue and says what it would do:

  keep    the chosen text already names a catalogue entry (or an ambiguous one the existing flow will ask about), or no other text names one;
  switch  the chosen text names nothing, and exactly ONE other heard text names exactly one entry -- that text is the better transcript;
  ask     another heard text fits two or more entries (two doctors named Sen): never guess, the turn should ask "which one?".

A sound-alike is NOT enough to switch (agent/gazetteer.py's rule stands: a suggestion is read back to the caller, never acted on). Plain string work on the CPU: no model call, no GPU, no
network. The clinic API stays the only source of doctors, tests, prices and availability.

PHASE: SHADOW. main.py runs this behind ASR_REFINE=shadow and only LOGS the decision; no caller hears any difference. Only after a week of logs have been scored (tools/asr_eval.py) is a switch
allowed to take effect. Departments are not yet matched (the alias table covers doctors and tests only).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from agent.transcript_rules import DOCTOR, TEST, AliasTable

MODES = ("off", "shadow", "on")


@dataclass(frozen=True)
class Decision:
    action: str  # none | keep | switch | ask
    text: str = ""  # for "switch": the heard text that would replace the chosen one
    kind: str = ""  # doctor | test
    entry: str = ""  # for "switch": the catalogue entry it names; for "ask": the candidates joined by " / "
    reason: str = ""


def parse_mode(value: str | None) -> str:
    v = (value or "off").strip().lower()
    return v if v in MODES else "off"


class AsrRefiner:
    """Built once per catalogue and reused across turns (rebuilding repeats the alias index for nothing)."""

    def __init__(self, cat: dict[str, Any]) -> None:
        self._table = AliasTable(cat)

    def _names(self, text: str, lang: str) -> tuple[list[tuple[str, str]], list[tuple[str, tuple[str, ...]]]]:
        """(exact entries named, ambiguous names) in `text`, doctors and tests only."""
        exact: list[tuple[str, str]] = []
        ambiguous: list[tuple[str, tuple[str, ...]]] = []
        for e in self._table.entities(text, lang):
            if e.kind not in (DOCTOR, TEST):
                continue
            if e.value is not None:
                if (e.kind, e.value) not in exact:
                    exact.append((e.kind, e.value))
            elif e.candidates:
                ambiguous.append((e.kind, tuple(e.candidates)))
        return exact, ambiguous

    def decide(self, chosen: str, alternatives: Sequence[str], lang: str = "bn") -> Decision:
        others = [a for a in dict.fromkeys(a.strip() for a in alternatives if a and a.strip()) if a != (chosen or "").strip()]
        if not others:
            return Decision("none", reason="only one text was heard")
        exact, ambiguous = self._names(chosen or "", lang)
        if exact or ambiguous:
            return Decision("keep", reason="the chosen text already names an entry")
        for alt in others:
            alt_exact, alt_ambiguous = self._names(alt, lang)
            if alt_ambiguous or len(alt_exact) >= 2:
                names = [c for _k, cs in alt_ambiguous for c in cs] or [v for _k, v in alt_exact]
                return Decision("ask", kind=(alt_ambiguous or alt_exact)[0][0], entry=" / ".join(dict.fromkeys(names)), reason="another heard text fits several entries")
            if len(alt_exact) == 1:
                kind, value = alt_exact[0]
                return Decision("switch", text=alt, kind=kind, entry=value, reason="only another heard text names an entry")
        return Decision("keep", reason="no heard text names an entry")
