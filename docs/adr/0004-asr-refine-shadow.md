# ADR 0004: ASR refinement layer, shadow phase first

Status: accepted (shadow only), 2026-10-28

## Context
`agent/asr.py` runs two decoders (CTC and RNNT) on every clip and kept one text. The cue and alias tables (3,400+ lines) grow with every mis-hearing. `agent/nbest_rescore.py` was written but never used.

## Decision
`ASRResult.alternatives` now carries the other decoder's text when it differs. `agent/asr_refine.py` compares both texts with the clinic catalogue (`AliasTable`, doctors and tests) and decides keep / switch / ask:
it never writes a word of its own, a sound-alike never switches (the gazetteer rule stands), and two or more fitting entries mean ask. Plain string work on the CPU: no model, no GPU, no network.

`ASR_REFINE=off|shadow|on` (`deploy/env.sh`, default shadow on the pod, default off in code). In shadow the layer only writes `ASR_REFINE shadow` lines to the agent log; callers hear no difference. `on` is NOT built; it behaves as shadow.

## Consequences
- Logs carry the decision and the catalogue entry, never the caller's words.
- Next: join the log lines with the labelled outcome of the turn, count good and wrong switches per language (tools/asr_eval.py), set thresholds from that, and only then let a switch take effect. Run the emergency check on every heard text when it does.
- Not covered yet: departments; cross-language runs as extra alternatives; a real N-best list (RNNT beam search is untried).
