"""Print every word the Bengali respelling layer changes, for a native reviewer to veto: python tools/respell_audit.py"""
import sys

sys.path.insert(0, ".")
from agent import bn_respell  # noqa: E402

print("EXPLICIT TABLE (chosen by a listener, or the same sound as one):")
for word, spelling in bn_respell.table().items():
    print(f"  {word:12} -> {spelling}")
print("\nSH RULE (native words only; plain স -> শ):")
for word, spelling in bn_respell.rule_changes().items():
    print(f"  {word:14} -> {spelling}")
print(f"\n{len(bn_respell.table())} explicit, {len(bn_respell.rule_changes())} by rule")
