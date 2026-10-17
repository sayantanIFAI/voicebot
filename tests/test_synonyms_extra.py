"""Each lab test has five MORE names in each of Bengali, Hindi and English (owner's request 2026-10-17), and no name belongs to two tests."""
import importlib
import pathlib
import sys
import unicodedata
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "clinic-api"))
sx = importlib.import_module("synonyms_extra")
i18n = importlib.import_module("i18n_content")
sf = importlib.import_module("spelled_forms")


def _seed_tests():
    import ast
    src = (ROOT / "clinic-api" / "seed.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "LAB_TESTS" for t in node.targets):
            return [(row[0], list(row[1])) for row in ast.literal_eval(node.value)]
    raise AssertionError("LAB_TESTS not found")


def _norm(s):
    return " ".join(unicodedata.normalize("NFC", s).lower().split())


TESTS = _seed_tests()


def test_every_test_has_five_names_in_each_language():
    for name, _ in TESTS:
        assert name in sx.EXTRA, f"{name}: no extra synonyms"
        for lang in ("bn", "hi", "en"):
            forms = sx.extra(name, lang)
            assert len(set(map(_norm, forms))) >= 5, (name, lang, forms)


def test_no_extra_name_is_in_the_wrong_script():
    for name, d in sx.EXTRA.items():
        for f in d["bn"]:
            assert any("\u0980" <= c <= "\u09ff" for c in f), (name, f)
        for f in d["hi"]:
            assert any("\u0900" <= c <= "\u097f" for c in f), (name, f)
        for f in d["en"]:
            assert f.isascii(), (name, f)


def test_extra_keys_are_real_tests():
    names = {n for n, _ in TESTS}
    assert set(sx.EXTRA) <= names, set(sx.EXTRA) - names


def test_a_new_name_belongs_to_one_test_only():
    """Every NEW name (the extras here and the generated spelled-letter forms) is held by exactly one test. A name the seed already gave to two
    tests on purpose (a bare "ইউএসজি" means either ultrasound, so the caller is asked which) is not this test's concern."""
    owners = defaultdict(set)
    new_forms = set()
    for name, bn in TESTS:
        old = list(bn) + i18n.TEST_ALIASES_HI.get(name, []) + i18n.TEST_ALIASES_EN.get(name, [])
        new = sf.forms_bn(name) + sf.forms_hi(name) + [f for lang in ("bn", "hi", "en") for f in sx.extra(name, lang)]
        for f in old + new:
            owners[_norm(f)].add(name)
        new_forms |= {_norm(f) for f in new}
    clashes = {f: sorted(owners[f]) for f in new_forms if len(owners[f]) > 1}
    assert not clashes, clashes
