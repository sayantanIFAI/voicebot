"""Matching a spoken place, police station or district against the geo_* tables (owner, 2026-10-23, item 19).

WHAT IT IS FOR. A caller gives an address, or answers "which police station?", in Bengali, Hindi or English -- and the recogniser writes it any way it heard it
("শ্যামপুকুর", "shampukur thana", "सल्ट लेक"). The tables hold the real names (geo_data.py, from the West Bengal administrative dataset); this finds which row a
spoken form belongs to, and from a row, its parents by foreign key: a place -> its police station -> its district.

WHAT IT NEVER DOES (CLAUDE.md, the truth boundary). It names no place the tables do not hold and invents no police station: a place with no station in the
dataset resolves to no station. A sound-alike is only ever a CANDIDATE (the same discipline as clinic-api/gazetteer.py); the one thing treated as settled is an
EXACT written-form match, and only when every candidate of the best match agrees on one police station. The agent stores the table's own written name, and speaks
nothing from here.
"""

from __future__ import annotations

import re
import time

import gazetteer as gz
import geo_data
from models import GeoDistrict, GeoPlace, GeoPoliceStation, GeoSubdivisionBlock
from sqlalchemy import func
from sqlalchemy.orm import Session

# Words that say "police station" in the three languages (and "near" / "at"): they carry no place, so they are dropped from the query and from every stored form alike.
_DROP = frozenset(
    {
        "police", "station", "stn", "thana", "thanay", "thanar", "thane", "ps", "p.s", "p.s.", "near", "nearby", "in", "at", "of", "the", "under", "area", "er", "ke", "ka", "ki", "pase", "kache",
        "থানা", "থানায়", "থানার", "থানাতে", "পুলিশ", "স্টেশন", "কাছে", "পাশে", "এলাকা", "এলাকায়",
        "थाना", "थाने", "थानेमें", "पुलिस", "स्टेशन", "के", "का", "की", "पास", "में", "इलाके", "इलाका",
    }
)  # fmt: skip
GEO_TTL_S = 30.0
# A sound-alike is trusted only above this. REASONED, NOT MEASURED -- from a probe over this table: real misspellings ("shamppukur" 0.90, "jangipore" 0.89, "nu market"
# 0.89, "kasbaa" 0.80) scored 0.79-0.90, while words that are no place here ("amar bari", "behala", "12 baker street") scored 0.57-0.70. Recalibrate on real calls.
SOUND_CONFIDENT = 0.78
_cache: dict[tuple, tuple[float, tuple, gz.Gazetteer, dict]] = {}


def invalidate() -> None:
    _cache.clear()


# ------------------------------------------------------------------------------------------------------------------------------ seed
def seed_geo(db: Session) -> dict:
    """Insert every row of geo_data that is missing (matched by its `code`) and fill an alias that was empty. Safe to run on every boot; never deletes or overwrites."""
    added = {"districts": 0, "police_stations": 0, "subdivisions_blocks": 0, "places": 0}
    districts = {d.code: d for d in db.query(GeoDistrict).all()}
    for code, name, division, en, bn, hi in geo_data.DISTRICTS:
        row = districts.get(code)
        if row is None:
            row = GeoDistrict(code=code, name=name, division=division, aliases_en=en, aliases_bn=bn, aliases_hi=hi)
            db.add(row)
            districts[code] = row
            added["districts"] += 1
    db.flush()
    stations = {s.code: s for s in db.query(GeoPoliceStation).all()}
    for code, dist, name, en, bn, hi in geo_data.POLICE_STATIONS:
        if code not in stations:
            row = GeoPoliceStation(
                code=code, district_id=districts[dist].id, name=name, aliases_en=en, aliases_bn=bn, aliases_hi=hi
            )
            db.add(row)
            stations[code] = row
            added["police_stations"] += 1
    db.flush()
    have = {b.code for b in db.query(GeoSubdivisionBlock).all()}
    for code, dist, name, unit in geo_data.SUBDIVISIONS_BLOCKS:
        if code not in have:
            db.add(GeoSubdivisionBlock(code=code, district_id=districts[dist].id, name=name, unit_type=unit))
            added["subdivisions_blocks"] += 1
    have = {p.code for p in db.query(GeoPlace).all()}
    for code, dist, ps, name, ptype, en, bn, hi in geo_data.PLACES:
        if code not in have:
            db.add(
                GeoPlace(
                    code=code,
                    district_id=districts[dist].id,
                    police_station_id=(stations[ps].id if ps else None),
                    name=name,
                    place_type=ptype,
                    aliases_en=en,
                    aliases_bn=bn,
                    aliases_hi=hi,
                )
            )
            added["places"] += 1
    db.commit()
    invalidate()
    return added


# ------------------------------------------------------------------------------------------------------------------------- the gazetteer
def _first(blob: str) -> str:
    return next((a.strip() for a in (blob or "").split("|") if a.strip()), "")


def short_name(name: str) -> str:
    """ "Bidhannagar / Salt Lake PS (Adj.)" -> "Bidhannagar / Salt Lake"; "Shyampukur PS" -> "Shyampukur" (the form spoken after the word "police station")."""
    name = re.sub(r"\(Adj\.?\)", "", name or "", flags=re.I)
    name = re.sub(r"\bPS\b", "", name)
    return " ".join(name.split())


def _forms(name: str, *alias_blobs: str) -> list[str]:
    base = short_name(name)
    forms = [base, re.sub(r"\s*\(.*?\)", "", base).strip()]
    forms += [p.strip() for p in re.split(r"\s*/\s*", base)]
    forms += [m for m in re.findall(r"\((.*?)\)", base)]
    for blob in alias_blobs:
        forms += [a.strip() for a in (blob or "").split("|")]
    seen: list[str] = []
    for f in forms:
        if f and f not in seen:
            seen.append(f)
    return seen


def _build(db: Session) -> tuple[gz.Gazetteer, dict]:
    entries: list[tuple[str, str]] = []
    info: dict[str, dict] = {}
    district = {d.id: d for d in db.query(GeoDistrict).all()}
    station = {s.id: s for s in db.query(GeoPoliceStation).all()}
    for d in district.values():
        key = f"district:{d.code}"
        info[key] = {"kind": "district", "name": d.name, "district": d.name, "police_station": None}
        entries += [(key, f) for f in _forms(d.name, d.aliases_en, d.aliases_bn, d.aliases_hi)]
    for s in station.values():
        key = f"police_station:{s.code}"
        info[key] = {
            "kind": "police_station",
            "name": s.name,
            "district": district[s.district_id].name,
            "police_station": short_name(s.name),
            "police_station_bn": _first(s.aliases_bn),
            "police_station_hi": _first(s.aliases_hi),
        }
        entries += [(key, f) for f in _forms(s.name, s.aliases_en, s.aliases_bn, s.aliases_hi)]
    for b in db.query(GeoSubdivisionBlock).all():
        key = f"block:{b.code}"
        info[key] = {"kind": "block", "name": b.name, "district": district[b.district_id].name, "police_station": None}
        entries += [(key, f) for f in _forms(b.name)]
    for p in db.query(GeoPlace).all():
        key = f"place:{p.code}"
        ps = station.get(p.police_station_id) if p.police_station_id else None
        info[key] = {
            "kind": "place",
            "name": p.name,
            "district": district[p.district_id].name,
            "police_station": short_name(ps.name) if ps else None,
            "police_station_bn": _first(ps.aliases_bn) if ps else None,
            "police_station_hi": _first(ps.aliases_hi) if ps else None,
        }
        entries += [(key, f) for f in _forms(p.name, p.aliases_en, p.aliases_bn, p.aliases_hi)]
    return gz.Gazetteer(entries, drop_words=_DROP), info


def _geo(db: Session) -> tuple[gz.Gazetteer, dict]:
    key = ("geo", id(db.get_bind()))
    stamp = tuple(
        db.query(func.count(m.id)).scalar() for m in (GeoDistrict, GeoPoliceStation, GeoSubdivisionBlock, GeoPlace)
    )
    now = time.monotonic()
    cached = _cache.get(key)
    if cached and cached[1] == stamp and now - cached[0] < GEO_TTL_S:
        return cached[2], cached[3]
    built, info = _build(db)
    _cache[key] = (now, stamp, built, info)
    return built, info


# ----------------------------------------------------------------------------------------------------------------------------- matching
def match_geo(db: Session, text: str, limit: int = 5) -> dict:
    """Which district / police station / block / place a sentence names.

    Any run of up to three words that EXACTLY equals a stored form counts, wherever it stands in the sentence ("near Shyambazar five point crossing"). A sound-alike or
    near spelling counts only when the whole answer is that one name (three words or fewer: the reply to "which police station?"). Result:
      matched        some row was found
      exact          the best match is a written-form match
      confident      safe to store the table's name for it: an exact match, or a sound-alike scoring at least SOUND_CONFIDENT
      police_station the police station (written name, without "PS") when the best match gives ONE -- a station is itself, a place's is its parent; else None
      district       the district when every best match agrees on one, else None
      candidates     the best matches, strongest first: kind, name, district, police_station, basis, score
    """
    gaz, info = _geo(db)
    words = gz.normalise(text or "", _DROP).split()
    if not words or not len(gaz):
        return {
            "matched": False,
            "exact": False,
            "confident": False,
            "police_station": None,
            "district": None,
            "candidates": [],
        }
    exact: dict[int, dict[str, float]] = {}
    fuzzy: dict[str, float] = {}
    for n in (3, 2, 1):
        for i in range(0, len(words) - n + 1):
            phrase = " ".join(words[i : i + n])
            for s in gaz.suggest(phrase, limit=4):
                if s.basis == "exact":
                    exact.setdefault(n, {})[s.canonical] = max(exact.get(n, {}).get(s.canonical, 0.0), s.score)
                elif len(words) <= 3 and n == len(words):
                    fuzzy[s.canonical] = max(fuzzy.get(s.canonical, 0.0), s.score)
    if exact:
        best = exact[max(exact)]  # the longest exact phrase: "howrah maidan" beats "howrah"
        basis = "exact"
    elif fuzzy:
        top = max(fuzzy.values())
        best = {k: v for k, v in fuzzy.items() if v >= top - gz.TIE_MARGIN}
        basis = "sound"
    else:
        return {
            "matched": False,
            "exact": False,
            "confident": False,
            "police_station": None,
            "district": None,
            "candidates": [],
        }
    keys = sorted(best, key=lambda k: -best[k])[:limit]
    candidates = [{**info[k], "code": k.split(":", 1)[1], "basis": basis, "score": round(best[k], 3)} for k in keys]
    # a place is more specific than a district or station word beside it ("Nabanna, Howrah": the place's station, not the district's namesake)
    pool = [c for c in candidates if c["kind"] == "place"] or [
        c for c in candidates if c["kind"] in ("police_station", "place")
    ]
    stations = {c["police_station"] for c in pool if c["police_station"]}
    districts = {c["district"] for c in candidates}
    one = len(stations) == 1 and all(c["police_station"] for c in pool)
    named = next((c for c in pool if c["police_station"]), None) if one else None
    return {
        "matched": True,
        "exact": basis == "exact",
        "confident": basis == "exact" or best[keys[0]] >= SOUND_CONFIDENT,
        "police_station": next(iter(stations)) if one else None,
        # the same station as a Bengali / Hindi voice can SAY it: those voices drop English letters, which blocks the whole reply
        "police_station_bn": (named or {}).get("police_station_bn") or None,
        "police_station_hi": (named or {}).get("police_station_hi") or None,
        "district": next(iter(districts)) if len(districts) == 1 else None,
        "candidates": candidates,
    }
