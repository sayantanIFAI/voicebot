"""Owner, 2026-10-23, item 19: the West Bengal geography tables (districts, police stations, subdivisions/blocks, places), their matching in Bengali, Hindi and English
(romanized, garbled, plain), and the police-station step of a booking using them.

python -m pytest tests/test_geo_tables.py -q
"""

import os
import sys
import tempfile

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLINIC_API_DIR = os.path.join(REPO_ROOT, "clinic-api")
sys.path.insert(0, REPO_ROOT)

from agent import booking_capture as bc  # noqa: E402
from agent.booking_flow import new_state  # noqa: E402


@pytest.fixture(scope="module")
def geo():
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.environ["CLINIC_DB_PATH"] = db_path
    os.environ.pop("DATABASE_URL", None)
    if CLINIC_API_DIR not in sys.path:
        sys.path.insert(0, CLINIC_API_DIR)
    for mod in ("db", "models", "geo_service", "geo_data"):
        sys.modules.pop(mod, None)
    import db as db_mod
    import geo_service
    import models as m

    m.Base.metadata.create_all(db_mod.engine)
    first = geo_service.seed_geo(db_mod.SessionLocal())
    session = db_mod.SessionLocal()
    yield geo_service, session, m, first
    session.close()


def test_the_dataset_is_loaded_and_loading_it_again_adds_nothing(geo):
    gs, db, m, first = geo
    assert first == {"districts": 7, "police_stations": 34, "subdivisions_blocks": 15, "places": 28}
    assert gs.seed_geo(db) == {"districts": 0, "police_stations": 0, "subdivisions_blocks": 0, "places": 0}


def test_the_tables_are_separate_and_joined_by_real_foreign_keys(geo):
    _gs, db, m, _ = geo
    for place in db.query(m.GeoPlace).all():
        if place.police_station_id:
            station = db.get(m.GeoPoliceStation, place.police_station_id)
            assert station.district_id == place.district_id, (
                place.name
            )  # a place's station lies in the place's district
    assert {d.name for d in db.query(m.GeoDistrict).all()} == {
        "Malda",
        "Uttar Dinajpur",
        "Dakshin Dinajpur",
        "Kolkata",
        "Howrah",
        "Hooghly",
        "Murshidabad",
    }
    assert (
        db.query(m.GeoPoliceStation).filter_by(code="PS_0406").one().name == "Kasba PS"
    )  # typed "PS0406 / DIST04" in the source
    assert (
        db.query(m.GeoPlace).filter_by(name="Gariahat").one().police_station_id is None
    )  # the source names a station it never lists: none is invented


STATIONS = [
    ("Shyampukur", "Shyampukur"), ("shampukur thana", "Shyampukur"), ("syampukur police station", "Shyampukur"), ("শ্যামপুকুর", "Shyampukur"), ("শ্যামপুকুর থানা", "Shyampukur"),
    ("श्यामपुकुर", "Shyampukur"), ("श्यामपुकुर थाना", "Shyampukur"), ("salt lake", "Bidhannagar / Salt Lake"), ("সল্টলেক", "Bidhannagar / Salt Lake"), ("साल्ट लेक", "Bidhannagar / Salt Lake"),
    ("park street", "Park Street"), ("পার্ক স্ট্রিট", "Park Street"), ("पार्क स्ट्रीट", "Park Street"), ("chuchura", "Chinsurah"), ("চুঁচুড়া", "Chinsurah"), ("baharampur", "Berhampore"),
    ("বহরমপুর", "Berhampore"), ("बहरमपुर", "Berhampore"), ("englishbazar", "English Bazar"), ("ইংলিশ বাজার", "English Bazar"), ("इंग्लिश बाजार", "English Bazar"), ("kaliyachak", "Kaliachak"),
    ("raiganj", "Raiganj"), ("বালুরঘাট", "Balurghat"), ("गंगारामपुर", "Gangarampur"), ("jangipur", "Jangipur"), ("kasba", "Kasba"), ("কসবা থানা", "Kasba"),
    ("shamppukur", "Shyampukur"), ("jangipore", "Jangipur"), ("nu market", "New Market"), ("kasbaa", "Kasba"),  # a recogniser's garbling: a sound-alike, still a sure one
]  # fmt: skip


@pytest.mark.parametrize("said,station", STATIONS)
def test_a_police_station_said_in_any_language_or_spelling_is_found(geo, said, station):
    gs, db, *_ = geo
    r = gs.match_geo(db, said)
    assert r["matched"] and r["confident"] and r["police_station"] == station, r


PLACES = [
    ("12 Shyambazar Kolkata", "Shyampukur"), ("বাড়ি 12 শ্যামবাজার মোড় কলকাতা", "Shyampukur"), ("श्यामबाजार कोलकाता", "Shyampukur"), ("house 5 near Hastings", "Alipore"),
    ("Howrah Maidan", "Howrah"), ("Nabanna Howrah", "Shibpur"), ("Bandel Hooghly", "Chinsurah"), ("উত্তরপাড়া", "Serampore"), ("Azimganj", "Jiaganj"), ("Esplanade", "New Market"),
    ("Camac Street", "Park Street"), ("Rathbari Malda", "English Bazar"), ("Pandua", "Gazole"), ("Karnajora", "Raiganj"), ("Buniadpur", "Buniadpur"),
]  # fmt: skip


@pytest.mark.parametrize("said,station", PLACES)
def test_a_place_in_an_address_leads_to_its_police_station_by_foreign_key(geo, said, station):
    gs, db, *_ = geo
    r = gs.match_geo(db, said)
    assert r["exact"] and r["police_station"] == station, r
    assert any(c["kind"] in ("place", "police_station") for c in r["candidates"])


def test_a_district_alone_names_no_police_station(geo):
    gs, db, *_ = geo
    for said, district in (
        ("Kolkata", "Kolkata"),
        ("কলকাতা", "Kolkata"),
        ("कोलकाता", "Kolkata"),
        ("hooghly", "Hooghly"),
        ("মুর্শিদাবাদ জেলা", "Murshidabad"),
        ("dakshin dinajpur", "Dakshin Dinajpur"),
    ):
        r = gs.match_geo(db, said)
        assert r["district"] in (district, None) and any(c["district"] == district for c in r["candidates"]), (said, r)
    assert gs.match_geo(db, "Kolkata")["police_station"] is None
    assert gs.match_geo(db, "Gariahat")["police_station"] is None  # a place the source gives no station to


def test_a_name_that_is_in_no_table_is_never_forced_onto_one(geo):
    gs, db, *_ = geo
    for said in (
        "jadavpur",
        "behala",
        "amar bari",
        "12 baker street",
        "road number five",
        "xyzzy",
        "tollygunge",
        "জানি না",
        "पता नहीं",
    ):
        r = gs.match_geo(db, said)
        assert not r["confident"], (said, r)


def test_police_station_words_alone_match_nothing(geo):
    gs, db, *_ = geo
    for said in ("police station", "thana", "থানা", "थाना", "ps"):
        assert not gs.match_geo(db, said)["matched"], said


def _waiting_for_the_station(**slots):
    st = new_state("book_appointment")
    st.slots.update(
        doctor_name="Sen",
        date="2026-10-30",
        time_slot="10:00",
        patient_name="Rahul Dey",
        phone="9876543210",
        patient_age=45,
    )
    st.slots["_ps_pending"] = "Benia Pukur Road, Kolkata"
    st.slots.update(slots)
    return st


def test_the_station_the_tables_recognise_is_stored_under_the_tables_own_name():
    st = _waiting_for_the_station(_ps_match="Shyampukur")
    plan = bc.before_model(st, "shampukur", "en")
    assert plan.skip_model and plan.slots == {"patient_address": "Benia Pukur Road, Kolkata, police station Shyampukur"}
    st = _waiting_for_the_station(_ps_match="Shyampukur")
    assert bc.before_model(st, "শ্যামপুকুর থানা", "bn").slots == {
        "patient_address": "Benia Pukur Road, Kolkata, থানা Shyampukur"
    }
    st = _waiting_for_the_station(_ps_match="Shyampukur")
    assert bc.before_model(st, "श्यामपुकुर", "hi").slots == {
        "patient_address": "Benia Pukur Road, Kolkata, थाना Shyampukur"
    }


def test_a_station_the_tables_do_not_know_is_kept_as_the_caller_said_it():
    st = _waiting_for_the_station()
    assert bc.before_model(st, "Jadavpur", "en").slots == {
        "patient_address": "Benia Pukur Road, Kolkata, police station Jadavpur"
    }


def test_a_caller_who_will_not_say_it_is_still_asked_twice_before_the_place_in_the_address_fills_it_in():
    st = _waiting_for_the_station(_ps_place="Shyampukur")
    assert bc.before_model(st, "jani na", "en").speak == bc.prompt("ps_again", "en")  # the insisting is unchanged
    st.slots["_ps_place"] = "Shyampukur"
    assert bc.before_model(st, "bolbo na", "en").speak == bc.prompt("ps_again", "en")
    st.slots["_ps_place"] = "Shyampukur"
    plan = bc.before_model(st, "nahi bolunga", "en")
    assert plan.skip_model and plan.slots == {"patient_address": "Benia Pukur Road, Kolkata, police station Shyampukur"}


def test_without_a_place_in_the_address_a_final_refusal_leaves_the_address_as_before():
    st = _waiting_for_the_station()
    st.slots["_ps_asks"] = 2
    plan = bc.before_model(st, "jani na", "en")
    assert plan.slots == {"patient_address": "Benia Pukur Road, Kolkata"}


def test_every_station_has_a_bengali_and_a_hindi_name_those_voices_can_say(geo):
    """The Bengali and Hindi voices drop English letters and the whole reply is then blocked (agent/tts.py, KCD-455): the station read back on such a call must be in script."""
    from agent import speech_norm

    gs, db, m, _ = geo
    for station in db.query(m.GeoPoliceStation).all():
        for lang, blob in (("bn", station.aliases_bn), ("hi", station.aliases_hi)):
            name = gs._first(blob)
            assert name, (station.code, lang)
            assert speech_norm.unspeakable_spans(speech_norm.verbalize(f"থানা {name} শুনেছি" if lang == "bn" else f"थाना {name} सुना", lang), lang) == [], (station.code, name)


def test_the_match_hands_back_the_station_in_script_for_a_bengali_or_hindi_call(geo):
    gs, db, *_ = geo
    r = gs.match_geo(db, "shampukur thana")
    assert (r["police_station"], r["police_station_bn"], r["police_station_hi"]) == ("Shyampukur", "শ্যামপুকুর", "श्यामपुकुर")
    r = gs.match_geo(db, "house 5 near Shyambazar Kolkata")  # a place: its station, in script
    assert (r["police_station"], r["police_station_bn"], r["police_station_hi"]) == ("Shyampukur", "শ্যামপুকুর", "श्यामपुकुर")
    r = gs.match_geo(db, "Kolkata")  # a district alone: no station, so no spelling of one either
    assert r["police_station"] is None and r["police_station_bn"] is None and r["police_station_hi"] is None
