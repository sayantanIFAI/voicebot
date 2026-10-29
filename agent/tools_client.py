"""REST client for the clinic's Java (Spring Boot) + PostgreSQL service.

This service owns the actual facts -- prices, schedules, slot availability
-- and is the only thing allowed to state them. The contract below is what
that service needs to implement; nothing here assumes it exists yet.

Every method returns a plain dict and NEVER raises for a normal "not
found" / "unavailable" outcome -- those are valid, expected answers a
caller can be told. It only raises ToolCallError for actual infrastructure
failure (timeout, connection refused, 5xx), which main.py maps to a
distinct "I couldn't check that right now" reply instead of a false
"not found".
"""

from __future__ import annotations

import os
import time
import uuid

import asyncio

import httpx
from pydantic import BaseModel

from agent.api_models import (
    ConflictAnswer,
    DoctorAnswer,
    FaqAnswer,
    FindAnswer,
    FoundAnswer,
    IdentifyAnswer,
    LookupAnswer,
    PrepAnswer,
    RouteAnswer,
    SuccessAnswer,
    TestAnswer,
    NextSerialAnswer,
    TestDaysAnswer,
    VerifyAnswer,
    validated,
)

DEFAULT_TIMEOUT_S = 4.0  # a phone caller will not wait much longer than this per lookup


class ToolCallError(Exception):
    """The backing service itself failed -- distinct from a normal
    not-found/unavailable result, which is not an error."""


class ClinicToolsClient:
    def __init__(self, base_url: str, timeout_s: float = DEFAULT_TIMEOUT_S):
        self.base_url = base_url.rstrip("/")
        # The clinic API's service token (clinic-api/main.py), read from the environment, never logged.
        token = os.environ.get("CLINIC_API_TOKEN", "")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout_s, headers=headers)

    async def aclose(self):
        await self._client.aclose()

    # ---- writes: ONE Idempotency-Key per operation, reused on every retry of it (review 2026-10-28) ----
    # The clinic API claims the key before it runs the operation (clinic-api/idempotency.py), so a retry after a timeout cannot run a booking, hold, cancel,
    # reschedule or message twice: if the first attempt finished, the retry gets its stored answer; if it is still running the API answers 409 and the retry waits.
    # An attempt that could not reach the server is repeated; an attempt that MAY have been executed (a read timeout) is repeated only for endpoints that carry the guard.
    # When the outcome is still unknown after that, the caller gets a ToolCallError -- never a guessed success.
    _GUARDED_PATHS = frozenset(
        {
            "/api/v1/appointments", "/api/v1/bookings/hold", "/api/v1/bookings/confirm", "/api/v1/bookings/reschedule", "/api/v1/bookings/cancel", "/api/v1/patients/senior",
            "/api/v1/bookings/tests", "/api/v1/bookings/add-test", "/api/v1/doctors/leave", "/api/v1/notifications/sms", "/api/v1/bookings/resend", "/api/v1/bookings/draft",
            "/api/v1/calls/out-of-scope", "/api/v1/callbacks", "/api/v1/payments/send-link", "/api/v1/bookings/serial/resend", "/api/v1/bookings/send-whatsapp-confirmation",
            "/api/v1/reports/request-otp", "/api/v1/reports/deliver",
        }
    )
    _WRITE_RETRIES = 1
    _IN_PROGRESS_WAITS = (0.4, 0.8, 1.2)

    @classmethod
    def _is_guarded(cls, path: str) -> bool:
        return path in cls._GUARDED_PATHS or (path.startswith("/api/v1/patients/") and path.endswith("/preferences"))

    async def _write(self, path: str, json=None, headers=None):
        key = uuid.uuid4().hex  # one key for this operation, whatever happens below
        merged = {**(headers or {}), "Idempotency-Key": key}
        guarded = self._is_guarded(path)
        attempts_left = (self._WRITE_RETRIES if guarded else 0) + 1
        waits = list(self._IN_PROGRESS_WAITS)
        while True:
            try:
                r = await self._client.post(path, json=json, headers=merged)
            except (httpx.ConnectError, httpx.ConnectTimeout):
                attempts_left -= 1  # never reached the server: safe to repeat
                if attempts_left <= 0 or not guarded:
                    raise
                await asyncio.sleep(0.3)
                continue
            except (httpx.ReadTimeout, httpx.RemoteProtocolError, httpx.ReadError, httpx.WriteError):
                attempts_left -= 1  # it may have been executed: only a guarded endpoint answers the same key with the stored result
                if attempts_left <= 0 or not guarded:
                    raise
                await asyncio.sleep(0.3)
                continue
            if r.status_code == 409 and guarded and waits:
                await asyncio.sleep(waits.pop(0))  # the first attempt is still running on the server: ask again with the same key
                continue
            return r

    # ---- Tool 1: GET /api/v1/tests/search?name=... ----
    # Expected response shape:
    #   found=true:  {"found": true, "test_name": "...", "rate_inr": 650,
    #                 "sample_type": "Blood", "report_time_hours": 24}
    #   found=false: {"found": false, "query": "...", "did_you_mean": ["..."]}
    async def get_test_rate(self, test_name: str) -> dict:
        try:
            r = await self._client.get("/api/v1/tests/search", params={"name": test_name})
            r.raise_for_status()
            return validated(TestAnswer, r.json(), "get_test_rate")
        except (httpx.HTTPError, ValueError) as e:  # ValueError: not JSON, or not the expected shape
            raise ToolCallError(f"get_test_rate({test_name!r}): {e}") from e

    # ---- Tool 2: GET /api/v1/doctors/availability?name=...&date=YYYY-MM-DD ----
    # date is OPTIONAL -- omit it to ask "when is this doctor next available".
    # Expected response shape:
    #   found=true:  {"found": true, "doctor_name": "...", "date": "...",
    #                 "available": true, "chamber_hours": "18:00-20:00",
    #                 "next_available_date": null}
    #                or, if not available that date:
    #                {"found": true, ..., "available": false,
    #                 "next_available_date": "2026-08-27"}
    #   found=false: {"found": false, "query": "..."}
    # Owner, 2026-10-29: "pre-load all the doctors' availability for today and tomorrow so that no database hit is needed". preload_availability() fills this for every doctor (by short and whole name) for
    # today, tomorrow and "next sitting" (no date), and is repeated in the background every few seconds (main._availability_preload_loop), so a turn finds its answer here. A name or date not in it (a spoken
    # variant, a later date) goes to the clinic API exactly as before. An entry older than AVAILABILITY_TTL_S is not used: today's answer changes as a chamber closes.
    AVAILABILITY_TTL_S = float(os.environ.get("AVAILABILITY_TTL_S", "45"))

    async def get_doctor_availability(self, doctor_name: str, date: str | None) -> dict:
        cache = self.__dict__.setdefault("_availability", {})
        hit = cache.get((doctor_name, date))
        if hit is not None and time.monotonic() - hit[0] <= self.AVAILABILITY_TTL_S:
            return hit[1]
        return await self._fetch_availability(doctor_name, date)

    async def _fetch_availability(self, doctor_name: str, date: str | None) -> dict:
        params = {"name": doctor_name}
        if date:
            params["date"] = date
        try:
            r = await self._client.get("/api/v1/doctors/availability", params=params)
            r.raise_for_status()
            return validated(DoctorAnswer, r.json(), "get_doctor_availability")
        except (httpx.HTTPError, ValueError) as e:  # ValueError: not JSON, or not the expected shape
            raise ToolCallError(f"get_doctor_availability({doctor_name!r}, {date!r}): {e}") from e

    async def preload_availability(self, names: list[str], dates: list[str | None], concurrency: int = 8) -> int:
        """Fetch every (name, date) and keep it; returns how many were stored. A failed fetch keeps the older entry (it simply expires)."""
        cache = self.__dict__.setdefault("_availability", {})
        gate = asyncio.Semaphore(concurrency)
        stored = 0

        async def one(name: str, date: str | None) -> None:
            nonlocal stored
            async with gate:
                try:
                    value = await self._fetch_availability(name, date)
                except ToolCallError:
                    return
                cache[(name, date)] = (time.monotonic(), value)
                stored += 1

        await asyncio.gather(*[one(n, d) for n in dict.fromkeys(names) for d in dates])
        return stored

    async def free_slots(self, doctor_name: str, date: str) -> dict:
        """The doctor's still-free slots on a date (GET /api/v1/doctors/free-slots); the agent chooses the time from these."""
        try:
            r = await self._client.get("/api/v1/doctors/free-slots", params={"name": doctor_name, "date": date})
            r.raise_for_status()
            data = r.json()
            if not isinstance(data, dict):
                raise ValueError("free-slots did not return an object")
            return data
        except (httpx.HTTPError, ValueError) as e:
            raise ToolCallError(f"free_slots({doctor_name!r}, {date!r}): {e}") from e

    async def doctor_weekdays(self, doctor_name: str) -> dict:
        """The weekdays a doctor sits, with hours (GET /api/v1/doctors/weekdays) -- the weekly pattern, never dates."""
        try:
            r = await self._client.get("/api/v1/doctors/weekdays", params={"name": doctor_name})
            r.raise_for_status()
            data = r.json()
            if not isinstance(data, dict):
                raise ValueError("weekdays did not return an object")
            return data
        except (httpx.HTTPError, ValueError) as e:
            raise ToolCallError(f"doctor_weekdays({doctor_name!r}): {e}") from e

    async def geo_match(self, text: str) -> dict:
        """Item 19: which district / police station / block / place of the clinic's geography tables a phrase names, and the police station that gives
        (GET /api/v1/geo/match). Candidates only -- see clinic-api/geo_service.py."""
        try:
            r = await self._client.get("/api/v1/geo/match", params={"q": text})
            r.raise_for_status()
            data = r.json()
            if not isinstance(data, dict):
                raise ValueError("geo/match did not return an object")
            return data
        except (httpx.HTTPError, ValueError) as e:
            raise ToolCallError(f"geo_match({text!r}): {e}") from e

    async def department_alternatives(self, doctor_name: str, date: str, lang: str = "bn") -> dict:
        """KCD-lead-6: other doctors in `doctor_name`'s department who actually have a free slot on `date`
        (GET /api/v1/doctors/department-alternatives) -- used when that doctor is full that day and the caller
        insists on the same day instead of the next day this doctor sits."""
        try:
            r = await self._client.get(
                "/api/v1/doctors/department-alternatives", params={"name": doctor_name, "date": date, "lang": lang}
            )
            r.raise_for_status()
            data = r.json()
            if not isinstance(data, dict):
                raise ValueError("department-alternatives did not return an object")
            return data
        except (httpx.HTTPError, ValueError) as e:
            raise ToolCallError(f"department_alternatives({doctor_name!r}, {date!r}): {e}") from e

    # ---- Tool 4: GET /api/v1/tests/prep?name=... ----
    # Expected response shape:
    #   found=true:  {"found": true, "test_name": "...", "test_name_bn": "...",
    #                 "fasting_required": true, "prep_instructions": "..."}
    #   found=false: {"found": false, "query": "...", "did_you_mean": ["..."]}
    async def get_test_prep(self, test_name: str, lang: str = "bn") -> dict:
        try:
            r = await self._client.get("/api/v1/tests/prep", params={"name": test_name, "lang": lang})
            r.raise_for_status()
            return validated(PrepAnswer, r.json(), "get_test_prep")
        except (httpx.HTTPError, ValueError) as e:  # ValueError: not JSON, or not the expected shape
            raise ToolCallError(f"get_test_prep({test_name!r}): {e}") from e

    # ---- Tool 5: GET /api/v1/faq?topic=... ----
    # `topic` is the STABLE KEY from /api/v1/catalogue's faq_topics, already
    # resolved by agent/fast_path.py's FAQCatalogue -- never free caller text.
    # Expected response shape:
    #   found=true:  {"found": true, "topic": "hours", "answer": "..."}
    #   found=false: {"found": false, "topic": "..."}
    async def get_faq(self, topic: str, lang: str = "bn") -> dict:
        # The FAQ rows are fixed text that changes only when the clinic edits it: remembered for a minute (24 callers asking the address together made 24 identical lookups).
        cache = self.__dict__.setdefault("_faq_cache", {})
        hit = cache.get((topic, lang))
        if hit is not None and time.monotonic() - hit[0] < 60.0:
            return hit[1]
        try:
            r = await self._client.get("/api/v1/faq", params={"topic": topic, "lang": lang})
            r.raise_for_status()
            result = validated(FaqAnswer, r.json(), "get_faq")
            cache[(topic, lang)] = (time.monotonic(), result)
            return result
        except (httpx.HTTPError, ValueError) as e:  # ValueError: not JSON, or not the expected shape
            raise ToolCallError(f"get_faq({topic!r}): {e}") from e

    # ---- Tool 3: POST /api/v1/appointments ----
    # Body: {"doctor_name", "date", "time_slot", "patient_name", "phone"}
    # Expected response shape:
    #   success=true:  {"success": true, "confirmation_id": "KCD-20260824-0031",
    #                    "doctor_name": "...", "date": "...", "time_slot": "..."}
    #   success=false: {"success": false, "reason": "slot_taken" | "missing_field" | "doctor_not_found",
    #                    "alternative_slots": ["17:30", "18:15"]}
    async def book_appointment(
        self, doctor_name: str, date: str, time_slot: str, patient_name: str, phone: str
    ) -> dict:
        body = {
            "doctor_name": doctor_name,
            "date": date,
            "time_slot": time_slot,
            "patient_name": patient_name,
            "phone": phone,
        }
        try:
            r = await self._write("/api/v1/appointments", json=body)
            r.raise_for_status()
            return validated(SuccessAnswer, r.json(), "book_appointment")
        except (httpx.HTTPError, ValueError) as e:  # ValueError: not JSON, or not the expected shape
            raise ToolCallError(f"book_appointment({body!r}): {e}") from e

    # =========================================================================
    # Epic E26: booking, rescheduling and cancellation. Two-phase (hold, then
    # confirm) for the same reason /api/v1/bookings/hold itself is two-phase
    # -- see clinic-api/booking_service.py and models.SlotLock's docstrings
    # for the concurrency reasoning (KCD-376).
    # =========================================================================
    async def hold_slot(self, doctor_name: str, date: str, time_slot: str) -> dict:
        body = {"doctor_name": doctor_name, "date": date, "time_slot": time_slot}
        try:
            r = await self._write("/api/v1/bookings/hold", json=body)
            r.raise_for_status()
            return validated(SuccessAnswer, r.json(), "hold_slot")
        except (httpx.HTTPError, ValueError) as e:  # ValueError: not JSON, or not the expected shape
            raise ToolCallError(f"hold_slot({body!r}): {e}") from e

    async def confirm_booking(
        self,
        hold_token: str,
        doctor_id: int,
        date: str,
        time_slot: str,
        patient_name: str,
        phone: str,
        caller_phone: str,
        patient_age: int | None = None,
        relationship: str = "self",
        patient_address: str = "",
        patient_sex: str = "",
    ) -> dict:
        body = {
            "hold_token": hold_token,
            "doctor_id": doctor_id,
            "date": date,
            "time_slot": time_slot,
            "patient_name": patient_name,
            "phone": phone,
            "caller_phone": caller_phone,
            "patient_age": patient_age,
            "relationship": relationship,
            "patient_address": patient_address or "",
            "patient_sex": patient_sex or "",
        }
        try:
            r = await self._write("/api/v1/bookings/confirm", json=body)
            r.raise_for_status()
            return validated(SuccessAnswer, r.json(), "confirm_booking")
        except (httpx.HTTPError, ValueError) as e:  # ValueError: not JSON, or not the expected shape
            raise ToolCallError(f"confirm_booking({body!r}): {e}") from e

    async def reschedule_appointment(self, confirmation_id: str, new_date: str, new_time_slot: str, caller_phone: str | None = None, call_id: str | None = None) -> dict:
        body = {"confirmation_id": confirmation_id, "new_date": new_date, "new_time_slot": new_time_slot}
        if caller_phone:
            body["caller_phone"] = caller_phone
        if call_id:
            body["call_id"] = call_id
        try:
            r = await self._write("/api/v1/bookings/reschedule", json=body)
            r.raise_for_status()
            return validated(SuccessAnswer, r.json(), "reschedule_appointment")
        except (httpx.HTTPError, ValueError) as e:  # ValueError: not JSON, or not the expected shape
            raise ToolCallError(f"reschedule_appointment({body!r}): {e}") from e

    async def cancel_appointment(self, confirmation_id: str, confirm_charge: bool = False, caller_phone: str | None = None, call_id: str | None = None) -> dict:
        body = {"confirmation_id": confirmation_id, "confirm_charge": confirm_charge}
        if caller_phone:
            body["caller_phone"] = caller_phone
        if call_id:
            body["call_id"] = call_id
        try:
            r = await self._write("/api/v1/bookings/cancel", json=body)
            r.raise_for_status()
            return validated(SuccessAnswer, r.json(), "cancel_appointment")
        except (httpx.HTTPError, ValueError) as e:  # ValueError: not JSON, or not the expected shape
            raise ToolCallError(f"cancel_appointment({body!r}): {e}") from e

    async def lookup_bookings(
        self, phone: str | None = None, confirmation_id: str | None = None, name: str | None = None
    ) -> dict:
        params = {k: v for k, v in {"phone": phone, "confirmation_id": confirmation_id, "name": name}.items() if v}
        try:
            r = await self._client.get("/api/v1/bookings/lookup", params=params)
            r.raise_for_status()
            return validated(LookupAnswer, r.json(), "lookup_bookings")
        except (httpx.HTTPError, ValueError) as e:  # ValueError: not JSON, or not the expected shape
            raise ToolCallError(f"lookup_bookings({params!r}): {e}") from e

    # KCD-084: senior mode persisted against the patient (a boolean only).
    async def set_patient_senior(self, phone: str, senior: bool = True, caller_phone: str | None = None) -> dict:
        try:
            r = await self._write(
                "/api/v1/patients/senior", json={"phone": phone, "senior": senior, "caller_phone": caller_phone}
            )
            r.raise_for_status()
            return r.json()
        except (httpx.HTTPError, ValueError) as e:  # ValueError: not JSON
            raise ToolCallError(f"set_patient_senior: {e}") from e

    async def get_patient_senior(self, phone: str, caller_phone: str | None = None) -> bool:
        try:
            r = await self._client.get(
                "/api/v1/patients/senior",
                params={"phone": phone, **({"caller_phone": caller_phone} if caller_phone else {})},
            )
            r.raise_for_status()
            return bool(r.json().get("senior"))
        except (httpx.HTTPError, ValueError) as e:  # ValueError: not JSON
            raise ToolCallError(f"get_patient_senior: {e}") from e

    async def booking_conflict(self, phone: str, date: str, time_slot: str) -> dict:
        params = {"phone": phone, "date": date, "time_slot": time_slot}
        try:
            r = await self._client.get("/api/v1/bookings/conflict", params=params)
            r.raise_for_status()
            return validated(ConflictAnswer, r.json(), "booking_conflict")
        except (httpx.HTTPError, ValueError) as e:  # ValueError: not JSON, or not the expected shape
            raise ToolCallError(f"booking_conflict({params!r}): {e}") from e

    async def book_tests(
        self,
        test_names: list[str],
        date: str,
        patient_name: str,
        phone: str,
        caller_phone: str,
        patient_age: int | None = None,
        relationship: str = "self",
    ) -> dict:
        body = {
            "test_names": test_names,
            "date": date,
            "patient_name": patient_name,
            "phone": phone,
            "caller_phone": caller_phone,
            "patient_age": patient_age,
            "relationship": relationship,
        }
        try:
            r = await self._write("/api/v1/bookings/tests", json=body)
            r.raise_for_status()
            return validated(SuccessAnswer, r.json(), "book_tests")
        except (httpx.HTTPError, ValueError) as e:  # ValueError: not JSON, or not the expected shape
            raise ToolCallError(f"book_tests({body!r}): {e}") from e

    async def add_test_to_booking(self, confirmation_id: str, test_name: str) -> dict:
        body = {"confirmation_id": confirmation_id, "test_name": test_name}
        try:
            r = await self._write("/api/v1/bookings/add-test", json=body)
            r.raise_for_status()
            return validated(SuccessAnswer, r.json(), "add_test_to_booking")
        except (httpx.HTTPError, ValueError) as e:  # ValueError: not JSON, or not the expected shape
            raise ToolCallError(f"add_test_to_booking({body!r}): {e}") from e

    async def doctor_earliest(self, doctor_name: str) -> dict:
        try:
            r = await self._client.get("/api/v1/doctors/earliest", params={"name": doctor_name})
            r.raise_for_status()
            return validated(FoundAnswer, r.json(), "doctor_earliest")
        except (httpx.HTTPError, ValueError) as e:  # ValueError: not JSON, or not the expected shape
            raise ToolCallError(f"doctor_earliest({doctor_name!r}): {e}") from e

    async def route_department(self, query_text: str, lang: str = "bn", age: int | None = None) -> dict:
        params = {"query": query_text, "lang": lang}
        if age is not None:
            params["age"] = age  # sourav, 2026-10-24: 13 or more is never sent to the paediatrician (clinic-api/lay_departments.py)
        try:
            r = await self._client.get("/api/v1/departments/route", params=params)
            r.raise_for_status()
            return validated(RouteAnswer, r.json(), "route_department")
        except (httpx.HTTPError, ValueError) as e:  # ValueError: not JSON, or not the expected shape
            raise ToolCallError(f"route_department({params!r}): {e}") from e

    # KCD-dentistry: the real doctors in a NAMED department -- used for the caller who gave no usable symptom
    # (route_department found nothing to match) and no doctor name either, routed to a general checkup instead.
    async def department_doctors(
        self, department_name: str, lang: str = "bn", sex: str | None = None, on_date: str | None = None, at_time: str | None = None
    ) -> dict:
        params = {"name": department_name, "lang": lang}
        if sex:
            params["sex"] = sex  # "F" | "M": a lady / gentleman doctor (owner, 2026-10-18)
        if on_date:
            params["on_date"] = on_date
        if at_time:
            params["at_time"] = at_time
        try:
            r = await self._client.get("/api/v1/departments/doctors", params=params)
            r.raise_for_status()
            return validated(RouteAnswer, r.json(), "department_doctors")
        except (httpx.HTTPError, ValueError) as e:
            raise ToolCallError(f"department_doctors({params!r}): {e}") from e

    async def doctor_department(self, name: str) -> dict:
        """sourav, 2026-10-24: {found, department_name} of a named doctor (used by the paediatric age rule)."""
        try:
            r = await self._client.get("/api/v1/doctors/department", params={"name": name})
            r.raise_for_status()
            return validated(FoundAnswer, r.json(), "doctor_department")
        except (httpx.HTTPError, ValueError) as e:
            raise ToolCallError(f"doctor_department({name!r}): {e}") from e

    async def doctor_visit(self, name: str) -> dict:
        """What a doctor's visit costs, from the doctors table."""
        try:
            r = await self._client.get("/api/v1/doctors/visit", params={"name": name})
            r.raise_for_status()
            return r.json()
        except (httpx.HTTPError, ValueError) as e:
            raise ToolCallError(f"doctor_visit({name!r}): {e}") from e

    async def relationships(self) -> dict:
        """The relationship words and the sex each implies (clinic-api relationship_rules)."""
        try:
            r = await self._client.get("/api/v1/relationships")
            r.raise_for_status()
            return r.json()
        except (httpx.HTTPError, ValueError) as e:
            raise ToolCallError(f"relationships(): {e}") from e

    async def clinic_cancelled(self, phone: str | None = None, confirmation_id: str | None = None) -> dict:
        """Bookings the clinic itself cancelled (a doctor on leave) for this phone / confirmation id."""
        params = {k: v for k, v in (("phone", phone), ("confirmation_id", confirmation_id)) if v}
        try:
            r = await self._client.get("/api/v1/bookings/cancelled-by-clinic", params=params)
            r.raise_for_status()
            return r.json()
        except (httpx.HTTPError, ValueError) as e:
            raise ToolCallError(f"clinic_cancelled({params!r}): {e}") from e

    async def resend_confirmation(self, confirmation_id: str) -> dict:
        try:
            r = await self._write(
                "/api/v1/bookings/resend",
                json={"confirmation_id": confirmation_id},
            )
            r.raise_for_status()
            return validated(SuccessAnswer, r.json(), "resend_confirmation")
        except (httpx.HTTPError, ValueError) as e:  # ValueError: not JSON, or not the expected shape
            raise ToolCallError(f"resend_confirmation({confirmation_id!r}): {e}") from e

    async def request_callback(
        self, phone: str, call_id: str, requested_window: str, reason: str = "", call_summary: str = ""
    ) -> dict:
        """Epic E27's callback queue (clinic-api/enquiry_service.request_callback): the durable half of "a human
        colleague will call you back" -- there is no outbound-calling system, so this only records the promise, never
        claims a call was placed. `call_summary` (agent/call_summary.py) rides along so the human has the call's
        context next to the phone number."""
        body = {
            "phone": phone,
            "call_id": call_id,
            "requested_window": requested_window,
            "reason": reason,
            "call_summary": call_summary,
        }
        try:
            r = await self._write("/api/v1/callbacks", json=body)
            r.raise_for_status()
            return validated(SuccessAnswer, r.json(), "request_callback")
        except (httpx.HTTPError, ValueError) as e:  # ValueError: not JSON, or not the expected shape
            raise ToolCallError(f"request_callback({body!r}): {e}") from e

    async def next_serial(self, doctor_name: str, date: str) -> dict:
        """sourav, 2026-10-24: "if I book today, what will my serial number be?" -- the number the next booking of this doctor on this date would get (clinic-api
        /api/v1/bookings/next-serial; serials are simulated). found, doctor_name(_bn/_hi), date, available, next_serial, next_available_date."""
        try:
            r = await self._client.get("/api/v1/bookings/next-serial", params={"name": doctor_name, "date": date})
            r.raise_for_status()
            return validated(NextSerialAnswer, r.json(), "next_serial")
        except (httpx.HTTPError, ValueError) as e:  # ValueError: not JSON, or not the expected shape
            raise ToolCallError(f"next_serial({doctor_name!r}): {e}") from e

    async def get_test_days(self, test_name: str) -> dict:
        """sourav, 2026-10-24: the weekdays a lab test is done on (clinic-api /api/v1/tests/availability). Why: "is the CT scan done on Thursday?" is answered from the table.
        The answer has found, test_name(_bn/_hi), available_days (Monday = 0), closed_days, every_day."""
        try:
            r = await self._client.get("/api/v1/tests/availability", params={"name": test_name})
            r.raise_for_status()
            return validated(TestDaysAnswer, r.json(), "get_test_days")
        except (httpx.HTTPError, ValueError) as e:  # ValueError: not JSON, or not the expected shape
            raise ToolCallError(f"get_test_days({test_name!r}): {e}") from e

    async def lookup_serial(self, phone: str) -> dict:
        """sourav, 2026-10-24: "I paid but my serial number never came on WhatsApp" -- what the database holds for this (confirmed) mobile number: found, bookings[] each with
        confirmation_id, doctor_name, date, time_slot, patient_name, paid, serial_number (clinic-api /api/v1/bookings/serial)."""
        try:
            r = await self._client.get("/api/v1/bookings/serial", params={"phone": phone})
            r.raise_for_status()
            return validated(LookupAnswer, r.json(), "lookup_serial")
        except (httpx.HTTPError, ValueError) as e:  # ValueError: not JSON, or not the expected shape
            raise ToolCallError(f"lookup_serial: {e}") from e

    async def resend_serial(self, phone: str, confirmation_id: str) -> dict:
        """sourav, 2026-10-24: queue a paid booking's serial number to its WhatsApp number again (clinic-api /api/v1/bookings/serial/resend). The channel is a placeholder: the
        answer says "success" when it was QUEUED, never that it reached the phone."""
        try:
            r = await self._write(
                "/api/v1/bookings/serial/resend",
                json={"phone": phone, "confirmation_id": confirmation_id},
            )
            r.raise_for_status()
            return validated(SuccessAnswer, r.json(), "resend_serial")
        except (httpx.HTTPError, ValueError) as e:  # ValueError: not JSON, or not the expected shape
            raise ToolCallError(f"resend_serial({confirmation_id!r}): {e}") from e

    async def send_payment_link(self, confirmation_id: str) -> dict:
        """Only a confirmed DOCTOR appointment ever gets a link (clinic-api/booking_service.send_payment_link) -- a
        lab test's confirmation_id belongs to a different table there and is never found, by design: a lab test is
        paid and booked at the counter, never over the call."""
        try:
            r = await self._write(
                "/api/v1/payments/send-link",
                json={"confirmation_id": confirmation_id},
            )
            r.raise_for_status()
            return validated(SuccessAnswer, r.json(), "send_payment_link")
        except (httpx.HTTPError, ValueError) as e:  # ValueError: not JSON, or not the expected shape
            raise ToolCallError(f"send_payment_link({confirmation_id!r}): {e}") from e

    async def send_booking_confirmation_whatsapp(self, confirmation_id: str) -> dict:
        """KCD-lead-2: the warm/friendly WhatsApp confirmation template (clinic-api/booking_service.
        send_booking_confirmation_whatsapp), sent once a doctor appointment is confirmed -- separate from
        send_payment_link's own message above."""
        try:
            r = await self._write(
                "/api/v1/bookings/send-whatsapp-confirmation",
                json={"confirmation_id": confirmation_id},
            )
            r.raise_for_status()
            return validated(SuccessAnswer, r.json(), "send_booking_confirmation_whatsapp")
        except (httpx.HTTPError, ValueError) as e:  # ValueError: not JSON, or not the expected shape
            raise ToolCallError(f"send_booking_confirmation_whatsapp({confirmation_id!r}): {e}") from e

    async def log_referral(
        self,
        patient_name: str = "",
        phone: str = "",
        doctor_name: str = "",
        symptom_description: str = "",
        notes: str = "",
    ) -> dict:
        """KCD-lead-1: staff at another Sonoscan center passing along a patient's number for a later outbound
        call (clinic-api/referral_log.py) -- a CSV worklist for staff, never a claim that the call was made."""
        try:
            r = await self._write(
                "/api/v1/referrals",
                json={
                    "patient_name": patient_name,
                    "phone": phone,
                    "doctor_name": doctor_name,
                    "symptom_description": symptom_description,
                    "notes": notes,
                },
            )
            r.raise_for_status()
            data = r.json()
            if not isinstance(data, dict):
                raise ValueError("referrals did not return an object")
            return data
        except (httpx.HTTPError, ValueError) as e:
            raise ToolCallError(f"log_referral({patient_name!r}, {phone!r}): {e}") from e

    async def save_draft_booking(self, caller_phone: str, call_id: str, slots_json: str) -> dict:
        body = {"caller_phone": caller_phone, "call_id": call_id, "slots_json": slots_json}
        try:
            r = await self._write("/api/v1/bookings/draft", json=body)
            r.raise_for_status()
            return r.json()
        except (httpx.HTTPError, ValueError) as e:  # ValueError: not JSON
            raise ToolCallError(f"save_draft_booking({body!r}): {e}") from e

    async def get_draft_booking(self, caller_phone: str) -> dict:
        try:
            r = await self._client.get("/api/v1/bookings/draft", params={"phone": caller_phone})
            r.raise_for_status()
            return r.json()
        except (httpx.HTTPError, ValueError) as e:  # ValueError: not JSON
            raise ToolCallError(f"get_draft_booking({caller_phone!r}): {e}") from e

    # ---- Epic E33: patient context and history (clinic-api/patient_context.py) ----
    async def _get(self, path: str, params: dict, what: str, model: type[BaseModel] | None = None) -> dict:
        try:
            r = await self._client.get(path, params={k: v for k, v in params.items() if v is not None})
            r.raise_for_status()
            return validated(model, r.json(), what) if model else r.json()
        except (httpx.HTTPError, ValueError) as e:  # ValueError: not JSON, or not the expected shape
            raise ToolCallError(f"{what}: {e}") from e

    async def _post(self, path: str, body: dict, what: str, model: type[BaseModel] | None = None) -> dict:
        try:
            r = await self._write(path, json=body)
            r.raise_for_status()
            return validated(model, r.json(), what) if model else r.json()
        except (httpx.HTTPError, ValueError) as e:  # ValueError: not JSON, or not the expected shape
            raise ToolCallError(f"{what}: {e}") from e

    async def identify_patient(self, phone: str) -> dict:
        """KCD-494: {"status": new|single|ambiguous, "record_exists", "count"} -- never a name."""
        return await self._get("/api/v1/patients/identify", {"phone": phone}, "identify_patient", IdentifyAnswer)

    async def resolve_patient(self, phone: str, name: str, age: int | None = None) -> dict:
        return await self._post(
            "/api/v1/patients/resolve", {"phone": phone, "name": name, "age": age}, "resolve_patient"
        )

    async def patient_timeline(self, patient_ref: int, caller_phone: str, call_id: str) -> dict:
        return await self._get(
            f"/api/v1/patients/{patient_ref}/timeline",
            {"caller_phone": caller_phone, "call_id": call_id},
            "patient_timeline",
        )

    async def patient_test_status(self, patient_ref: int, test_name: str, caller_phone: str, call_id: str) -> dict:
        return await self._get(
            f"/api/v1/patients/{patient_ref}/test-status",
            {"test_name": test_name, "caller_phone": caller_phone, "call_id": call_id},
            "patient_test_status",
        )

    async def get_preferences(self, patient_ref: int, caller_phone: str) -> dict:
        return await self._get(
            f"/api/v1/patients/{patient_ref}/preferences", {"caller_phone": caller_phone}, "get_preferences"
        )

    async def set_preferences(self, patient_ref: int, caller_phone: str, **fields) -> dict:
        return await self._post(
            f"/api/v1/patients/{patient_ref}/preferences", {"caller_phone": caller_phone, **fields}, "set_preferences"
        )

    async def continuity(self, caller_phone: str, call_id: str | None = None, patient_ref: int | None = None) -> dict:
        """KCD-496: unfinished work, the last interactions, and the last calls cached for one day."""
        return await self._get(
            "/api/v1/continuity",
            {"caller_phone": caller_phone, "call_id": call_id, "patient_ref": patient_ref},
            "continuity",
        )

    async def verify_patient(
        self, call_id: str, patient_ref: int, answers: dict, caller_phone: str | None = None
    ) -> dict:
        """KCD-495: the SERVER checks the security answers. {"verified", "attempts_left", "locked",
        and after a pass "age_years", "is_senior"} -- never which answer was wrong."""
        return await self._post(
            "/api/v1/patients/verify",
            {"call_id": call_id, "patient_ref": patient_ref, "caller_phone": caller_phone, **answers},
            "verify_patient",
            VerifyAnswer,
        )

    async def find_patient(self, call_id: str, answers: dict) -> dict:
        """KCD-497: find a patient from a patient id, or a date of birth with a name. Opaque reference only."""
        return await self._post("/api/v1/patients/find", {"call_id": call_id, **answers}, "find_patient", FindAnswer)

    async def agent_messages(self, lang: str | None = None) -> dict:
        """KCD-353/500/513: the operator-editable wording (agent/messages.py)."""
        return await self._get("/api/v1/agent/messages", {"lang": lang}, "agent_messages")

    async def search_bookings(self, caller_phone: str, **criteria) -> dict:
        """KCD-497: any combination of phone, name, approx_date, test_name, branch."""
        return await self._post(
            "/api/v1/bookings/search", {"caller_phone": caller_phone, **criteria}, "search_bookings"
        )

    async def write_call_event(
        self, call_id: str, seq: int, kind: str, payload: dict | None = None, caller_phone: str | None = None
    ) -> dict:
        """KCD-501: idempotent by (call_id, seq)."""
        return await self._post(
            f"/api/v1/calls/{call_id}/events",
            {"seq": seq, "kind": kind, "payload": payload or {}, "caller_phone": caller_phone},
            "write_call_event",
        )
