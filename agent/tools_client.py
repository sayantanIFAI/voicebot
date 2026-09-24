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

import httpx

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

    # ---- Tool 1: GET /api/v1/tests/search?name=... ----
    # Expected response shape:
    #   found=true:  {"found": true, "test_name": "...", "rate_inr": 650,
    #                 "sample_type": "Blood", "report_time_hours": 24}
    #   found=false: {"found": false, "query": "...", "did_you_mean": ["..."]}
    async def get_test_rate(self, test_name: str) -> dict:
        try:
            r = await self._client.get("/api/v1/tests/search", params={"name": test_name})
            r.raise_for_status()
            return r.json()
        except httpx.HTTPError as e:
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
    async def get_doctor_availability(self, doctor_name: str, date: str | None) -> dict:
        params = {"name": doctor_name}
        if date:
            params["date"] = date
        try:
            r = await self._client.get("/api/v1/doctors/availability", params=params)
            r.raise_for_status()
            return r.json()
        except httpx.HTTPError as e:
            raise ToolCallError(f"get_doctor_availability({doctor_name!r}, {date!r}): {e}") from e

    # ---- Tool 4: GET /api/v1/tests/prep?name=... ----
    # Expected response shape:
    #   found=true:  {"found": true, "test_name": "...", "test_name_bn": "...",
    #                 "fasting_required": true, "prep_instructions": "..."}
    #   found=false: {"found": false, "query": "...", "did_you_mean": ["..."]}
    async def get_test_prep(self, test_name: str, lang: str = "bn") -> dict:
        try:
            r = await self._client.get("/api/v1/tests/prep",
                                       params={"name": test_name, "lang": lang})
            r.raise_for_status()
            return r.json()
        except httpx.HTTPError as e:
            raise ToolCallError(f"get_test_prep({test_name!r}): {e}") from e

    # ---- Tool 5: GET /api/v1/faq?topic=... ----
    # `topic` is the STABLE KEY from /api/v1/catalogue's faq_topics, already
    # resolved by agent/fast_path.py's FAQCatalogue -- never free caller text.
    # Expected response shape:
    #   found=true:  {"found": true, "topic": "hours", "answer": "..."}
    #   found=false: {"found": false, "topic": "..."}
    async def get_faq(self, topic: str, lang: str = "bn") -> dict:
        try:
            r = await self._client.get("/api/v1/faq", params={"topic": topic, "lang": lang})
            r.raise_for_status()
            return r.json()
        except httpx.HTTPError as e:
            raise ToolCallError(f"get_faq({topic!r}): {e}") from e

    # ---- Tool 3: POST /api/v1/appointments ----
    # Body: {"doctor_name", "date", "time_slot", "patient_name", "phone"}
    # Expected response shape:
    #   success=true:  {"success": true, "confirmation_id": "KCD-20260824-0031",
    #                    "doctor_name": "...", "date": "...", "time_slot": "..."}
    #   success=false: {"success": false, "reason": "slot_taken" | "missing_field" | "doctor_not_found",
    #                    "alternative_slots": ["17:30", "18:15"]}
    async def book_appointment(self, doctor_name: str, date: str, time_slot: str,
                                patient_name: str, phone: str) -> dict:
        body = {
            "doctor_name": doctor_name, "date": date, "time_slot": time_slot,
            "patient_name": patient_name, "phone": phone,
        }
        try:
            r = await self._client.post("/api/v1/appointments", json=body)
            r.raise_for_status()
            return r.json()
        except httpx.HTTPError as e:
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
            r = await self._client.post("/api/v1/bookings/hold", json=body)
            r.raise_for_status()
            return r.json()
        except httpx.HTTPError as e:
            raise ToolCallError(f"hold_slot({body!r}): {e}") from e

    async def confirm_booking(self, hold_token: str, doctor_id: int, date: str, time_slot: str,
                               patient_name: str, phone: str, caller_phone: str,
                               patient_age: int | None = None, relationship: str = "self") -> dict:
        body = {
            "hold_token": hold_token, "doctor_id": doctor_id, "date": date, "time_slot": time_slot,
            "patient_name": patient_name, "phone": phone, "caller_phone": caller_phone,
            "patient_age": patient_age, "relationship": relationship,
        }
        try:
            r = await self._client.post("/api/v1/bookings/confirm", json=body)
            r.raise_for_status()
            return r.json()
        except httpx.HTTPError as e:
            raise ToolCallError(f"confirm_booking({body!r}): {e}") from e

    async def reschedule_appointment(self, confirmation_id: str, new_date: str, new_time_slot: str) -> dict:
        body = {"confirmation_id": confirmation_id, "new_date": new_date, "new_time_slot": new_time_slot}
        try:
            r = await self._client.post("/api/v1/bookings/reschedule", json=body)
            r.raise_for_status()
            return r.json()
        except httpx.HTTPError as e:
            raise ToolCallError(f"reschedule_appointment({body!r}): {e}") from e

    async def cancel_appointment(self, confirmation_id: str, confirm_charge: bool = False) -> dict:
        body = {"confirmation_id": confirmation_id, "confirm_charge": confirm_charge}
        try:
            r = await self._client.post("/api/v1/bookings/cancel", json=body)
            r.raise_for_status()
            return r.json()
        except httpx.HTTPError as e:
            raise ToolCallError(f"cancel_appointment({body!r}): {e}") from e

    async def lookup_bookings(self, phone: str | None = None, confirmation_id: str | None = None,
                               name: str | None = None) -> dict:
        params = {k: v for k, v in
                  {"phone": phone, "confirmation_id": confirmation_id, "name": name}.items() if v}
        try:
            r = await self._client.get("/api/v1/bookings/lookup", params=params)
            r.raise_for_status()
            return r.json()
        except httpx.HTTPError as e:
            raise ToolCallError(f"lookup_bookings({params!r}): {e}") from e

    # KCD-084: senior mode persisted against the patient (a boolean only).
    async def set_patient_senior(self, phone: str, senior: bool = True,
                                 caller_phone: str | None = None) -> dict:
        try:
            r = await self._client.post("/api/v1/patients/senior", json={"phone": phone, "senior": senior, "caller_phone": caller_phone})
            r.raise_for_status()
            return r.json()
        except httpx.HTTPError as e:
            raise ToolCallError(f"set_patient_senior: {e}") from e

    async def get_patient_senior(self, phone: str, caller_phone: str | None = None) -> bool:
        try:
            r = await self._client.get("/api/v1/patients/senior", params={"phone": phone, **({"caller_phone": caller_phone} if caller_phone else {})})
            r.raise_for_status()
            return bool(r.json().get("senior"))
        except httpx.HTTPError as e:
            raise ToolCallError(f"get_patient_senior: {e}") from e

    async def booking_conflict(self, phone: str, date: str, time_slot: str) -> dict:
        params = {"phone": phone, "date": date, "time_slot": time_slot}
        try:
            r = await self._client.get("/api/v1/bookings/conflict", params=params)
            r.raise_for_status()
            return r.json()
        except httpx.HTTPError as e:
            raise ToolCallError(f"booking_conflict({params!r}): {e}") from e

    async def book_tests(self, test_names: list[str], date: str, patient_name: str, phone: str,
                          caller_phone: str, patient_age: int | None = None,
                          relationship: str = "self") -> dict:
        body = {
            "test_names": test_names, "date": date, "patient_name": patient_name, "phone": phone,
            "caller_phone": caller_phone, "patient_age": patient_age, "relationship": relationship,
        }
        try:
            r = await self._client.post("/api/v1/bookings/tests", json=body)
            r.raise_for_status()
            return r.json()
        except httpx.HTTPError as e:
            raise ToolCallError(f"book_tests({body!r}): {e}") from e

    async def add_test_to_booking(self, confirmation_id: str, test_name: str) -> dict:
        body = {"confirmation_id": confirmation_id, "test_name": test_name}
        try:
            r = await self._client.post("/api/v1/bookings/add-test", json=body)
            r.raise_for_status()
            return r.json()
        except httpx.HTTPError as e:
            raise ToolCallError(f"add_test_to_booking({body!r}): {e}") from e

    async def doctor_earliest(self, doctor_name: str) -> dict:
        try:
            r = await self._client.get("/api/v1/doctors/earliest", params={"name": doctor_name})
            r.raise_for_status()
            return r.json()
        except httpx.HTTPError as e:
            raise ToolCallError(f"doctor_earliest({doctor_name!r}): {e}") from e

    async def route_department(self, query_text: str, lang: str = "bn") -> dict:
        params = {"query": query_text, "lang": lang}
        try:
            r = await self._client.get("/api/v1/departments/route", params=params)
            r.raise_for_status()
            return r.json()
        except httpx.HTTPError as e:
            raise ToolCallError(f"route_department({params!r}): {e}") from e

    async def resend_confirmation(self, confirmation_id: str) -> dict:
        try:
            r = await self._client.post("/api/v1/bookings/resend",
                                        params={"confirmation_id": confirmation_id})
            r.raise_for_status()
            return r.json()
        except httpx.HTTPError as e:
            raise ToolCallError(f"resend_confirmation({confirmation_id!r}): {e}") from e

    async def save_draft_booking(self, caller_phone: str, call_id: str, slots_json: str) -> dict:
        body = {"caller_phone": caller_phone, "call_id": call_id, "slots_json": slots_json}
        try:
            r = await self._client.post("/api/v1/bookings/draft", json=body)
            r.raise_for_status()
            return r.json()
        except httpx.HTTPError as e:
            raise ToolCallError(f"save_draft_booking({body!r}): {e}") from e

    async def get_draft_booking(self, caller_phone: str) -> dict:
        try:
            r = await self._client.get("/api/v1/bookings/draft", params={"phone": caller_phone})
            r.raise_for_status()
            return r.json()
        except httpx.HTTPError as e:
            raise ToolCallError(f"get_draft_booking({caller_phone!r}): {e}") from e

    # ---- Epic E33: patient context and history (clinic-api/patient_context.py) ----
    async def _get(self, path: str, params: dict, what: str) -> dict:
        try:
            r = await self._client.get(path, params={k: v for k, v in params.items() if v is not None})
            r.raise_for_status()
            return r.json()
        except httpx.HTTPError as e:
            raise ToolCallError(f"{what}: {e}") from e

    async def _post(self, path: str, body: dict, what: str) -> dict:
        try:
            r = await self._client.post(path, json=body)
            r.raise_for_status()
            return r.json()
        except httpx.HTTPError as e:
            raise ToolCallError(f"{what}: {e}") from e

    async def identify_patient(self, phone: str) -> dict:
        """KCD-494: {"status": new|single|ambiguous, "record_exists", "count"} -- never a name."""
        return await self._get("/api/v1/patients/identify", {"phone": phone}, "identify_patient")

    async def resolve_patient(self, phone: str, name: str, age: int | None = None) -> dict:
        return await self._post("/api/v1/patients/resolve", {"phone": phone, "name": name, "age": age}, "resolve_patient")

    async def patient_timeline(self, patient_ref: int, caller_phone: str, call_id: str) -> dict:
        return await self._get(f"/api/v1/patients/{patient_ref}/timeline",
                               {"caller_phone": caller_phone, "call_id": call_id}, "patient_timeline")

    async def patient_test_status(self, patient_ref: int, test_name: str, caller_phone: str, call_id: str) -> dict:
        return await self._get(f"/api/v1/patients/{patient_ref}/test-status",
                               {"test_name": test_name, "caller_phone": caller_phone, "call_id": call_id},
                               "patient_test_status")

    async def get_preferences(self, patient_ref: int, caller_phone: str) -> dict:
        return await self._get(f"/api/v1/patients/{patient_ref}/preferences", {"caller_phone": caller_phone},
                               "get_preferences")

    async def set_preferences(self, patient_ref: int, caller_phone: str, **fields) -> dict:
        return await self._post(f"/api/v1/patients/{patient_ref}/preferences",
                                {"caller_phone": caller_phone, **fields}, "set_preferences")

    async def continuity(self, caller_phone: str, call_id: str | None = None) -> dict:
        return await self._get("/api/v1/continuity", {"caller_phone": caller_phone, "call_id": call_id}, "continuity")

    async def search_bookings(self, caller_phone: str, **criteria) -> dict:
        """KCD-497: any combination of phone, name, approx_date, test_name, branch."""
        return await self._post("/api/v1/bookings/search", {"caller_phone": caller_phone, **criteria}, "search_bookings")

    async def write_call_event(self, call_id: str, seq: int, kind: str, payload: dict | None = None,
                               caller_phone: str | None = None) -> dict:
        """KCD-501: idempotent by (call_id, seq)."""
        return await self._post(f"/api/v1/calls/{call_id}/events",
                                {"seq": seq, "kind": kind, "payload": payload or {}, "caller_phone": caller_phone},
                                "write_call_event")

