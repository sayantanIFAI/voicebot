-- booking_log.sql -- one readable row per booking, so you can SEE that a call really logged. (sourav, 2026-10-24)
--
-- Why: after a test call on Colab / RunPod you want to open ONE place and see what the voice agent captured: who was booked, with which doctor, on which day and
-- time, and the automatic confirmation number. This view joins appointments + doctors + departments + patients (and lab test bookings) so no join is needed by hand.
-- The data itself is never copied: it is a VIEW over the real tables, so it is always current and cannot disagree with the system of record.
--
-- Run it by hand (SQLite is the default database):
--     sqlite3 /workspace/clinic.db < clinic-api/booking_log.sql
--     sqlite3 -header -column /workspace/clinic.db "SELECT * FROM booking_log ORDER BY booked_at DESC LIMIT 10;"
-- The clinic API also applies it by itself at every start (booking_log.ensure_view), so the view always exists after the API has been started once.
--
-- Columns: kind (doctor | lab_test), confirmation_id (generated automatically at booking), booked_at, patient_name, age, sex,
--   address (the police station, when the caller said it, is part of this text -- there is no separate police-station column in the patient table),
--   patient_phone, caller_phone, doctor, department, tests (lab bookings), visit_date, visit_day, time_slot, status, paid, serial_number.
-- SQLite dialect (strftime, ||, GROUP_CONCAT). For Postgres replace strftime('%w', ...) with EXTRACT(DOW FROM ...) and GROUP_CONCAT with STRING_AGG.

DROP VIEW IF EXISTS booking_log;

CREATE VIEW booking_log AS
SELECT
    'doctor'                                   AS kind,
    a.confirmation_id                          AS confirmation_id,
    a.created_at                               AS booked_at,
    a.patient_name                             AS patient_name,
    p.age                                      AS age,
    p.sex                                      AS sex,
    p.address                                  AS address,
    a.phone                                    AS patient_phone,
    a.caller_phone                             AS caller_phone,
    COALESCE(NULLIF(d.full_name, ''), d.name)  AS doctor,
    dep.name                                   AS department,
    ''                                         AS tests,
    a.date                                     AS visit_date,
    CASE strftime('%w', a.date)
        WHEN '0' THEN 'Sunday' WHEN '1' THEN 'Monday' WHEN '2' THEN 'Tuesday' WHEN '3' THEN 'Wednesday'
        WHEN '4' THEN 'Thursday' WHEN '5' THEN 'Friday' WHEN '6' THEN 'Saturday' END AS visit_day,
    a.time_slot                                AS time_slot,
    a.status                                   AS status,
    CASE WHEN a.paid_at IS NULL THEN 'no' ELSE 'yes' END AS paid,
    CASE WHEN a.paid_at IS NULL THEN NULL ELSE a.token_number END AS serial_number
FROM appointments a
LEFT JOIN doctors d      ON d.id = a.doctor_id
LEFT JOIN departments dep ON dep.id = d.department_id
LEFT JOIN patients p     ON p.id = a.patient_id

UNION ALL

SELECT
    'lab_test'                                 AS kind,
    tb.confirmation_id                         AS confirmation_id,
    MIN(tb.created_at)                         AS booked_at,
    tb.patient_name                            AS patient_name,
    p.age                                      AS age,
    p.sex                                      AS sex,
    p.address                                  AS address,
    tb.phone                                   AS patient_phone,
    tb.caller_phone                            AS caller_phone,
    ''                                         AS doctor,
    'Lab'                                      AS department,
    GROUP_CONCAT(lt.name, ', ')                AS tests,
    tb.date                                    AS visit_date,
    CASE strftime('%w', tb.date)
        WHEN '0' THEN 'Sunday' WHEN '1' THEN 'Monday' WHEN '2' THEN 'Tuesday' WHEN '3' THEN 'Wednesday'
        WHEN '4' THEN 'Thursday' WHEN '5' THEN 'Friday' WHEN '6' THEN 'Saturday' END AS visit_day,
    ''                                         AS time_slot,
    MIN(tb.status)                             AS status,
    'no'                                       AS paid,
    NULL                                       AS serial_number
FROM test_bookings tb
LEFT JOIN lab_tests lt ON lt.id = tb.lab_test_id
LEFT JOIN patients p   ON p.id = tb.patient_id
GROUP BY tb.confirmation_id, tb.patient_name, tb.phone, tb.caller_phone, tb.date, p.age, p.sex, p.address;
