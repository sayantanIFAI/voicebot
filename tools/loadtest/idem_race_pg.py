import asyncio, os, sys, uuid, httpx, psycopg2
tok = os.environ.get("CLINIC_API_TOKEN", "")
H = {"Authorization": f"Bearer {tok}"} if tok else {}
PW = open("/workspace/.pg_clinic_password").read().strip()
def db():
    return psycopg2.connect(f"postgresql://clinic:{PW}@127.0.0.1:5432/clinic")
async def one(c, key, body, results):
    for attempt in range(8):
        r = await c.post("/api/v1/callbacks", json=body, headers={**H, "Idempotency-Key": key})
        if r.status_code == 409:
            await asyncio.sleep(0.4); continue
        results.append(r.status_code); return
    results.append("gave-up-409")
async def main(n, rounds):
    bad = 0
    async with httpx.AsyncClient(base_url="http://localhost:8080", timeout=60, limits=httpx.Limits(max_connections=100)) as c:
        for rd in range(rounds):
            key = uuid.uuid4().hex; call_id = "race-" + key[:10]
            body = {"phone": "9000000000", "call_id": call_id, "requested_window": "asap", "reason": "idempotency race test", "call_summary": "x"}
            res = []
            await asyncio.gather(*[one(c, key, body, res) for _ in range(n)])
            con = db(); cur = con.cursor()
            cur.execute("select count(*) from callback_requests where call_id=%s", (call_id,)); rows = cur.fetchone()[0]
            ok = rows == 1 and set(res) <= {200}
            bad += 0 if ok else 1
            print(f"round {rd}: {n} concurrent same-key writes -> responses {sorted(set(res), key=str)}, rows created = {rows} {'OK' if ok else 'FAIL'}")
            cur.execute("delete from callback_requests where call_id=%s", (call_id,)); cur.execute("delete from idempotency_records where key=%s", (key,)); con.commit(); con.close()
    print("RESULT:", "all rounds OK" if bad == 0 else f"{bad} ROUND(S) FAILED")
asyncio.run(main(int(sys.argv[1]), int(sys.argv[2])))
