import asyncio, os, time, statistics, httpx
tok = os.environ.get("CLINIC_API_TOKEN", "")
h = {"Authorization": f"Bearer {tok}"} if tok else {}
async def one(c, i, out):
    t = time.perf_counter()
    try:
        if i % 3 == 0:
            r = await c.get("/api/v1/bookings/conflict", params={"phone": f"96355002{i%100:02d}", "date": "2026-10-30", "time_slot": "17:10"})
        elif i % 3 == 1:
            r = await c.get("/api/v1/doctors/availability", params={"name": "Dr. Prabir Ghosh", "date": "2026-10-30"})
        else:
            r = await c.get("/api/v1/doctors/availability", params={"name": "দত্ত"})
        out.append(((time.perf_counter() - t) * 1000, r.status_code))
    except Exception as e:
        out.append(((time.perf_counter() - t) * 1000, type(e).__name__))
async def main(n):
    async with httpx.AsyncClient(base_url="http://localhost:8080", headers=h, timeout=30, limits=httpx.Limits(max_connections=200)) as c:
        out = []
        await asyncio.gather(*[one(c, i, out) for i in range(n)])
    ms = sorted(x for x, _ in out)
    bad = [s for _, s in out if s != 200]
    print(f"n={n} p50={ms[len(ms)//2]:.0f} p95={ms[int(len(ms)*.95)]:.0f} max={ms[-1]:.0f} ms  non200={len(bad)} {set(bad)}")
for n in (30, 90, 200):
    asyncio.run(main(n))
