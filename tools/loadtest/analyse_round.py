import json, re, statistics as st, sys, bisect


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * (len(v) - 1) + 0.5))] if v else None


def line(label, v):
    if not v:
        return f"{label}: no data"
    return f"{label}: n={len(v)} p50={pct(v,.5):.0f} p90={pct(v,.9):.0f} p99={pct(v,.99):.0f} max={max(v):.0f}"


def secs(t):
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s.replace(",", "."))


def analyse(label):
    d = json.load(open(f"round_{label}_calls.json", encoding="utf-8"))
    R = d["results"]
    out = [f"===== ROUND {label} ====="]
    launched = sum(n for _t, n in d["launched"])
    oc = {}
    for r in R:
        oc[r["outcome"]] = oc.get(r["outcome"], 0) + 1
    out.append(f"calls launched {launched} over {d['stop_launch_s']:.0f}s (ramp 10 then +5 every 10s); outcomes {oc}; client loop lag p95 {d['client_lag_ms_p95']} ms max {d['client_lag_ms_max']} ms; test total {d['total_s']:.0f}s")
    turns = [(r["call"], t) for r in R for t in r["turns"]]
    ans = [t["ai_text_ms"] for _c, t in turns if t["ai_text_ms"] is not None and t["done"]]
    aud = [t["ai_audio_ms"] for _c, t in turns if t["ai_audio_ms"] is not None and t["done"]]
    out.append(f"turns {len(turns)}, finished {sum(1 for _c,t in turns if t['done'])}, timed out {sum(1 for _c,t in turns if not t['done'])}")
    out.append(line("end of caller speech -> first AI text (ms)", ans))
    out.append(line("end of caller speech -> first AI audio (ms)", aud))
    for name in ("g1", "d1", "d2", "d3", "d4", "d5"):
        v = [t["ai_text_ms"] for _c, t in turns if t["file"] == name and t["ai_text_ms"] is not None and t["done"]]
        out.append("   " + line(name, v))
    out.append("by time of the turn (end_s window):")
    for lo in range(0, 200, 20):
        v = [t["ai_text_ms"] for _c, t in turns if lo <= t["end_s"] < lo + 20 and t["ai_text_ms"] is not None and t["done"]]
        if v:
            out.append(f"   {lo:>3}-{lo+20:<3}s  n={len(v):>3}  p50={pct(v,.5):>6.0f}  p90={pct(v,.9):>6.0f}  max={max(v):>6.0f}")
    # samples
    S = [json.loads(x) for x in open(f"round_{label}_samples.jsonl", encoding="utf-8") if x.strip()]
    if S:
        num = lambda k: [float(s[k]) for s in S if s.get(k) not in (None, "")]
        out.append(f"samples: peak active calls {max(num('active')):.0f}, backlog max {max(num('backlog_ms')):.0f} ms (mean {st.mean(num('backlog_ms')):.0f}), loop lag max {max(num('lag_max_ms')):.0f} ms, "
                   f"GPU util mean {st.mean(num('gpu_util')):.0f}% max {max(num('gpu_util')):.0f}%, load1 max {max(num('load1')):.1f}, agent CPU max {max(num('agent_cpu')):.0f}% mean {st.mean(num('agent_cpu')):.0f}%, "
                   f"overruns {max(num('overruns')):.0f}, admission p95 max {max(num('p95_s')):.1f}s")
    # per-stage
    rows = []
    for ln in open(f"round_{label}_timing.log", encoding="utf-8", errors="replace"):
        m = re.search(r"(\d\d:\d\d:\d\d,\d+).*turn timing: (.*)", ln)
        if not m:
            continue
        stages = {k: int(v) for k, v in re.findall(r"(\w+) (\d+) ms", m.group(2))}
        rows.append((secs(m.group(1)), stages))
    if rows and S:
        st_times = [secs(s["t"]) for s in S]
        def active_at(t):
            i = min(bisect.bisect_left(st_times, t), len(S) - 1)
            return float(S[i]["active"] or 0)
        for name, test in (("<=15 active calls", lambda a: a <= 15), ("16-30 active", lambda a: 15 < a <= 30), (">30 active", lambda a: a > 30)):
            sel = [s for t, s in rows if test(active_at(t))]
            if not sel:
                continue
            out.append(f"per-stage (agent log) with {name}: {len(sel)} turns")
            for k in ("prep", "lid", "asr", "intent", "reply", "tts", "send", "total"):
                v = [s[k] for s in sel if k in s]
                if v:
                    out.append(f"      {k:<7} p50={pct(v,.5):>6.0f} p95={pct(v,.95):>6.0f} max={max(v):>6.0f}  (n={len(v)})")
    return "\n".join(out)


if __name__ == "__main__":
    for lab in sys.argv[1:]:
        print(analyse(lab))
        print()
