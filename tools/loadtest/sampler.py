"""Every 2 s: admission, ingress counters, GPU, load, agent CPU -> JSONL. usage: sampler.py OUT.jsonl"""
import json, subprocess, sys, time, urllib.request, os

OUT = sys.argv[1]


def get(url):
    try:
        return json.loads(urllib.request.urlopen(url, timeout=3).read())
    except Exception:
        return {}


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=4).stdout.strip()
    except Exception:
        return ""


with open(OUT, "w") as f:
    while True:
        t0 = time.time()
        adm = get("http://127.0.0.1:8101/api/admission")
        hl = get("http://127.0.0.1:8101/api/health").get("ingress", {})
        gpu = sh("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits").split(",")
        pid = sh("ss -ltnpH 'sport = :8101' | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2")
        cpu = sh(f"ps -o pcpu= -p {pid}") if pid else ""
        row = {
            "t": time.strftime("%H:%M:%S", time.gmtime(t0 + 19800)),  # IST, the agent log's clock
            "active": adm.get("active_calls"), "p95_s": adm.get("turn_latency_p95_s"), "p50_s": adm.get("turn_latency_p50_s"), "closed": adm.get("closed_reason"),
            "backlog_ms": hl.get("max_backlog_ms"), "lag_max_ms": hl.get("event_loop_lag_ms_max"), "lag_p95_ms": hl.get("event_loop_lag_ms_p95"), "overruns": hl.get("total_overruns"),
            "gpu_util": gpu[0].strip() if len(gpu) > 1 else None, "gpu_mem": gpu[1].strip() if len(gpu) > 1 else None,
            "load1": os.getloadavg()[0], "agent_cpu": cpu,
        }
        f.write(json.dumps(row) + "\n"); f.flush()
        time.sleep(max(0.0, 2.0 - (time.time() - t0)))
