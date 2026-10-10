# Password-protected front door + restart tooling (RunPod)

RunPod's HTTP proxy only reaches ports exposed in the pod config; on the pilot pod that is **8888 only**. These files put
the voice agent behind HTTP basic auth on that port, keep it alive, and rebuild it after a restart.

| File | What it does |
|---|---|
| `../restart_voicebot.sh` | The one command after any pod stop/restart. Archives logs (copy, never delete), snapshots the DB, runs `bootstrap_pod.sh`, starts the gateway and watchdog, prints the URL and login. |
| `gateway.sh` | `start`/`stop`/`status`/`pcm`/`webm`. Runs its own nginx (`nginx.conf`) on :8888 -> the PCM agent (:8101) or the WebM agent (:8100). Stops the pod template's Jupyter if it holds :8888 (stops it, deletes nothing). |
| `watchdog.sh` | Relaunches a dead service by checking its *process* (never the health URL: a service still loading models has no port yet). Snapshots the DB every 6 h. |
| `backup_db.sh` | Consistent SQLite snapshot into `/workspace/backups/`. Never deletes older snapshots. |

## Install on a pod

```bash
mkdir -p /workspace/gateway
cp deploy/gateway/*.sh deploy/gateway/nginx.conf /workspace/gateway/
cp deploy/restart_voicebot.sh /workspace/restart_voicebot.sh
chmod +x /workspace/gateway/*.sh /workspace/restart_voicebot.sh
bash /workspace/restart_voicebot.sh
```

The first `gateway.sh start` generates a random password: the hash goes to `/workspace/gateway/htpasswd`, the plain text
to `/workspace/gateway/CREDENTIALS.txt`. **Neither is in git; never commit them.** Delete both files to rotate.

The public URL is `https://<RUNPOD_POD_ID>-8888.proxy.runpod.net/` (the pod id changes if the pod is recreated).
