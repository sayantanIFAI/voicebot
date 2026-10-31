# tools/loadtest — measuring the voice agent under many calls

All of these run ON THE POD (they talk to `localhost`). Copy the repo's latest `main` there first (`cd /workspace/kolkata-care-voice-agent && git pull`).

| File | What it does |
|---|---|
| `rampload.py` | Starts simulated callers. Each caller plays its recordings in order and speaks the next one only after the AI has finished answering. `rampload.py OUT.json [seconds] [first_wave] [add_each_step] [step_seconds] [files]`. |
| `sampler.py` | Every 2 seconds writes the agent's backlog, event-loop lag, GPU and CPU load to a `.jsonl` file. |
| `round.sh` | One whole test: restarts the agent with test settings, warms it up, runs `rampload.py` and `sampler.py` together, collects the agent's own timing lines into `/tmp/round_LABEL_*`. **It lifts the call limits and switches off load shedding for the test, so put the normal settings back afterwards** (restart the agents the normal way). |
| `analyse_round.py` | Prints the result tables. Run it in the folder that holds the `round_LABEL_*` files: `python3 tools/loadtest/analyse_round.py LABEL`. |
| `clinic_load.py` | Fires 30 / 90 / 200 requests at the clinic API at once and prints median / 95th percentile / slowest time. |
| `idem_race_pg.py` | Sends the same write with the same Idempotency-Key from many clients at once; passes when exactly one row is created. `idem_race_pg.py 24 6` = 24 clients, 6 rounds. |
| `barge_test.py` | One call; the caller talks over the AI. Prints whether playback stopped and how fast. `barge_test.py ws://localhost:8101/ws/audio question.wav interruption.wav 150 0 hp` (the last word `hp` = phone-line rule). |

## The recordings
`rampload.py` reads `/tmp/<name>.pcm` (16 kHz, mono, 16-bit, no header). The owner's six test recordings are `g1`, `d1`, `d2`, `d3`, `d4`, `d5` (WAV files; ask the owner, or copy them from `C:\Users\Dell\Downloads`). Keep the WAVs on the persistent volume so a pod restart does not lose them:

```bash
mkdir -p /workspace/loadtest_audio          # upload g1.wav d1.wav ... d5.wav here (scp)
for n in g1 d1 d2 d3 d4 d5; do
  ffmpeg -loglevel error -y -i /workspace/loadtest_audio/$n.wav -ar 16000 -ac 1 -f s16le /tmp/$n.pcm
done
```
(Run this loop again after every pod restart: `/tmp` is wiped.)

## The standard runs
* **Thirty calls at once (the release gate):** `python3 tools/loadtest/rampload.py /tmp/gate30.json 5 30 0 10`
* **The ramp (10 calls, +5 every 10 s, up to about 65):** `bash tools/loadtest/round.sh LABEL 1` (the second word: `1` = echo canceller on, `0` = off), then `cd /tmp && python3 /workspace/kolkata-care-voice-agent/tools/loadtest/analyse_round.py LABEL`.

Never start a second `uvicorn main_pcm:app` on the pod: the watchdog matches that name and a test copy hides a dead real agent.
