#!/bin/bash
# Front door for the voice agent on the one public port (8888).
#   gateway.sh start|stop|status|pcm|webm
#     pcm  (default) -> main_pcm on :8101 (raw PCM; works in every browser incl. Safari)
#     webm           -> main     on :8100 (MediaRecorder/WebM; Chrome/Firefox)
# Idempotent. Credentials live in /workspace/gateway/CREDENTIALS.txt (hash in htpasswd).
GW=/workspace/gateway
CONF=$GW/nginx.conf
PORT=8888

pid_on_port() { ss -ltnpH "sport = :$PORT" 2>/dev/null | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2; }

set_upstream() {  # set_upstream <8101|8100>
    printf 'upstream voice_app { server 127.0.0.1:%s; keepalive 8; }\n' "$1" > "$GW/upstream.conf"
    echo "$1" > "$GW/active_upstream"
}

ensure_creds() {
    if [ ! -s "$GW/htpasswd" ]; then
        local pw; pw=$(openssl rand -base64 36 | tr -dc 'A-Za-z0-9' | head -c 20)
        printf 'tester:%s\n' "$(openssl passwd -apr1 "$pw")" > "$GW/htpasswd"
        printf 'id: tester\npassword: %s\n' "$pw" > "$GW/CREDENTIALS.txt"
    fi
}

do_start() {
    mkdir -p /tmp/gw/body /tmp/gw/proxy /tmp/gw/fcgi /tmp/gw/uwsgi /tmp/gw/scgi
    ensure_creds
    [ -s "$GW/upstream.conf" ] || set_upstream 8101
    if [ -s /tmp/voicebot-gw.pid ] && kill -0 "$(cat /tmp/voicebot-gw.pid)" 2>/dev/null; then
        nginx -t -c "$CONF" >/dev/null 2>&1 && nginx -c "$CONF" -s reload && echo "gateway reloaded"
        return
    fi
    # The pod's template starts Jupyter on :8888 at every boot. It is only stopped (never deleted); its files are untouched.
    local p; p=$(pid_on_port)
    if [ -n "$p" ]; then
        echo "port $PORT held by pid $p ($(tr '\0' ' ' < /proc/$p/cmdline | cut -c1-60)) -- stopping it"
        kill "$p"; sleep 2
    fi
    nginx -t -c "$CONF" && nginx -c "$CONF" && echo "gateway up on :$PORT -> $(cat $GW/active_upstream)"
}

case "${1:-start}" in
    start)  do_start ;;
    stop)   [ -s /tmp/voicebot-gw.pid ] && nginx -c "$CONF" -s stop && echo stopped ;;
    pcm)    set_upstream 8101; do_start ;;
    webm)   set_upstream 8100; do_start ;;
    status) echo "upstream: $(cat $GW/active_upstream 2>/dev/null)"; curl -s -o /dev/null -w "gateway (no auth) -> %{http_code}\n" http://127.0.0.1:$PORT/ ;;
esac
