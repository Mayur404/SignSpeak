# Shared port selection for start.sh / run.sh (sourced, not executed).
#
# Port 8080 on the edge VM is used by the NiralOS web console, so SignSpeak
# defaults to 8090. WEB_PORT in .env (or the environment) overrides it; if
# that port is already taken, the next free port from a short list is used.

port_in_use() {
    local port="$1"
    if command -v ss >/dev/null 2>&1; then
        ss -ltnH 2>/dev/null | awk '{print $4}' | grep -qE "[:.]${port}\$"
    else
        python3 - "$port" <<'PY' 2>/dev/null
import socket, sys
s = socket.socket()
try:
    s.bind(("0.0.0.0", int(sys.argv[1])))
except OSError:
    sys.exit(0)   # in use
finally:
    s.close()
sys.exit(1)       # free
PY
    fi
}

pick_port() {
    local wanted="${WEB_PORT:-}"
    if [[ -z "$wanted" && -f .env ]]; then
        wanted=$(grep -E '^WEB_PORT=' .env | tail -1 | cut -d= -f2- | tr -d '[:space:]"'"'" || true)
    fi
    wanted="${wanted:-8090}"
    local port
    for port in "$wanted" 8090 8091 8092 8501 5050 9090; do
        if ! port_in_use "$port"; then
            if [[ "$port" != "$wanted" ]]; then
                echo "Port $wanted is already used by another service; using port $port instead." >&2
            fi
            echo "$port"
            return 0
        fi
    done
    echo "Error: no free port found (tried $wanted 8090 8091 8092 8501 5050 9090)." >&2
    return 1
}
