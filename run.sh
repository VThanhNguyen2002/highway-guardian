#!/usr/bin/env bash
# =============================================================
# Highway Guardian — One-Click Startup (Linux / macOS)
# Starts FastAPI backend + Vue dashboard concurrently.
# Streamlit is optional; uncomment the block below to enable it.
# =============================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Colour helpers ────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${RED}[WARN]${NC}  $*"; }

echo ""
echo "========================================================="
echo "         HIGHWAY GUARDIAN — SYSTEM STARTUP"
echo "========================================================="
echo ""

# ── Pre-flight: .env ─────────────────────────────────────────
if [[ ! -f "$PROJECT_ROOT/.env" ]]; then
  if [[ -f "$PROJECT_ROOT/.env.example" ]]; then
    warn ".env not found — copying from .env.example. Fill in your keys!"
    cp "$PROJECT_ROOT/.env.example" "$PROJECT_ROOT/.env"
  else
    warn ".env missing and no .env.example found. The backend may fail to start."
  fi
fi

# ── Pre-flight: uploads dir ───────────────────────────────────
mkdir -p "$PROJECT_ROOT/uploads"

# ── Pre-flight: frontend node_modules ────────────────────────
if [[ ! -d "$PROJECT_ROOT/frontend/node_modules" ]]; then
  info "node_modules not found — installing Vue dependencies..."
  cd "$PROJECT_ROOT/frontend" && npm install
fi

# ── Python environment ────────────────────────────────────────
export PYTHONPATH="$PROJECT_ROOT"

VENV_ACTIVATE=""
for candidate in "$PROJECT_ROOT/.venv/bin/activate" \
                  "$PROJECT_ROOT/venv/bin/activate" \
                  "$PROJECT_ROOT/env/bin/activate"; do
  if [[ -f "$candidate" ]]; then
    VENV_ACTIVATE="$candidate"
    break
  fi
done

if [[ -n "$VENV_ACTIVATE" ]]; then
  ok "Found virtualenv: $VENV_ACTIVATE"
  # shellcheck source=/dev/null
  source "$VENV_ACTIVATE"
else
  warn "No virtualenv found — using system Python. Consider: python3 -m venv .venv"
fi

# ── Launch services ───────────────────────────────────────────
info "Starting FastAPI backend on http://localhost:8000 ..."
(cd "$PROJECT_ROOT" && python3 -m uvicorn backend.main:app \
    --host 0.0.0.0 --port 8000 --reload 2>&1 | \
    sed 's/^/[BACKEND] /' ) &
BACKEND_PID=$!

info "Starting Vue dashboard on http://localhost:5173 ..."
(cd "$PROJECT_ROOT/frontend" && npm run dev 2>&1 | \
    sed 's/^/[FRONTEND] /') &
FRONTEND_PID=$!

# ── Optional: Streamlit inference UI ─────────────────────────
# Uncomment the block below to also launch Streamlit on :8501
#
# info "Starting Streamlit inference UI on http://localhost:8501 ..."
# (cd "$PROJECT_ROOT/streamlit_app" && streamlit run app.py \
#     --server.port 8501 2>&1 | sed 's/^/[STREAMLIT] /') &
# STREAMLIT_PID=$!

echo ""
echo "========================================================="
echo "  Services started:"
echo "    API Docs      → http://localhost:8000/docs"
echo "    Vue Dashboard → http://localhost:5173"
echo "    Streamlit UI  → http://localhost:8501 (if enabled)"
echo ""
echo "  Press Ctrl+C to stop all services."
echo "========================================================="
echo ""

# ── Graceful shutdown on Ctrl-C ───────────────────────────────
trap 'echo ""; info "Shutting down..."; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0' SIGINT SIGTERM

wait
