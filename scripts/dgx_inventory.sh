#!/usr/bin/env bash
# DGX inventory: hardware, Python, local LLM runtime, listening ports, disk. No sudo, no data access.
# Run on the DGX:  bash scripts/dgx_inventory.sh | tee dgx_inventory.txt
# It never lists or opens the contents of data/ or $MTAH_DATA_DIR (only the directory permission bits).
set -u

sec() { printf '\n=== %s ===\n' "$1"; }
have() { command -v "$1" >/dev/null 2>&1; }

sec "SYSTEM"
uname -srm
[ -r /etc/os-release ] && . /etc/os-release && echo "${PRETTY_NAME:-unknown}"
echo "user: $(id -un)  sudo-capable: $(sudo -n true 2>/dev/null && echo yes || echo no)"
echo "docker: $(have docker && echo present || echo absent)"

sec "GPU (nvidia-smi summary)"
if have nvidia-smi; then
  nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu \
             --format=csv,noheader 2>/dev/null || nvidia-smi | head -n 15
  echo "-- compute processes:"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true
else
  echo "nvidia-smi not found"
fi

sec "MEMORY (unified on Spark)"
free -h 2>/dev/null || cat /proc/meminfo | head -n 3

sec "PYTHON"
for py in python3 python; do
  have "$py" && echo "$py -> $($py --version 2>&1) at $(command -v $py)"
done
echo "active venv: ${VIRTUAL_ENV:-none}"
have conda && echo "conda: $(conda --version 2>&1)"
have uv && echo "uv: $(uv --version 2>&1)"

sec "PIP (filtered)"
PIP="python3 -m pip"
[ -n "${VIRTUAL_ENV:-}" ] && PIP="python -m pip"
$PIP list 2>/dev/null | grep -i -E '^(torch|torchvision|vllm|transformers|kloppy|numpy|pandas|pyarrow|llama|ollama|openai|anthropic)' \
  || echo "(none of torch/vllm/transformers/kloppy/... found in this interpreter)"

sec "LOCAL LLM RUNTIME"
if have ollama; then
  echo "ollama: $(ollama --version 2>&1 | head -n 1)"
  ollama list 2>&1 | head -n 40
else
  echo "ollama: not in PATH"
  ls "$HOME/.local/bin" 2>/dev/null | grep -i -E 'ollama|llama|vllm|opencode' || true
fi
echo "-- ollama API on loopback:"
if have curl; then
  curl -s --max-time 3 http://127.0.0.1:11434/api/version || echo "no answer on 127.0.0.1:11434"
  echo
fi
echo "-- model caches (names only):"
du -sh "$HOME/.ollama" "$HOME/.cache/huggingface" 2>/dev/null || echo "(no ~/.ollama or HF cache found)"

sec "LISTENING PORTS (want 127.0.0.1, NOT 0.0.0.0 / [::] / *)"
if have ss; then
  ss -ltnH 2>/dev/null | awk '{print $4}' | sort -u | while read -r addr; do
    case "$addr" in
      127.0.0.1:*|\[::1\]:*) echo "OK     $addr" ;;
      *)                     echo "EXPOSED $addr" ;;
    esac
  done
  echo "-- owning processes (your user only):"
  ss -ltnp 2>/dev/null | awk 'NR>1{print $4, $NF}' | grep -E 'ollama|vllm|llama|jupyter|python' || echo "(none of yours listening)"
else
  echo "ss not available"
fi

sec "DISK"
df -h "$HOME" 2>/dev/null
echo "MTAH_DATA_DIR set: $([ -n "${MTAH_DATA_DIR:-}" ] && echo yes || echo no)"
if [ -n "${MTAH_DATA_DIR:-}" ] && [ -d "$MTAH_DATA_DIR" ]; then
  echo "permissions (want 700): $(stat -c '%a' "$MTAH_DATA_DIR")"
  df -h "$MTAH_DATA_DIR" | tail -n 1
fi

sec "TOOLS"
for t in git ssh scp rsync tmux curl jq; do
  printf '%-6s %s\n' "$t" "$(have $t && command -v $t || echo MISSING)"
done
echo
echo "done. Paste back ONLY this output (it contains no match data)."
