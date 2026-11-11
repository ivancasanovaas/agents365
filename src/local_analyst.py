"""Ask a LOCAL LLM (Ollama on loopback) for analysis proposals from AGGREGATE material only.

    python -m src.local_analyst [--profile outputs/profile.md] [--out outputs/local_proposals.md]

Safety by construction:
  - refuses any endpoint that is not 127.0.0.1 / localhost / ::1 (no cloud, no remote host);
  - reads only prompts/local_analyst.md, docs/INFORME_PLAN.md and the profile file (aggregates);
  - sends no match data: it never opens MTAH_DATA_DIR or data/;
  - the model only returns text; nothing it writes is executed here.
Environment: OLLAMA_URL (default http://127.0.0.1:11435), OLLAMA_MODEL (default qwen3.8:27b).
Standard library only.
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOOPBACK = {"127.0.0.1", "localhost", "::1"}
MAX_CHARS = 60_000      # context guard for the profile; a profile larger than this is not an aggregate
MAX_LINE = 400          # a very long line is a sign of raw rows, not aggregates


def check_loopback(url: str) -> None:
    host = urllib.parse.urlparse(url).hostname
    if host not in LOOPBACK:
        raise SystemExit("refusing non-loopback endpoint: {!r}".format(host))


def read_aggregate(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if len(text) > MAX_CHARS:
        raise SystemExit("{} is {} chars (> {}): not an aggregate profile".format(path, len(text), MAX_CHARS))
    longest = max((len(l) for l in text.splitlines()), default=0)
    if longest > MAX_LINE:
        raise SystemExit("{} has a line of {} chars (> {}): looks like raw rows, aborting".format(path, longest, MAX_LINE))
    return text


def chat(url: str, model: str, system: str, user: str, timeout: int = 1800) -> str:
    body = json.dumps({
        "model": model, "stream": False,
        "options": {"temperature": 0.2, "seed": 0},
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
    }).encode("utf-8")
    req = urllib.request.Request(url.rstrip("/") + "/api/chat", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))["message"]["content"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default=str(REPO_ROOT / "outputs" / "profile.md"))
    ap.add_argument("--plan", default=str(REPO_ROOT / "docs" / "INFORME_PLAN.md"))
    ap.add_argument("--prompt", default=str(REPO_ROOT / "prompts" / "local_analyst.md"))
    ap.add_argument("--out", default=str(REPO_ROOT / "outputs" / "local_proposals.md"))
    ap.add_argument("--url", default=os.environ.get("OLLAMA_URL", "http://127.0.0.1:11435"))
    ap.add_argument("--model", default=os.environ.get("OLLAMA_MODEL", "qwen3.8:27b"))
    args = ap.parse_args(argv)

    check_loopback(args.url)
    system = read_aggregate(Path(args.prompt))
    user = "## PLAN\n{}\n\n## DATA PROFILE (aggregates only)\n{}".format(
        read_aggregate(Path(args.plan)), read_aggregate(Path(args.profile)))
    answer = chat(args.url, args.model, system, user)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("<!-- model: {} @ {} -->\n{}\n".format(args.model, args.url, answer), encoding="utf-8")
    print("wrote {} ({} chars)".format(out, len(answer)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
