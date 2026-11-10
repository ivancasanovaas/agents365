"""PreToolUse hook: stop the agent from touching competition data.

The T&C forbid sending competition data (or extracts) to cloud LLMs. This is a best-effort
guard: it blocks any shell command that mentions the data folder, unless the command is an
allow-listed aggregate-only entry point. The human runs data-touching code locally.

Exit code 2 = block, message on stderr is shown to the model.
"""
import json
import os
import re
import sys

ALLOW = (
    r"^\s*python\s+(-I\s+)?evaluation/(run_evaluation|validate_submission)\.py\b",
    r"^\s*git\s+(status|diff|log|add|commit)\b",
)
SENSITIVE = re.compile(
    r"(\bdata[\\/])|MTAH_DATA_DIR|_atd\.(txt|xml)|_smart_tagging|_homography|\.mp4\b|evaluation[\\/]gt",
    re.IGNORECASE,
)


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0
    cmd = (payload.get("tool_input") or {}).get("command", "") or ""
    extra = os.environ.get("MTAH_DATA_DIR", "")
    # Remote URLs (e.g. public GitHub paths containing "data/") are not local competition data.
    scan = re.sub(r"https?://\S+", " ", cmd)
    hit = SENSITIVE.search(scan) or (extra and extra.lower() in scan.lower())
    if not hit:
        return 0
    if any(re.search(p, cmd) for p in ALLOW):
        return 0
    sys.stderr.write(
        "BLOCKED: this command touches competition data (data/, ATD, tagging, homography, "
        "video or GT). Do not run it. Write the script and give the user the command to run "
        "locally; scripts must print aggregates only.\n"
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
