---
name: compliance-guard
description: Use before reading files, running code, adding features/thresholds, or committing in this hackathon repo. Checks the work against RULES.md and the T&C (no competition data to cloud LLMs, no hand labels, no GT tuning).
---

# compliance-guard

Run this checklist before acting. If any answer is "yes" or "unsure": STOP and ask the user.

1. Does the action read, list, print or send anything from `data/`, `*_atd.*`, `*_smart_tagging.*`,
   `*_homography.csv`, `*.mp4` or `evaluation/gt/`? -> Do not run it. Give the user the local command.
2. Does a script print rows of competition data? -> Rewrite to print aggregates only.
3. Does a threshold, rule or label come from watching the video, hand-correcting the match, or from the
   5-minute public GT? -> Forbidden. Use FIFA Laws, physics or other-game data.
4. Is an external model/dataset/API used? -> It must be accessible to organisers and documented in the report.
5. Does the change bake in frame-specific knowledge of this match? -> Forbidden.
6. Will anything derived from the match (frame-level exports, clips) be committed? -> Forbidden.
7. Is the rule unclear or borderline? -> Ask the user to open an issue / email the organisers
   (hackathon.talentarena@mobileworldcapital.com) before building on it.

Report outcome as: PASS / BLOCKED (reason) / NEEDS-USER-DECISION.
