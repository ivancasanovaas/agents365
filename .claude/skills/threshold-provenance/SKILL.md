---
name: threshold-provenance
description: Use whenever a numeric threshold, window size, probability or rule constant is added or changed in the possession pipeline. Forces a documented, rule-compliant origin in PROVENANCE.md.
---

# threshold-provenance

For every new or changed constant:

1. Identify its origin. Accepted: FIFA Laws of the Game (cite law), physical/kinematic limits
   (e.g. max ball speed, pitch dimensions 105.3 x 68.0 m from `games.py`), open datasets of OTHER
   games (cite dataset, size, licence), or unsupervised estimation on the provided game with no human labels.
2. Rejected: "looked right in the video", "best score on the public 5 minutes", hand-picked frames.
3. Add a row to `PROVENANCE.md` (constant, value, origin, where used, date/author).
4. Reference the constant by name in code, with a comment `# see PROVENANCE.md`. No magic numbers.
5. If no acceptable origin exists, propose an acceptable estimation procedure or ask the user.

Never pick a value by comparing scores on the public GT; use experiment design from other-game data.
