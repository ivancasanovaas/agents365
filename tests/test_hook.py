"""The pre-commit hook must block data-like files and notebooks with outputs, and let clean files through.
Runs git in a throw-away repository (needs git and a POSIX shell on PATH; skipped otherwise)."""
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOOK = ROOT / "scripts" / "hooks" / "pre-commit"


def _git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)


@unittest.skipUnless(shutil.which("git") and HOOK.exists(), "git or hook not available")
class HookTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        _git(self.repo, "init", "-q")
        _git(self.repo, "config", "user.email", "t@example.invalid")
        _git(self.repo, "config", "user.name", "t")
        hooks = self.repo / "hooks"
        hooks.mkdir()
        shutil.copy(HOOK, hooks / "pre-commit")
        _git(self.repo, "config", "core.hooksPath", str(hooks))
        probe = _git(self.repo, "commit", "--allow-empty", "-q", "-m", "probe")
        if probe.returncode != 0:
            self.skipTest("hooks cannot run here: " + probe.stderr[:200])

    def tearDown(self):
        self.tmp.cleanup()

    def _commit(self, name, content):
        (self.repo / name).write_text(content, encoding="utf-8")
        _git(self.repo, "add", "-f", name)
        return _git(self.repo, "commit", "-q", "-m", "t")

    @staticmethod
    def _nb(outputs):
        return json.dumps({"cells": [{"cell_type": "code", "metadata": {}, "execution_count": None,
                                      "outputs": outputs, "source": ["1"]}],
                           "metadata": {}, "nbformat": 4, "nbformat_minor": 4})

    def test_clean_notebook_passes(self):
        self.assertEqual(self._commit("ok.ipynb", self._nb([])).returncode, 0)

    def test_notebook_with_outputs_blocked(self):
        res = self._commit("dirty.ipynb", self._nb([{"output_type": "stream", "name": "stdout", "text": ["x"]}]))
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("notebook with outputs", res.stderr)

    def test_data_like_files_blocked(self):
        for name in ("frames.parquet", "BAR-ATH_tactical_atd.txt", "clip.mp4"):
            self.assertNotEqual(self._commit(name, "x").returncode, 0, name)

    def test_plain_source_passes(self):
        self.assertEqual(self._commit("a.py", "print(1)\n").returncode, 0)


if __name__ == "__main__":
    unittest.main()
