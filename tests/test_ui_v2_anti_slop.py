"""Anti-AI-slop and V2 Scholarly Document Chat UI contract verification for cortex-rag-agent."""

from __future__ import annotations

import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestCortexUiV2Contracts(unittest.TestCase):
    def test_scholarly_reader_identity_and_safe_dom(self) -> None:
        html = (REPO_ROOT / "ui" / "index.html").read_text(encoding="utf-8")
        self.assertIn("Cortex", html)
        self.assertIn("Source Serif 4", html)
        self.assertIn("Atkinson Hyperlegible", html)
        self.assertIn("#1E3A5F", html.upper())
        self.assertNotIn("#F7F6F3", html.upper())
        for sink in ("innerHTML", "outerHTML", "insertAdjacentHTML", "eval("):
            self.assertNotIn(sink, html)
        self.assertTrue((REPO_ROOT / "PRODUCT.md").is_file())
        self.assertTrue((REPO_ROOT / "DESIGN.md").is_file())


if __name__ == "__main__":
    unittest.main()
