"""Failure-path tests for the public review runner; no optional dependencies."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("employer_review", HERE / "run.py")
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Cannot load review runner")
RUN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN)


class ReviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "source"
        (self.root / "review").mkdir(parents=True)
        for name in ("run.py", "test_run.py", "template.html"):
            shutil.copyfile(HERE / name, self.root / "review" / name)
        (self.root / "proof.py").write_text('print("PUBLIC_CHECK_PASSED")\n')
        (self.root / "evidence.json").write_text('{"kind":"synthetic"}\n')
        (self.root / "README.md").write_text("# Public fixture\n")
        (self.root / "review/README.md").write_text("Public instructions\n")
        self.project = {
            "schema_version": 1,
            "repository": "alvaromendizabal/review-fixture",
            "reviewed_base_commit": "a" * 40,
            "title": "Fixture",
            "discipline": "Test fixture",
            "summary": "Synthetic test project, not research evidence.",
            "scope": "Synthetic test only.",
            "private_boundary": "No private artifacts.",
            "signals": [{"value": "1", "label": "synthetic check"}],
            "read_more": [{"path": "README.md", "label": "Overview"}],
            "cards": [
                {
                    "audience": "ML engineering",
                    "title": "Safe fixture",
                    "decision": "Execute the actual fixture.",
                    "boundary": "Synthetic only.",
                    "implementation": "proof.py",
                    "tests": "review/test_run.py",
                    "evidence": "evidence.json",
                }
            ],
            "commands": [{"id": "fixture", "runtime": "python", "entry": "proof.py"}],
        }
        self.output = self.base / "receipt"
        self.save()

    def save(self) -> None:
        (self.root / "review/project.json").write_text(json.dumps(self.project))

    def run_review(self) -> dict:
        return RUN.execute(self.root, RUN.load_project(self.root), self.output, False, 5)

    def test_valid_run_and_reuse(self) -> None:
        self.assertFalse(self.run_review()["reused"])
        self.assertTrue(self.run_review()["reused"])
        self.assertEqual(
            RUN.verify(self.output, self.root, self.project, False)["status"], "passed"
        )

    def test_nested_manifests_are_verified(self) -> None:
        self.project["commands"][0]["args"] = ["{output}/demo"]
        self.save()
        (self.root / "proof.py").write_text(
            "from pathlib import Path\nimport sys\n"
            "p = Path(sys.argv[1]); p.mkdir()\n"
            "(p / 'manifest.json').write_text('{}')\n"
        )
        self.run_review()
        self.assertTrue(self.run_review()["reused"])
        (self.output / "demo/manifest.json").write_text("changed")
        with self.assertRaisesRegex(ValueError, "Changed review artifact"):
            self.run_review()

    def test_real_command_timeout_never_publishes_success(self) -> None:
        (self.root / "proof.py").write_text("import time\ntime.sleep(5)\n")
        with self.assertRaises(RUN.subprocess.TimeoutExpired):
            RUN.execute(self.root, self.project, self.output, False, 1)
        self.assertFalse(self.output.exists())
        self.assertEqual(len(list(self.base.glob("receipt-failed-*/failure.json"))), 1)

    def test_changed_source_rejected(self) -> None:
        self.run_review()
        (self.root / "evidence.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "source changed"):
            self.run_review()

    def test_changed_explicit_dependency_rejects_reuse(self) -> None:
        (self.root / "dependency.json").write_text('{"parameter":1}')
        self.project["source_dependencies"] = ["dependency.json"]
        self.save()
        self.run_review()
        (self.root / "dependency.json").write_text('{"parameter":2}')
        with self.assertRaisesRegex(ValueError, "source changed"):
            self.run_review()

    def test_missing_dependency_is_rejected_before_execution(self) -> None:
        self.project["source_dependencies"] = ["missing.json"]
        self.save()
        with self.assertRaisesRegex(ValueError, "Missing reviewed file"):
            self.run_review()
        self.assertFalse(self.output.exists())

    def test_invalid_dependency_lists_are_rejected(self) -> None:
        for value in ("README.md", [42], ["README.md", "README.md"], ["../README.md"]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.project["source_dependencies"] = value
                self.save()
                RUN.load_project(self.root)

    def test_artifact_tampering_rejected(self) -> None:
        self.run_review()
        (self.output / "index.html").write_text("modified")
        with self.assertRaisesRegex(ValueError, "Changed review artifact"):
            RUN.verify(self.output, self.root, self.project, False)

    def test_unlisted_artifact_rejected(self) -> None:
        self.run_review()
        (self.output / "extra.txt").write_text("unexpected")
        with self.assertRaisesRegex(ValueError, "Unexpected or missing"):
            RUN.verify(self.output, self.root, self.project, False)

    def test_failed_command_never_publishes_success(self) -> None:
        (self.root / "proof.py").write_text("raise SystemExit(7)\n")
        with self.assertRaisesRegex(ValueError, "Check failed"):
            self.run_review()
        self.assertFalse(self.output.exists())
        failures = list(self.base.glob("receipt-failed-*/failure.json"))
        self.assertEqual(len(failures), 1)
        self.assertEqual(json.loads(failures[0].read_text())["status"], "failed")
        self.assertTrue((failures[0].parent / "fixture.log").is_file())

    def test_path_traversal_rejected(self) -> None:
        for path in ("../outside.txt", "/etc/passwd", "review/../proof.py", "review\\run.py"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                RUN.local_file(self.root, path)

    def test_symlink_rejected(self) -> None:
        if sys.platform == "win32":
            self.skipTest("Symlink permission is platform-specific")
        (self.root / "link.py").symlink_to(self.root / "proof.py")
        with self.assertRaisesRegex(ValueError, "Symlink"):
            RUN.local_file(self.root, "link.py")

    def test_missing_evidence_rejected(self) -> None:
        (self.root / "evidence.json").unlink()
        with self.assertRaisesRegex(ValueError, "Missing reviewed file"):
            RUN.load_project(self.root)

    def test_duplicate_command_rejected(self) -> None:
        self.project["commands"] *= 2
        self.save()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            RUN.load_project(self.root)

    def test_unsupported_runtime_rejected(self) -> None:
        self.project["commands"][0]["runtime"] = "shell"
        self.save()
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            RUN.load_project(self.root)

    def test_inside_checkout_output_rejected(self) -> None:
        self.output = self.root / "generated"
        with self.assertRaisesRegex(ValueError, "outside the checkout"):
            self.run_review()

    def test_check_mode_mismatch_rejected(self) -> None:
        self.run_review()
        with self.assertRaisesRegex(ValueError, "Different check mode"):
            RUN.verify(self.output, self.root, self.project, True)

    def test_missing_node_rejected_before_execution(self) -> None:
        with (
            patch.object(RUN.shutil, "which", return_value=None),
            self.assertRaisesRegex(ValueError, "Node.js"),
        ):
            RUN.execute(self.root, self.project, self.output, True, 5)
        self.assertFalse(self.output.exists())

    def test_timeout_rejected_before_execution(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            RUN.execute(self.root, self.project, self.output, False, 0)

    def test_html_escapes_untrusted_text(self) -> None:
        self.project["title"] = "<script>alert(1)</script>"
        page = RUN.page(self.project, root=self.root)
        self.assertNotIn("<script>", page)
        self.assertIn("&lt;script&gt;", page)
        self.assertIn("No checks were executed", page)
        self.assertNotIn("Selected public checks passed", page)

    def test_render_is_deterministic(self) -> None:
        self.assertEqual(
            RUN.page(self.project, root=self.root), RUN.page(self.project, root=self.root)
        )

    def test_assertions_not_used_as_runtime_gates(self) -> None:
        import ast

        tree = ast.parse((HERE / "run.py").read_text())
        self.assertFalse(any(isinstance(node, ast.Assert) for node in ast.walk(tree)))


if __name__ == "__main__":
    unittest.main()
