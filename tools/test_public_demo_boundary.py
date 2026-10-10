"""Negative fixtures for the static demo packaging boundary; no project dependencies."""

import tempfile
import unittest
from pathlib import Path

from check_public_demo import verify


class DemoBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.write("index.html", '<h1>Synthetic demo</h1><script src="app.js"></script>')
        self.write("app.js", "import {run} from './engine.js'; run();")
        self.write("engine.js", "export const run=()=>({evidence_type:'SYNTHETIC_ONLY'});")
        self.write("styles.css", "body{color:#123}")

    def write(self, name, text):
        (self.root / name).write_text(text)

    def test_self_contained_demo_has_content_pins(self):
        report = verify(self.root)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(len(report["files"]), 4)
        self.assertTrue(all(len(item["sha256"]) == 64 for item in report["files"]))

    def test_remote_script_is_rejected(self):
        self.write(
            "index.html", '<h1>Synthetic</h1><script src="https://example.org/sdk.js"></script>'
        )
        with self.assertRaisesRegex(ValueError, "Remote runtime"):
            verify(self.root)

    def test_canonical_identity_metadata_is_not_a_runtime_dependency(self):
        self.write(
            "index.html",
            '<h1>Synthetic demo</h1><link rel="canonical" href="https://example.org/demo/">',
        )
        self.assertEqual(verify(self.root)["status"], "PASS")

    def test_canonical_label_cannot_hide_a_remote_stylesheet(self):
        self.write(
            "index.html",
            '<h1>Synthetic demo</h1><link rel="canonical stylesheet" href="https://example.org/a.css">',
        )
        with self.assertRaisesRegex(ValueError, "Remote runtime"):
            verify(self.root)

    def test_escaped_import_is_rejected(self):
        self.write("app.js", "import '../private.js';")
        with self.assertRaisesRegex(ValueError, "escapes"):
            verify(self.root)

    def test_missing_import_is_rejected(self):
        self.write("app.js", "import './missing.js';")
        with self.assertRaisesRegex(ValueError, "Missing runtime"):
            verify(self.root)

    def test_remote_css_is_rejected(self):
        self.write("styles.css", "body{background:url(https://example.org/track.png)}")
        with self.assertRaisesRegex(ValueError, "Remote runtime"):
            verify(self.root)

    def test_network_calls_are_rejected(self):
        self.write("app.js", "fetch('https://example.org/input');")
        with self.assertRaisesRegex(ValueError, "Runtime network"):
            verify(self.root)

    def test_remote_reexport_is_rejected(self):
        self.write("app.js", "export {run} from 'https://example.org/code.js';")
        with self.assertRaisesRegex(ValueError, "Remote runtime"):
            verify(self.root)

    def test_svg_cannot_pull_a_remote_image(self):
        self.write(
            "figure.svg",
            '<svg xmlns="http://www.w3.org/2000/svg"><image href="https://example.org/a.png"/></svg>',
        )
        with self.assertRaisesRegex(ValueError, "Remote runtime"):
            verify(self.root)

    def test_credential_material_is_rejected(self):
        self.write("app.js", "const key='" + "AKIA" + "A" * 16 + "';")
        with self.assertRaisesRegex(ValueError, "Credential-like"):
            verify(self.root)

    def test_private_blob_extension_is_rejected(self):
        self.write("weights.npz", "private")
        with self.assertRaisesRegex(ValueError, "Unreviewed public"):
            verify(self.root)

    def test_operational_location_is_rejected(self):
        self.write("app.js", "const path='s3://example-private/inputs';")
        with self.assertRaisesRegex(ValueError, "Private operational"):
            verify(self.root)

    def test_missing_synthetic_marker_is_rejected(self):
        self.write("engine.js", "export const run=()=>({});")
        with self.assertRaisesRegex(ValueError, "machine-readable synthetic"):
            verify(self.root)

    def test_symlink_cannot_import_outside_artifact(self):
        (self.root / "copy.js").symlink_to(self.root / "app.js")
        with self.assertRaisesRegex(ValueError, "Symlink"):
            verify(self.root)


if __name__ == "__main__":
    unittest.main()
