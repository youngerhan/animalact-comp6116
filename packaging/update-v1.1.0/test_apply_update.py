"""Synthetic checks for the update installer; no real project files are modified."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import apply_update as update


class UpdateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.target = self.folder / "旧包 with space"
        self.root = self.folder / "更新 package"
        self.payload = self.root / "payload"
        required = ["runtime/python.exe", "AnimalAct/action_classes.json",
                    "AnimalAct-SlowFast/data/main.json", "AnimalAct/annotations/a.json",
                    "AnimalAct/splits/train.txt", "AnimalAct/videos/test/a.mp4", "README.md"]
        required += [f"AnimalAct-SlowFast/runs/{name}/best.pth" for name in update.LEGACY_RUNS]
        self.base = {}
        for relative in required:
            file = self.target / relative
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes((relative + " original\n").encode("utf-8"))
            self.base[relative] = update.fingerprint(file)
        baseline = self.target / "SHA256SUMS.json"
        baseline.write_text(json.dumps(self.base), encoding="utf-8")
        self.base["SHA256SUMS.json"] = update.fingerprint(baseline)
        self.files = {}
        for relative, value in (("README.md", b"new\n"),
                                ("AnimalAct-SlowFast/runs/improve-context32/best.pth", b"weights\n")):
            file = self.payload / relative
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(value)
            self.files[relative] = update.fingerprint(file)
        merged = dict(self.base)
        merged.pop("SHA256SUMS.json")
        merged.update(self.files)
        checksum = self.payload / "SHA256SUMS.json"
        checksum.write_text(json.dumps(merged), encoding="utf-8")
        self.files["SHA256SUMS.json"] = update.fingerprint(checksum)
        self.manifest = dict(version="v1.1.0-20261125", base_version="v1.0.0-20261125",
                             files=self.files, base_files=self.base)
        self.write_manifest()

    def write_manifest(self):
        (self.root / "update-manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def apply(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            update.apply(self.target, root=self.root, **kwargs)

    def inventory(self):
        return {str(p.relative_to(self.target)): update.fingerprint(p)
                for p in self.target.rglob("*") if p.is_file()}

    def test_dry_run_writes_nothing(self):
        before = self.inventory()
        self.apply(dry_run=True)
        self.assertEqual(before, self.inventory())

    def test_apply_backup_verify_and_repeat(self):
        self.apply()
        self.apply(verify_only=True)
        original = self.target / "packaging/update-backups/v1.1.0-20261125/README.md"
        self.assertEqual(update.fingerprint(original), self.base["README.md"])
        before = self.inventory()
        self.apply()
        self.assertEqual(before, self.inventory())
        for relative, expected in self.files.items():
            self.assertEqual(update.fingerprint(self.target / relative), expected)

    def test_user_edit_is_preserved(self):
        file = self.target / "README.md"
        file.write_bytes(b"custom user edit\n")
        before = self.inventory()
        with self.assertRaisesRegex(ValueError, "Local file differs"):
            self.apply()
        self.assertEqual(before, self.inventory())

    def test_corrupt_payload_is_rejected_before_writes(self):
        (self.payload / "README.md").write_bytes(b"bad payload\n")
        before = self.inventory()
        with self.assertRaisesRegex(ValueError, "mismatch"):
            self.apply()
        self.assertEqual(before, self.inventory())

    def test_path_traversal_is_rejected(self):
        self.manifest["files"]["../outside.txt"] = self.files["README.md"]
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "Unsafe manifest path"):
            self.apply()

    def test_case_collision_is_rejected(self):
        self.manifest["files"]["readme.md"] = self.files["README.md"]
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "Case-colliding"):
            self.apply()

    def test_changed_base_checkpoint_is_rejected(self):
        file = self.target / f"AnimalAct-SlowFast/runs/{update.LEGACY_RUNS[0]}/best.pth"
        file.write_bytes(b"other checkpoint")
        with self.assertRaisesRegex(ValueError, "mismatch"):
            self.apply()


if __name__ == "__main__":
    unittest.main()
