"""Verify archived data against the repository's manifest before installing it."""

import hashlib
import io
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INPUT = "data/yeoh/Yeoh2002.rds"
EXPECTED = b"verified input\n"
W8A_INPUT = "data/libsvm/w8a"
W8A_EXPECTED = b"+1 1:1\n-1 2:1\n"


def manifest(contents):
    return f"{hashlib.sha256(contents).hexdigest()}  {INPUT}\n".encode()


class FetchDataTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(prefix="fetch data ")
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)
        self.experiments = self.root / "experiments"
        self.experiments.mkdir()
        shutil.copy2(ROOT / "experiments" / "fetch-data.sh", self.experiments)
        (self.root / INPUT).parent.mkdir(parents=True)
        (self.root / INPUT).write_bytes(b"existing input\n")
        (self.root / "data" / "MANIFEST.sha256").write_bytes(manifest(EXPECTED))
        (self.root / "data" / "README.md").write_text("committed documentation\n")
        (self.root / "data" / "libsvm").mkdir()
        (self.root / "data" / "libsvm" / "manifest.json").write_text(
            "repository metadata\n"
        )
        self.expected_manifest = manifest(EXPECTED)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        rscript = self.bin / "Rscript"
        rscript.write_text(
            f"#!{sys.executable}\nfrom pathlib import Path\nPath('derived').touch()\n"
        )
        rscript.chmod(0o755)
        self.archive = self.root / "inputs.tar.gz"
        self.env = {
            **os.environ,
            "PATH": f"{self.bin}{os.pathsep}{os.environ['PATH']}",
            "INTERCEPTS_DATA_ARCHIVE": str(self.archive),
        }

    def build_archive(self, contents, extras=None):
        members = {
            "data/MANIFEST.sha256": manifest(contents),
            "data/README.md": b"archived documentation\n",
            "data/unlisted.txt": b"unverified extra file\n",
            "data/libsvm/manifest.json": b"old archived metadata\n",
            INPUT: contents,
        }
        members.update(extras or {})
        with tarfile.open(self.archive, "w:gz") as archive:
            for name, data in members.items():
                member = tarfile.TarInfo(name)
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))

    def run_fetch(self):
        return subprocess.run(
            ["bash", str(self.experiments / "fetch-data.sh")],
            env=self.env,
            capture_output=True,
            text=True,
            check=False,
        )

    def assert_repository_metadata_preserved(self):
        self.assertEqual(
            (self.root / "data" / "MANIFEST.sha256").read_bytes(),
            self.expected_manifest,
        )
        self.assertEqual(
            (self.root / "data" / "README.md").read_text(),
            "committed documentation\n",
        )
        self.assertFalse((self.root / "data" / "unlisted.txt").exists())
        self.assertEqual(
            (self.root / "data" / "libsvm" / "manifest.json").read_text(),
            "repository metadata\n",
        )

    def add_w8a(self, contents):
        self.expected_manifest += (
            f"{hashlib.sha256(W8A_EXPECTED).hexdigest()}  {W8A_INPUT}\n".encode()
        )
        (self.root / "data" / "MANIFEST.sha256").write_bytes(self.expected_manifest)
        (self.root / W8A_INPUT).write_bytes(b"existing w8a\n")
        source = self.root / "w8a source"
        source.write_bytes(contents)
        self.env["INTERCEPTS_W8A_SOURCE"] = str(source)

    def test_v1_archive_fetches_and_verifies_w8a(self):
        self.add_w8a(W8A_EXPECTED)
        self.build_archive(EXPECTED)
        run = self.run_fetch()
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertEqual((self.root / W8A_INPUT).read_bytes(), W8A_EXPECTED)
        self.assert_repository_metadata_preserved()

    def test_changed_w8a_preserves_all_existing_inputs(self):
        self.add_w8a(b"changed w8a\n")
        self.build_archive(EXPECTED)
        run = self.run_fetch()
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("FAILED", run.stdout)
        self.assertEqual((self.root / INPUT).read_bytes(), b"existing input\n")
        self.assertEqual((self.root / W8A_INPUT).read_bytes(), b"existing w8a\n")
        self.assertFalse((self.root / "derived").exists())
        self.assert_repository_metadata_preserved()

    def test_new_archive_supplies_w8a_without_a_separate_download(self):
        self.add_w8a(b"unused source\n")
        self.build_archive(EXPECTED, {W8A_INPUT: W8A_EXPECTED})
        run = self.run_fetch()
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertEqual((self.root / W8A_INPUT).read_bytes(), W8A_EXPECTED)
        self.assert_repository_metadata_preserved()

    def test_valid_archive_installs_only_verified_inputs(self):
        self.build_archive(EXPECTED)
        run = self.run_fetch()
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertEqual((self.root / INPUT).read_bytes(), EXPECTED)
        self.assertTrue((self.root / "derived").exists())
        self.assert_repository_metadata_preserved()

    def test_archive_cannot_replace_manifest_to_validate_changed_inputs(self):
        self.build_archive(b"changed input with matching archived checksum\n")
        run = self.run_fetch()
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("FAILED", run.stdout)
        self.assertEqual((self.root / INPUT).read_bytes(), b"existing input\n")
        self.assertFalse((self.root / "derived").exists())
        self.assert_repository_metadata_preserved()


if __name__ == "__main__":
    unittest.main()
