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
        self.bin = self.root / "bin"
        self.bin.mkdir()
        rscript = self.bin / "Rscript"
        rscript.write_text(
            f"#!{sys.executable}\n"
            "from pathlib import Path\n"
            "Path('derived').touch()\n"
        )
        rscript.chmod(0o755)
        self.archive = self.root / "inputs.tar.gz"
        self.env = {
            **os.environ,
            "PATH": f"{self.bin}{os.pathsep}{os.environ['PATH']}",
            "INTERCEPTS_DATA_ARCHIVE": str(self.archive),
        }

    def build_archive(self, contents):
        members = {
            "data/MANIFEST.sha256": manifest(contents),
            "data/README.md": b"archived documentation\n",
            "data/unlisted.txt": b"unverified extra file\n",
            INPUT: contents,
        }
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
            manifest(EXPECTED),
        )
        self.assertEqual(
            (self.root / "data" / "README.md").read_text(),
            "committed documentation\n",
        )
        self.assertFalse((self.root / "data" / "unlisted.txt").exists())

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
