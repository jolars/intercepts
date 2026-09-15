"""Exercise the reproduction wrapper without running the numerical experiment."""

import csv
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INDEX = "cell_id,cell_name\n1,cell_001\n2,cell_002\n"


class SkglmWrapperTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(prefix="skglm wrapper ")
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)
        self.results = self.root / "results" / "skglm-controlled"
        self.results.mkdir(parents=True)
        self.experiments = self.root / "experiments"
        self.experiments.mkdir()
        shutil.copy2(ROOT / "experiments" / "run-skglm-controlled.sh", self.experiments)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        generator = self.bin / "julia"
        generator.write_text(
            f"#!{sys.executable}\n"
            "import os\n"
            "from pathlib import Path\n"
            "root = Path(__file__).resolve().parent.parent\n"
            "(root / 'generated').touch()\n"
            "results = root / 'results' / 'skglm-controlled'\n"
            f"(results / 'index.csv').write_text({INDEX!r})\n"
            "if not os.environ.get('INCOMPLETE_GENERATION'):\n"
            "    for name in ['cell_001', 'cell_002']:\n"
            "        cell = results / 'cells' / name\n"
            "        cell.mkdir(parents=True, exist_ok=True)\n"
            "        for file in ['X.csv', 'y.csv', 'meta.json']:\n"
            "            (cell / file).write_text('generated input\\n')\n"
        )
        generator.chmod(0o755)
        (self.experiments / "sim-skglm-controlled.py").write_text(
            "import argparse, csv\n"
            "from pathlib import Path\n"
            "parser = argparse.ArgumentParser()\n"
            "for flag in ['--commit', '--out', '--index']:\n"
            "    parser.add_argument(flag, required=True)\n"
            "args = parser.parse_args()\n"
            "index = Path(args.index)\n"
            "with index.open() as stream:\n"
            "    for row in csv.DictReader(stream):\n"
            "        for file in ['X.csv', 'y.csv', 'meta.json']:\n"
            "            path = index.parent / 'cells' / row['cell_name'] / file\n"
            "            assert path.read_text()\n"
            "with Path(args.out).open('a') as stream:\n"
            "    stream.write(args.commit + '\\n')\n"
        )
        source = self.root / "source" / "skglm"
        source.mkdir(parents=True)
        (source / "__init__.py").touch()
        self.env = {
            **os.environ,
            "PATH": f"{self.bin}{os.pathsep}{os.environ['PATH']}",
            "SKGLM_PRE_FIX_SOURCE": str(source.parent),
            "SKGLM_POST_FIX_SOURCE": str(source.parent),
        }

    def seed_inputs(self):
        (self.results / "index.csv").write_text(INDEX)
        for row in csv.DictReader(INDEX.splitlines()):
            cell = self.results / "cells" / row["cell_name"]
            cell.mkdir(parents=True)
            for file in ["X.csv", "y.csv", "meta.json"]:
                (cell / file).write_text("existing input\n")

    def run_wrapper(self):
        return subprocess.run(
            ["bash", str(self.experiments / "run-skglm-controlled.sh")],
            env=self.env,
            capture_output=True,
            text=True,
            check=False,
        )

    def assert_completed(self, run):
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertEqual(
            (self.results / "results.csv").read_text(),
            "b03644fe\n8f9cbd77\n",
        )

    def test_missing_index_generates_inputs(self):
        self.assert_completed(self.run_wrapper())
        self.assertTrue((self.root / "generated").exists())

    def test_committed_index_without_cells_generates_inputs(self):
        (self.results / "index.csv").write_text(INDEX)
        self.assert_completed(self.run_wrapper())
        self.assertTrue((self.root / "generated").exists())

    def test_missing_cell_file_regenerates_inputs(self):
        self.seed_inputs()
        for file in ["X.csv", "y.csv", "meta.json"]:
            with self.subTest(file=file):
                (self.results / "cells" / "cell_002" / file).unlink()
                self.assert_completed(self.run_wrapper())

    def test_complete_inputs_are_reused(self):
        self.seed_inputs()
        self.assert_completed(self.run_wrapper())
        self.assertFalse((self.root / "generated").exists())

    def test_empty_cell_file_regenerates_inputs(self):
        self.seed_inputs()
        (self.results / "cells" / "cell_002" / "meta.json").write_text("")
        self.assert_completed(self.run_wrapper())
        self.assertTrue((self.root / "generated").exists())

    def test_incomplete_generation_preserves_cached_results(self):
        (self.results / "index.csv").write_text(INDEX)
        (self.results / "results.csv").write_text("cached results\n")
        self.env["INCOMPLETE_GENERATION"] = "1"
        run = self.run_wrapper()
        self.assertNotEqual(run.returncode, 0)
        self.assertEqual((self.results / "results.csv").read_text(), "cached results\n")


if __name__ == "__main__":
    unittest.main()
