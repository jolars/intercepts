"""Check the source packet without rendering or compiling the numerical paper."""

import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = r"""\documentclass{article}
\setmathfont{Libertinus Math}
\setmonofont[Scale=0.92]{Latin Modern Mono}
\DraftwatermarkOptions{stamp=true, text={submitted}}
\linenumbers
\begin{document}
\includegraphics{figures/example}
\newpage
We submitted a patch.
Bibliography: OLD
\end{document}
"""


class ArxivPackagingTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(prefix="arxiv packet ")
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)
        (self.root / "dev").mkdir()
        shutil.copy2(ROOT / "dev" / "make-arxiv.sh", self.root / "dev")
        (self.root / "main.qmd").write_text("Paper\n")
        (self.root / "main.tex").write_text(SOURCE)
        (self.root / "main.pdf").write_bytes(b"journal PDF")
        (self.root / "references.bib").write_text("CORRECTED")
        # A bibliography-only correction must invalidate a newer TeX artifact.
        for filename, timestamp in [
            ("main.qmd", 100), ("main.tex", 200), ("references.bib", 300)
        ]:
            os.utime(self.root / filename, (timestamp, timestamp))
        (self.root / "figures").mkdir()
        (self.root / "figures" / "example.pdf").write_bytes(b"figure PDF")
        (self.root / "figures" / "unused.pdf").write_bytes(b"unused figure")
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.mock_tool("quarto", """
source = Path('main.tex')
source.write_text(source.read_text().replace('OLD', Path('references.bib').read_text()))
""")
        self.mock_tool("xelatex", "")
        self.mock_tool("latexmk", """
assert '-xelatex' in sys.argv
assert '-no-shell-escape' in sys.argv
Path('main.pdf').write_text('Title page\\fWe submitted a patch.')
Path('main.log').write_text('Clean compilation\\n')
for suffix in ['aux', 'out', 'toc', 'fls', 'fdb_latexmk', 'xdv']:
    Path('main.' + suffix).write_text('build artifact')
""")
        self.mock_tool("pdftotext", """
pages = Path('main.pdf').read_text().split('\\f')
if os.environ.get('ARXIV_TEST_WATERMARK'):
    pages[0] += '\\nsubmitted'
if '-f' in sys.argv and '-l' in sys.argv:
    first = int(sys.argv[sys.argv.index('-f') + 1])
    last = int(sys.argv[sys.argv.index('-l') + 1])
    pages = pages[first - 1:last]
print('\\n'.join(pages))
""")
        self.mock_tool("pdfinfo", "print('Pages: 2')\n")
        self.env = {**os.environ, "PATH": f"{self.bin}{os.pathsep}{os.environ['PATH']}"}
        for key in ["ARXIV_SKIP_RENDER", "ARXIV_SKIP_VERIFY", "ARXIV_TEST_WATERMARK"]:
            self.env.pop(key, None)

    def mock_tool(self, name, body):
        tool = self.bin / name
        tool.write_text(
            f"#!{sys.executable}\nimport os, sys\nfrom pathlib import Path\n" + body
        )
        tool.chmod(0o755)

    def run_builder(self):
        return subprocess.run(
            ["bash", str(self.root / "dev" / "make-arxiv.sh")],
            env=self.env,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_fresh_self_contained_packet_preserves_journal_source(self):
        run = self.run_builder()
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        with tarfile.open(self.root / "intercepts-arxiv.tar.gz") as packet:
            files = {
                member.name.removeprefix("./"): packet.extractfile(member).read()
                for member in packet.getmembers() if member.isfile()
            }
        self.assertEqual(set(files), {"main.tex", "00README.json", "figures/example.pdf"})
        self.assertEqual(files["figures/example.pdf"], b"figure PDF")
        metadata = json.loads(files["00README.json"])
        self.assertEqual(metadata["process"]["compiler"], "xelatex")
        self.assertEqual(metadata["sources"], [{"filename": "main.tex", "usage": "toplevel"}])
        self.assertEqual(metadata["texlive_version"], 2025)
        staged = files["main.tex"].decode()
        self.assertIn("Bibliography: CORRECTED", staged)
        self.assertIn("We submitted a patch.", staged)
        self.assertIn("LibertinusMath-Regular.otf", staged)
        self.assertIn("lmmono10-regular.otf", staged)
        self.assertNotIn("stamp=true", staged)
        self.assertNotIn(r"\linenumbers", staged)
        self.assertEqual((self.root / "main.tex").read_text(), SOURCE.replace("OLD", "CORRECTED"))
        self.assertEqual((self.root / "main.pdf").read_bytes(), b"journal PDF")
        self.assertIn("We submitted a patch.", (self.root / "intercepts-arxiv.pdf").read_text())

    def test_watermark_on_title_page_rejects_packet(self):
        self.env["ARXIV_TEST_WATERMARK"] = "1"
        run = self.run_builder()
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("still contains 'submitted'", run.stderr)
        self.assertFalse((self.root / "intercepts-arxiv.tar.gz").exists())


if __name__ == "__main__":
    unittest.main()
