"""Replace the inline figure blocks with \input of the caption files.

The manuscripts carry one figure* block per panel, inserted earlier with
a short caption.  The long captions now live in
figures/<name>-captions.tex.  This removes the inline blocks and inputs
the caption file at the first figure's position, so figure order and
labels are preserved.
"""

from __future__ import annotations

import os
import re

HERE = os.path.dirname(__file__)
DOCS = os.path.join(HERE, "..", "..", "docs")

FIGBLOCK = re.compile(
    r"\n?\\begin\{figure\*\}\[!htbp\]\s*\n\\centering\s*\n"
    r"\\includegraphics\[width=\\textwidth\]\{[^}]*\}\s*\n"
    r"\\caption\{.*?\}\s*\n\\label\{[^}]*\}\s*\n\\end\{figure\*\}\n",
    re.DOTALL,
)


def swap(path: str, caption_file: str) -> tuple[int, int]:
    s = open(path, encoding="utf-8").read()
    blocks = FIGBLOCK.findall(s)
    n = len(blocks)
    if n == 0:
        return 0, s.count("includegraphics")

    first = s.index(blocks[0])
    s = FIGBLOCK.sub("", s)
    # re-insert the caption file where the first figure stood
    s = s[:first] + "\n\\input{figures/" + caption_file + "}\n" + s[first:]
    open(path, "w", encoding="utf-8").write(s)
    return n, s.count("input{figures/")


if __name__ == "__main__":
    a = swap(os.path.join(DOCS, "cannonical-ranking-algorithm",
                          "canonical-ranking-algorithm.tex"),
             "ranking-captions")
    b = swap(os.path.join(DOCS, "structural-correspondence",
                          "structural-correspondence.tex"),
             "correspondence-captions")
    print("ranking: removed", a[0], "inline blocks, inputs:", a[1])
    print("correspondence: removed", b[0], "inline blocks, inputs:", b[1])
