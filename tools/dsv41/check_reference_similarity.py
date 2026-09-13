# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Verbatim-similarity check of the DeepSeek-V4.1 branch against reference implementations.

The source policy in ``docs/dsv41/PROVENANCE.md`` allows public drafts to be read for
design only. This tool reports every run of at least ``--min-run`` consecutive identical
lines (whitespace-normalised, blank and comment lines dropped) between the files added by
the branch and the files touched by the reference trees, so a reviewer can classify each
run (interface signature / upstream boilerplate / must-match constant / copied code).

Reference trees are git refs of this repository (e.g. PR heads fetched read-only into
``refs/reference-prs/<n>``) or a directory checkout of another repository::

    git fetch upstream pull/7224/head:refs/reference-prs/7224 pull/7231/head:refs/reference-prs/7231
    python tools/dsv41/check_reference_similarity.py --base 0cd11658f \
        --ref refs/reference-prs/7224 --ref refs/reference-prs/7231 \
        --dir /path/to/automodel-pr-3855 --dir-filter engram --dir-filter deepseek
"""

import argparse
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

Line = Tuple[int, str]


def normalise(text: str) -> List[Line]:
    """Keep (line number, whitespace-collapsed content) for code lines only."""
    out: List[Line] = []
    for number, raw in enumerate(text.splitlines(), 1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        out.append((number, " ".join(stripped.split())))
    return out


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    ).stdout


def added_files(repo: Path, base: str) -> List[str]:
    names = git(repo, "diff", "--name-only", "--diff-filter=A", base, "HEAD").split()
    return [n for n in names if n.endswith(".py")]


def ref_files(repo: Path, ref: str, merge_base: str) -> Iterable[Tuple[str, str]]:
    names = git(repo, "diff", "--name-only", "--diff-filter=AM", merge_base, ref).split()
    for name in names:
        if not name.endswith(".py"):
            continue
        blob = subprocess.run(
            ["git", "-C", str(repo), "show", f"{ref}:{name}"], capture_output=True, text=True
        )
        if blob.returncode == 0:
            yield f"{ref}:{name}", blob.stdout


def dir_files(root: Path, filters: List[str]) -> Iterable[Tuple[str, str]]:
    for path in root.rglob("*.py"):
        rel = str(path.relative_to(root))
        if not filters or any(f in rel.lower() for f in filters):
            yield f"{root.name}:{rel}", path.read_text(errors="ignore")


def matching_runs(a: List[Line], b: List[Line], min_run: int):
    index: Dict[str, List[int]] = {}
    for j, (_, line) in enumerate(b):
        index.setdefault(line, []).append(j)
    seen = set()
    for i, (_, line) in enumerate(a):
        for j in index.get(line, ()):
            if (i, j) in seen:
                continue
            k = 0
            while i + k < len(a) and j + k < len(b) and a[i + k][1] == b[j + k][1]:
                seen.add((i + k, j + k))
                k += 1
            if k >= min_run:
                yield k, a[i][0], a[i + k - 1][0], b[j][0], b[j + k - 1][0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".", help="repository root (default: cwd)")
    parser.add_argument("--base", required=True, help="branch base commit")
    parser.add_argument("--ref", action="append", default=[], help="git ref of a reference tree")
    parser.add_argument(
        "--dir", action="append", default=[], help="directory checkout of a reference"
    )
    parser.add_argument(
        "--dir-filter", action="append", default=[], help="substring filter for --dir paths"
    )
    parser.add_argument("--min-run", type=int, default=10)
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    mine = {name: normalise((repo / name).read_text()) for name in added_files(repo, args.base)}
    references: List[Tuple[str, List[Line]]] = []
    for ref in args.ref:
        merge_base = git(repo, "merge-base", args.base, ref).strip()
        references.extend((n, normalise(t)) for n, t in ref_files(repo, ref, merge_base))
    for directory in args.dir:
        references.extend((n, normalise(t)) for n, t in dir_files(Path(directory), args.dir_filter))

    print(f"branch files: {len(mine)}; reference files: {len(references)}; min run: {args.min_run}")
    total = 0
    for mine_name, mine_lines in mine.items():
        for ref_name, ref_lines in references:
            for k, a0, a1, b0, b1 in matching_runs(mine_lines, ref_lines, args.min_run):
                total += 1
                print(f"MATCH {k:3d} lines  {mine_name}:{a0}-{a1}  <->  {ref_name}:{b0}-{b1}")
    print(f"total matching runs: {total}")


if __name__ == "__main__":
    main()
