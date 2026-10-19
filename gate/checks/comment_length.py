#!/usr/bin/env python3
"""Fail when a comment in a tracked Python, Go or shell file is over its limit.

- A file docstring has at most MAX_FILE_DOC lines.
- A function or class docstring has at most MAX_DEF_DOC lines.
- Any other run of consecutive whole-line comments has at most MAX_COMMENT_RUN lines.

A Python docstring is the string literal that opens the module, function or class. Go and
shell have no docstrings. There, the comment block that opens the file is the file
docstring. A block directly above a declaration is the docstring of that declaration: a Go
`func`, `type`, `var` or `const`, or a shell function.

Exit codes: 0 ok, 1 a comment over its limit.
"""

from __future__ import annotations

import ast
import io
import re
import subprocess
import sys
import tokenize
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

MAX_FILE_DOC = 20
MAX_DEF_DOC = 5
MAX_COMMENT_RUN = 2

Span = tuple[int, int]
GO_DECLARATION = re.compile(r"(func|type|var|const)\b")
SHELL_FUNCTION = re.compile(r"\s*(function\s+)?[A-Za-z_][\w-]*\s*\(\)")


def runs(lines: list[int]) -> list[Span]:
    """The (first, last) runs of consecutive numbers in `lines`, which must be sorted."""
    out: list[Span] = []
    for n in lines:
        if out and n == out[-1][1] + 1:
            out[-1] = (out[-1][0], n)
        else:
            out.append((n, n))
    return out


def python(text: str) -> tuple[Span | None, list[Span], list[int]]:
    """The module docstring, each function and class docstring, and the comment-only lines."""

    def docstring(node: ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> Span | None:
        first = node.body[0] if node.body else None
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            return first.lineno, first.end_lineno or first.lineno
        return None

    tree = ast.parse(text)
    definitions = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    defs = [span for node in ast.walk(tree) if isinstance(node, definitions) and (span := docstring(node))]
    comments = [
        tok.start[0]
        for tok in tokenize.generate_tokens(io.StringIO(text).readline)
        if tok.type == tokenize.COMMENT and not tok.line[: tok.start[1]].strip()
    ]
    return docstring(tree), defs, comments


def go_comment_lines(text: str) -> list[int]:
    """The lines that hold only a comment. Comment markers in strings and runes do not count."""
    comments = []
    in_raw = in_block = False
    for n, line in enumerate(text.splitlines(), 1):
        code = comment = False
        i = 0
        while i < len(line):
            c = line[i]
            if in_block:
                comment = True
                end = line.find("*/", i)
                if end < 0:
                    break
                in_block, i = False, end + 2
            elif in_raw:
                code = True
                end = line.find("`", i)
                if end < 0:
                    break
                in_raw, i = False, end + 1
            elif line.startswith("//", i):
                comment = True
                break
            elif line.startswith("/*", i):
                in_block, i = True, i + 2
            elif c == "`":
                code, in_raw, i = True, True, i + 1
            elif c in "\"'":
                code, i = True, i + 1
                while i < len(line) and line[i] != c:
                    i += 2 if line[i] == "\\" else 1
                i += 1
            else:
                code = code or not c.isspace()
                i += 1
        if comment and not code:
            comments.append(n)
    return comments


def hash_comment_lines(text: str) -> list[int]:
    """The lines whose first non-blank character is `#`, except a shebang on line 1.

    The lines are not parsed as shell, so a `#` line in a heredoc counts too.
    """
    return [
        n
        for n, line in enumerate(text.splitlines(), 1)
        if line.lstrip().startswith("#") and not (n == 1 and line.startswith("#!"))
    ]


def blocks(
    text: str, comments: list[int], declaration: re.Pattern[str]
) -> tuple[Span | None, list[Span], list[int]]:
    """For Go and shell: the opening block, the blocks above a declaration, and the comment lines."""
    lines = text.splitlines()
    first = next(
        (n for n, line in enumerate(lines, 1) if line.strip() and not (n == 1 and line.startswith("#!"))),
        None,
    )
    found = runs(comments)
    opening = found[0] if found and found[0][0] == first else None
    defs = [(a, b) for a, b in found if (a, b) != opening and b < len(lines) and declaration.match(lines[b])]
    return opening, defs, comments


def is_shell(path: Path) -> bool:
    if path.suffix == ".sh":
        return True
    if path.suffix:
        return False
    with path.open("rb") as f:
        head = f.readline()
    return head.startswith(b"#!") and head.rstrip().endswith((b"sh", b"bash"))


def check(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".py":
        file_doc, def_docs, comments = python(text)
    elif path.suffix == ".go":
        file_doc, def_docs, comments = blocks(text, go_comment_lines(text), GO_DECLARATION)
    else:
        file_doc, def_docs, comments = blocks(text, hash_comment_lines(text), SHELL_FUNCTION)
    problems = []
    if file_doc and file_doc[1] - file_doc[0] + 1 > MAX_FILE_DOC:
        problems.append(
            f"{file_doc[0]}: file docstring is {file_doc[1] - file_doc[0] + 1} lines (max {MAX_FILE_DOC})"
        )
    for first, last in def_docs:
        if last - first + 1 > MAX_DEF_DOC:
            problems.append(f"{first}: docstring is {last - first + 1} lines (max {MAX_DEF_DOC})")
    for span in runs(comments):
        if span != file_doc and span not in def_docs and span[1] - span[0] + 1 > MAX_COMMENT_RUN:
            problems.append(
                f"{span[0]}: {span[1] - span[0] + 1} consecutive comment lines (max {MAX_COMMENT_RUN})"
            )
    return problems


def main() -> int:
    tracked = subprocess.run(
        ["git", "ls-files", "-z"], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.split("\0")
    failed = False
    for name in filter(None, tracked):
        path = REPO / name
        if path.is_file() and (path.suffix in (".py", ".go") or is_shell(path)):
            for problem in check(path):
                failed = True
                print(f"{name}:{problem}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
