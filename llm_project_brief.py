#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
llm_project_brief.py
Create an LLM-friendly Markdown brief of a Python project:
  1) Project overview + tree
  2) Classes & functions (parsed via AST)
  3) File inventory (path, size, sha256)
  4) Optional code previews (first N lines) in delimited blocks

Usage:
  python llm_project_brief.py [ROOT] > project_brief.md

Options:
  --preview                  include code previews (default on)
  --no-preview               disable code previews
  --preview-max-lines N      lines per file (default 60)
  --preview-max-bytes N      byte cap per file (default 40000)
  --preview-only-py          preview only .py files (default on)
  --no-preview-only-py       preview any text file
  --exclude-dir NAME ...     directory names to exclude (repeatable)
  --exclude-file GLOB ...    file glob patterns to exclude (repeatable)
  --json JSON_PATH           also write a machine-readable JSON with the same sections
"""
from __future__ import annotations

import argparse
import ast
import fnmatch
import hashlib
import io
import json
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Iterator, List, Tuple, Dict, Any
from datetime import datetime, UTC


# ------------------------ Configuration ------------------------
DEFAULT_EXCLUDE_DIRS = [
    ".git", "__pycache__", ".venv", "venv", "data", "build", "dist", ".tox",
    ".idea", ".vscode", ".ruff_cache", ".pytest_cache", ".mypy_cache",
]
DEFAULT_EXCLUDE_FILES = [
    "*.pyc", "*.pyo", "*.pyd", "*.so", "*.dll", "*.dylib", "*.egg-info", "*.egg",
]

# --------------------------- Helpers ---------------------------

def human_path(p: Path, root: Path) -> str:
    try:
        return str(p.relative_to(root))
    except ValueError:
        return str(p)

def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def is_text_file(path: Path, max_probe_bytes: int = 4096) -> bool:
    # Simple heuristic: if it decodes as utf-8 (with errors ignored) and has few NULs
    try:
        b = path.open("rb").read(max_probe_bytes)
    except Exception:
        return False
    if not b:
        return True
    nul = b.count(b"\x00")
    try:
        b.decode("utf-8", errors="ignore")
    except Exception:
        return False
    return nul == 0

def walk_tree(root: Path, exclude_dirs: List[str]) -> Iterator[Tuple[Path, List[Path], List[Path]]]:
    for dirpath, dirnames, filenames in os.walk(root):
        p = Path(dirpath)
        # dirs-first, apply excludes by name
        dirnames[:] = sorted([d for d in dirnames if d not in exclude_dirs])
        yield p, [p / d for d in dirnames], sorted(p / f for f in filenames)

def render_tree(root: Path, exclude_dirs: List[str], exclude_files: List[str]) -> str:
    # Minimal tree printer (dirs first)
    lines: List[str] = []
    def _should_skip_file(name: str) -> bool:
        return any(fnmatch.fnmatch(name, pat) for pat in exclude_files)

    prefix_map: Dict[Path, str] = {}

    def list_dir(base: Path, prefix: str = ""):
        entries_dirs: List[Path] = []
        entries_files: List[Path] = []
        try:
            for entry in sorted(base.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
                if entry.is_dir():
                    if entry.name in exclude_dirs:
                        continue
                    entries_dirs.append(entry)
                else:
                    if _should_skip_file(entry.name):
                        continue
                    entries_files.append(entry)
        except PermissionError:
            return
        entries = entries_dirs + entries_files
        total = len(entries)
        for i, entry in enumerate(entries):
            last = (i == total - 1)
            connector = "└── " if last else "├── "
            lines.append(f"{prefix}{connector}{entry.name}")
            if entry.is_dir():
                extension = "    " if last else "│   "
                list_dir(entry, prefix + extension)
    lines.append(root.name)
    list_dir(root)
    return "\n".join(lines)

# ----------------------- AST extraction -----------------------

@dataclass
class MethodInfo:
    name: str
    args: List[str]

@dataclass
class ClassInfo:
    name: str
    bases: List[str]
    methods: List[MethodInfo]

@dataclass
class FunctionInfo:
    name: str
    args: List[str]

def parse_api(py_path: Path) -> Tuple[List[ClassInfo], List[FunctionInfo]]:
    try:
        src = py_path.read_text(encoding="utf-8")
    except Exception:
        return [], []
    try:
        tree = ast.parse(src, filename=str(py_path))
    except SyntaxError:
        return [], []

    classes: List[ClassInfo] = []
    functions: List[FunctionInfo] = []

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            methods: List[MethodInfo] = []
            for cnode in node.body:
                if isinstance(cnode, ast.FunctionDef):
                    args = [a.arg for a in cnode.args.args]  # no *args/**kwargs expansion here
                    methods.append(MethodInfo(cnode.name, args))
            bases: List[str] = []
            for b in node.bases:
                if isinstance(b, ast.Name):
                    bases.append(b.id)
                elif isinstance(b, ast.Attribute):
                    parts = []
                    cur: Any = b
                    while isinstance(cur, ast.Attribute):
                        parts.append(cur.attr)
                        cur = cur.value
                    if isinstance(cur, ast.Name):
                        parts.append(cur.id)
                    bases.append(".".join(reversed(parts)))
                else:
                    bases.append(ast.dump(b, annotate_fields=False))
            classes.append(ClassInfo(node.name, bases, methods))
        elif isinstance(node, ast.FunctionDef):
            args = [a.arg for a in node.args.args]
            functions.append(FunctionInfo(node.name, args))
    return classes, functions

# ----------------------- Inventory & previews -----------------------

@dataclass
class FileRecord:
    path: str
    size: int
    sha256: str

def collect_files(root: Path, exclude_dirs: List[str], exclude_files: List[str]) -> List[Path]:
    out: List[Path] = []
    for dp, dnames, fnames in walk_tree(root, exclude_dirs):
        for f in fnames:
            if any(fnmatch.fnmatch(f.name, pat) for pat in exclude_files):
                continue
            out.append(f)
    return sorted(out)

def preview_file(path: Path, max_lines: int, max_bytes: int) -> str:
    """Preview file with optional caps. <=0 disables that cap."""
    no_line_cap = max_lines is None or max_lines <= 0
    no_byte_cap = max_bytes is None or max_bytes <= 0

    # Fast path: both caps disabled → read all
    if no_line_cap and no_byte_cap:
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return ""

    buf = io.StringIO()
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            total_bytes = 0
            for i, line in enumerate(f, 1):
                enc = line
                new_bytes = len(enc.encode("utf-8"))

                # only enforce a cap if it's enabled
                if (not no_line_cap and i > max_lines) or (not no_byte_cap and (total_bytes + new_bytes) > max_bytes):
                    break

                buf.write(line.rstrip("\n"))
                buf.write("\n")
                total_bytes += new_bytes
    except Exception:
        return ""
    return buf.getvalue()

# ----------------------------- Main -----------------------------

def main(argv: List[str]) -> int:
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("root", nargs="?", default=".", help="Project root")
    p.add_argument("--preview", dest="preview", action="store_true", default=True)
    p.add_argument("--no-preview", dest="preview", action="store_false")
    p.add_argument("--preview-max-lines", type=int, default=60)
    p.add_argument("--preview-max-bytes", type=int, default=40000)
    p.add_argument("--preview-only-py", dest="preview_only_py", action="store_true", default=True)
    p.add_argument("--no-preview-only-py", dest="preview_only_py", action="store_false")
    p.add_argument("--exclude-dir", action="append", default=None)
    p.add_argument("--exclude-file", action="append", default=None)
    p.add_argument("--json", dest="json_out", default=None, help="Optional JSON output path")
    args = p.parse_args(argv)

    root = Path(args.root).resolve()
    exclude_dirs = args.exclude_dir or DEFAULT_EXCLUDE_DIRS[:]
    exclude_files = args.exclude_file or DEFAULT_EXCLUDE_FILES[:]

    # Header
    print("# Project Overview\n")
    print("### Detected Environment")
    print(f"- Root: `{root}`")
    print(f"- Date: `{datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S %Z')}`")
    print(f"- Python: `{sys.version.split()[0]}`\n")


    # Tree
    print("## Structure (tree)")
    print("```text")
    print(render_tree(root, exclude_dirs, exclude_files))
    print("```\n---\n")

    # API map
    print("## Python API Map (classes & functions)")
    api_map: Dict[str, Dict[str, Any]] = {}
    any_api = False
    for py in collect_files(root, exclude_dirs, exclude_files):
        if py.suffix != ".py":
            continue
        classes, functions = parse_api(py)
        if not classes and not functions:
            continue
        any_api = True
        print(f"**File:** `{human_path(py, root)}`")
        if classes:
            print("- **Classes**")
            for c in classes:
                base_str = f"({', '.join(c.bases)})" if c.bases else ""
                print(f"  - {c.name}{base_str}")
                for m in c.methods:
                    print(f"    - `{m.name}({', '.join(m.args)})`")
        if functions:
            print("- **Functions**")
            for f in functions:
                print(f"  - `{f.name}({', '.join(f.args)})`")
        print()
        api_map[str(py)] = {
            "classes": [asdict(c) for c in classes],
            "functions": [asdict(f) for f in functions],
        }
    if not any_api:
        print("_No top-level classes or functions found in .py files._")
    print("\n---\n")

    # File inventory
    print("## File Inventory")
    files = collect_files(root, exclude_dirs, exclude_files)
    print("| Path | Size (bytes) | SHA256 |")
    print("|------|---------------|--------|")
    inventory: List[FileRecord] = []
    for f in files:
        try:
            size = f.stat().st_size
            digest = sha256sum(f)
        except Exception:
            size, digest = -1, ""
        print(f"| `{human_path(f, root)}` | {size} | `{digest}` |")
        inventory.append(FileRecord(path=human_path(f, root), size=size, sha256=digest))
    print("\n---\n")

    # Optional previews
    preview_blocks: Dict[str, str] = {}
    if args.preview:
        print("## Code Previews")
        print(f"_First {args.preview_max_lines} lines per file (capped at {args.preview_max_bytes} bytes)._")
        print()
        for f in files:
            if args.preview_only_py and f.suffix != ".py":
                continue
            if not is_text_file(f):
                continue
            content = preview_file(f, args.preview_max_lines, args.preview_max_bytes)
            if not content.strip():
                continue
            rel = human_path(f, root)
            print(f"<<<FILE:{rel}>>>")
            # Don’t wrap in Markdown fences; use explicit markers for easier chunking
            sys.stdout.write(content)
            if not content.endswith("\n"):
                print()
            print(f"<<<END:{rel}>>>\n")
            preview_blocks[rel] = content

    # Optional JSON mirror
    if args.json_out:
        payload = {
            "root": str(root),
            "environment": {
                "python": sys.version.split()[0],
                "utc": __import__('datetime').datetime.utcnow().isoformat() + "Z",
            },
            "exclude_dirs": exclude_dirs,
            "exclude_files": exclude_files,
            "api_map": api_map,
            "inventory": [asdict(rec) for rec in inventory],
            "previews": preview_blocks,
        }
        Path(args.json_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
