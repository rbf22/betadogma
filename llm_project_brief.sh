#!/usr/bin/env bash
set -euo pipefail

# llm_project_brief.sh
# Create an LLM-friendly Markdown brief of a Python project:
#  1) Project overview + tree
#  2) Classes & functions (parsed via AST)
#  3) File inventory (path, size, sha256)
#  4) Optional code previews (first N lines) in delimited blocks
#
# Usage:
#   ./llm_project_brief.sh [PROJECT_ROOT] > project_brief.md
#
# Tunables (env vars):
#   PREVIEW_ENABLE=1           # include code previews (default 1)
#   PREVIEW_MAX_LINES=60       # lines per preview block (default 60)
#   PREVIEW_MAX_BYTES=40000    # hard cap per file preview (default 40k)
#   PREVIEW_ONLY_PY=1          # preview only .py files when enabled (default 1)
#   EXCLUDE_DIRS=".git .mypy_cache __pycache__ .venv venv build dist .tox .idea .vscode .ruff_cache .pytest_cache"
#   EXCLUDE_FILES="*.pyc *.pyo *.pyd *.so *.dll *.dylib *.egg-info *.egg"
#
# Requires: bash, python3, tree, sha256sum (or shasum -a 256 on macOS)

ROOT="${1:-.}"

PREVIEW_ENABLE="${PREVIEW_ENABLE:-1}"
PREVIEW_MAX_LINES="${PREVIEW_MAX_LINES:-60}"
PREVIEW_MAX_BYTES="${PREVIEW_MAX_BYTES:-40000}"
PREVIEW_ONLY_PY="${PREVIEW_ONLY_PY:-1}"

EXCLUDE_DIRS="${EXCLUDE_DIRS:-.git .hg .svn .mypy_cache .pytest_cache __pycache__ .venv venv build dist data .tox .idea .vscode .ruff_cache}"
EXCLUDE_FILES="${EXCLUDE_FILES:-*.pyc *.pyo *.pyd *.so *.dll *.dylib *.egg-info *.egg}"

# Tools checks
command -v python3 >/dev/null 2>&1 || { echo "python3 required" >&2; exit 1; }
command -v tree    >/dev/null 2>&1 || { echo "tree required (e.g., apt-get install tree)" >&2; exit 1; }

# sha256 helper (Linux/macOS)
if command -v sha256sum >/dev/null 2>&1; then
  SHA256="sha256sum"
elif command -v shasum >/dev/null 2>&1; then
  SHA256="shasum -a 256"
else
  echo "sha256sum or shasum required" >&2; exit 1
fi

# Build ignore regex for tree
TREE_IGNORE="$(printf '%s|' $EXCLUDE_DIRS $EXCLUDE_FILES | sed 's/|$//')"

# Markdown helpers
hrule() { echo; printf -- '---\n\n'; }

title() { echo "# $1"; echo; }
h2() { echo "## $1"; echo; }
h3() { echo "### $1"; echo; }

# 1) Project Overview + Tree
title "Project Overview"

# Basic detected metadata
h3 "Detected Environment"
echo "- Root: \`$(cd "$ROOT" && pwd)\`"
echo "- Date: \`$(date -u +"%Y-%m-%d %H:%M:%S UTC")\`"
echo "- Python: \`$(python3 -V 2>&1)\`"
echo

h2 "Structure (tree)"
echo '```text'
# --dirsfirst for readability; -I to ignore patterns; -a to include dotfiles
tree -a --dirsfirst -I "$TREE_IGNORE" "$ROOT" || true
echo '```'
hrule

# 2) Classes & Functions (AST-based)
h2 "Python API Map (classes & functions)"
python3 - "$ROOT" $EXCLUDE_DIRS $EXCLUDE_FILES <<'PY'
import ast, os, sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
parts = sys.argv[2:]
half = len(parts)//2
ex_dirs = set(parts[:half])
ex_files = set(parts[half:])

def is_ignored_dir(name):
    return name in ex_dirs

def is_ignored_file(path: Path):
    name = path.name
    for pat in ex_files:
        if Path(name).match(pat):
            return True
    return False

def walk_py_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not is_ignored_dir(d)]
        for fn in filenames:
            p = Path(dirpath) / fn
            if p.suffix == ".py" and not is_ignored_file(p):
                yield p

def extract_api(py_path: Path):
    try:
        src = py_path.read_text(encoding="utf-8")
    except Exception:
        return None
    try:
        tree = ast.parse(src, filename=str(py_path))
    except SyntaxError:
        return None

    classes, functions = [], []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            methods = []
            for cnode in node.body:
                if isinstance(cnode, ast.FunctionDef):
                    args = [a.arg for a in cnode.args.args]
                    methods.append((cnode.name, args))
            bases = []
            for b in node.bases:
                if isinstance(b, ast.Name):
                    bases.append(b.id)
                elif isinstance(b, ast.Attribute):
                    # module.Class
                    parts = []
                    cur = b
                    while isinstance(cur, ast.Attribute):
                        parts.append(cur.attr)
                        cur = cur.value
                    if isinstance(cur, ast.Name):
                        parts.append(cur.id)
                    bases.append(".".join(reversed(parts)))
                else:
                    bases.append(ast.dump(b, annotate_fields=False))
            classes.append((node.name, bases, methods))
        elif isinstance(node, ast.FunctionDef):
            args = [a.arg for a in node.args.args]
            functions.append((node.name, args))
    return classes, functions

any_output = False
for f in sorted(walk_py_files(root)):
    res = extract_api(f)
    if not res:
        continue
    classes, functions = res
    if not classes and not functions:
        continue
    any_output = True
    print(f"**File:** `{f}`")
    if classes:
        print("- **Classes**")
        for cname, bases, methods in classes:
            base_str = f"({', '.join(bases)})" if bases else ""
            print(f"  - {cname}{base_str}")
            for mname, margs in methods:
                print(f"    - `{mname}({', '.join(margs)})`")
    if functions:
        print("- **Functions**")
        for fname, fargs in functions:
            print(f"  - `{fname}({', '.join(fargs)})`")
    print()
if not any_output:
    print("_No top-level classes or functions found in .py files._")
PY
hrule

# 3) File Inventory (path, size, sha256)
h2 "File Inventory"

# Build prune for find
PRUNE_DIRS=()
for d in $EXCLUDE_DIRS; do PRUNE_DIRS+=( -name "$d" -o ); done
[[ ${#PRUNE_DIRS[@]} -gt 0 ]] && unset 'PRUNE_DIRS[${#PRUNE_DIRS[@]}-1]'

# Collect files, excluding binary junk patterns later
ALL_FILES=$(mktemp)
if [[ ${#PRUNE_DIRS[@]} -gt 0 ]]; then
  eval find "\"$ROOT\"" -type d \( "${PRUNE_DIRS[@]}" \) -prune -o -type f -print | sort > "$ALL_FILES"
else
  find "$ROOT" -type f -print | sort > "$ALL_FILES"
fi

# Build exclude regex for files
EX_RE="$(printf '%s|' $EXCLUDE_FILES | sed 's/|$//; s/\./\\./g; s/\*/.*/g')"

echo "| Path | Size (bytes) | SHA256 |"
echo "|------|---------------|--------|"
while IFS= read -r f; do
  rel="${f#"$ROOT"/}"
  [[ -n "$EX_RE" ]] && echo "$rel" | grep -Eq "$EX_RE" && continue
  sz="$(wc -c < "$f" | tr -d ' ')"
  sum="$($SHA256 "$f" 2>/dev/null | awk '{print $1}')"
  echo "| \`$rel\` | $sz | \`$sum\` |"
done < "$ALL_FILES"
hrule

# 4) (Optional) Code Previews in explicit LLM blocks
if [[ "$PREVIEW_ENABLE" == "1" ]]; then
  h2 "Code Previews"
  echo "_First ${PREVIEW_MAX_LINES} lines per file (capped at ${PREVIEW_MAX_BYTES} bytes)._"
  echo

  while IFS= read -r f; do
    rel="${f#"$ROOT"/}"
    [[ -n "$EX_RE" ]] && echo "$rel" | grep -Eq "$EX_RE" && continue
    if [[ "$PREVIEW_ONLY_PY" == "1" && "${f##*.}" != "py" ]]; then
      continue
    fi
    # ensure readable text-ish files only
    if file -b "$f" | grep -qiE 'text|ascii|utf-8|unicode'; then
      echo "<<<FILE:$rel>>>"
      # head -n with byte cap safeguard
      awk -v max="$PREVIEW_MAX_BYTES" -v n="$PREVIEW_MAX_LINES" '
        BEGIN{bytes=0; lines=0}
        {
          len=length($0)+1;
          if (bytes+len>max || lines>=n) {exit}
          print $0;
          bytes+=len; lines++
        }' "$f"
      echo "<<<END:$rel>>>"
      echo
    fi
  done < "$ALL_FILES"
fi

# Cleanup
rm -f "$ALL_FILES"
