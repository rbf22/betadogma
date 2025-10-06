"""
Encode variants (SNP/indel) into an auxiliary input channel aligned with the sequence window.
"""

from __future__ import annotations
import re
from typing import Any, Dict, Optional, List, Tuple


# -----------------------------
# Parsing / normalization
# -----------------------------

_VAR_RE = re.compile(
    r'^(?P<chrom>[^:]+):(?P<pos>\d+)\s*(?P<ref>[ACGTN\-]+)>(?P<alt>[ACGTN\-]+)$',
    re.IGNORECASE,
)


def parse_variant_spec(spec: str) -> Dict[str, Any]:
    """
    Parse a simple VCF-like variant spec "chr:POSREF>ALT".
    Returns: dict with chrom, pos0, pos1, ref, alt, type, spec
    """
    m = _VAR_RE.match(spec.replace(" ", ""))
    if not m:
        raise ValueError(f"Unrecognized variant spec: {spec}")
    chrom = m.group("chrom")
    pos0 = int(m.group("pos")) - 1
    ref = m.group("ref").upper()
    alt = m.group("alt").upper()

    if ref != "-" and alt != "-" and len(ref) == len(alt) == 1:
        vtype = "SNP"
    elif ref == "-" and alt != "-":
        vtype = "INS"
    elif alt == "-" and ref != "-":
        vtype = "DEL"
    else:
        vtype = "INDEL"

    pos1 = pos0 if ref == "-" else pos0 + len(ref)
    return {"chrom": chrom, "pos0": pos0, "pos1": pos1, "ref": ref, "alt": alt, "type": vtype, "spec": spec}


def encode_variant(spec: str, window: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Align variant to a sequence window.
    window = {"chrom": "...", "start": int, "end": int}
    """
    v = parse_variant_spec(spec)
    in_window = False
    in_idx = None
    span = None
    if window and window.get("chrom") == v["chrom"]:
        w0, w1 = window["start"], window["end"]
        if w0 <= v["pos0"] < w1:
            in_window = True
            in_idx = v["pos0"] - w0
        if v["ref"] != "-" and w0 < v["pos1"] and v["pos0"] < w1:
            span = (max(0, v["pos0"] - w0), min(w1 - w0, v["pos1"] - w0))
    return {**v, "in_window": in_window, "in_window_idx": in_idx, "span_in_window": span}


# -----------------------------
# Sequence application
# -----------------------------

def apply_variant_to_sequence(seq: str, window_start: int, var: Dict[str, Any]) -> str:
    """Apply a parsed variant to a sequence window string."""
    idx = var.get("in_window_idx", var["pos0"] - window_start)
    if idx < 0 or idx > len(seq):
        return seq

    ref, alt, vtype = var["ref"], var["alt"], var["type"]

    if vtype == "SNP" and 0 <= idx < len(seq):
        return seq[:idx] + alt + seq[idx + 1 :]
    elif vtype == "INS":
        return seq[:idx] + alt + seq[idx :]
    elif vtype in {"DEL", "INDEL"}:
        span = var.get("span_in_window")
        if span:
            s, e = span
        else:
            s, e = idx, idx + len(ref)
        return seq[:s] + ("" if alt == "-" else alt) + seq[e:]
    return seq


# -----------------------------
# Channel encoding
# -----------------------------

def build_variant_channels(seq_len: int, var: Dict[str, Any]) -> Dict[str, List[int]]:
    """
    Build simple binary per-base channels (snp/ins/del/any).
    """
    ch = {k: [0] * seq_len for k in ("snp", "ins", "del", "any")}
    idx, span = var.get("in_window_idx"), var.get("span_in_window")

    if var["type"] == "SNP" and idx is not None and 0 <= idx < seq_len:
        ch["snp"][idx] = 1
    elif var["type"] == "INS" and idx is not None and 0 <= idx < seq_len:
        ch["ins"][idx] = 1
    elif var["type"] in {"DEL", "INDEL"} and span:
        s, e = span
        for i in range(max(0, s), min(seq_len, e)):
            ch["del"][i] = 1

    for k in ("snp", "ins", "del"):
        for i, v in enumerate(ch[k]):
            if v:
                ch["any"][i] = 1
    return ch