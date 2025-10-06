"""
Convenience CLI: prepare structural training data from FASTA + GTF.

This script delegates to prepare_gencode.main() with the same arguments.
Usage:
  python -m betadogma.data.prepare_data \
      --fasta /path/GRCh38.fa \
      --gtf /path/gencode.v44.gtf \
      --out data/cache/gencode_v44_structural \
      --window 131072 --stride 65536 --bin-size 128 --chroms chr1,chr2
"""

from __future__ import annotations
import argparse
from . import prepare_gencode


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--gtf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=int, default=131072)
    ap.add_argument("--stride", type=int, default=65536)
    ap.add_argument("--bin-size", type=int, default=128)
    ap.add_argument("--chroms", type=str, default="")
    ap.add_argument("--max-shard-bases", type=int, default=50_000_000)
    return ap.parse_args()


def main():
    # This simply forwards to prepare_gencode with the same signature.
    args = parse_args()
    # prepare_gencode.main() already handles the exact same arg set.
    prepare_gencode.main()


if __name__ == "__main__":
    main()