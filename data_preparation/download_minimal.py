#!/usr/bin/env python3
"""
BetaDogma Data Downloader - Downloads all required genomic data files.

This script downloads:
  - Reference genome (GRCh38)
  - GENCODE annotations (v26 for GTEx compatibility + v44 latest)
  - GTEx v8 transcript expression, junctions, and metadata
  - Variant databases (ClinVar, 1000 Genomes)
  - SpliceVarDB (local copy)

Usage:
    python download_data.py
    python download_data.py --skip-existing
    python download_data.py --output-dir /path/to/data
    python download_data.py --essential-only  # Skip large optional files
"""

import argparse
import hashlib
import requests
import shutil
from pathlib import Path
from tqdm import tqdm
from typing import Optional


# =============================================================================
# FILE DEFINITIONS
# =============================================================================

DOWNLOADS = {
    # -------------------------------------------------------------------------
    # REFERENCE GENOME
    # -------------------------------------------------------------------------
    'genome': [
        {
            'name': 'Reference Genome (GRCh38)',
            'url': 'https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/GRCh38.primary_assembly.genome.fa.gz',
            'output': 'genome/GRCh38.primary_assembly.genome.fa.gz',
            'size': '938 MB',
            'essential': True,
            'description': 'Human reference genome GRCh38 primary assembly'
        },
    ],
    
    # -------------------------------------------------------------------------
    # GENCODE ANNOTATIONS
    # -------------------------------------------------------------------------
    'gencode': [
        {
            'name': 'GENCODE v26 Comprehensive Annotation',
            'url': 'https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_26/gencode.v26.annotation.gtf.gz',
            'output': 'gencode/v26/gencode.v26.annotation.gtf.gz',
            'size': '41 MB',
            'essential': True,
            'description': 'GENCODE v26 - matches GTEx v8 annotation'
        },
        {
            'name': 'GENCODE v26 Transcript Sequences',
            'url': 'https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_26/gencode.v26.transcripts.fa.gz',
            'output': 'gencode/v26/gencode.v26.transcripts.fa.gz',
            'size': '58 MB',
            'essential': True,
            'description': 'Transcript sequences for v26'
        },
        {
            'name': 'GENCODE v26 Protein Sequences',
            'url': 'https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_26/gencode.v26.pc_translations.fa.gz',
            'output': 'gencode/v26/gencode.v26.pc_translations.fa.gz',
            'size': '17 MB',
            'essential': True,
            'description': 'Protein-coding translations for v26'
        },
        {
            'name': 'GENCODE v44 Comprehensive Annotation',
            'url': 'https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/gencode.v44.annotation.gtf.gz',
            'output': 'gencode/v44/gencode.v44.annotation.gtf.gz',
            'size': '49 MB',
            'essential': False,
            'description': 'GENCODE v44 - latest annotation (optional)'
        },
    ],
    
    # -------------------------------------------------------------------------
    # GTEx v8 DATA
    # -------------------------------------------------------------------------
    'gtex': [
        {
            'name': 'GTEx v8 Transcript TPM',
            'url': 'https://storage.googleapis.com/adult-gtex/bulk-gex/v8/rna-seq/GTEx_Analysis_2017-06-05_v8_RSEMv1.3.0_transcript_tpm.gct.gz',
            'output': 'gtex/v8/expression/GTEx_v8_transcript_tpm.gct.gz',
            'size': '~6 GB',
            'essential': True,
            'description': '⭐ Transcript-level TPM for all GTEx samples - NEEDED FOR ISOFORM TRAINING'
        },
        {
            'name': 'GTEx v8 Gene TPM',
            'url': 'https://storage.googleapis.com/adult-gtex/bulk-gex/v8/rna-seq/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz',
            'output': 'gtex/v8/expression/GTEx_v8_gene_tpm.gct.gz',
            'size': '126 MB',
            'essential': False,
            'description': 'Gene-level TPM (optional, transcript-level is more useful)'
        },
        {
            'name': 'GTEx v8 Sample Attributes',
            'url': 'https://storage.googleapis.com/adult-gtex/annotations/v8/metadata-files/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt',
            'output': 'gtex/v8/metadata/GTEx_v8_sample_attributes.txt',
            'size': '8.8 MB',
            'essential': True,
            'description': 'Sample metadata - maps sample IDs to tissues'
        },
        {
            'name': 'GTEx v8 Subject Phenotypes',
            'url': 'https://storage.googleapis.com/adult-gtex/annotations/v8/metadata-files/GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt',
            'output': 'gtex/v8/metadata/GTEx_v8_subject_phenotypes.txt',
            'size': '152 KB',
            'essential': False,
            'description': 'Subject-level phenotype data'
        },
        {
            'name': 'GTEx v8 Splice Junctions',
            'url': 'https://storage.googleapis.com/adult-gtex/bulk-gex/v8/rna-seq/GTEx_Analysis_2017-06-05_v8_STARv2.5.3a_junctions.gct.gz',
            'output': 'gtex/v8/junctions/GTEx_v8_junctions.gct.gz',
            'size': '2.4 GB',
            'essential': False,
            'description': 'Splice junction read counts (optional)'
        },
    ],
    
    # -------------------------------------------------------------------------
    # VARIANT DATABASES
    # -------------------------------------------------------------------------
    'variants': [
        {
            'name': 'ClinVar (GRCh38)',
            'url': 'https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz',
            'output': 'variants/clinvar/clinvar.vcf.gz',
            'size': '~2 GB',
            'essential': True,
            'description': 'ClinVar pathogenic and benign variants'
        },
        {
            'name': 'ClinVar Index',
            'url': 'https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz.tbi',
            'output': 'variants/clinvar/clinvar.vcf.gz.tbi',
            'size': '~5 MB',
            'essential': True,
            'description': 'ClinVar index file'
        },
        {
            'name': '1000 Genomes Phase 3',
            'url': 'https://ftp.ensembl.org/pub/current_variation/vcf/homo_sapiens/1000GENOMES-phase_3.vcf.gz',
            'output': 'variants/1000genomes/1000GENOMES-phase_3.vcf.gz',
            'size': '~2.5 GB',
            'essential': True,
            'description': '1000 Genomes common variants (for benign controls)'
        },
        {
            'name': '1000 Genomes Index',
            'url': 'https://ftp.ensembl.org/pub/current_variation/vcf/homo_sapiens/1000GENOMES-phase_3.vcf.gz.csi',
            'output': 'variants/1000genomes/1000GENOMES-phase_3.vcf.gz.csi',
            'size': '~500 KB',
            'essential': True,
            'description': '1000 Genomes index file'
        },
    ],
}

# Local files to copy (if available)
LOCAL_FILES = [
    {
        'name': 'SpliceVarDB VCF',
        'source': 'splicevar/splicevar_hg38.vcf.gz',
        'output': 'variants/splicevar/splicevar_hg38.vcf.gz',
        'essential': True,
        'description': 'Splice variants with functional annotations'
    },
    {
        'name': 'SpliceVarDB Index',
        'source': 'splicevar/splicevar_hg38.vcf.gz.tbi',
        'output': 'variants/splicevar/splicevar_hg38.vcf.gz.tbi',
        'essential': True,
        'description': 'SpliceVarDB index file'
    },
]


# =============================================================================
# DOWNLOAD FUNCTIONS
# =============================================================================

def download_file(url: str, output_path: Path, expected_size: Optional[str] = None) -> bool:
    """Download a file with progress bar and resume capability."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if file exists and get size for resume
    existing_size = 0
    if output_path.exists():
        existing_size = output_path.stat().st_size
    
    headers = {}
    if existing_size > 0:
        headers['Range'] = f'bytes={existing_size}-'
    
    try:
        response = requests.get(url, stream=True, headers=headers, timeout=60)
        
        # Handle resume
        if response.status_code == 416:  # Range not satisfiable
            print(f"  ✓ Already complete ({existing_size / (1024**2):.1f} MB)")
            return True
        elif response.status_code == 206:  # Partial content (resume)
            print(f"  ↻ Resuming from {existing_size / (1024**2):.1f} MB")
            mode = 'ab'
        else:
            response.raise_for_status()
            mode = 'wb'
            existing_size = 0
        
        total_size = int(response.headers.get('content-length', 0)) + existing_size
        
        with open(output_path, mode) as f:
            with tqdm(
                total=total_size,
                initial=existing_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
                desc=f"     "
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        final_size = output_path.stat().st_size / (1024**2)
        print(f"  ✓ Complete ({final_size:.1f} MB)\n")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"  ✗ Download failed: {e}")
        if output_path.exists() and existing_size == 0:
            output_path.unlink()
        return False
    except KeyboardInterrupt:
        print(f"\n  ⚠ Interrupted - progress saved, can resume later")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def copy_local_file(source_path: Path, output_path: Path) -> bool:
    """Copy a local file if it exists."""
    if not source_path.exists():
        print(f"  ⚠ Source not found: {source_path}")
        print(f"     Place the file in the expected location and re-run")
        return False
    
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, output_path)
        size_mb = output_path.stat().st_size / (1024**2)
        print(f"  ✓ Copied ({size_mb:.1f} MB)\n")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}\n")
        return False


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Download all required data for BetaDogma",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Essential files (~12 GB):
  ✓ Reference genome (GRCh38)
  ✓ GENCODE v26 annotations (matches GTEx)
  ✓ GTEx v8 transcript TPM (~6GB) ⭐ KEY FILE FOR ISOFORMS
  ✓ GTEx v8 sample metadata
  ✓ ClinVar variants
  ✓ 1000 Genomes variants
  ✓ SpliceVarDB variants (local)

Optional files (~3 GB):
  - GTEx gene-level TPM
  - GTEx junctions
  - GENCODE v44 (latest)

Total: ~15 GB
        """
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="data/raw",
        help="Output directory (default: data/raw)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files that already exist"
    )
    parser.add_argument(
        "--essential-only",
        action="store_true",
        help="Only download essential files (skip optional)"
    )
    parser.add_argument(
        "--category",
        choices=['genome', 'gencode', 'gtex', 'variants', 'all'],
        default='all',
        help="Only download specific category"
    )
    parser.add_argument(
        "--no-local",
        action="store_true",
        help="Skip local file copying"
    )
    
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    
    print("=" * 80)
    print("🧬 BetaDogma Data Downloader")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Skip existing:    {args.skip_existing}")
    print(f"Essential only:   {args.essential_only}")
    print(f"Category:         {args.category}")
    print("=" * 80)
    print()
    
    stats = {'success': 0, 'skip': 0, 'fail': 0, 'total': 0}
    
    # Process each category
    categories = [args.category] if args.category != 'all' else DOWNLOADS.keys()
    
    for category in categories:
        if category not in DOWNLOADS:
            continue
            
        print(f"\n{'='*80}")
        print(f"📦 {category.upper()}")
        print(f"{'='*80}\n")
        
        for item in DOWNLOADS[category]:
            stats['total'] += 1
            
            # Skip non-essential if requested
            if args.essential_only and not item.get('essential', False):
                print(f"⊘ {item['name']}")
                print(f"   Skipping (optional file)")
                print(f"   {item['description']}\n")
                stats['skip'] += 1
                continue
            
            output_path = output_dir / item['output']
            
            # Skip if exists
            if output_path.exists() and args.skip_existing:
                size_mb = output_path.stat().st_size / (1024**2)
                print(f"⊙ {item['name']}")
                print(f"   Already exists ({size_mb:.1f} MB)")
                print(f"   {item['description']}\n")
                stats['skip'] += 1
                continue
            
            # Download
            print(f"📥 {item['name']}")
            print(f"   {item['description']}")
            print(f"   Size: {item['size']}")
            print(f"   URL: {item['url'][:80]}...")
            
            if download_file(item['url'], output_path, item['size']):
                stats['success'] += 1
            else:
                stats['fail'] += 1
    
    # Copy local files
    if not args.no_local:
        print(f"\n{'='*80}")
        print("📋 LOCAL FILES")
        print(f"{'='*80}\n")
        
        for item in LOCAL_FILES:
            stats['total'] += 1
            
            # Skip non-essential if requested
            if args.essential_only and not item.get('essential', False):
                print(f"⊘ {item['name']}")
                print(f"   Skipping (optional)")
                print(f"   {item['description']}\n")
                stats['skip'] += 1
                continue
            
            source_path = Path(item['source']).resolve()
            output_path = output_dir / item['output']
            
            # Skip if exists
            if output_path.exists() and args.skip_existing:
                size_mb = output_path.stat().st_size / (1024**2)
                print(f"⊙ {item['name']}")
                print(f"   Already exists ({size_mb:.1f} MB)\n")
                stats['skip'] += 1
                continue
            
            print(f"📋 {item['name']}")
            print(f"   {item['description']}")
            print(f"   Source: {source_path}")
            
            if copy_local_file(source_path, output_path):
                stats['success'] += 1
            else:
                stats['fail'] += 1
    
    # Summary
    print()
    print("=" * 80)
    print("📊 DOWNLOAD SUMMARY")
    print("=" * 80)
    print(f"  Total files:  {stats['total']}")
    print(f"  ✓ Success:    {stats['success']}")
    print(f"  ⊙ Skipped:    {stats['skip']}")
    print(f"  ✗ Failed:     {stats['fail']}")
    print("=" * 80)
    
    if stats['fail'] == 0:
        print("✅ All files processed successfully!")
        print()
        print("📁 Directory structure:")
        print(f"   {output_dir}/")
        print(f"   ├── genome/              # Reference genome")
        print(f"   ├── gencode/")
        print(f"   │   ├── v26/             # GTEx-compatible annotations ⭐")
        print(f"   │   └── v44/             # Latest annotations")
        print(f"   ├── gtex/")
        print(f"   │   └── v8/")
        print(f"   │       ├── expression/  # Transcript & gene TPM ⭐")
        print(f"   │       ├── metadata/    # Sample attributes ⭐")
        print(f"   │       └── junctions/   # Splice junctions")
        print(f"   └── variants/")
        print(f"       ├── clinvar/         # Pathogenic variants ⭐")
        print(f"       ├── 1000genomes/     # Benign variants ⭐")
        print(f"       └── splicevar/       # Splice variants ⭐")
        print()
        print("🎯 Next steps:")
        print("   1. Run preprocessing to combine GTEx + GENCODE")
        print("   2. Generate isoform annotations for training data")
        print("   3. Update training pipeline to use isoform labels")
        print()
        return 0
    else:
        print(f"❌ {stats['fail']} file(s) failed to download.")
        print()
        print("💡 Tips:")
        print("   - Check your internet connection")
        print("   - Re-run with --skip-existing to resume")
        print("   - Some files may require direct browser download")
        print()
        return 1


if __name__ == "__main__":
    exit(main())