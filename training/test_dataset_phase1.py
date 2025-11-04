#!/usr/bin/env python3
"""
Test script to verify Phase 1 dataset changes work correctly.

This tests:
1. Loading parquet files with new fields (isoforms, variants)
2. Parsing isoform metadata
3. Creating protein labels
4. Creating CDS boundary labels
5. Extracting NMD and expression labels

Run with: poetry run python training/test_dataset_phase1.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import json
import numpy as np
import pandas as pd
import torch
from training.dataset_helpers import (
    parse_isoforms,
    create_protein_labels,
    create_cds_boundary_labels,
    extract_canonical_isoform,
    to_tensor,
    AA_TO_IDX
)

def test_data_loading():
    """Test that we can load parquet files with new fields."""
    print("\n" + "="*60)
    print("TEST 1: Loading Parquet Files")
    print("="*60)
    
    data_dir = Path(__file__).parent.parent / 'data' / 'processed'
    
    # Try to find a parquet file
    train_files = list((data_dir / 'train').glob("*.parquet"))
    
    if not train_files:
        print("❌ No training files found!")
        print(f"   Looking in: {data_dir / 'train'}")
        return False
    
    print(f"✓ Found {len(train_files)} training files")
    test_file = train_files[0]
    print(f"  Testing with: {test_file.name}")
    
    # Try to load with new columns
    try:
        df = pd.read_parquet(
            test_file,
            columns=[
                'seq', 'chrom', 'start', 'end',
                'transcript_id', 'gene_name', 'strand',
                'donor', 'acceptor', 'tss', 'polya',
                'isoforms',  # NEW
                'variants'   # NEW
            ]
        )
        print(f"✓ Successfully loaded {len(df)} rows")
        print(f"  Columns: {list(df.columns)}")
        return True, df
        
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return False, None


def test_isoform_parsing(df):
    """Test parsing isoform metadata."""
    print("\n" + "="*60)
    print("TEST 2: Parsing Isoform Metadata")
    print("="*60)
    
    if df is None or len(df) == 0:
        print("❌ No data to test")
        return False
    
    # Get first row with isoforms that has a protein
    for idx, row in df.iterrows():
        isoforms_json = row.get('isoforms', '[]')
        if isoforms_json and isoforms_json != '[]':
            # Parse isoforms
            isoform_data = parse_isoforms(isoforms_json)
            
            # Extract canonical
            canonical = extract_canonical_isoform(isoform_data)
            
            # Skip if no protein (non-coding RNA)
            if not canonical['protein']:
                continue
            
            print(f"✓ Found row with protein-coding isoforms: index {idx}")
            print(f"  Number of isoforms: {len(isoform_data['proteins'])}")
            print(f"  Has canonical: {any(isoform_data['is_canonical'])}")
            
            print(f"\n  Canonical isoform:")
            protein = canonical.get('protein', '') or ''
            print(f"    Protein length: {len(protein)} AA")
            print(f"    CDS: {canonical['cds_start']} - {canonical['cds_end']}")
            print(f"    NMD: {canonical['nmd']}")
            print(f"    TPM: {canonical['tpm']:.2f}")
            
            if canonical['protein']:
                print(f"    Protein (first 20 AA): {canonical['protein'][:20]}...")
            
            return True, canonical
    
    print("⚠️  No rows with isoforms found")
    return False, None


def test_protein_labels(canonical, seq_len=300000):
    """Test creating protein labels."""
    print("\n" + "="*60)
    print("TEST 3: Creating Protein Labels")
    print("="*60)
    
    if not canonical or not canonical['protein']:
        print("❌ No canonical isoform to test")
        return False
    
    protein_labels = create_protein_labels(
        canonical['protein'],
        canonical['cds_start'],
        canonical['cds_end'],
        seq_len
    )
    
    print(f"✓ Created protein labels: shape {protein_labels.shape}")
    print(f"  Data type: {protein_labels.dtype}")
    print(f"  Unique values: {np.unique(protein_labels)}")
    print(f"  Non-ignore positions: {(protein_labels != -1).sum()}")
    
    # Check that CDS region has valid labels
    cds_start = canonical['cds_start']
    cds_end = canonical['cds_end']
    
    if cds_start >= 0 and cds_end > cds_start:
        cds_labels = protein_labels[cds_start:cds_end]
        print(f"  CDS region labels: {np.unique(cds_labels[cds_labels != -1])}")
        print(f"  Expected ~{(cds_end - cds_start) // 3} codons")
        print(f"  Got {(cds_labels != -1).sum() // 3} codons")
    
    return True


def test_cds_boundaries(canonical, seq_len=300000):
    """Test creating CDS boundary labels."""
    print("\n" + "="*60)
    print("TEST 4: Creating CDS Boundary Labels")
    print("="*60)
    
    if not canonical:
        print("❌ No canonical isoform to test")
        return False
    
    start_labels, end_labels = create_cds_boundary_labels(
        canonical['cds_start'],
        canonical['cds_end'],
        seq_len
    )
    
    print(f"✓ Created CDS boundary labels")
    print(f"  Start labels shape: {start_labels.shape}")
    print(f"  End labels shape: {end_labels.shape}")
    print(f"  Start positions marked: {start_labels.sum()}")
    print(f"  End positions marked: {end_labels.sum()}")
    
    if canonical['cds_start'] >= 0:
        print(f"  CDS start at position: {canonical['cds_start']}")
        print(f"  Label value: {start_labels[canonical['cds_start']]}")
    
    if canonical['cds_end'] is not None and canonical['cds_end'] >= 0:
        if canonical['cds_end'] < seq_len:
            print(f"  CDS end at position: {canonical['cds_end']}")
            print(f"  Label value: {end_labels[canonical['cds_end']]}")
        else:
            print(f"  CDS end at position: {canonical['cds_end']} (extends beyond window)")
    
    return True


def test_nmd_expression(canonical):
    """Test NMD and expression labels."""
    print("\n" + "="*60)
    print("TEST 5: NMD and Expression Labels")
    print("="*60)
    
    if not canonical:
        print("❌ No canonical isoform to test")
        return False
    
    # NMD label (handle both bool and string)
    nmd_value = canonical['nmd']
    if isinstance(nmd_value, str):
        nmd_value = nmd_value.lower() == 'true'
    nmd_label = torch.tensor(float(nmd_value), dtype=torch.float32)
    print(f"✓ NMD label: {nmd_label.item()} (type: {nmd_label.dtype})")
    
    # Expression label (log TPM)
    expression_label = torch.tensor(
        np.log1p(canonical['tpm']),
        dtype=torch.float32
    )
    print(f"✓ Expression label:")
    print(f"  Raw TPM: {canonical['tpm']:.2f}")
    print(f"  Log(TPM+1): {expression_label.item():.4f}")
    print(f"  Type: {expression_label.dtype}")
    
    return True


def test_tensor_conversion():
    """Test to_tensor helper function."""
    print("\n" + "="*60)
    print("TEST 6: Tensor Conversion")
    print("="*60)
    
    # Test with list
    data_list = [0.0, 1.0, 0.0, 1.0]
    tensor = to_tensor(data_list, length=10)
    print(f"✓ List to tensor: {tensor.shape}, dtype: {tensor.dtype}")
    
    # Test with JSON string
    data_json = json.dumps([0.0, 1.0, 0.0, 1.0])
    tensor = to_tensor(data_json, length=10)
    print(f"✓ JSON to tensor: {tensor.shape}, dtype: {tensor.dtype}")
    
    # Test with numpy array
    data_np = np.array([0.0, 1.0, 0.0, 1.0])
    tensor = to_tensor(data_np, length=10)
    print(f"✓ Numpy to tensor: {tensor.shape}, dtype: {tensor.dtype}")
    
    # Test padding
    data_short = [1.0, 2.0, 3.0]
    tensor = to_tensor(data_short, length=10)
    print(f"✓ Padding test: input len={len(data_short)}, output shape={tensor.shape}")
    
    # Test truncation
    data_long = list(range(100))
    tensor = to_tensor(data_long, length=10)
    print(f"✓ Truncation test: input len={len(data_long)}, output shape={tensor.shape}")
    
    return True


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("PHASE 1 DATASET TESTING")
    print("="*60)
    print("\nThis script tests that the new dataset changes work correctly.")
    print("It verifies:")
    print("  1. Loading parquet files with isoforms/variants fields")
    print("  2. Parsing isoform metadata (proteins, NMD, TPM, CDS)")
    print("  3. Creating protein sequence labels")
    print("  4. Creating CDS boundary labels")
    print("  5. Creating NMD and expression labels")
    print("  6. Tensor conversion utilities")
    
    # Run tests
    success, df = test_data_loading()
    if not success:
        print("\n❌ Data loading failed. Cannot continue.")
        return
    
    success, canonical = test_isoform_parsing(df)
    if success and canonical:
        test_protein_labels(canonical)
        test_cds_boundaries(canonical)
        test_nmd_expression(canonical)
    
    test_tensor_conversion()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print("✓ All tests completed!")
    print("\nNext steps:")
    print("  1. Review the test output above")
    print("  2. If all looks good, integrate into train.py")
    print("  3. See PHASE1_INTEGRATION_GUIDE.md for details")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
