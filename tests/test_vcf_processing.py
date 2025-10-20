#!/usr/bin/env python3
"""Test VCF processing pipeline."""
import os
import sys
import unittest
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from betadogma.data.vcf_processor import process_vcf
from betadogma.config import init_config

class TestVCFProcessing(unittest.TestCase):
    """Test VCF processing functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        # Initialize configuration
        cls.test_dir = Path(__file__).parent / "data"
        cls.output_dir = cls.test_dir / "output"
        cls.output_dir.mkdir(exist_ok=True)
        
        # Test VCF file
        cls.test_vcf = cls.test_dir / "test.vcf"
        
        # Create a custom config that only includes chr21
        from betadogma.config import init_config
        config = init_config()
        config.set_path(['chromosomes', 'include'], ['chr21'])
        
        # Process the test VCF with the custom config
        # Use a slightly lower min_af to account for floating point precision
        cls.results = process_vcf(
            input_vcf=cls.test_vcf,
            output_dir=cls.output_dir,
            min_qual=20.0,
            min_af=0.0099,  # Slightly below 0.01 to account for floating point precision
            max_af=0.99,
            filter_pass=True,
            force_overwrite=True,
            config=config
        )
    
    def test_output_files_exist(self):
        """Test that output files were created."""
        self.assertTrue(Path(self.results['normalized']).exists())
        self.assertTrue(Path(self.results['filtered']).exists())
    
    def test_chromosome_normalization(self):
        """Test that chromosome names are normalized."""
        import pysam
        
        # Check normalized VCF
        with pysam.VariantFile(self.results['normalized']) as vcf:
            chroms = {rec.chrom for rec in vcf}
            
        # Should only include chr21 (chr22 and non-prefixed 21 should be excluded)
        self.assertEqual(chroms, {'chr21'})
    
    def test_variant_filtering(self):
        """Test that variants are filtered correctly."""
        import pysam
        
        # Print the normalized VCF for debugging
        print("\nNormalized VCF content:")
        with pysam.VariantFile(self.results['normalized']) as vcf:
            for i, record in enumerate(vcf, 1):
                af = record.info.get('AF', [None])[0]
                print(f"  {i}. {record.chrom}:{record.pos} (AF={af}, FILTER={record.filter}, QUAL={record.qual})")
        
        # Check filtered VCF
        print("\nFiltered VCF content:")
        with pysam.VariantFile(self.results['filtered']) as vcf:
            variants = list(vcf)
            for i, v in enumerate(variants, 1):
                af = v.info.get('AF', [None])[0]
                print(f"  {i}. {v.chrom}:{v.pos} (AF={af}, FILTER={v.filter}, QUAL={v.qual})")
            
        # Should include these variants:
        # - chr21:1000000 (AF=0.5) - passes
        # - chr21:2000000 (AF~=0.01) - passes (with floating point tolerance)
        # - 21:3000000 - passes after being normalized to chr21
        # And exclude:
        # - chr22:1500000 - filtered out (wrong chromosome)
        
        # Check that we have the expected variants
        pos = {v.pos: v for v in variants}
        print(f"\nVariant positions: {sorted(pos.keys())}")
        
        # The variant at 1,000,000 should be present with AF=0.5
        self.assertIn(1000000, pos, f"Variant at position 1000000 not found in {sorted(pos.keys())}")
        self.assertAlmostEqual(pos[1000000].info['AF'][0], 0.5)
        
        # The variant at 2,000,000 should be present with AF~=0.01
        self.assertIn(2000000, pos, f"Variant at position 2000000 not found in {sorted(pos.keys())}")
        self.assertAlmostEqual(pos[2000000].info['AF'][0], 0.01, places=2)
        
        # The variant at 3,000,000 should be present after chromosome normalization
        self.assertIn(3000000, pos, f"Variant at position 3000000 not found in {sorted(pos.keys())}")
        self.assertEqual(pos[3000000].chrom, 'chr21', "Chromosome should be normalized to chr21")
        self.assertAlmostEqual(pos[3000000].info['AF'][0], 0.8, places=2)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test files."""
        # Remove output files
        for path in cls.output_dir.glob("*"):
            path.unlink()
        cls.output_dir.rmdir()

if __name__ == "__main__":
    unittest.main()
