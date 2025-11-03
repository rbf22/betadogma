# debug_chr1_0_1650000.py
import pandas as pd
from data_preparation.prepare_training_data import TrainingDataGenerator

def _parse_attr(s: str):
    out = {}
    for item in str(s).split(';'):
        item = item.strip()
        if not item:
            continue
        if ' "' in item:
            k, v = item.split(' "', 1)
            v = v.rstrip('"')
        elif '=' in item:
            k, v = item.split('=', 1)
        else:
            parts = item.split(' ', 1)
            if len(parts) == 2:
                k, v = parts[0], parts[1].strip('"')
            else:
                continue
        out[k.strip()] = v.strip()
    return out

gen = TrainingDataGenerator(
    reference_fasta="data/processed/genome/GRCh38.fa.gz",
    gencode_gtf="data/processed/annotation/gencode_v26/gencode.v26.annotation.gtf.gz",
    clinvar_vcf="data/processed/clinvar/clinvar.vcf.gz",
    splicevar_vcf="data/processed/splicevar/splicevar.vcf.gz",
    gtex_tpm="data/processed/gtex/v8/expression/transcript_tpm_summary.parquet",
    thousand_genomes_vcf="data/processed/1kg/1kg.vcf.gz",
    gtex_junctions="data/processed/gtex/v8/junctions/junctions_filtered.parquet",
    output_dir="data/processed",
)

chrom, start, end = "chr1", 0, 1_650_000

def ensure_versionless(df):
    if 'transcript_id_versionless' not in df.columns and 'transcript_id' in df.columns:
        df['transcript_id'] = df['transcript_id'].fillna('').astype(str)
        df['transcript_id_versionless'] = df['transcript_id'].str.split('.').str[0]
    return df

chrom_features = gen.transcripts[gen.transcripts.Chromosome == chrom]

# Transcripts
tx_df = chrom_features[chrom_features.Feature == 'transcript'].df.copy()
if 'transcript_id' not in tx_df.columns and 'Attributes' in tx_df.columns:
    attrs = tx_df['Attributes'].apply(_parse_attr)
    tx_df['transcript_id'] = attrs.apply(lambda d: d.get('transcript_id', ''))
    tx_df['gene_id'] = attrs.apply(lambda d: d.get('gene_id', ''))
    tx_df['gene_name'] = attrs.apply(lambda d: d.get('gene_name', ''))
tx_df = ensure_versionless(tx_df)
for col in ['Start','End']:
    if col in tx_df.columns:
        tx_df[col] = pd.to_numeric(tx_df[col], errors='coerce')
tx_df = tx_df.dropna(subset=['Start','End']).astype({'Start': int, 'End': int})
tx_df['Start0'] = tx_df['Start'] - 1
tx_df['End0'] = tx_df['End']
over_tx = tx_df[(tx_df['End0'] >= start) & (tx_df['Start0'] < end)]

print("Overlapping transcripts:", len(over_tx))
print("tx columns:", list(tx_df.columns)[:12])
print("Sample transcript_ids (versionless):", over_tx['transcript_id_versionless'].head(5).tolist())

# Exons
ex_df = chrom_features[chrom_features.Feature == 'exon'].df.copy()
if 'transcript_id' not in ex_df.columns and 'Attributes' in ex_df.columns:
    eattrs = ex_df['Attributes'].apply(_parse_attr)
    ex_df['transcript_id'] = eattrs.apply(lambda d: d.get('transcript_id', ''))
ex_df = ensure_versionless(ex_df)
for col in ['Start','End']:
    if col in ex_df.columns:
        ex_df[col] = pd.to_numeric(ex_df[col], errors='coerce')
ex_df = ex_df.dropna(subset=['Start','End']).astype({'Start': int, 'End': int})
ex_df['Start0'] = ex_df['Start'] - 1
ex_df['End0'] = ex_df['End']
over_ex = ex_df[(ex_df['End0'] >= start) & (ex_df['Start0'] < end)]

print("Overlapping exons:", len(over_ex))
print("exon columns:", list(ex_df.columns)[:12])

# Map overlap: how many exon transcript_ids intersect over_tx transcript_ids
tx_ids_versionless = set(over_tx['transcript_id_versionless'])
ex_ids_versionless = set(over_ex['transcript_id_versionless'])
intersect_ids = tx_ids_versionless & ex_ids_versionless
print("Intersecting transcript_ids (versionless) between tx and exon:", len(intersect_ids))
print("Sample intersect ids:", list(sorted(intersect_ids))[:5])

# Try builder
isos = gen._get_transcripts_in_region(chrom, start, end)
print("Isoforms found:", len(isos))
if isos:
    iso0 = isos[0]
    print("Example isoform keys:", sorted(iso0.keys()))
    print("Example exons (first 3):", iso0['exons'][:3])
    print("Example transcript_id:", iso0['transcript_id'], "TPM:", iso0['expression_tpm'])