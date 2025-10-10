# Data Directory

This directory contains all data-related files and scripts for the BetaDogma project.

## Directory Structure

```
data/
├── data_chr22/          # Full dataset for chromosome 22
│   ├── raw/             # Raw data files
│   └── processed/       # Processed data files
├── data_mini/           # Minimal dataset for testing (chromosome 21 only)
│   ├── gencode/         # GENCODE annotations (chr21 only)
│   ├── genome/          # Reference genome (chr21 only)
│   ├── raw/             # Raw data files including junction data
│   └── processed/       # Processed data files
├── scripts/             # Data processing scripts
└── README.md            # This file
```

## Available Scripts

- `create_mini_dataset.py`: Creates a minimal dataset with chromosome 21 data
- `create_mini_training_data.py`: Prepares training data for the minimal dataset
- `create_minimal_dataset_direct.py`: Directly creates a minimal dataset without dependencies
- `create_minimal_dataset_with_junctions.py`: Creates a dataset with junction data for testing

## Usage

### For Testing/Development

Use the minimal dataset in `data_mini/` for quick testing and development.

### For Full Training

Use the full dataset in `data_chr22/` for actual model training.

### Regenerating the Minimal Dataset

To regenerate the minimal dataset:

```bash
# Make sure you have the required dependencies
poetry install

# Run the dataset creation script
python data/scripts/create_minimal_dataset_with_junctions.py
```

## Notes

- The minimal dataset is designed to be small (under 100MB) for quick testing
- The full dataset contains more comprehensive data for actual training
- All paths in the scripts are relative to the project root
