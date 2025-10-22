"""
Configuration management for the BetaDogma project.

This module handles loading and validating configuration settings for data processing.
"""
from pathlib import Path
from typing import Dict, List, Optional, Set, Union
import yaml
from .data.chrom_utils import get_chrom_set, normalize_chrom_convention, UCSC, NCBI

# Default configuration values
DEFAULT_CONFIG = {
    "project": {
        "name": "betadogma",
        "version": "0.1.0"
    },
    "chromosomes": {
        "include": None,  # None means include all standard chromosomes
        "exclude": ["chrM"],  # Default: exclude mitochondrial DNA
        "include_sex_chromosomes": True,
        "naming_convention": "ucsc"  # or "ncbi"
    },
    "paths": {
        "data_dir": "data",
        "raw_data": "data/raw",
        "processed_data": "data/processed",
        "references": "data/references"
    },
    "reference_genome": {
        "assembly": "GRCh38",
        "fasta": None,  # Will be set based on assembly
        "index": None   # Will be set based on assembly
    },
    "resources": {
        "threads": 4,
        "memory_gb": 16
    }
}

class Config:
    """Configuration manager for BetaDogma."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize configuration.
        
        Args:
            config_path: Path to a YAML configuration file
        """
        self._config = DEFAULT_CONFIG.copy()
        self._config_path = str(config_path) if config_path else None
        
        if config_path:
            self.load(config_path)
        
        # Initialize paths
        self._init_paths()
    
    def _init_paths(self):
        """Initialize and validate data paths."""
        base_dir = Path(__file__).parent.parent.parent
        
        # Resolve relative paths
        paths = self._config["paths"]
        for key, path in paths.items():
            if not Path(path).is_absolute():
                paths[key] = str(base_dir / path)
        
        # Set reference genome paths if not specified
        ref = self._config["reference_genome"]
        if not ref.get("fasta"):
            ref["fasta"] = f"{paths['references']}/genomes/{ref['assembly']}.fa"
        if not ref.get("index"):
            ref["index"] = f"{ref['fasta']}.fai"
    
    def load(self, config_path: Union[str, Path]):
        """Load configuration from a YAML file."""
        with open(config_path, 'r') as f:
            user_config = yaml.safe_load(f) or {}
        
        # Deep update of the configuration
        self._update_dict(self._config, user_config)
        self._config_path = str(config_path)
        self._init_paths()
    
    def _update_dict(self, d: Dict, u: Dict) -> Dict:
        """Recursively update a dictionary."""
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._update_dict(d[k], v)
            else:
                d[k] = v
        return d
    
    def get_chromosomes(self) -> Set[str]:
        """Get the set of chromosomes to process."""
        chrom_config = self._config["chromosomes"]
        convention = str(chrom_config.get("naming_convention", "ucsc")).lower()
        
        # Get base set of chromosomes
        if chrom_config.get("include"):
            # Normalize included list to target convention
            included = set(
                normalize_chrom_convention(c, convention) for c in chrom_config["include"]
            )
            chroms = included
        else:
            # Build default UCSC set then convert to target convention if needed
            base = get_chrom_set(
                include_sex=chrom_config.get("include_sex_chromosomes", True),
                include_mito="chrM" not in chrom_config.get("exclude", [])
            )
            chroms = set(normalize_chrom_convention(c, convention) for c in base)
        
        # Apply exclusions (normalize to same convention)
        excludes = set(
            normalize_chrom_convention(c, convention) for c in chrom_config.get("exclude", [])
        )
        chroms -= excludes

        return chroms
    
    def get_path(self, *keys: str) -> str:
        """Get a configuration value using dot notation."""
        value = self._config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                raise KeyError(f"Config key not found: {'.'.join(keys)}")
        return value
    
    def set_path(self, keys: List[str], value):
        """Set a configuration value using dot notation."""
        d = self._config
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = value
    
    def to_dict(self) -> Dict:
        """Return a deep copy of the configuration as a dictionary."""
        import copy
        return copy.deepcopy(self._config)
    
    def __getitem__(self, key):
        return self._config[key]
    
    def __contains__(self, key):
        return key in self._config
    
    def __str__(self):
        return yaml.dump(self._config, default_flow_style=False, sort_keys=False)

# Global configuration instance
_config = None

def get_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """Get or create the global configuration."""
    global _config
    if _config is None or config_path is not None:
        _config = Config(config_path)
    return _config

def init_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """Initialize the global configuration."""
    global _config
    _config = Config(config_path)
    return _config

# Initialize the default configuration when the module is imported
init_config()