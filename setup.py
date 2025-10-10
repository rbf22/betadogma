from setuptools import setup, find_packages

setup(
    name="betadogma",
    version="0.0.1",
    description="Revising the central dogma through data.",
    author="BetaDogma Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.6",
        "transformers>=4.42",
        "tokenizers>=0.14",
        "numpy>=1.24,<2.0",
        "pandas>=2.0",
        "pyarrow>=14.0",
        "pyfaidx>=0.8.1",
        "pyyaml>=6.0",
        "tqdm>=4.66",
        "networkx>=3.2",
        "pytorch-lightning>=2.5.5,<3.0.0",
        "tensorboard>=2.20.0,<3.0.0",
        "psutil>=7.1.0,<8.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.1.1",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.12.0",
            "pytest-xdist>=3.5.0",
            "hypothesis>=6.97.0",
            "ruff>=0.5.3",
            "mypy>=1.8.0",
            "types-PyYAML>=6.0.12.12",
            "types-requests>=2.31.0.20240125",
            "codecov>=2.1.13",
            "pre-commit>=3.6.0",
            "black>=24.1.1",
            "isort>=5.13.2",
        ]
    },
)
