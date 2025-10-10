# Contributing to BetaDogma

We welcome contributions from computational biologists, ML researchers, and data engineers. This document provides guidelines for contributing to the BetaDogma project.

## Project Structure

| Path | Responsibility |
|------|----------------|
| `core/` | Model architecture and training utilities |
| `decoder/` | Isoform graph + ψ |
| `nmd/` | Transcript decay prediction |
| `variant/` | Variant encoding and simulation |
| `data/` | Dataset loading, normalization, caching |
| `experiments/` | Training configs, hyperparameters |
| `notebooks/` | Analysis notebooks |
| `tests/` | Unit and integration tests |
| `.github/workflows/` | CI/CD pipeline configuration |

## Development Workflow

### Prerequisites

- Python 3.10+
- [Poetry](https://python-poetry.org/) for dependency management
- [Pre-commit](https://pre-commit.com/) for git hooks

### Getting Started

1. Fork the repository and create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Install dependencies and set up pre-commit hooks:
   ```bash
   make install
   make pre-commit-install
   ```

3. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

### Development Commands

Use the provided Makefile for common tasks:

```bash
# Install dependencies
make install

# Run tests with coverage
make test

# Lint and format code
make lint
make format

# Run type checking
make type-check

# Run all checks (tests, lint, type-check)
make check

# Clean up temporary files
make clean
```

## Testing Guidelines

### Writing Tests

- Place test files in the `tests/` directory mirroring the source structure
- Use descriptive test function names starting with `test_`
- Follow the Arrange-Act-Assert pattern
- Use fixtures for common test data (see `tests/conftest.py`)
- For property-based testing, use `hypothesis`

Example test structure:
```python
def test_functionality():
    # Arrange
    input_data = ...
    expected = ...
    
    # Act
    result = function_under_test(input_data)
    
    # Assert
    assert result == expected

# Parametrized test
@pytest.mark.parametrize("input_a, input_b, expected", [
    (1, 2, 3),
    (0, 0, 0),
])
def test_addition(input_a, input_b, expected):
    assert input_a + input_b == expected
```

### Running Tests

Run all tests:
```bash
make test
```

Run a specific test file:
```bash
pytest tests/path/to/test_file.py -v
```

Run tests with coverage report:
```bash
pytest --cov=betadogma --cov-report=term-missing
```

## Code Quality

### Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) style guide
- Use type hints for all function signatures
- Keep line length to 100 characters
- Use `black` for code formatting
- Use `isort` for import sorting

### Linting

We use `ruff` for linting. To check for issues:
```bash
make lint
```

### Type Checking

We use `mypy` for static type checking:
```bash
make type-check
```

## Pull Request Process

1. Ensure all tests pass (`make check`)
2. Update documentation if needed
3. Open a pull request with a clear description of changes
4. Reference any related issues
5. Ensure CI passes before merging

## Code Review Guidelines

- Be constructive and respectful
- Focus on code quality and correctness
- Check for potential performance issues
- Verify test coverage for new code
- Ensure documentation is clear and up-to-date

## Reporting Issues

When reporting issues, please include:
- A clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment details (Python version, OS, etc.)
- Any relevant error messages or logs
