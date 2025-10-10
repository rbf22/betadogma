.PHONY: install test lint format type-check check-all clean

# Install dependencies
install:
	poetry install

# Run tests with coverage
coverage:
	poetry run pytest --cov=betadogma --cov-report=term-missing --cov-report=xml tests/

# Run tests
pytest:
	poetry run pytest -v tests/

# Run linting
lint:
	poetry run ruff check .
	poetry run black --check .
	poetry run isort --check-only .

# Format code
format:
	poetry run black .
	poetry run isort .
	poetry run ruff --fix .

# Run type checking
type-check:
	poetry run mypy betadogma/

# Run all checks
check-all: lint type-check test

# Clean up build artifacts
clean:
	rm -rf build/ dist/ .pytest_cache/ .mypy_cache/ .ruff_cache/ .coverage coverage.xml htmlcov/ *.egg-info/
	find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete

# Pre-commit hook installation
install-hooks:
	poetry run pre-commit install

# Update dependencies
update:
	poetry update

# Run Jupyter notebook server
notebook:
	poetry run jupyter notebook

# Run the application
run:
	poetry run python -m betadogma

# Build documentation
docs:
	cd docs && make html

# Show help
help:
	@echo "Available commands:"
	@echo "  install       - Install dependencies"
	@echo "  test          - Run tests"
	@echo "  coverage      - Run tests with coverage"
	@echo "  lint          - Run linting"
	@echo "  format        - Format code"
	@echo "  type-check    - Run type checking"
	@echo "  check-all     - Run all checks (lint, type-check, test)"
	@echo "  clean         - Clean build artifacts"
	@echo "  install-hooks - Install pre-commit hooks"
	@echo "  update        - Update dependencies"
	@echo "  notebook      - Start Jupyter notebook"
	@echo "  run           - Run the application"
	@echo "  docs          - Build documentation"
	@echo "  help          - Show this help message"
