# Project Development Guidelines

## Dependency Management with Poetry

This project uses [Poetry](https://python-poetry.org/) for dependency management. Please follow these guidelines:

### Key Points to Remember
- **Always use Poetry** for managing Python dependencies
- The project's dependencies are defined in `pyproject.toml`
- **Do not** use `pip install` directly - instead, use `poetry add <package>`

### Common Commands
```bash
# Install dependencies (after cloning the repository)
poetry install

# Add a new dependency
poetry add <package-name>

# Add a development dependency
poetry add --group dev <package-name>

# Run a command within the virtual environment
poetry run python script.py

# Activate the virtual environment
poetry shell
```

### For IDEs
- Configure your IDE to use the Poetry virtual environment
- The virtual environment is typically located in `~/.cache/pypoetry/virtualenvs/`

### For CI/CD
- Use `poetry install --no-dev` in your CI pipeline for production builds
- Use `poetry install` for development and testing environments

### Troubleshooting
- If you encounter dependency conflicts, run `poetry lock --no-update` to refresh the lock file
- To update all dependencies: `poetry update`

---
*This file serves as a reminder for AI assistants and developers working on this project.*
