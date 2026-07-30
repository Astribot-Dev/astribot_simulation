# Contributing to Astribot Simulation

Thank you for your interest in contributing to Astribot Simulation!

## Development Setup

1. Fork and clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/astribot_simulation.git
cd astribot_simulation
```

2. Create a virtual environment:
```bash
conda create -n astribot_simu python=3.10
conda activate astribot_simu
```

3. Install development dependencies:
```bash
pip install -e .[dev]
```

4. Install pre-commit hooks:
```bash
pre-commit install
```

## Code Style

- Follow PEP 8 guidelines
- Use Black for code formatting (line length: 100)
- Use isort for import sorting
- Add type hints to function signatures
- Write docstrings in Google style
- All comments and documentation must be in English

## Testing

- Write tests for new features
- Ensure all tests pass: `pytest`
- Maintain test coverage >80% for new code
- Test simulator imports to verify installations

## Pull Request Process

1. Create a feature branch from `develop`
2. Make your changes with clear commit messages
3. Add tests for new functionality
4. Update documentation if needed
5. Ensure CI passes (linting, tests)
6. Submit PR with clear description

## Reporting Issues

Use GitHub Issues with appropriate templates for:
- Bug reports
- Feature requests
- Documentation improvements
