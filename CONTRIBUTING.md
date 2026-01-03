# Contributing to Oculi

Thank you for your interest in contributing to Oculi! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Testing](#testing)
- [Code Style](#code-style)
- [Submitting Changes](#submitting-changes)
- [Project Structure](#project-structure)
- [Documentation](#documentation)

## Code of Conduct

This project adheres to a code of professional conduct. By participating, you are expected to:

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Accept constructive criticism gracefully
- Focus on what is best for the community
- Show empathy towards other community members

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues. When you create a bug report, include:

**Bug Report Template:**
```markdown
**Description:**
A clear description of the bug.

**To Reproduce:**
Steps to reproduce the behavior:
1. Load model '...'
2. Call method '...'
3. See error

**Expected behavior:**
What you expected to happen.

**Actual behavior:**
What actually happened.

**Environment:**
- OS: [e.g., Ubuntu 22.04]
- Python version: [e.g., 3.10.8]
- PyTorch version: [e.g., 2.1.0]
- Oculi version: [e.g., 0.3.0-dev]
- CUDA version (if applicable): [e.g., 11.8]

**Additional context:**
Any other information that might be helpful.
```

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, include:

- **Use case:** Why is this enhancement needed?
- **Proposed solution:** How should it work?
- **Alternatives considered:** What other approaches did you consider?
- **Additional context:** Screenshots, references, examples

### Pull Requests

1. **Discuss first** for major changes - open an issue to discuss before implementing
2. **Follow the style guide** - maintain consistency with existing code
3. **Write tests** - all new features must include tests
4. **Update documentation** - document new features and API changes
5. **Keep commits atomic** - one logical change per commit
6. **Write clear commit messages** - explain what and why

## Development Setup

### Prerequisites

- Python 3.10+
- Git
- PyTorch 2.0.0+

### Setup Steps

1. **Fork and clone:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/oculi.git
   cd oculi
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install in development mode:**
   ```bash
   pip install -e ".[all]"
   ```

4. **Verify installation:**
   ```bash
   pytest tests/ -v
   ```

### Development Workflow

1. **Create a branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make changes and test:**
   ```bash
   # Make your changes
   pytest tests/ -v
   ```

3. **Commit with clear messages:**
   ```bash
   git add .
   git commit -m "feat: Add feature X

   - Implement core functionality
   - Add unit tests
   - Update documentation"
   ```

4. **Push and create PR:**
   ```bash
   git push origin feature/your-feature-name
   ```

## Testing

### Test Structure

```
tests/
├── contract_tests/     # API contract tests (shape, semantics)
├── integration/        # Integration tests with mock models
├── unit/               # Unit tests for individual components
└── mocks/              # Mock models and fixtures
```

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/unit/test_attribution.py -v

# Specific test
pytest tests/unit/test_attribution.py::TestAttentionFlow::test_output_shape -v

# With coverage
pytest tests/ --cov=oculi --cov-report=html
```

### Writing Tests

All new features must include tests. Follow these guidelines:

**Contract Tests** (for public API):
```python
def test_method_output_shape():
    """Verify method returns documented shape."""
    capture = mock_capture(n_layers=4, n_heads=8, n_tokens=16)
    result = MyAnalysis.my_method(capture)
    assert result.shape == (4, 8, 16)  # [L, H, T]
```

**Unit Tests** (for implementation):
```python
def test_method_edge_case():
    """Test behavior with edge case input."""
    capture = mock_capture(n_tokens=1)
    result = MyAnalysis.my_method(capture)
    assert torch.isnan(result[0, 0, 0])  # First token should be NaN
```

**Test on Mock Models:**
```python
from tests.mocks import MockLlamaAdapter

def test_integration():
    """Test end-to-end with mock model."""
    adapter = MockLlamaAdapter()
    input_ids = adapter.tokenize("Test")
    capture = adapter.capture(input_ids)
    assert capture.n_tokens == 4  # "Test" tokenized
```

**Important:** DO NOT test on real GPU models in CI. Use mocks for all automated tests.

## Code Style

### Python Style

- **PEP 8** compliance (enforced by linters)
- **Type hints** for all public functions
- **Docstrings** in Google style for all public classes/methods
- **Line length:** 88 characters (Black default)

### Docstring Format

```python
def my_function(capture: AttentionCapture, threshold: float = 0.5) -> torch.Tensor:
    """
    Brief one-line description.

    More detailed description if needed. Explain what the function does,
    any important algorithms, and key considerations.

    Args:
        capture: AttentionCapture with patterns
        threshold: Minimum value for detection (default: 0.5)

    Returns:
        Tensor of shape [L, H] with detection scores

    Raises:
        ValueError: If threshold not in [0, 1]

    Example:
        >>> capture = adapter.capture(input_ids)
        >>> scores = my_function(capture, threshold=0.7)
        >>> print(scores.shape)
        torch.Size([32, 32])
    """
    if not 0 <= threshold <= 1:
        raise ValueError(f"Threshold must be in [0, 1], got {threshold}")

    # Implementation
    ...
```

### Code Organization

**Public API:**
- Clean, well-documented interfaces
- No implementation details exposed
- Consistent naming across modules

**Private Implementation:**
- Can be refactored without public API changes
- Implementation details in `oculi/_private/`

### Naming Conventions

- **Classes:** PascalCase (`AttributionMethods`, `CompositionAnalysis`)
- **Functions/methods:** snake_case (`attention_flow`, `detect_circuit`)
- **Constants:** UPPER_SNAKE_CASE (`MAX_SEQUENCE_LENGTH`)
- **Private:** Leading underscore (`_internal_helper`)

## Submitting Changes

### Commit Message Format

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation only
- `style:` Code style (formatting, no logic change)
- `refactor:` Code change that neither fixes bug nor adds feature
- `test:` Adding or updating tests
- `chore:` Maintenance tasks

**Examples:**
```bash
feat(analysis): Add head composition analysis

- Implement QK/OV composition methods
- Add virtual attention computation
- Include induction circuit detection

Closes #42
```

```bash
fix(capture): Handle empty sequences correctly

Previously, capturing attention for empty sequences would raise
an unhelpful error. Now it validates input and provides clear
error message.

Fixes #123
```

### Pull Request Process

1. **Update documentation** for any API changes
2. **Add tests** for new functionality
3. **Ensure all tests pass** locally
4. **Update CHANGELOG.md** with your changes
5. **Fill out PR template** completely
6. **Request review** from maintainers

**PR Title Format:**
```
feat: Add feature name
```

**PR Description Template:**
```markdown
## Description
Brief description of changes.

## Motivation
Why is this change needed?

## Changes
- Change 1
- Change 2
- Change 3

## Testing
- [ ] Unit tests added
- [ ] Integration tests added
- [ ] All tests passing locally
- [ ] Tested on mock model

## Documentation
- [ ] Docstrings updated
- [ ] User guide updated (if needed)
- [ ] API reference updated (if needed)

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex logic
- [ ] Documentation updated
- [ ] No new warnings
- [ ] CHANGELOG.md updated

## Related Issues
Closes #issue_number
```

## Project Structure

```
oculi/
├── models/          # PUBLIC: Model adapters
│   ├── base.py      # Adapter interface
│   └── llama/       # LLaMA implementation
├── capture/         # Capture data structures
├── analysis/        # PUBLIC: Analysis methods
├── intervention/    # PUBLIC: Interventions
├── visualize/       # Visualization utilities
└── _private/        # PRIVATE: Implementation details
```

**Guidelines:**
- **Public modules** (`oculi/models/`, `oculi/analysis/`, etc.) - Stable, documented APIs
- **Private modules** (`oculi/_private/`) - Implementation details, can change
- **Never import from `_private/`** in public modules

## Documentation

### Types of Documentation

1. **API Documentation** - Docstrings (required for all public APIs)
2. **User Guides** - In `docs/guides/` (for features)
3. **Tutorials** - In `docs/tutorials/` (step-by-step examples)
4. **Examples** - In `examples/` (working code)

### Building Documentation

```bash
# Install docs dependencies
pip install -e ".[docs]"

# Build and serve locally
mkdocs serve

# Build static site
mkdocs build
```

### Adding New Features

When adding a new feature, provide:

1. **Docstrings** with clear examples
2. **User guide** in `docs/guides/`
3. **Tutorial** in `docs/tutorials/` (if major feature)
4. **Example code** in `examples/`
5. **Tests** in `tests/`
6. **Update README.md** if major feature

## Questions?

- **GitHub Issues:** [Ask a question](https://github.com/ajayspatil7/oculi/issues/new)
- **GitHub Discussions:** [Start a discussion](https://github.com/ajayspatil7/oculi/discussions)
- **Email:** ajayspatil7@gmail.com

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to Oculi! 🔬
