# Contributing to Sheikh-2.5-Coder

Thank you for your interest in contributing to Sheikh-2.5-Coder! This document provides guidelines for contributing to the project.

## Code of Conduct

By participating in this project, you are expected to uphold our Code of Conduct:

- **Be respectful and inclusive**
- **Be collaborative and constructive**
- **Focus on what's best for the community**
- **Be open to feedback and different perspectives**

## How to Contribute

### 1. Reporting Issues

Before creating a new issue, please check if the issue already exists:

- Use the search function to find existing issues
- Provide detailed information about the problem
- Include steps to reproduce the issue
- Specify your environment details

**Issue Templates:**
- Bug reports should include steps to reproduce and expected vs actual behavior
- Feature requests should include use cases and implementation suggestions

### 2. Submitting Pull Requests

#### Development Setup

```bash
# Fork and clone the repository
git clone https://github.com/your-username/Sheikh-2.5-Coder.git
cd Sheikh-2.5-Coder

# Add upstream remote
git remote add upstream https://github.com/likhonsdevbd/Sheikh-2.5-Coder.git

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install development dependencies
pip install -r requirements-dev.txt
pre-commit install
```

#### Branch Naming Convention

- `feature/description-of-feature` - New features
- `bugfix/description-of-bug` - Bug fixes
- `docs/description-of-docs` - Documentation updates
- `refactor/description-of-refactor` - Code refactoring
- `test/description-of-tests` - Test additions

#### Code Style

We use the following tools for code formatting:

```bash
# Format code
black src/
isort src/

# Check code style
flake8 src/

# Run type checking
mypy src/
```

#### Commit Message Guidelines

Follow the conventional commits format:

```
type(scope): description

Types:
- feat: new feature
- fix: bug fix
- docs: documentation changes
- style: code style changes
- refactor: code refactoring
- test: adding tests
- chore: maintenance tasks

Examples:
feat(model): add quantization support for on-device deployment
fix(preprocessing): resolve memory leak in tokenization pipeline
docs(readme): update installation instructions
```

#### Pull Request Guidelines

1. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Write clean, well-documented code
   - Add tests for new functionality
   - Update documentation as needed

3. **Test Your Changes**
   ```bash
   # Run all tests
   pytest
   
   # Run specific test categories
   pytest tests/test_model.py
   pytest tests/test_preprocessing.py
   pytest tests/test_optimization.py
   ```

4. **Commit Changes**
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   ```

5. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

#### PR Requirements

- [ ] Code passes all tests
- [ ] Code follows style guidelines
- [ ] Documentation is updated
- [ ] Tests are added for new functionality
- [ ] Commit messages follow conventions
- [ ] PR description clearly explains changes

### 3. Code Review Process

All PRs require review from maintainers:

1. **Automated Checks**: CI/CD pipeline runs tests and linting
2. **Manual Review**: Maintainers review code quality and design
3. **Changes Requested**: Make requested changes and update PR
4. **Approval**: At least one maintainer approval required
5. **Merge**: Squash and merge into main branch

### 4. Development Areas

#### High Priority Areas

- **Model Architecture**: Transformer improvements and optimizations
- **Preprocessing Pipeline**: Data processing and tokenization
- **On-Device Optimization**: Quantization and memory optimization
- **Quality Metrics**: Evaluation and benchmarking improvements

#### How to Help

1. **Documentation**: Improve docstrings, README, and guides
2. **Tests**: Add test coverage for new features
3. **Bug Reports**: Reproduce and debug reported issues
4. **Performance**: Profile and optimize bottlenecks
5. **Examples**: Create usage examples and tutorials

### 5. Setting Up Development Environment

#### Requirements

- Python 3.8+
- CUDA-capable GPU (recommended for training/fine-tuning)
- 16GB+ RAM
- 50GB+ storage for datasets and models

#### Optional Dependencies

```bash
# For Flash Attention
pip install flash-attn

# For advanced optimization
pip install apex

# For monitoring and visualization
pip install tensorboard wandb
```

### 6. Testing Guidelines

#### Test Structure

```
tests/
├── test_model.py          # Model functionality tests
├── test_preprocessing.py  # Data processing tests
├── test_optimization.py   # Performance optimization tests
├── test_benchmarks.py     # Evaluation metric tests
└── test_integration.py    # End-to-end integration tests
```

#### Writing Tests

```python
import pytest
from src.model import Sheikh2_5Coder

def test_model_initialization():
    """Test model can be initialized with default parameters"""
    model = Sheikh2_5Coder()
    assert model.config.total_parameters == 3_090_000_000
    assert model.config.context_length == 32_768

def test_code_generation():
    """Test basic code generation functionality"""
    model = Sheikh2_5Coder()
    prompt = "function hello() {"
    result = model.generate(prompt, max_new_tokens=50)
    assert len(result) > len(prompt)
    assert result.startswith(prompt)
```

### 7. Performance Considerations

When contributing optimizations:

1. **Profile Before Optimizing**: Use profiling tools to identify bottlenecks
2. **Benchmark Changes**: Ensure optimizations don't break existing functionality
3. **Memory Management**: Be mindful of memory usage, especially for on-device deployment
4. **Backward Compatibility**: Maintain compatibility with existing features

### 8. Documentation Standards

#### Code Documentation

- **Docstrings**: Use Google-style docstrings
- **Type Hints**: Include type annotations for all functions
- **Comments**: Explain complex logic and algorithm choices

#### Example Docstring

```python
def preprocess_code(
    code: str,
    language: str,
    max_length: int = 1024
) -> Dict[str, Any]:
    """Preprocess code for model input.
    
    Args:
        code: Raw code string to preprocess
        language: Programming language identifier
        max_length: Maximum token length after tokenization
        
    Returns:
        Dictionary containing preprocessed tokens and metadata
        
    Raises:
        ValueError: If language is not supported
        RuntimeError: If tokenization fails
    """
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language: {language}")
    
    # Implementation details...
```

### 9. Security Considerations

- **No API Keys**: Never commit API keys or sensitive credentials
- **Input Validation**: Validate all user inputs
- **Dependency Security**: Keep dependencies updated
- **Code Scanning**: Run security scans before merging

### 10. Getting Help

- **GitHub Discussions**: Use for questions and feature discussions
- **GitHub Issues**: Report bugs and request features
- **Email**: likhonsheikh.dev@gmail.com for private matters

### 11. Recognition

Contributors will be recognized in:

- **README Contributors Section**: Listed alphabetically by contribution
- **Release Notes**: Major contributors mentioned in releases
- **Special Mention**: Outstanding contributions highlighted

## License Agreement

By contributing to Sheikh-2.5-Coder, you agree that your contributions will be licensed under the MIT License.

## Questions?

Don't hesitate to ask questions! The maintainers and community are here to help.

Thank you for contributing to Sheikh-2.5-Coder! 🎉
