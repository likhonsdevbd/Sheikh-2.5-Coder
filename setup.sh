#!/bin/bash

# Setup script for Sheikh-2.5-Coder development environment

set -e

echo "🚀 Setting up Sheikh-2.5-Coder development environment..."

# Check Python version
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
if [[ ! $(echo "$python_version >= 3.8" | bc -l) ]]; then
    echo "❌ Python 3.8+ required. Found: $python_version"
    exit 1
fi

echo "✅ Python version check passed: $python_version"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Install development dependencies
if [ -f "requirements-dev.txt" ]; then
    echo "🛠️  Installing development dependencies..."
    pip install -r requirements-dev.txt
fi

# Install pre-commit hooks
echo "🔧 Setting up pre-commit hooks..."
pre-commit install

# Create necessary directories
echo "📁 Creating directory structure..."
mkdir -p {data/{raw,processed,tokenized},logs,notebooks,evaluation/reports}

# Download NLTK data
echo "📖 Downloading NLTK data..."
python3 -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Setup Git hooks (optional)
read -p "🤖 Do you want to setup Git hooks? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🔗 Setting up Git hooks..."
    # Add git hooks setup here if needed
fi

# Final setup verification
echo "🔍 Verifying setup..."
python3 -c "import torch, transformers, datasets; print('✅ All packages imported successfully')"

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "   1. Activate virtual environment: source venv/bin/activate"
echo "   2. Review configuration: configs/data_prep_config.yaml"
echo "   3. Start data preparation: python scripts/prepare_data.py"
echo "   4. Check documentation: docs/"
echo ""
echo "📚 Useful commands:"
echo "   - Run data preparation: python scripts/prepare_data.py"
echo "   - Run tests: pytest"
echo "   - Format code: black src/ && isort src/"
echo "   - Check code style: flake8 src/"
echo ""
echo "🐛 Report issues: https://github.com/likhonsdevbd/Sheikh-2.5-Coder/issues"
