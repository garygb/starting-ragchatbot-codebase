#!/bin/bash
# Check code quality without modifying files

set -e

echo "Checking imports with isort..."
uv run isort backend/ main.py --profile black --check --diff

echo "Checking formatting with black..."
uv run black backend/ main.py --check --diff

echo "Linting with ruff..."
uv run ruff check backend/ main.py

echo "All checks passed!"
