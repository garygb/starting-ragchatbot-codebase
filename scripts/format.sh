#!/bin/bash
# Auto-format all Python files with isort and black

set -e

echo "Sorting imports with isort..."
uv run isort backend/ main.py --profile black

echo "Formatting with black..."
uv run black backend/ main.py

echo "Done! All files formatted."
