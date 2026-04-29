#!/bin/bash
# Run all quality checks: lint + format check + tests

set -e

echo "=== Lint Check ==="
uv run ruff check backend/ main.py

echo ""
echo "=== Format Check ==="
uv run isort backend/ main.py --profile black --check --diff
uv run black backend/ main.py --check --diff

echo ""
echo "=== Tests ==="
cd backend && uv run pytest tests/ -v

echo ""
echo "All quality checks passed!"
