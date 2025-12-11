.PHONY: help install test lint format check clean bench bench-save bench-compare

help:
	@echo "Available commands:"
	@echo "  make install        Install package and dev dependencies"
	@echo "  make test           Run tests"
	@echo "  make lint           Run linter (ruff check)"
	@echo "  make format         Format code (ruff format)"
	@echo "  make check          Run linter and formatter check"
	@echo "  make bench          Run performance benchmarks"
	@echo "  make bench-save     Save benchmark baseline"
	@echo "  make bench-compare  Compare to saved baseline"
	@echo "  make clean          Clean build artifacts and cache"

install:
	uv sync

test:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff format .

check:
	uv run ruff check .
	uv run ruff format --check .

bench:
	uv sync --group bench
	uv run python benchmarks/run_benchmarks.py

bench-save:
	uv sync --group bench
	uv run pytest benchmarks/ --benchmark-only --benchmark-save=baseline

bench-compare:
	uv sync --group bench
	uv run pytest benchmarks/ --benchmark-only --benchmark-compare=baseline

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache .mypy_cache
	rm -rf dist build *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.bloblog" -delete
