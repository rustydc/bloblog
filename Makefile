.PHONY: help install test lint format check clean

help:
	@echo "Available commands:"
	@echo "  make install    Install package and dev dependencies"
	@echo "  make test       Run tests"
	@echo "  make lint       Run linter (ruff check)"
	@echo "  make format     Format code (ruff format)"
	@echo "  make check      Run linter and formatter check"
	@echo "  make clean      Clean build artifacts and cache"

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

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache .mypy_cache
	rm -rf dist build *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.bloblog" -delete
