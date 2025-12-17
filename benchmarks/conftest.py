"""pytest configuration for benchmarks."""

pytest_plugins = ["pytest_benchmark"]

# Exclude exploratory benchmarks from default collection
collect_ignore = ["exploratory"]
