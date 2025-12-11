"""Script to run benchmarks and generate reports."""

import subprocess
import sys
from pathlib import Path


def main():
    """Run benchmarks with various options."""
    benchmarks_dir = Path(__file__).parent

    print("=" * 80)
    print("BlobLog Performance Benchmarks")
    print("=" * 80)

    # Find all bench_*.py files
    bench_files = list(benchmarks_dir.glob("bench_*.py"))
    if not bench_files:
        print("No benchmark files found!")
        sys.exit(1)

    # Default: run all benchmarks with comparison
    cmd = [
        "pytest",
        *[str(f) for f in bench_files],
        "--benchmark-only",
        "--benchmark-sort=mean",
        "--benchmark-columns=min,max,mean,stddev,ops,median",
        "--benchmark-group-by=group",
    ]

    # Add any additional arguments from command line
    if len(sys.argv) > 1:
        cmd.extend(sys.argv[1:])

    print(f"\nRunning: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print("\n" + "=" * 80)
        print("Benchmark Tips:")
        print("=" * 80)
        print("- Save baseline:    pytest benchmarks/ --benchmark-save=baseline")
        print("- Compare to saved: pytest benchmarks/ --benchmark-compare=baseline")
        print("- Histogram:        pytest benchmarks/ --benchmark-histogram")
        print("- JSON output:      pytest benchmarks/ --benchmark-json=output.json")
        print("- Verbose:          pytest benchmarks/ -v")
        print("- Specific group:   pytest benchmarks/ --benchmark-only -k 'write-small'")

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
