# BlobLog Benchmarks

Comprehensive performance benchmarks for the BlobLog framework.

## Setup

Install benchmark dependencies:

```bash
uv sync --group bench
```

## Running Benchmarks

### Quick Start

Run all benchmarks:
```bash
uv run python benchmarks/run_benchmarks.py
```

Or use pytest directly:
```bash
uv run pytest benchmarks/ --benchmark-only
```

### Benchmark Categories

**Logging Performance** (`bench_logging.py`):
- Small message writes (100 bytes)
- Medium message writes (10KB)
- Large message writes (1MB)
- Burst writing patterns
- Multi-channel concurrent writes
- writev batching efficiency
- IOV_MAX boundary testing

**Reading Performance** (`bench_reading.py`):
- Small message reads
- Large message reads
- Memoryview vs copy overhead
- Read throughput (MB/s)
- BlobLogReader callbacks
- amerge multi-channel merging
- Batch size impact

**Codec Performance** (`bench_codecs.py`):
- String codec encode/decode
- Pickle codec simple types
- Pickle codec complex structures
- Binary no-op codec (baseline)
- Roundtrip overhead
- Memoryview conversion costs

**Pipeline Performance** (`bench_pipelines.py`):
- Simple producer-consumer
- Transform pipeline
- Fan-out patterns
- Logging overhead
- Multi-channel pipelines
- Queue backpressure
- Framework overhead vs raw asyncio

## Advanced Usage

### Save Baseline

Save current performance as baseline:
```bash
uv run pytest benchmarks/ --benchmark-save=baseline
```

### Compare to Baseline

After making changes, compare:
```bash
uv run pytest benchmarks/ --benchmark-compare=baseline
```

This will show performance regressions/improvements.

### Generate Histogram

```bash
uv run pytest benchmarks/ --benchmark-histogram
```

Generates HTML histograms in `.benchmarks/`.

### Run Specific Category

```bash
# Only logging benchmarks
uv run pytest benchmarks/bench_logging.py --benchmark-only

# Only specific group
uv run pytest benchmarks/ --benchmark-only -k "write-small"
```

### JSON Output

Export results as JSON for analysis:
```bash
uv run pytest benchmarks/ --benchmark-json=results.json
```

### Verbose Output

```bash
uv run pytest benchmarks/ --benchmark-only -v
```

## Interpreting Results

Results show:
- **min/max**: Range of execution times
- **mean**: Average execution time
- **stddev**: Standard deviation (consistency)
- **median**: Middle value (less affected by outliers)
- **ops**: Operations per second

### What to Look For

1. **Regressions**: Mean time increases significantly
2. **Variance**: High stddev indicates inconsistent performance
3. **Comparisons**: Within-group comparisons show relative costs
4. **Bottlenecks**: Identify which operations dominate runtime

## Performance Questions Answered

These benchmarks help answer:

- ✅ Is writev batching actually faster than individual writes?
- ✅ What's the overhead of memoryview slicing vs copying?
- ✅ How much does logging slow down a pipeline?
- ✅ What's the framework overhead vs raw asyncio?
- ✅ How do different codecs compare in speed?
- ✅ Does the 1000-item batch size make sense?
- ✅ What message sizes work best?
- ✅ Is IOV_MAX batching beneficial?

## CI Integration

Add to your CI pipeline:
```yaml
- name: Run performance benchmarks
  run: |
    uv sync --group bench
    uv run pytest benchmarks/ --benchmark-only --benchmark-json=bench.json
```

## Tips

- Run benchmarks multiple times for consistency
- Close other applications to reduce noise
- Use `--benchmark-warmup=on` for more stable results
- Save baselines before major changes
- Focus on relative performance, not absolute numbers
