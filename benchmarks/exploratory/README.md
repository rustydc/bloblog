# Exploratory Benchmarks

This directory contains exploratory and research benchmarks that are **not run in CI**.

These benchmarks were useful for:
- Comparing implementation alternatives (decisions already made)
- Testing Python/asyncio primitives (won't change from our code)
- Detailed format comparisons (e.g., pickle vs pyarrow vs feather)
- Memory profiling and zero-copy proofs

## Running Exploratory Benchmarks

```bash
# Run all exploratory benchmarks
uv run pytest benchmarks/exploratory/ --benchmark-only

# Run a specific file
uv run pytest benchmarks/exploratory/bench_codecs.py --benchmark-only

# Run without benchmarking (just verify they pass)
uv run pytest benchmarks/exploratory/ --benchmark-disable
```

## Contents

- `bench_codecs.py` - Tests Python stdlib codecs (pickle, string encoding)
- `bench_decode_only.py` - Isolated decode performance tests
- `bench_pandas_codec.py` - Pandas codec format comparisons (pickle, pyarrow, feather)
- `bench_pubsub_micro.py` - Asyncio primitive overhead measurements
- `bench_pubsub_optimizations.py` - Pub/sub optimization comparisons
- `bench_slow_consumer.py` - Queue backpressure behavioral tests
