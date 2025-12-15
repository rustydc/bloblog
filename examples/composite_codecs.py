"""Example of using composite codecs for nested serialization.

This demonstrates how to use DictCodec, ListCodec, and other composite utilities
to serialize complex nested structures while maintaining zero-copy for NumPy arrays.
"""

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from tinman import ObLog
from tinman.codecs import (
    DictCodec,
    FloatCodec,
    IntCodec,
    ListCodec,
    NumpyArrayCodec,
    OptionalCodec,
    StringCodec,
    TupleCodec,
)


async def example_dict_with_arrays():
    """Example 1: Dict containing NumPy arrays (zero-copy!)."""
    print("\n=== Example 1: Dict with NumPy Arrays ===")

    with TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir)

        # Define schema for sensor readings
        schema = {
            "accelerometer": NumpyArrayCodec(),
            "gyroscope": NumpyArrayCodec(),
            "timestamp": IntCodec(),
            "device_id": StringCodec(),
        }
        codec = DictCodec(schema)

        # Write sensor data
        oblog = ObLog(log_dir)
        write = oblog.get_writer("sensors", codec)

        for i in range(3):
            data = {
                "accelerometer": np.random.rand(100, 3),  # 100 samples, x/y/z
                "gyroscope": np.random.rand(100, 3),
                "timestamp": 1000000 + i * 1000,
                "device_id": f"sensor_{i}",
            }
            write(data)
            print(f"  Wrote reading {i}: {data['accelerometer'].shape}, {data['gyroscope'].shape}")

        await oblog.close()

        # Read sensor data (arrays are zero-copy views!)
        oblog = ObLog(log_dir)
        print("\n  Reading back (zero-copy):")
        async for _, reading in oblog.read_channel("sensors"):
            print(f"    Device {reading['device_id']}: "
                  f"accel shape {reading['accelerometer'].shape}, "
                  f"gyro shape {reading['gyroscope'].shape}")
            # Arrays are read-only views into mmap
            assert not reading['accelerometer'].flags.writeable

        await oblog.close()


async def example_list_of_arrays():
    """Example 2: List of arrays (like a batch)."""
    print("\n=== Example 2: List of Arrays ===")

    with TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir)

        # Codec for batches of images (as arrays)
        codec = ListCodec(NumpyArrayCodec())

        # Write batches
        oblog = ObLog(log_dir)
        write = oblog.get_writer("batches", codec)

        for batch_num in range(3):
            # Simulate a batch of 4 images (28x28)
            batch = [np.random.rand(28, 28) for _ in range(4)]
            write(batch)
            print(f"  Wrote batch {batch_num}: {len(batch)} images")

        await oblog.close()

        # Read batches
        oblog = ObLog(log_dir)
        print("\n  Reading back:")
        async for _, batch in oblog.read_channel("batches"):
            print(f"    Batch with {len(batch)} images, each {batch[0].shape}")

        await oblog.close()


async def example_nested_structure():
    """Example 3: Nested dict of lists of arrays."""
    print("\n=== Example 3: Nested Structure ===")

    with TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir)

        # Complex nested structure: dict containing lists of arrays
        schema = {
            "train_batch": ListCodec(NumpyArrayCodec()),
            "val_batch": ListCodec(NumpyArrayCodec()),
            "epoch": IntCodec(),
            "loss": FloatCodec(),
        }
        codec = DictCodec(schema)

        # Write training data
        oblog = ObLog(log_dir)
        write = oblog.get_writer("training", codec)

        for epoch in range(3):
            data = {
                "train_batch": [np.random.rand(10, 5) for _ in range(3)],
                "val_batch": [np.random.rand(10, 5) for _ in range(2)],
                "epoch": epoch,
                "loss": 0.5 - epoch * 0.1,  # Decreasing loss
            }
            write(data)
            print(f"  Epoch {epoch}: "
                  f"{len(data['train_batch'])} train batches, "
                  f"{len(data['val_batch'])} val batches, "
                  f"loss={data['loss']:.2f}")

        await oblog.close()

        # Read training data
        oblog = ObLog(log_dir)
        print("\n  Reading back:")
        async for _, data in oblog.read_channel("training"):
            print(f"    Epoch {data['epoch']}: loss={data['loss']:.2f}, "
                  f"train={len(data['train_batch'])}, val={len(data['val_batch'])}")

        await oblog.close()


async def example_tuple_format():
    """Example 4: Tuple format for positional data."""
    print("\n=== Example 4: Tuple Format ===")

    with TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir)

        # Tuple: (image_array, label_string, confidence_float)
        codec = TupleCodec([
            NumpyArrayCodec(),
            StringCodec(),
            FloatCodec(),
        ])

        # Write predictions
        oblog = ObLog(log_dir)
        write = oblog.get_writer("predictions", codec)

        predictions = [
            (np.random.rand(28, 28), "cat", 0.95),
            (np.random.rand(28, 28), "dog", 0.88),
            (np.random.rand(28, 28), "bird", 0.72),
        ]

        for img, label, conf in predictions:
            write((img, label, conf))
            print(f"  Predicted: {label} (confidence: {conf:.2f})")

        await oblog.close()

        # Read predictions
        oblog = ObLog(log_dir)
        print("\n  Reading back:")
        async for _, (img, label, conf) in oblog.read_channel("predictions"):
            print(f"    {label}: {conf:.2f}, image shape {img.shape}")

        await oblog.close()


async def example_optional_fields():
    """Example 5: Optional fields in schema."""
    print("\n=== Example 5: Optional Fields ===")

    with TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir)

        # Schema with optional error field
        schema = {
            "data": NumpyArrayCodec(),
            "success": IntCodec(),  # 1 or 0
            "error_msg": OptionalCodec(StringCodec()),  # Only if failed
        }
        codec = DictCodec(schema)

        # Write results
        oblog = ObLog(log_dir)
        write = oblog.get_writer("results", codec)

        results = [
            {"data": np.array([1, 2, 3]), "success": 1, "error_msg": None},
            {"data": np.array([4, 5, 6]), "success": 0, "error_msg": "Timeout"},
            {"data": np.array([7, 8, 9]), "success": 1, "error_msg": None},
        ]

        for result in results:
            write(result)
            status = "✓" if result["success"] else "✗"
            msg = f" ({result['error_msg']})" if result['error_msg'] else ""
            print(f"  {status} Data: {result['data']}{msg}")

        await oblog.close()

        # Read results
        oblog = ObLog(log_dir)
        print("\n  Reading back:")
        async for _, result in oblog.read_channel("results"):
            status = "✓" if result["success"] else "✗"
            msg = f" ({result['error_msg']})" if result['error_msg'] else ""
            print(f"    {status} {result['data']}{msg}")

        await oblog.close()


async def main():
    """Run all examples."""
    print("=" * 60)
    print("Composite Codec Examples")
    print("Zero-copy NumPy arrays inside dicts, lists, and tuples!")
    print("=" * 60)

    await example_dict_with_arrays()
    await example_list_of_arrays()
    await example_nested_structure()
    await example_tuple_format()
    await example_optional_fields()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("Key takeaway: Arrays maintain zero-copy even when nested!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
