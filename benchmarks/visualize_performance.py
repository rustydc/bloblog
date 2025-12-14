#!/usr/bin/env python3
"""Visualize zero-copy performance benefits."""

import matplotlib.pyplot as plt
import numpy as np

# Data from benchmarks
array_sizes_mb = [0.001, 1.0, 10.0]  # KB, MB, MB
array_sizes_labels = ["1 KB", "1 MB", "10 MB"]

# Decode times in microseconds (from benchmark results)
pickle_times = [1.6, 13.5, 123.6]
copy_times = [0.82, 28.8, 248.3]
zerocopy_times = [0.97, 0.97, 0.99]

# Calculate speedups
speedup_vs_pickle = [p/z for p, z in zip(pickle_times, zerocopy_times)]
speedup_vs_copy = [c/z for c, z in zip(copy_times, zerocopy_times)]

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Decode times (log scale)
x = np.arange(len(array_sizes_labels))
width = 0.25

bars1 = ax1.bar(x - width, pickle_times, width, label='Pickle', color='#e74c3c')
bars2 = ax1.bar(x, copy_times, width, label='Copy', color='#f39c12')
bars3 = ax1.bar(x + width, zerocopy_times, width, label='Zero-Copy', color='#27ae60')

ax1.set_ylabel('Decode Time (μs, log scale)', fontsize=12)
ax1.set_xlabel('Array Size', fontsize=12)
ax1.set_title('Decode Performance: Zero-Copy vs Copying', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(array_sizes_labels)
ax1.legend()
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3, which='both')

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=9)

# Plot 2: Speedup factors
bars1 = ax2.bar(x - width/2, speedup_vs_pickle, width, label='vs Pickle', color='#e74c3c', alpha=0.7)
bars2 = ax2.bar(x + width/2, speedup_vs_copy, width, label='vs Copy', color='#f39c12', alpha=0.7)

ax2.set_ylabel('Speedup Factor (x faster)', fontsize=12)
ax2.set_xlabel('Array Size', fontsize=12)
ax2.set_title('Zero-Copy Speedup', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(array_sizes_labels)
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}x',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('benchmarks/numpy_codec_performance.png', dpi=150, bbox_inches='tight')
print("✓ Saved visualization to benchmarks/numpy_codec_performance.png")

# Also create a simpler chart showing constant-time nature
fig2, ax = plt.subplots(figsize=(10, 6))

ax.plot(array_sizes_mb, pickle_times, 'o-', linewidth=2, markersize=8, label='Pickle (O(n))', color='#e74c3c')
ax.plot(array_sizes_mb, copy_times, 's-', linewidth=2, markersize=8, label='Copy (O(n))', color='#f39c12')
ax.plot(array_sizes_mb, zerocopy_times, '^-', linewidth=2, markersize=8, label='Zero-Copy (O(1))', color='#27ae60')

ax.set_xlabel('Array Size (MB, log scale)', fontsize=12)
ax.set_ylabel('Decode Time (μs)', fontsize=12)
ax.set_title('Zero-Copy Has Constant Decode Time', fontsize=14, fontweight='bold')
ax.set_xscale('log')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Annotate the zero-copy line
ax.annotate('Constant ~1μs\nregardless of size!', 
            xy=(1.0, 0.97), xytext=(0.1, 50),
            arrowprops=dict(arrowstyle='->', color='#27ae60', lw=2),
            fontsize=11, color='#27ae60', fontweight='bold')

plt.tight_layout()
plt.savefig('benchmarks/numpy_codec_constant_time.png', dpi=150, bbox_inches='tight')
print("✓ Saved visualization to benchmarks/numpy_codec_constant_time.png")

print("\nKey Insights:")
print(f"• Zero-copy decode time is CONSTANT at ~1μs")
print(f"• For 10MB arrays: {speedup_vs_pickle[-1]:.0f}x faster than pickle, {speedup_vs_copy[-1]:.0f}x faster than copy")
print(f"• Zero-copy is O(1), copying is O(n)")
