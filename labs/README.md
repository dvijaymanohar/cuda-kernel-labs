# Expanded CUDA lab set

Each file is standalone and intentionally small.

- `scan.cu`: one-block inclusive scan; exercise is multi-block generalization.
- `histogram.cu`: shared-memory aggregation plus global atomics.
- `transpose.cu`: naive vs padded shared-memory tile.
- `divergence.cu`: intra-warp divergence vs grouped work.
- `streams_overlap.cu`: multiple streams for H2D → kernel → D2H.
- `pinned_vs_pageable.cu`: host-memory transfer experiment.

Build with `nvcc -O3 <file> -o <name>`. Validate correctness before measuring, then use CUDA events/Nsight tools rather than relying on a single wall-clock sample.
