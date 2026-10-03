# CUDA Kernel Labs

Implement CUDA primitives progressively from indexing to asynchronous pipelines.

## Sequence
vector add → grid-stride loops → reduction → scan → histogram → transpose → shared tiling → divergence → atomics → streams/events → copy/compute overlap.

## Build
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/vector_add 1000000
./build/reduction 1000000
```

Every kernel must have a CPU/reference result and CUDA error checking before performance work.
