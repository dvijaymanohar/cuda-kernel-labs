# Profiling

Start with correctness, then benchmark with CUDA events. Use Nsight Systems for transfer/launch/synchronization timelines and Nsight Compute only on dominant kernels.

Suggested commands:
```bash
nsys profile -o profiles/vector ./build/vector_add 50000000
ncu -o profiles/reduction ./build/reduction 50000000
```
