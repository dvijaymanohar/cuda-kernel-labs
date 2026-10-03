# DSP / signal-processing extension

Optional advanced module connecting CUDA kernels to practical signal processing.

## Concepts
sampling and Nyquist; time vs frequency domain; FIR/IIR concepts; convolution; correlation; FFT; fixed vs floating point; CPU/GPU/FPGA trade-offs.

## Practical ladder
1. Implement CPU and CUDA 1D convolution.
2. Validate edge handling and numerical tolerance.
3. Implement correlation and compare with convolution.
4. Use a trusted FFT library for spectral analysis.
5. Compare direct convolution with FFT-based convolution across signal/filter sizes.
6. Sweep precision where supported.
7. Optional: process chunks with CUDA streams and overlap transfer/compute.

Deep DSP theory remains reference material; this repository owns executable code, correctness, and measured evidence.
