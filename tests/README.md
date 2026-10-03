# Correctness strategy

Test tiny sizes, zero/one element where supported, odd and non-block-aligned sizes, randomized values, boundary conditions, and precision-appropriate tolerances. Call `cudaGetLastError` after launches and synchronize where errors must be surfaced.
