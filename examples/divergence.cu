// Build: nvcc -O3 examples/divergence.cu -o divergence
#include <cuda_runtime.h>
#include <iostream>
__global__ void divergent(float*x,int n,int iters){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float v=x[i];if(threadIdx.x&1)for(int k=0;k<iters;++k)v=v*1.000001f+1e-6f;else for(int k=0;k<iters/8;++k)v=v*1.000001f+1e-6f;x[i]=v;}}
__global__ void grouped(float*x,int n,int iters){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float v=x[i];bool heavy=((i/32)&1);int loops=heavy?iters:iters/8;for(int k=0;k<loops;++k)v=v*1.000001f+1e-6f;x[i]=v;}}
int main(){int n=1<<24;float*x;cudaMalloc(&x,(size_t)n*4);cudaMemset(x,0,(size_t)n*4);divergent<<<(n+255)/256,256>>>(x,n,256);grouped<<<(n+255)/256,256>>>(x,n,256);cudaDeviceSynchronize();std::cout<<"Profile both kernels and compare branch/warp execution efficiency.\n";cudaFree(x);}
