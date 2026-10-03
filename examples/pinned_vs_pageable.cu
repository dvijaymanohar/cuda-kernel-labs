// Build: nvcc -O3 examples/pinned_vs_pageable.cu -o pinned_vs_pageable
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
static float copy_ms(void*h,float*d,size_t bytes){cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);cudaEventRecord(a);cudaMemcpyAsync(d,h,bytes,cudaMemcpyHostToDevice);cudaEventRecord(b);cudaEventSynchronize(b);float ms;cudaEventElapsedTime(&ms,a,b);cudaEventDestroy(a);cudaEventDestroy(b);return ms;}
int main(){size_t n=1<<24,bytes=n*4;std::vector<float>pageable(n,1);float*pinned=nullptr,*d=nullptr;cudaMallocHost(&pinned,bytes);cudaMalloc(&d,bytes);for(size_t i=0;i<n;++i)pinned[i]=1;
 std::cout<<"pageable_ms="<<copy_ms(pageable.data(),d,bytes)<<" pinned_ms="<<copy_ms(pinned,d,bytes)<<"\n";
 cudaFreeHost(pinned);cudaFree(d);}
