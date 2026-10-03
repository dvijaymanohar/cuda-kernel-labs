// Build: nvcc -O3 examples/histogram.cu -o histogram
#include <cuda_runtime.h>
#include <array>
#include <iostream>
#include <random>
#include <vector>
__global__ void hist_global(const unsigned char*x,int*nout,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)atomicAdd(&nout[x[i]],1);
}
__global__ void hist_shared(const unsigned char*x,int*nout,int n){
    __shared__ int local[256];
    for(int i=threadIdx.x;i<256;i+=blockDim.x)local[i]=0;__syncthreads();
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x)atomicAdd(&local[x[i]],1);
    __syncthreads();
    for(int i=threadIdx.x;i<256;i+=blockDim.x)atomicAdd(&nout[i],local[i]);
}
int main(){
    int n=1<<20;std::vector<unsigned char>h(n);std::mt19937 g(1);for(auto&v:h)v=(unsigned char)(g()%256);
    std::array<int,256>ref{};for(auto v:h)ref[v]++;
    unsigned char*d;int*out;cudaMalloc(&d,n);cudaMalloc(&out,256*4);cudaMemcpy(d,h.data(),n,cudaMemcpyHostToDevice);cudaMemset(out,0,1024);
    hist_shared<<<256,256>>>(d,out,n);std::array<int,256>got{};cudaMemcpy(got.data(),out,1024,cudaMemcpyDeviceToHost);
    bool ok=got==ref;std::cout<<(ok?"PASS":"FAIL")<<"\n";cudaFree(d);cudaFree(out);return ok?0:2;
}
