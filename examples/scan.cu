// Build: nvcc -O3 examples/scan.cu -o scan
#include <cuda_runtime.h>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>
__global__ void block_scan(const int*in,int*out,int n){
    extern __shared__ int s[]; int t=threadIdx.x,i=blockIdx.x*blockDim.x+t;
    s[t]=i<n?in[i]:0; __syncthreads();
    for(int off=1;off<blockDim.x;off<<=1){
        int v=t>=off?s[t-off]:0; __syncthreads(); s[t]+=v; __syncthreads();
    }
    if(i<n)out[i]=s[t];
}
int main(){
    int n=1000;std::vector<int>h(n,1),got(n),ref(n);std::partial_sum(h.begin(),h.end(),ref.begin());
    int *a,*b;cudaMalloc(&a,n*4);cudaMalloc(&b,n*4);cudaMemcpy(a,h.data(),n*4,cudaMemcpyHostToDevice);
    // This first lab intentionally supports one block only. Generalize it in the exercise.
    if(n>1024){std::cerr<<"n must be <=1024 for this teaching kernel\n";return 1;}
    block_scan<<<1,1024,1024*sizeof(int)>>>(a,b,n);cudaMemcpy(got.data(),b,n*4,cudaMemcpyDeviceToHost);
    std::cout<<(got==ref?"PASS":"FAIL")<<" last="<<got.back()<<"\n";cudaFree(a);cudaFree(b);return got==ref?0:2;
}
