#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <vector>

__global__ void reduce_sum(const float* in,float* partial,size_t n){
    extern __shared__ float s[];
    unsigned tid=threadIdx.x; size_t i=(size_t)blockIdx.x*blockDim.x*2+tid;
    float v=0; if(i<n)v+=in[i]; if(i+blockDim.x<n)v+=in[i+blockDim.x];
    s[tid]=v; __syncthreads();
    for(unsigned step=blockDim.x/2;step;step>>=1){ if(tid<step)s[tid]+=s[tid+step]; __syncthreads(); }
    if(tid==0) partial[blockIdx.x]=s[0];
}
int main(int argc,char** argv){
    size_t n=argc>1?std::strtoull(argv[1],nullptr,10):1<<20;
    std::vector<float> h(n); for(size_t i=0;i<n;++i) h[i]=float((i%11)-5);
    const float ref=std::accumulate(h.begin(),h.end(),0.0f);
    const int threads=256; int blocks=(n+threads*2-1)/(threads*2);
    float *din,*dpartial; cudaMalloc(&din,n*4); cudaMalloc(&dpartial,blocks*4);
    cudaMemcpy(din,h.data(),n*4,cudaMemcpyHostToDevice);
    reduce_sum<<<blocks,threads,threads*sizeof(float)>>>(din,dpartial,n);
    std::vector<float> partial(blocks); cudaMemcpy(partial.data(),dpartial,blocks*4,cudaMemcpyDeviceToHost);
    float got=std::accumulate(partial.begin(),partial.end(),0.0f);
    std::cout<<"got="<<got<<" ref="<<ref<<" abs_err="<<std::fabs(got-ref)<<"\n";
    cudaFree(din);cudaFree(dpartial);
    return std::fabs(got-ref)<=1e-2f*std::max(1.0f,std::fabs(ref))?0:3;
}
