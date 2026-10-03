#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>
#define CUDA_CHECK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){std::cerr<<cudaGetErrorString(e)<<"\n"; return 2;}}while(0)

__global__ void add(const float* a,const float* b,float* c,size_t n){
    for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(size_t)blockDim.x*gridDim.x) c[i]=a[i]+b[i];
}
int main(int argc,char** argv){
    size_t n=argc>1?std::strtoull(argv[1],nullptr,10):1<<20;
    std::vector<float>a(n),b(n),c(n);
    for(size_t i=0;i<n;++i){a[i]=float(i%97); b[i]=float(i%53);}
    float *da,*db,*dc; CUDA_CHECK(cudaMalloc(&da,n*4)); CUDA_CHECK(cudaMalloc(&db,n*4)); CUDA_CHECK(cudaMalloc(&dc,n*4));
    CUDA_CHECK(cudaMemcpy(da,a.data(),n*4,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db,b.data(),n*4,cudaMemcpyHostToDevice));
    add<<<std::min<size_t>((n+255)/256,4096),256>>>(da,db,dc,n);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaMemcpy(c.data(),dc,n*4,cudaMemcpyDeviceToHost));
    for(size_t i=0;i<n;++i) if(std::fabs(c[i]-(a[i]+b[i]))>1e-5){std::cerr<<"mismatch "<<i<<"\n"; return 3;}
    std::cout<<"PASS n="<<n<<"\n"; cudaFree(da);cudaFree(db);cudaFree(dc);
}
