// Build: nvcc -O3 examples/streams_overlap.cu -o streams_overlap
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
__global__ void scale(float*x,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)x[i]*=2.0f;}
int main(){
 int n=1<<24,chunks=4,chunk=n/chunks;std::vector<float>h(n,1);float*d;cudaMalloc(&d,(size_t)n*4);
 cudaStream_t s[chunks];for(auto&x:s)cudaStreamCreate(&x);
 for(int c=0;c<chunks;++c){int off=c*chunk;cudaMemcpyAsync(d+off,h.data()+off,(size_t)chunk*4,cudaMemcpyHostToDevice,s[c]);scale<<<(chunk+255)/256,256,0,s[c]>>>(d+off,chunk);cudaMemcpyAsync(h.data()+off,d+off,(size_t)chunk*4,cudaMemcpyDeviceToHost,s[c]);}
 cudaDeviceSynchronize();bool ok=true;for(float v:h)if(v!=2){ok=false;break;}std::cout<<(ok?"PASS":"FAIL")<<"\n";
 for(auto&x:s)cudaStreamDestroy(x);cudaFree(d);return ok?0:2;
}
