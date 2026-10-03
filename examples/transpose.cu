// Build: nvcc -O3 examples/transpose.cu -o transpose
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <vector>
__global__ void naive(const float*in,float*out,int w,int h){
 int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;
 if(x<w&&y<h)out[x*h+y]=in[y*w+x];
}
__global__ void tiled(const float*in,float*out,int w,int h){
 __shared__ float tile[32][33];
 int x=blockIdx.x*32+threadIdx.x,y=blockIdx.y*32+threadIdx.y;
 for(int j=0;j<32;j+=8)if(x<w&&y+j<h)tile[threadIdx.y+j][threadIdx.x]=in[(y+j)*w+x];
 __syncthreads();x=blockIdx.y*32+threadIdx.x;y=blockIdx.x*32+threadIdx.y;
 for(int j=0;j<32;j+=8)if(x<h&&y+j<w)out[(y+j)*h+x]=tile[threadIdx.x][threadIdx.y+j];
}
int main(){int w=1000,h=777;size_t bytes=(size_t)w*h*4;std::vector<float>a(w*h),b(w*h),c(w*h);for(size_t i=0;i<a.size();++i)a[i]=float(i%101);
 float *x,*y;cudaMalloc(&x,bytes);cudaMalloc(&y,bytes);cudaMemcpy(x,a.data(),bytes,cudaMemcpyHostToDevice);dim3 th(32,8),bl((w+31)/32,(h+31)/32);
 naive<<<bl,th>>>(x,y,w,h);cudaMemcpy(b.data(),y,bytes,cudaMemcpyDeviceToHost);tiled<<<bl,th>>>(x,y,w,h);cudaMemcpy(c.data(),y,bytes,cudaMemcpyDeviceToHost);
 bool ok=b==c;std::cout<<(ok?"PASS":"FAIL")<<"\n";cudaFree(x);cudaFree(y);return ok?0:2;}
