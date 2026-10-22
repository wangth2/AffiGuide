// Explicit FP32 round-to-nearest multiply, then add; no atomics, FMA or parallel reduction.
#include <cuda_runtime.h>
#include <stdint.h>
#include <limits.h>

extern "C" __global__ void ag_fixed_linear_forward(const float* x,const float* w,const float* b,float* y,
                              int64_t M,int64_t N,int64_t K) {
  int64_t z=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;
  if(z>=M*N)return;
  int64_t m=z/N,n=z%N;float acc=0.0f;
  for(int64_t k=0;k<K;k++)acc=__fadd_rn(acc,__fmul_rn(x[m*K+k],w[n*K+k]));
  if(b)acc=__fadd_rn(acc,b[n]);
  y[z]=acc;
}
extern "C" __global__ void ag_fixed_linear_dx(const float* g,const float* w,float* dx,
                         int64_t M,int64_t N,int64_t K) {
  int64_t z=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;
  if(z>=M*K)return;
  int64_t m=z/K,k=z%K;float acc=0.0f;
  for(int64_t n=0;n<N;n++)acc=__fadd_rn(acc,__fmul_rn(g[m*N+n],w[n*K+k]));
  dx[z]=acc;
}
extern "C" __global__ void ag_fixed_linear_dw(const float* x,const float* g,float* dw,
                         int64_t M,int64_t N,int64_t K) {
  int64_t z=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;
  if(z>=N*K)return;
  int64_t n=z/K,k=z%K;float acc=0.0f;
  for(int64_t m=0;m<M;m++)acc=__fadd_rn(acc,__fmul_rn(g[m*N+n],x[m*K+k]));
  dw[z]=acc;
}
extern "C" __global__ void ag_fixed_linear_db(const float* g,float* db,int64_t M,int64_t N) {
  int64_t n=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;
  if(n>=N)return;
  float acc=0.0f;
  for(int64_t m=0;m<M;m++)acc=__fadd_rn(acc,g[m*N+n]);
  db[n]=acc;
}
