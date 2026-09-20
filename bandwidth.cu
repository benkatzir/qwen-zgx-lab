#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#define CHECK(x) do {cudaError_t check_error=(x);if(check_error!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(check_error));return 1;}}while(0)
int main(){
  const size_t bytes=1ULL<<30; const int iterations=80;
  void *a,*b; cudaEvent_t s,e;
  CHECK(cudaMalloc(&a,bytes)); CHECK(cudaMalloc(&b,bytes));
  CHECK(cudaMemset(a,123,bytes)); CHECK(cudaMemset(b,45,bytes));
  CHECK(cudaEventCreate(&s)); CHECK(cudaEventCreate(&e));
  for(int i=0;i<5;i++) CHECK(cudaMemcpyAsync(b,a,bytes,cudaMemcpyDeviceToDevice));
  CHECK(cudaDeviceSynchronize()); CHECK(cudaEventRecord(s));
  for(int i=0;i<iterations;i++) CHECK(cudaMemcpyAsync(b,a,bytes,cudaMemcpyDeviceToDevice));
  CHECK(cudaEventRecord(e)); CHECK(cudaEventSynchronize(e)); float ms;
  CHECK(cudaEventElapsedTime(&ms,s,e));
  cudaDeviceProp p; CHECK(cudaGetDeviceProperties(&p,0));
  printf("{\"device\":\"%s\",\"bytes_per_copy\":%zu,\"copies\":%d,\"total_ms\":%.3f,\"payload_GBps\":%.3f,\"read_plus_write_GBps\":%.3f,\"not_attention_benchmark\":true}\n",p.name,bytes,iterations,ms,bytes*iterations/(ms*1e6),2.0*bytes*iterations/(ms*1e6));
  cudaFree(a);cudaFree(b);
}
