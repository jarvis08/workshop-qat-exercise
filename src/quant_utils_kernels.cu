//#include "cuda_runtime.h"
//#include "curand.h"
#include "cublas_v2.h"
extern "C" {
#include "quant_utils.h"
#include "cuda.h"
}

__global__ void fake_quantize_int4_gpu_kernel(float* input, const int n, const float QS, const int QZ)
{
    // 작성 X
}

__global__ void fake_quantize_int8_gpu_kernel(float* input, const int n,
        const float QS, const int8_t QZ) {
    // 작성 X
}

__global__ void momentum_kernel(const int N, const float M, const float LR, float *X_GRAD, float *V)
{
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) V[i] = M * V[i] + LR * X_GRAD[i];
}

__global__ void fill_quant_kernel(int N, int8_t ALPHA, int8_t *X, int INCX)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) X[i*INCX] = ALPHA;
}

void fill_quant_gpu(int N, int8_t ALPHA, int8_t * X, int INCX)
{
    fill_quant_kernel<<<cuda_gridsize(N), BLOCK>>>(N, ALPHA, X, INCX);
    check_error(cudaPeekAtLastError());
}

void fake_quantize_int8_gpu(float *input, const int n, float QS, int8_t QZ) {
    fake_quantize_int8_gpu_kernel<<<cuda_gridsize(n), BLOCK>>>(input, n, QS, QZ);
}

void fake_quantize_int4_gpu(float *input, const int n, const float QS, const int QZ) {
    fake_quantize_int4_gpu_kernel<<<cuda_gridsize(n), BLOCK>>>(input, n, QS, QZ);
}

void momentum_gpu(int N, float M, float LR, float* X_GRAD, float *V)
{
    momentum_kernel<<<cuda_gridsize(N), BLOCK>>>(N, M, LR, X_GRAD, V);
    check_error(cudaPeekAtLastError());
    fill_gpu(N, 0, X_GRAD, 1);
}
