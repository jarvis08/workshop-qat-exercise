#ifndef CUDA_H
#define CUDA_H

#include "darknet.h"

#ifdef GPU

// quantization ========================================================================================================
int *cuda_make_array_int(int *x, size_t n);
int8_t *cuda_make_array_int8(int8_t *x, size_t n);
int32_t *cuda_make_array_int32(int32_t *x, size_t n);

void cuda_free_int(int *x_gpu);
void cuda_push_array_int(int *x_gpu, int *x, size_t n);
void cuda_pull_array_int(int *x_gpu, int *x, size_t n);

void cuda_free_int8(int8_t *x_gpu);
void cuda_push_array_int8(int8_t *x_gpu, int8_t *x, size_t n);
void cuda_pull_array_int8(int8_t *x_gpu, int8_t *x, size_t n);

void cuda_free_int32(int32_t *x_gpu);
void cuda_push_array_int32(int32_t *x_gpu, int32_t *x, size_t n);
void cuda_pull_array_int32(int32_t *x_gpu, int32_t *x, size_t n);
// =====================================================================================================================

void check_error(cudaError_t status);
cublasHandle_t blas_handle();
int *cuda_make_int_array(int *x, size_t n);
void cuda_random(float *x_gpu, size_t n);
float cuda_compare(float *x_gpu, float *x, size_t n, char *s);
dim3 cuda_gridsize(size_t n);


#ifdef CUDNN
cudnnHandle_t cudnn_handle();
#endif

#endif
#endif
