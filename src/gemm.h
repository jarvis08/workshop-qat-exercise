#ifndef GEMM_H
#define GEMM_H

#include <stdint.h>
#include <darknet.h>

#ifdef __cplusplus
extern "C" {
#endif

void cal_qsz(const float _min, const float _max, float *QS, int8_t *QZ);
void cal_qsz_int4(const float _min, const float _max, float *QS, int4_t *QZ);

void gemm_bin(int M, int N, int K, float ALPHA, 
        char  *A, int lda, 
        float *B, int ldb,
        float *C, int ldc);
        
void gemm(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc);

void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc);
        
void gemm_nt_int8(int M, int N, int K, float ALPHA,
        int8_t *A, int lda,
        int8_t *B, int ldb,
        int32_t *C, int ldc);

void gemm_nt(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc);

void gemm_nt_quant(int M, int N, int K, 
        int8_t *A, int lda, 
        int8_t *B, int ldb,
        int8_t *C, int ldc,
        int32_t *biases, float *QS, int8_t *QZ);

void gemm_nn_grad_first(int M, int N, int K,
        float *A, int lda, 
        float *B, int ldb,
        int8_t *C, int ldc,
        float *biases, float* QS, int8_t* QZ);

void gemm_nn_grad_second(int M, int N, int K,
        float *A, int lda, 
        int8_t *B, int ldb,
        int8_t *C, int ldc,
        float *biases, float* QS, int8_t* QZ);

void gemm_nn_quant(int M, int N, int K,
        int8_t *A, int lda, 
        int8_t *B, int ldb,
        int8_t *C, int ldc,
        int32_t *biases, float *QS, int8_t *QZ);

void quantize_matrix_int4(float *input, int4_t *output, int rows, int cols, float s, int4_t z);

void fake_quantization(float *input, const int rows, const int cols, float* act_range);
void make_quantized_matrix(float *input, int8_t *output, int rows, int cols, float* s, int8_t* z);
void quantize_matrix(float *input, int8_t *output, int rows, int cols, float s, int8_t z);
void quantize_biases(float *input, int32_t *output, int rows, int cols, float s, int8_t z);
void dequantize_matrix(int8_t *input, float *output, int rows, int cols, float s, int8_t z);

void compare_output(int8_t *QuantizedOutput, float *OriginOutput, int rows, int cols, float s, int8_t z);

void test();

#ifdef GPU
void get_nn_totalsum_gpu(int M, int N, int K,
                       int8_t *A, int lda,
                       int8_t *B, int ldb,
                       int8_t *C, int ldc,
                       int32_t *C32,
                       int32_t *biases, float *QS, int8_t *QZ);

void get_nt_totalsum_gpu(int M, int N, int K,
                       int8_t *A, int lda,
                       int8_t *B, int ldb,
                       int8_t *C, int ldc,
                       int32_t *C32,
                       int32_t *biases, float *QS, int8_t *QZ);

void gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A_gpu, int lda, 
        float *B_gpu, int ldb,
        float BETA,
        float *C_gpu, int ldc);

void gemm_quant_gpu(int TA, int TB, int M, int N, int K, float ALPHA,
                    int8_t *A_gpu, int lda,
                    int8_t *B_gpu, int ldb,
                    float BETA,
                    int32_t *C_gpu, int ldc);

void gemm_gpu_cublasGemmEx(int TA, int TB, int M, int N, int K, float ALPHA,
                           float *A_gpu, int lda,
                           float *B_gpu, int ldb,
                           float BETA,
                           float *C_gpu, int ldc);

void gemm_gpu_cublasGemmEx_int32(int TA, int TB,
                                 int M, int N, int K,
                                 int ALPHA,
                                 int8_t *A_gpu, int lda,
                                 int8_t *B_gpu, int ldb,
                                 int BETA,
                                 int32_t *C_gpu, int ldc);

void gemm_nn_cublasGemmEx_int32(int TA, int TB,
                                 int M, int N, int K,
                                 int ALPHA,
                                 int8_t *A_gpu, int lda,
                                 int8_t *B_gpu, int ldb,
                                 int BETA,
                                 int32_t *C_gpu, int ldc);

void test_gpu_accuracy(int TA, int TB, int m, int k, int n);

void test_gpu_accuracy_float();
void test_gpu_accuracy_int();
void test_gpu_accuracy_int32();
#endif
#ifdef __cplusplus
}
#endif
#endif
