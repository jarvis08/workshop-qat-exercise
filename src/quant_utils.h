#ifndef QUANT_UTILS_H
#define QUANT_UTILS_H

#include <stdint.h>
#include <darknet.h>

void quantize_int4_cpu(const float* input, int4_t *output, const int n, const float QS, const int QZ);
void quantize_int8_cpu(const float* input, int8_t *output, const int n, const float QS, const int8_t QZ);
void quantize_int32_cpu(const float* input, int32_t *output, const int n, const float QS, const int32_t QZ);
void dequantize_int4_cpu(const int4_t *input, float *output, const int n, const float QS, const int QZ);
void dequantize_int8_cpu(const int8_t *input, float *output, const int n, const float QS, const int8_t QZ);
void dequantize_int32_cpu(const int32_t *input, float *output, const int n, const float QS, const int32_t QZ);

void fake_quantize_int4(float *input, const int n, const float _min, const float _max);
void fake_quantize_int8(float *input, const int n, const float _min, const float _max);
void fake_quantize_int4_cpu(float* input, const int n, const float QS, const int QZ);
void fake_quantize_int8_cpu(float* input, const int n, const float QS, const int8_t QZ);

void totalsum_int4_cpu(const int M, const int N, const int K,
                       const int4_t *lhs,
                       const int4_t *rhs,
                       int4_t *C,
                       const int32_t *C32,
                       const int32_t *biases, const float *QS, const int4_t *QZ,
                       const int mode);

void totalsum_int8_cpu(const int M, const int N, const int K,
                       const int8_t *lhs,
                       const int8_t *rhs,
                       int8_t *C,
                       const int32_t *C32,
                       const int32_t *biases, const float *QS, const int8_t *QZ,
                       const int mode);

void float_to_int32(float *FP, int32_t *INT32, int n);
void int4_to_float(int4_t *INT4, float *FP, int n);
void int8_to_float(int8_t *INT8, float *FP, int n);
char *replaceString(char *s, const char *prev, const char *nxt);
void get_min_max_cpu(const float *mat, float *pmin, float *max, const int n);
void set_min_max(const float *mat, float *pmin, float *max, const int n);
void set_batch_range(float* mat, float* min, float* max, const int n);
void ema_cpu(const float *mat, float *pmin, float *max, const int n, const float smooth_param, const int is_relu);

void random_shuffle(int* arr, int n);
int static qsort_cmp(const void *first, const void *second);
float get_mean(float *mat, int n);
float sum_squared_array(float *a, int n);
void clamped_ema(float *origin_mat, int mat_size, float *min_max_arr, const float clamping_level, const float smooth_param);
void clamp_mat_qsz(float *origin_mat, int mat_size, float *min_max_arr, const float clamping_level);
void clamp_weight(float *origin_mat, int mat_size, float *min_max_arr, const int layer_idx);
void restore_gap(float *mat, int n, const float gap);

void test_fq();

#ifdef GPU
void fake_quantize_int4_gpu(float *input, const int n, const float QS, const int QZ);
void fake_quantize_int8_gpu(float *input, const int n, float QS, int8_t QZ);
void momentum_gpu(int N, float M, float LR, float* X_GRAD, float *V);
void fill_quant_gpu(int N, int8_t ALPHA, int8_t * X, int INCX);

void set_range_gpu(float *mat_gpu, float *target, const int n, const int is_relu);
void set_min_max_gpu(float *mat_gpu, int n, float *target);
void ema_gpu(float *mat_gpu, int n, float *target, float smooth_param);
void prune_matrix(float *mat, int mat_size, float *gap, const float prune_ratio);
void mask_matrix(float *origin_mat, float *mask_mat, int mat_size);
void clamp_mat_gpu(float *mat_gpu, int mat_size, float *min_max);
#endif
#endif
