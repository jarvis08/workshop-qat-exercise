#include <string.h> // memcpy
#include <math.h>   // round
#include <stdlib.h> // qsort
#include <stdio.h>
#include <stdint.h>
#include <assert.h>

#include <float.h> // int8_t
#include <limits.h> // for int32(int or long int) limit

#include <time.h>

#include "darknet.h"
#include "gemm.h"
#include "utils.h"
#include "quant_utils.h"


void dequantize_int4_cpu(const int4_t *input, float *output, const int n, const float QS, const int QZ)
{
    int i;
    #pragma omp parallel for
    for (i = 0; i < n; i++) {
        output[i] = QS * (input[i].el - QZ);
    }
}

void dequantize_int8_cpu(const int8_t *input, float *output, const int n, const float QS, const int8_t QZ)
{
    int i;
    #pragma omp parallel for
    for (i = 0; i < n; i++) {
        output[i] = QS * (input[i] - QZ);
    }
}

void dequantize_int32_cpu(const int32_t *input, float *output, const int n, const float QS, const int32_t QZ)
{
    int i;
    #pragma omp parallel for
    for (i = 0; i < n; i++) {
        output[i] = QS * (input[i] - QZ);
    }
}


void quantize_int4_cpu(const float* input, int4_t *output, const int n, const float QS, const int QZ)
{
    int i;
    #pragma omp parallel for
    for (i = 0; i < n; i++) {
        output[i].el = round(fmax(0.f, fmin(15.f, QZ + input[i] / QS)));
    }
}

void quantize_int8_cpu(const float* input, int8_t *output, const int n, const float QS, const int8_t QZ)
{
    int i;
    #pragma omp parallel for
    for (i = 0; i < n; i++) {
        output[i] = round(fmax(-128.f, fmin(127.f, QZ + input[i] / QS)));
    }
}

void quantize_int32_cpu(const float* input, int32_t *output, const int n, const float QS, const int32_t QZ)
{
    int i;
    #pragma omp parallel for
    for (i = 0; i < n; i++) {
        output[i] = round(fmax(INT_MIN, fmin(INT_MAX, QZ + input[i] / QS)));
    }
}

void fake_quantize_int4_cpu(float* input, const int n, const float QS, const int QZ)
{
    // 작성 X
}

void fake_quantize_int8_cpu(float* input, const int n, const float QS, const int8_t QZ)
{
    // input matirx를 quantize, dequantize 하는 과정을 작성하세요.
    int i;
    // 아래 #pragma ... 라인은 나머지 코드를 모두 작성한 후 주석을 제거해 주세요.
    //#pragma omp parallel for
    // line 85: input 배열의 크기 만큼 반복문을 시작
    // line 86: input matrix 요소 하나를 quantized 값으로 변경
    // line 87: qauntized 요소를 다시 dequantize하여 input matrix에 삽입
}

void fake_quantize_int4(float *input, const int n, const float _min, const float _max)
{
    float QS;
    int4_t QZ;
    cal_qsz_int4(_min, _max, &QS, &QZ);
#ifdef GPU
    fake_quantize_int4_gpu(input, n, QS, QZ.el);
#else
    fake_quantize_int4_cpu(input, n, QS, QZ.el);
#endif
}

void fake_quantize_int8(float *input, const int n, const float _min, const float _max)
{
    float QS;
    int8_t QZ;
    cal_qsz(_min, _max, &QS, &QZ);
#ifdef GPU
    fake_quantize_int8_gpu(input, n, QS, QZ);
#else
    fake_quantize_int8_cpu(input, n, QS, QZ);
#endif
}

void quantize_M(float real_multiplier, int32_t* quantized_multiplier, int* right_shift) {

    assert(real_multiplier > 0.f);
    assert(real_multiplier < 1.f);

    int s = 0;

    while(real_multiplier < 0.5f) {
        real_multiplier *= 2.0f;
        s++;
    }

    int64_t q = (int64_t)(round(real_multiplier * (1ll << 31)));
    assert(q <= (1ll << 31));

    if (q == (1ll << 31)) {
        q /= 2;
        s--;
    }
    assert(s >= 0);
    assert(q <= INT_MAX);
    *quantized_multiplier = (int32_t)(q);
    *right_shift = s;
}

int32_t multiply_M(int32_t subSum, int32_t multiplier)
{
    int overflow = subSum == multiplier && subSum == INT_MIN;
    int64_t subSum_64 = subSum;
    int64_t multiplier_64 = multiplier;
    int64_t subSumMultiplier = subSum_64 * multiplier_64;
    int32_t nudge = subSumMultiplier >= 0 ? (1 << 30) : (1 - (1 << 30));
    int32_t subSumMultiplier_high = (int32_t)((subSumMultiplier + nudge) / (1ll << 31));

    return overflow ? INT_MAX : subSumMultiplier_high;
}

int32_t BitAnd(int32_t a, int32_t b) {
    return a & b;
}

int32_t ShiftRight(int32_t a, int offset) {
    return a >> offset;
}

int32_t MaskIfNonZero(int32_t a) {
    static int32_t zero = 0;
    return a ? ~zero : zero;
}

int32_t MaskIfLessThan(int32_t a, int32_t b) {
    return MaskIfNonZero(a < b);
}

int32_t MaskIfGreaterThan(int32_t a, int32_t b) {
    return MaskIfNonZero(a > b);
}

int32_t shifting(int32_t x, int32_t exponent) {
    assert(exponent >= 0);
    assert(exponent <= 31);

    const int32_t mask = (1ll << exponent) - 1;
    const int32_t zero = 0;
    const int32_t one = 1;
    const int32_t remainder = BitAnd(x, mask);
    const int32_t threshold = ShiftRight(mask, 1) + BitAnd(MaskIfLessThan(x, zero), one);

    return ShiftRight(x, exponent) + BitAnd(MaskIfGreaterThan(remainder, threshold), one);
}

void quantized_gemm_int8_cpu(const int M, const int N, const int K,
                       const int8_t *lhs,
                       const int8_t *rhs,
                       int8_t *C,
                       const int32_t *C32,
                       const int32_t *biases,
                       const float *QS, const int8_t *QZ,
                       const int mode)
{
    int l_stride = K;
    int r_stride = 0;
    int o_stride = N;
    if ( mode ) r_stride = K;
    else r_stride = N;

    int32_t sumQ1Q2 = 0;
    int32_t subSum = 0;
    int32_t totalSum = 0;
    int32_t total = 0;
    int row, depth, col;

    // line 207: FP로 표현되는 M 계산을 0이 아닌, 올바른 계산으로 수정하세요.
    //const float real_M = 0;
    const float real_M = QS[0] * QS[1] / QS[2];

    int32_t quantized_M;
    int right_shift;
    quantize_M(real_M, &quantized_M, &right_shift);

    int32_t NZ1Z2 = K * QZ[0] * QZ[1];
    int32_t* a1row = calloc(M, sizeof(int32_t));
    int32_t* a2col = calloc(N, sizeof(int32_t));

    int _row = 0;
    #pragma omp parallel for
    for (row = 0; row < M; row++) {
        _row = row * l_stride;
        for (depth = 0; depth < K; depth++) {
            a1row[row] += lhs[_row + depth];
        }
    }

    if ( mode ) {
        gemm_nt_int8(M, N, K, 1, lhs, K, rhs, K, C32, N); 

        int _col = 0;
        #pragma omp parallel for
        for (col = 0; col < N; col++) {
            _col = col * r_stride;
            for (depth = 0; depth < K; depth++) {
                a2col[col] += rhs[_col + depth];
            }
        }

        // 아래 for문 내에서 biases[col]을 올바른 위치에서 더해 주세요.
        #pragma omp parallel for
        for (row = 0; row < M; row++) {
            for (col = 0; col < N; col++) {
                sumQ1Q2 = C32[row * o_stride + col];
                subSum = (NZ1Z2 - QZ[0] * a2col[col] - QZ[1] * a1row[row] + sumQ1Q2);
                subSum = multiply_M(subSum, quantized_M);
                subSum += biases[col];

                total = shifting(subSum, right_shift);
                totalSum = QZ[2] + total;
                totalSum = totalSum > 127 ? 127 : totalSum < -128 ? -128 : totalSum;
                C[row * o_stride + col] = totalSum;
            }
        }
    } else {
        // 아래 내용은 포함되지 않습니다.
        #pragma omp parallel for
        for (col = 0; col < N; col++) {
            for (depth = 0; depth < K; depth++) {
                a2col[col] += rhs[col + depth * r_stride];
            }
        }

        #pragma omp parallel for
        for (row = 0; row < M; row++) {
            for (col = 0; col < N; col++) {
                sumQ1Q2 = C32[row * o_stride + col];
                subSum = (NZ1Z2 - QZ[1] * a2col[col] - QZ[0] * a1row[row] + sumQ1Q2);
                subSum = multiply_M(subSum, quantized_M);

                total = shifting(subSum, right_shift);
                totalSum = QZ[2] + total;
                totalSum = totalSum > 127 ? 127 : totalSum < -128 ? -128 : totalSum;
                C[row * o_stride + col] = totalSum;
            }
        }
    }
    free(a1row);
    free(a2col);
}

void totalsum_int4_cpu(const int M, const int N, const int K,
                       const int4_t *lhs,
                       const int4_t *rhs,
                       int4_t *C,
                       const int32_t *C32,
                       const int32_t *biases,
                       const float *QS, const int4_t *QZ,
                       const int mode)
{
    // 작성 X
}

void totalsum_int8_cpu(const int M, const int N, const int K,
                       const int8_t *lhs,
                       const int8_t *rhs,
                       int8_t *C,
                       const int32_t *C32,
                       const int32_t *biases,
                       const float *QS, const int8_t *QZ,
                       const int mode)
{
    // 작성 X
}

void float_to_int32(float *FP, int32_t *INT32, int n)
{
    int i;
    #pragma omp parallel for
    for(i = 0; i < n; ++i) INT32[i] = (int32_t) FP[i];
}

void int4_to_float(int4_t *INT4, float *FP, int n)
{
    int i;
    #pragma omp parallel for
    for(i = 0; i < n; ++i) FP[i] = (float) INT4[i].el; 
}

void int8_to_float(int8_t *INT8, float *FP, int n)
{
    int i;
    #pragma omp parallel for
    for(i = 0; i < n; ++i) FP[i] = (float) INT8[i];
}

char *replaceString(char *s, const char *prev, const char *nxt)
{
  char *result, *sr;
  size_t i, count = 0;
  size_t prev_len = strlen(prev); if (prev_len < 1) return s;
  size_t nxt_len = strlen(nxt);


  if (nxt_len != prev_len) {
    for (i = 0; s[i] != '\0';) {
      if (memcmp(&s[i], prev, prev_len) == 0) count++, i += prev_len;
      else i++;
    }
  } else i = strlen(s);

  result = (char *) malloc(i + 1 + count * (nxt_len - prev_len));
  if (result == NULL) return NULL;

  sr = result;
  while (*s) {
    if (memcmp(s, prev, prev_len) == 0) {
      memcpy(sr, nxt, nxt_len);
      sr += nxt_len;
      s  += prev_len;
    } else *sr++ = *s++;
  }
  *sr = '\0';
  return result;
}

void get_min_max_cpu(const float *mat, float *pmin, float *pmax, const int n)
{
    int i;
    float _min = mat[0];
    float _max = mat[0];
    for(i = 0; i < n; ++i)
    {
       if(mat[i] > _max) _max = mat[i]; 
       else if(mat[i] < _min) _min = mat[i]; 
    }
    *pmin = _min;
    *pmax = _max;
}

void set_min_max(const float *mat, float *pmin, float *pmax, const int n)
{
    int i;
    float _min = mat[0];
    float _max = mat[0];
    for(i = 0; i < n; ++i)
    {
       if(mat[i] > _max) _max = mat[i]; 
       else if(mat[i] < _min) _min = mat[i]; 
    }
    if(*pmin > _min) *pmin = _min;
    if(*pmax < _max) *pmax = _max;
}

void set_batch_range(float* mat, float* min, float* max, const int n)
{
    int i;
    float _min = mat[0];
    float _max = mat[0];
    for(i = 0; i < n; ++i)
    {
       if(mat[i] > _max) _max = mat[i]; 
       else if(mat[i] < _min) _min = mat[i]; 
    }
    if(*min > _min) *min = _min;
    if(*max < _max) *max = _max;
}

void ema_cpu(const float *mat, float *pmin, float *pmax, const int n, const float smooth_param, const int is_relu)
{
    // Exponential Moving Average를 계산하는 코드를 완성하세요.
    int i;
    float _max = mat[0];
    if(is_relu) {
        for(i = 0; i < n; ++i)
        {
           if(mat[i] > _max) _max = mat[i]; 
        }
        // line 408: relu의 경우 min 값은 항상 0이다.
        // line 409: max값을 누적하는 변수에 ema를 적용하여 max 값을 갱신한다.
    } else {
        float _min = mat[0];
        for(i = 0; i < n; ++i)
        {
           if(mat[i] > _max) _max = mat[i]; 
           else if(mat[i] < _min) _min = mat[i]; 
        }
        // line 417: min 값을 누적하는 변수에 ema를 적용하여 min 값을 갱신한다.
        // line 418: max 값을 누적하는 변수에 ema를 적용하여 max 값을 갱신한다.
    }
}

void clamped_ema(float *origin_mat, int mat_size, float *min_max_arr, const float clamping_level, const float smooth_param)
{
    // 작성 X
}

void clamp_mat_qsz(float *origin_mat, int mat_size, float *min_max_arr, const float clamping_level)
{
    float *mat = calloc(mat_size, sizeof(float));
    memcpy(mat, origin_mat, mat_size * sizeof(float));

    float outliers_ratio = (1 - clamping_level) / 2;
    int skip = (int)round(mat_size * outliers_ratio);

	qsort(mat, mat_size, sizeof(float), qsort_cmp);
    min_max_arr[0] = mat[skip];
    min_max_arr[1] = mat[mat_size - 1 - skip];
    free(mat);
}

void clamp_weight(float *origin_mat, int mat_size, float *min_max_arr, const int layer_idx)
{
	int i;

    float *mat = calloc(mat_size, sizeof(float));
    memcpy(mat, origin_mat, mat_size * sizeof(float));
	qsort(mat, mat_size, sizeof(float), qsort_cmp);
    
    int skip_1_min = 0;
    int skip_2_min = 1;
    int skip_3_min = 1;
    int skip_4_min = 1;
    int skip_1_max = 0;
    int skip_2_max = 0;
    int skip_3_max = 1;
    int skip_4_max = 1;
    if(layer_idx == 0) {
        min_max_arr[0] = mat[skip_1_min];
        min_max_arr[1] = mat[mat_size - 1 - skip_1_max];
        printf("layer-%d, skipped before index %d(%f)\n", layer_idx+1, skip_1_min, mat[skip_1_min]);
        printf("layer-%d, skipped after index %d(%f)\n", layer_idx+1, mat_size - 1 - skip_1_max, mat[skip_1_min]);
        for(i=0;i<skip_1_min;i++) printf("Skipped = %f\n", mat[i]);
        for(i=0;i<skip_1_max;i++) printf("Skipped = %f\n", mat[mat_size - 1 - i]);
    } else if (layer_idx == 1) {
        min_max_arr[0] = mat[skip_2_min];
        min_max_arr[1] = mat[mat_size - 1 - skip_2_max];
        printf("layer-%d, skipped before index %d(%f)\n", layer_idx+1, skip_2_min, mat[skip_2_min]);
        printf("layer-%d, skipped after index %d(%f)\n", layer_idx+1, mat_size - 1 - skip_2_max, mat[skip_2_min]);
        for(i=0;i<skip_2_min;i++) printf("Skipped = %f\n", mat[i]);
        for(i=0;i<skip_2_max;i++) printf("Skipped = %f\n", mat[mat_size - 1 - i]);
    } else if (layer_idx == 2) {
        min_max_arr[0] = mat[skip_3_min];
        min_max_arr[1] = mat[mat_size - 1 - skip_3_max];
        printf("layer-%d, skipped before index %d(%f)\n", layer_idx+1, skip_3_min, mat[skip_3_min]);
        printf("layer-%d, skipped after index %d(%f)\n", layer_idx+1, mat_size - 1 - skip_3_max, mat[skip_3_min]);
        for(i=0;i<skip_3_min;i++) printf("Skipped = %f\n", mat[i]);
        for(i=0;i<skip_3_max;i++) printf("Skipped = %f\n", mat[mat_size - 1 - i]);
    } else if (layer_idx == 3) {
        min_max_arr[0] = mat[skip_4_min];
        min_max_arr[1] = mat[mat_size - 1 - skip_4_max];
        printf("layer-%d, skipped before index %d(%f)\n", layer_idx+1, skip_4_min, mat[skip_4_min]);
        printf("layer-%d, skipped after index %d(%f)\n", layer_idx+1, mat_size - 1 - skip_4_max, mat[skip_4_min]);
        for(i=0;i<skip_4_min;i++) printf("Skipped = %f\n", mat[i]);
        for(i=0;i<skip_4_max;i++) printf("Skipped = %f\n", mat[mat_size - 1 - i]);
    }
    free(mat);
}

void restore_gap(float *mat, int n, const float gap)
{
    int i;
    #pragma omp parallel for
    for(i = 0; i < n; i++){
        if(mat[i] > 0) mat[i] += gap;
        else if(mat[i] < 0) mat[i] -= gap;
    }
}

// For Fisher–Yates shuffle Algorithm
void swap(int *a, int *b) 
{ 
    int temp = *a; 
    *a = *b; 
    *b = temp; 
}

// Fisher–Yates shuffle Algorithm
// Ref: https://www.geeksforgeeks.org/shuffle-a-given-array-using-fisher-yates-shuffle-algorithm/
void random_shuffle(int* arr, int n)
{ 
	// duplicated srand makes random generate same numbers
    //srand ( time(NULL) ); 
    for (int i = n-1; i > 0; i--) 
    { 
        int j = rand() % (i+1); 
        swap(&arr[i], &arr[j]); 
    } 
}

int static qsort_cmp(const void* first, const void* second)
{
    if (*(float*)first > *(float*)second)
        return 1;
    else if (*(float*)first < *(float*)second)
        return -1;
    else
        return 0;
}


float get_mean(float *mat, int n)
{
    int i;
    float _sum = 0;
    for(i = 0; i < n; ++i)
    {
        _sum += mat[i];
    }
    return _sum / n;
}

float sum_squared_array(float *a, int n)
{
    int i;
    float sum = 0;
    for(i = 0; i < n; ++i) sum += a[i] * a[i];
    return sum;
}

#ifdef GPU
void set_range_gpu(float *mat_gpu, float *range, const int n, const int is_relu)
{
    float *mat = calloc(n, sizeof(float));
    cuda_pull_array(mat_gpu, mat, n);
    float _min = mat[0];
    float _max = mat[0];

    int i;
    if(is_relu) {
        for(i = 0; i < n; ++i)
        {
           if(mat[i] > _max) _max = mat[i]; 
        }
        if(range[1] < _max) range[1] = _max;
    } else {
        for(i = 0; i < n; ++i)
        {
           if(mat[i] > _max) _max = mat[i]; 
           else if(mat[i] < _min) _min = mat[i]; 
        }
        if(range[0] > _min) range[0] = _min;
        if(range[1] < _max) range[1] = _max;
    }
    free(mat);
}

void set_min_max_gpu(float *mat_gpu, int n, float *target)
{
    float *mat = calloc(n, sizeof(float));
    cuda_pull_array(mat_gpu, mat, n);
    int i;
    float _min = mat[0];
    float _max = mat[0];
    for(i = 0; i < n; ++i)
    {
       if(mat[i] > _max) _max = mat[i]; 
       else if(mat[i] < _min) _min = mat[i]; 
    }
    free(mat);
    if(target[0] > _min) target[0] = _min;
    if(target[1] < _max) target[1] = _max;
}

void ema_gpu(float *mat_gpu, int n, float *target, float smooth_param)
{
    // 작성 X
}

void prune_matrix(float *origin_mat, int mat_size, float *gap, const float prune_ratio)
{
    float *mat = calloc(mat_size, sizeof(float));
    memcpy(mat, origin_mat, mat_size * sizeof(float));

    int i;
    for(i = 0; i < mat_size; i++) mat[i] = fabs(mat[i]);
	qsort(mat, mat_size, sizeof(float), qsort_cmp);
    
    int num_prune = round(mat_size * prune_ratio);
    *gap = mat[num_prune];
    for(i = 0; i < mat_size; i++) {
        if(origin_mat[i] >=  - *gap && origin_mat[i] <= *gap) origin_mat[i] = 0;
    }
    free(mat);
}

void mask_matrix(float *origin_mat, float *mask_mat, int mat_size)
{
    float *mat = calloc(mat_size, sizeof(float));
    cuda_pull_array(origin_mat, mat, mat_size);

    int i;
    #pragma omp parallel for
    for(i = 0; i < mat_size; i++) mat[i] *= mask_mat[i];

    cuda_push_array(origin_mat, mat, mat_size);
    free(mat);
}

void clamp_mat_gpu(float *mat_gpu, int mat_size, float *min_max)
{
    int i;
    float *mat = calloc(mat_size, sizeof(float));
    cuda_pull_array(mat_gpu, mat, mat_size);

    // clamp outside of pruned mat's min/max
    #pragma omp parallel for
    for(i = 0; i < mat_size; ++i)
    {
       if(mat[i] > min_max[1]) {
           mat[i] = min_max[1]; 
       }
       else if(mat[i] < min_max[0]) {
           mat[i] = min_max[0]; 
       }
    }
    /*
    #pragma omp parallel for
    for(i = 0; i < mat_size; ++i)
    {
       if(mat[i] > 0.2) {
           mat[i] = 0.2; 
       }
       else if(mat[i] < - 0.2) {
           mat[i] = - 0.2; 
       }
    }
    */
    cuda_push_array(mat_gpu, mat, mat_size);
    free(mat);
}

#endif

float *make_random_matrix(int rows, int cols)
{
    int i;
    float *m = calloc(rows*cols, sizeof(float));

    for(i = 0; i < rows * cols; ++i){
        m[i] = rand_uniform(-1.0, 1.0);
    }
    return m;
}

void test_fq()
{
    //double time;
    clock_t start1, start2, end1, end2;
    float res1, res2;
    int i;

    float *A = NULL;  // (4 x 8)
    float *B = NULL;  // (8 x 4)
    int m = 75;
    int k = 1024;


    A = make_random_matrix(m,k);
    B = make_random_matrix(m,k);
    start1 = clock();
    for(i=0; i<5; i++) fake_quantization(A, m, k, 0);
    end1 = clock();
    res1 = (float)(end1 - start1)/CLOCKS_PER_SEC;
    printf("C++\t| Fake Quantization = %f\n", res1);

    start2 = clock();
    //for(i=0; i<5; i++) fake_quantize_weights_int8(B, m * k);
    end2 = clock();
    res2 = (float)(end2 - start2)/CLOCKS_PER_SEC;
    printf("C\t| Fake Quantization = %f\n", res2);

    free(A);
    free(B);

    // Value test;
    A = make_random_matrix(m,k);
    B = calloc(m*k, sizeof(float));
    memcpy(B, A, m * k * sizeof(float));

    fake_quantization(A, m, k, 0);
    //fake_quantize_weights_int8(B, m * k);
    for(i = 0; i < m * k; i ++) {
        printf("\n");
        printf("A[%d] = %f\n", i, A[i]);
        printf("B[%d] = %f\n", i, B[i]);
    }


    free(A);
    free(B);
    exit(0);
}
