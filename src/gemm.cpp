#include "gemm.h"
#include "utils.h"
extern "C" {
#include "cuda.h"
}
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdint.h>

#include "map.h"
#include "multiplication.h"
#include "MatrixMap.h"
#include "MatrixWithStorage.h"
#include "quantize.h"

void cal_qsz_int4(const float _min, const float _max, float *QS, int4_t *QZ)
{
    // 작성 X
}

void cal_qsz(const float _min, const float _max, float *QS, int8_t *QZ)
{
    // S, Z를 계산하는 코드를 작성하세요.
}

void gemm_bin(int M, int N, int K, float ALPHA,
        char  *A, int lda,
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            char A_PART = A[i*lda+k];
            if(A_PART){
                for(j = 0; j < N; ++j){
                    C[i*ldc+j] += B[k*ldb+j];
                }
            } else {
                for(j = 0; j < N; ++j){
                    C[i*ldc+j] -= B[k*ldb+j];
                }
            }
        }
    }
}

void test(){
    using namespace quantization;
    std::cout.precision(3);
    const int rows = 2;
    const int depth = 3;
    const int cols = 2;
    const auto kOrder = MapOrder::RowMajor;
    const auto cOrder = MapOrder::ColMajor;

    MatrixWithStorage<float, kOrder> float_lhs(rows, depth);
    float_lhs.lhs();
    MatrixWithStorage<float, cOrder> float_rhs(depth, cols);
    float_rhs.rhs();
    MatrixWithStorage<float, kOrder> float_result(rows, cols);
    auto float_result_map = float_result.Map();

    float biases[cols] = {1};
    MatrixWithStorage<float, kOrder> float_biases(biases, 1, cols);

    FloatMatrixMultiplication(float_lhs.ConstMap(), float_rhs.ConstMap(), &float_result_map, float_biases.ConstMap());
    std::cout << float_lhs << std::endl;
    std::cout << float_rhs << std::endl;
    std::cout << "Original Flaot MM Result : \n" << float_result << std::endl;

    FindMinMax(float_lhs.Map(), float_lhs.setMin(), float_lhs.setMax());
    FindMinMax(float_rhs.Map(), float_rhs.setMin(), float_rhs.setMax());
    FindMinMax(float_result.Map(), float_result.setMin(), float_result.setMax());

    const auto lhs_qparams = ChooseQuantizationParams(float_lhs.Min(), float_lhs.Max());
    const auto rhs_qparams = ChooseQuantizationParams(float_rhs.Min(), float_rhs.Max());
    const auto result_qparams = ChooseQuantizationParams(float_result.Min(), float_result.Max());

    MatrixWithStorage<std::int8_t, kOrder> uint8_lhs(rows, depth);
    MatrixWithStorage<std::int8_t, cOrder> uint8_rhs(depth, cols);
    MatrixWithStorage<std::int8_t, kOrder> uint8_result(rows, cols);
    MatrixWithStorage<std::int32_t, kOrder> int32_biases(1, cols);

    uint8_lhs.setQParams(lhs_qparams);
    uint8_rhs.setQParams(rhs_qparams);
    uint8_result.setQParams(result_qparams);

    std::cout << "LHS    : zero-point = " << static_cast<float>(uint8_lhs.getQParams().zero_point) <<
                 ", scale = " << uint8_lhs.getQParams().scale << std::endl;
    std::cout << "RHS    : zero-point = " << static_cast<float>(uint8_rhs.getQParams().zero_point) <<
              ", scale = " << uint8_rhs.getQParams().scale << std::endl;
    std::cout << "Result : zero-point = " << static_cast<float>(uint8_result.getQParams().zero_point) <<
              ", scale = " << uint8_result.getQParams().scale << std::endl;

    // Quantize
    quantization::quantize(uint8_lhs.getQParams(), float_lhs.Storage(), &uint8_lhs.Storage());
    quantization::quantize(uint8_rhs.getQParams(), float_rhs.Storage(), &uint8_rhs.Storage());
    quantization::quantize(uint8_result.getQParams(), float_result.Storage(), &uint8_result.Storage());

    std::cout << "Quantized uint8 LHS : \n" << uint8_lhs << std::endl;
    std::cout << "Quantized uint8 RHS : \n" << uint8_rhs << std::endl;

    const float real_M = uint8_lhs.getQParams().scale * uint8_rhs.getQParams().scale / uint8_result.getQParams().scale;
    std::int32_t quantized_M;
    int right_shift;
    quantization::QuantizeMultiplierSmallerThanOne(real_M, &quantized_M, &right_shift);

    std::cout << "ream_M : " << real_M << std::endl;
    std::cout << "quantized_M : " << quantized_M << std::endl;
    std::cout << "right_shift : " << right_shift << std::endl;

    OutputStageQuantizeDownInt32ByFixedPoint quantize_down_stage;
    quantize_down_stage.result_fixedpoint_multiplier = quantized_M;
    quantize_down_stage.result_shift = right_shift;

    // quantize Bias
    const quantization::quantizationParams biases_qparams = {.scale=lhs_qparams.scale * rhs_qparams.scale, .zero_point=0};
    int32_biases.setQParams(biases_qparams);
    quantization::quantize_bias(int32_biases.getQParams(), float_biases.Storage(), &int32_biases.Storage());

    // quantize GEMM
    auto uint8_result_map = uint8_result.Map();
//    quantization::QuantizeGemm(uint8_lhs.ConstMap(),
//                               uint8_rhs.ConstMap(),
//                               &uint8_result_map,
//                               uint8_lhs.getQParams(),
//                               uint8_rhs.getQParams(),
//                               uint8_result.getQParams(),
//                               quantize_down_stage,
//                               int32_biases.ConstMap());
    std::cout << "\nquantized uint8 Result : \n" << uint8_result << std::endl;

    // dequantize
    quantization::MatrixWithStorage<float, kOrder> actual_float_result(rows, cols);
    quantization::dequantize(uint8_result.getQParams(), uint8_result.Storage(), &actual_float_result.Storage());
    std::cout << "dequantized float_result : \n" << actual_float_result << std::endl;

}

float *random_matrix(int rows, int cols)
{
    int i;
    float *m = (float*)calloc(rows*cols, sizeof(float));
    for(i = 0; i < rows*cols; ++i){
        m[i] = i;
    }
    return m;
}

int *random_matrix_int(int rows, int cols)
{
    int i;
    int *m = (int*)calloc(rows*cols, sizeof(int));
    for(i = 0; i < rows*cols; ++i){
        m[i] = i;
    }
    return m;
}

int8_t *random_matrix_int8(int rows, int cols)
{
    int i;
    int8_t *m = (int8_t*)calloc(rows*cols, sizeof(int8_t));
    for(i = 0; i < rows*cols; ++i){
        m[i] = i;
    }
    return m;
}

int32_t *random_matrix_int32(int rows, int cols)
{
    int i;
    int32_t *m = (int32_t*)calloc(rows*cols, sizeof(int32_t));
    for(i = 0; i < rows*cols; ++i){
        m[i] = i;
    }
    return m;
}

float *random_matrix_zero(int rows, int cols)
{
    int i;
    float *m = (float*)calloc(rows*cols, sizeof(float));
    for(i = 0; i < rows*cols; ++i){
        m[i] = 0;
    }
    return m;
}

int *random_matrix_zero_int(int rows, int cols)
{
    int i;
    int *m = (int*)calloc(rows*cols, sizeof(int));
    for(i = 0; i < rows*cols; ++i){
        m[i] = 0;
    }
    return m;
}

int32_t *random_matrix_zero_int32(int rows, int cols)
{
    int i;
    int32_t *m = (int32_t*)calloc(rows*cols, sizeof(int32_t));
    for(i = 0; i < rows*cols; ++i){
        m[i] = 0;
    }
    return m;
}


void time_random_matrix(int TA, int TB, int m, int k, int n)
{
    float *a;
    if(!TA) a = random_matrix(m,k);
    else a = random_matrix(k,m);
    int lda = (!TA)?k:m;
    float *b;
    if(!TB) b = random_matrix(k,n);
    else b = random_matrix(n,k);
    int ldb = (!TB)?n:k;

    float *c = random_matrix(m,n);
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++)
            c[i*ldb + j] = 0;
    }
    int i;
    clock_t start = clock(), end;
    for(i = 0; i<1; ++i){
        gemm_cpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c,n);
    }
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++)
            printf("%f ", c[i*ldb + j]);
        printf("%c",'\n');
    }
    end = clock();
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %lf ms\n",m,k,k,n, TA, TB, (float)(end-start)/CLOCKS_PER_SEC);
    free(a);
    free(b);
    free(c);
}

void make_quantized_matrix(float *input, int8_t *output, int rows, int cols, float* s, int8_t* z)
{
    const auto kOrder = quantization::MapOrder::RowMajor;
    quantization::MatrixWithStorage<float, kOrder> float_mat(input, rows, cols);
    FindMinMax(float_mat.Map(), float_mat.setMin(), float_mat.setMax());
    const auto mat_qparams = quantization::ChooseQuantizationParams(float_mat.Min(), float_mat.Max());

    quantization::MatrixWithStorage<int8_t, kOrder> int8_mat(rows, cols);
    int8_mat.setQParams(mat_qparams);

    quantization::quantize(int8_mat.getQParams(), float_mat.Storage(), &int8_mat.Storage());
    int8_mat.toMatrix(&output);
    *s = mat_qparams.scale;
    *z = mat_qparams.zero_point;
}


void fake_quantization(float *input, const int rows, const int cols, float* act_range)
{
    const auto kOrder = quantization::MapOrder::RowMajor;
    quantization::MatrixWithStorage<float, kOrder> float_mat(input, rows, cols);
    quantization::MatrixWithStorage<int8_t, kOrder> quantized_mat(rows, cols);
    quantization::MatrixWithStorage<float, kOrder> dequantized_mat(rows, cols);

    quantization::quantizationParams qparams;
    if(act_range) {
        float QS;
        int8_t QZ;
        cal_qsz(act_range[0], act_range[1], &QS, &QZ);
        qparams = {.scale=QS, .zero_point=QZ};
    } else {
        FindMinMax(float_mat.Map(), float_mat.setMin(), float_mat.setMax());
        qparams = quantization::ChooseQuantizationParams(float_mat.Min(), float_mat.Max());
    }

    quantization::quantize(qparams, float_mat.Storage(), &quantized_mat.Storage());
    quantization::dequantize(qparams, quantized_mat.Storage(), &dequantized_mat.Storage());

    dequantized_mat.toMatrix(&input);
}

void quantize_matrix(float *input, int8_t *output, int rows, int cols, float s, int8_t z)
{
    const auto kOrder = quantization::MapOrder::RowMajor;
    quantization::MatrixWithStorage<float, kOrder> float_mat(input, rows, cols);
    quantization::MatrixWithStorage<int8_t, kOrder> int8_mat(rows, cols);
    const quantization::quantizationParams qparams = {.scale=s, .zero_point=z};
    quantization::quantize(qparams, float_mat.Storage(), &int8_mat.Storage());
    int8_mat.toMatrix(&output);
}

void quantize_biases(float *input, int32_t *output, int rows, int cols, float s, int8_t z)
{
    const auto kOrder = quantization::MapOrder::RowMajor;
    quantization::MatrixWithStorage<float, kOrder> float_mat(input, rows, cols);
    quantization::MatrixWithStorage<int32_t, kOrder> int32_mat(rows, cols);
    const quantization::quantizationParams qparams = {.scale=s, .zero_point=z};
    quantization::quantize_bias(qparams, float_mat.Storage(), &int32_mat.Storage());
    int32_mat.toMatrix(&output);
}

void dequantize_matrix(int8_t *input, float *output, int rows, int cols, float s, int8_t z)
{
    const auto kOrder = quantization::MapOrder::RowMajor;
    quantization::MatrixWithStorage<int8_t, kOrder> int8_mat(input, rows, cols);
    quantization::MatrixWithStorage<float, kOrder> float_mat(rows, cols);
    const quantization::quantizationParams qparams = {.scale=s, .zero_point=z};
    quantization::dequantize(qparams, int8_mat.Storage(), &float_mat.Storage());
    float_mat.toMatrix(&output);
}

void dequantize_tuned_matrix(int8_t *input, float *output, int rows, int cols, float s, int8_t z, const float tuned_gap)
{
    const auto kOrder = quantization::MapOrder::RowMajor;
    quantization::MatrixWithStorage<int8_t, kOrder> int8_mat(input, rows, cols);
    quantization::MatrixWithStorage<float, kOrder> float_mat(rows, cols);
    const quantization::quantizationParams qparams = {.scale=s, .zero_point=z};
    quantization::dequantize(qparams, int8_mat.Storage(), &float_mat.Storage());
    float_mat.toMatrix(&output);
}

void compare_output(int8_t *QuantizedOutput, float *OriginOutput, int rows, int cols, float s, int8_t z)
{
    const auto kOrder = quantization::MapOrder::RowMajor;

    quantization::MatrixWithStorage<float, kOrder> origin_float_result(OriginOutput, rows, cols);
    quantization::MatrixWithStorage<float, kOrder> dequantized_result(rows, cols);
    quantization::MatrixWithStorage<int8_t, kOrder> int8_result(QuantizedOutput, rows, cols);

    const quantization::quantizationParams result_qparams = {.scale=s, .zero_point=z};
    quantization::dequantize(result_qparams, int8_result.Storage(), &dequantized_result.Storage());

    float diff = 0;
    auto origMap = origin_float_result.ConstMap();
    auto dequantMap = dequantized_result.ConstMap();
    for (int row = 0; row < origMap.rows(); row++) {
        for (int col = 0; col < origMap.cols(); col++) {
            diff += fabs(origMap(row, col) - dequantMap(row, col));
        }
    }
    diff /= (origMap.rows() * origMap.cols());
    printf("Avg-diff = %f\n", diff);
}

void gemm_nn(int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float *C, int ldc)
{
    int i, j, k;
    #pragma omp parallel for

    //gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            register float A_PART = ALPHA*A[i*lda+k];
            for(j = 0; j < N; ++j){
                // A: RowMajor, B: RowMajor, C: RowMajor
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}

void gemm_nt_int8(int M, int N, int K, float ALPHA,
        int8_t *A, int lda,
        int8_t *B, int ldb,
        int32_t *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            register float sum = 0;
            for(k = 0; k < K; ++k){
                sum += ALPHA*A[i*lda+k]*B[j*ldb + k];
            }
            C[i*ldc+j] += sum;
        }
    }
}

void gemm_nt(int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            register float sum = 0;
            for(k = 0; k < K; ++k){
                sum += ALPHA*A[i*lda+k]*B[j*ldb + k];
            }
            C[i*ldc+j] += sum;
        }
    }
}

void gemm_nt_quant(int M, int N, int K,
        int8_t *A, int lda,
        int8_t *B, int ldb,
        int8_t *C, int ldc,
        int32_t *biases, float *QS, int8_t *QZ)
{
    const auto RM = quantization::MapOrder::RowMajor;
    const auto CM = quantization::MapOrder::ColMajor;
    int rows = M, cols = N, depth = K;

    quantization::MatrixWithStorage<int8_t, RM> int8_lhs(A, rows, depth);
    quantization::MatrixWithStorage<int8_t, CM> int8_rhs(B, depth, cols);
    quantization::MatrixWithStorage<int32_t, RM> int32_biases(biases, 1, cols);
    quantization::MatrixWithStorage<int8_t, RM> int8_result(rows, cols);

    const quantization::quantizationParams lhs_qparams = {.scale=QS[0], .zero_point=QZ[0]};
    const quantization::quantizationParams rhs_qparams = {.scale=QS[1], .zero_point=QZ[1]};
    const quantization::quantizationParams biases_qparams = {.scale=QS[0]*QS[1], .zero_point=0};
    const quantization::quantizationParams result_qparams = {.scale=QS[2], .zero_point=QZ[2]};

    int8_lhs.setQParams(lhs_qparams);
    int8_rhs.setQParams(rhs_qparams);
    int32_biases.setQParams(biases_qparams);
    int8_result.setQParams(result_qparams);

    const float real_M = int8_lhs.getQParams().scale * int8_rhs.getQParams().scale / int8_result.getQParams().scale;
    int32_t quantized_M;
    int right_shift;
    quantization::QuantizeMultiplierSmallerThanOne(real_M, &quantized_M, &right_shift);

    quantization::OutputStageQuantizeDownInt32ByFixedPoint quantize_down_stage;
    quantize_down_stage.result_fixedpoint_multiplier = quantized_M;
    quantize_down_stage.result_shift = right_shift;

    auto int8_result_map = int8_result.Map();
    quantization::QuantizeGemm_nt(int8_lhs.ConstMap(),
                                  int8_rhs.ConstMap(),
                                  &int8_result_map,
                                  int8_lhs.getQParams(),
                                  int8_rhs.getQParams(),
                                  int8_result.getQParams(),
                                  quantize_down_stage,
                                  int32_biases.ConstMap());
    int8_result.toMatrix(&C);
}

void gemm_nn_grad_first(int M, int N, int K,
        float *A, int lda, 
        float *B, int ldb,
        int8_t *C, int ldc,
        float *biases, float* QS, int8_t* QZ)
{
    const auto kOrder = quantization::MapOrder::RowMajor;
    int rows = M, cols = N, depth = K;

    quantization::MatrixWithStorage<float, kOrder> float_lhs(A, rows, depth);
    quantization::MatrixWithStorage<float, kOrder> float_rhs(B, depth, cols);
    quantization::MatrixWithStorage<float, kOrder> float_biases(biases, rows, 1);

    const quantization::quantizationParams lhs_qparams = {.scale=QS[1], .zero_point=QZ[1]};
    const quantization::quantizationParams rhs_qparams = {.scale=QS[0], .zero_point=QZ[0]};
    const quantization::quantizationParams biases_qparams = {.scale=QS[0]*QS[1], .zero_point=0};
    const quantization::quantizationParams result_qparams = {.scale=QS[2], .zero_point=QZ[2]};

    quantization::MatrixWithStorage<std::int8_t, kOrder> int8_lhs(rows, depth);
    quantization::MatrixWithStorage<std::int8_t, kOrder> int8_rhs(depth, cols);
    quantization::MatrixWithStorage<std::int32_t, kOrder> int32_biases(rows, 1);
    quantization::MatrixWithStorage<std::int8_t, kOrder> int8_result(rows, cols); 

    int8_lhs.setQParams(lhs_qparams);
    int8_rhs.setQParams(rhs_qparams);
    int32_biases.setQParams(biases_qparams);
    int8_result.setQParams(result_qparams);

    // Quantize
    quantization::quantize(int8_lhs.getQParams(), float_lhs.Storage(), &int8_lhs.Storage());
    quantization::quantize(int8_rhs.getQParams(), float_rhs.Storage(), &int8_rhs.Storage());
    quantization::quantize_bias(int32_biases.getQParams(), float_biases.Storage(), &int32_biases.Storage());
    
    const float real_M = int8_lhs.getQParams().scale * int8_rhs.getQParams().scale / int8_result.getQParams().scale;
    std::int32_t quantized_M;
    int right_shift;
    quantization::QuantizeMultiplierSmallerThanOne(real_M, &quantized_M, &right_shift);

    quantization::OutputStageQuantizeDownInt32ByFixedPoint quantize_down_stage;
    quantize_down_stage.result_fixedpoint_multiplier = quantized_M;
    quantize_down_stage.result_shift = right_shift;

    auto int8_result_map = int8_result.Map();
    quantization::QuantizeGemm(int8_lhs.ConstMap(),
                               int8_rhs.ConstMap(),
                               &int8_result_map,
                               int8_lhs.getQParams(),
                               int8_rhs.getQParams(),
                               int8_result.getQParams(),
                               quantize_down_stage,
                               int32_biases.ConstMap());

    int8_result.toMatrix(&C);
}

void gemm_nn_grad_second(int M, int N, int K,
        float *A, int lda, 
        int8_t *B, int ldb,
        int8_t *C, int ldc,
        float *biases, float* QS, int8_t* QZ)
{
    const auto kOrder = quantization::MapOrder::RowMajor;
    int rows = M, cols = N, depth = K;

    quantization::MatrixWithStorage<float, kOrder> float_lhs(A, rows, depth);
    quantization::MatrixWithStorage<std::int8_t, kOrder> int8_rhs(B, depth, cols);
    quantization::MatrixWithStorage<float, kOrder> float_biases(biases, rows, 1);

    const quantization::quantizationParams lhs_qparams = {.scale=QS[1], .zero_point=QZ[1]};
    const quantization::quantizationParams rhs_qparams = {.scale=QS[0], .zero_point=QZ[0]};
    const quantization::quantizationParams biases_qparams = {.scale=QS[0]*QS[1], .zero_point=0};
    const quantization::quantizationParams result_qparams = {.scale=QS[2], .zero_point=QZ[2]};

    quantization::MatrixWithStorage<std::int8_t, kOrder> int8_lhs(rows, depth);
    quantization::MatrixWithStorage<std::int32_t, kOrder> int32_biases(rows, 1);
    quantization::MatrixWithStorage<std::int8_t, kOrder> int8_result(rows, cols); 

    int8_lhs.setQParams(lhs_qparams);
    int8_rhs.setQParams(rhs_qparams);
    int32_biases.setQParams(biases_qparams);
    int8_result.setQParams(result_qparams);

    // Quantize
    quantization::quantize(int8_lhs.getQParams(), float_lhs.Storage(), &int8_lhs.Storage());
    quantization::quantize_bias(int32_biases.getQParams(), float_biases.Storage(), &int32_biases.Storage());
    
    const float real_M = int8_lhs.getQParams().scale * int8_rhs.getQParams().scale / int8_result.getQParams().scale;
    std::int32_t quantized_M;
    int right_shift;
    quantization::QuantizeMultiplierSmallerThanOne(real_M, &quantized_M, &right_shift);

    quantization::OutputStageQuantizeDownInt32ByFixedPoint quantize_down_stage;
    quantize_down_stage.result_fixedpoint_multiplier = quantized_M;
    quantize_down_stage.result_shift = right_shift;

    auto int8_result_map = int8_result.Map();
    quantization::QuantizeGemm(int8_lhs.ConstMap(),
                               int8_rhs.ConstMap(),
                               &int8_result_map,
                               int8_lhs.getQParams(),
                               int8_rhs.getQParams(),
                               int8_result.getQParams(),
                               quantize_down_stage,
                               int32_biases.ConstMap());

    int8_result.toMatrix(&C);
}

void gemm_nn_quant(int M, int N, int K,
        int8_t *A, int lda,
        int8_t *B, int ldb,
        int8_t *C, int ldc,
        int32_t *biases, float *QS, int8_t *QZ)
{
    const auto kOrder = quantization::MapOrder::RowMajor;
    int rows = M, cols = N, depth = K;

    quantization::MatrixWithStorage<int8_t, kOrder> int8_lhs(A, rows, depth);
    quantization::MatrixWithStorage<int8_t, kOrder> int8_rhs(B, depth, cols);
    quantization::MatrixWithStorage<int32_t, kOrder> int32_biases(biases, rows, 1);
    quantization::MatrixWithStorage<int8_t, kOrder> int8_result(rows, cols);
 
    const quantization::quantizationParams lhs_qparams = {.scale=QS[1], .zero_point=QZ[1]};
    const quantization::quantizationParams rhs_qparams = {.scale=QS[0], .zero_point=QZ[0]};
    const quantization::quantizationParams biases_qparams = {.scale=QS[0]*QS[1], .zero_point=0};
    const quantization::quantizationParams result_qparams = {.scale=QS[2], .zero_point=QZ[2]};

    int8_lhs.setQParams(lhs_qparams);
    int8_rhs.setQParams(rhs_qparams);
    int8_result.setQParams(result_qparams);
    int32_biases.setQParams(biases_qparams);

    const float real_M = int8_lhs.getQParams().scale * int8_rhs.getQParams().scale / int8_result.getQParams().scale;
    std::int32_t quantized_M;
    int right_shift;
    quantization::QuantizeMultiplierSmallerThanOne(real_M, &quantized_M, &right_shift);

    quantization::OutputStageQuantizeDownInt32ByFixedPoint quantize_down_stage;
    quantize_down_stage.result_fixedpoint_multiplier = quantized_M;
    quantize_down_stage.result_shift = right_shift;

    auto int8_result_map = int8_result.Map();
    quantization::QuantizeGemm_nn(int8_lhs.ConstMap(),
                                  int8_rhs.ConstMap(),
                                  &int8_result_map,
                                  int8_lhs.getQParams(),
                                  int8_rhs.getQParams(),
                                  int8_result.getQParams(),
                                  quantize_down_stage,
                                  int32_biases.ConstMap());
    int8_result.toMatrix(&C);
}

void gemm_tn(int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            register float A_PART = ALPHA*A[k*lda+i];
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}

void gemm_tt(int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            register float sum = 0;
            for(k = 0; k < K; ++k){
                sum += ALPHA*A[i+k*lda]*B[k+j*ldb];
            }
            C[i*ldc+j] += sum;
        }
    }
}


void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    //printf("cpu: %d %d %d %d %d %f %d %d %f %d\n",TA, TB, M, N, K, ALPHA, lda, ldb, BETA, ldc);
    int i, j;
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            C[i*ldc + j] *= BETA;
        }
    }
    if(!TA && !TB)
        gemm_nn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if(TA && !TB)
        gemm_tn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if(!TA && TB)
        gemm_nt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else
        gemm_tt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
}

void gemm(int TA, int TB, int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    gemm_cpu(TA, TB, M, N, K, ALPHA, A, lda, B, ldb, BETA, C, ldc);
}


//void printFloatMatrix(float *C, int m, int n)
//{
//    for (int row = 0; row < m; row++) {
//        for (int col = 0; col < n; col++) {
//            printf("%.2f", C[row * n + col]);
//        }
//    }
//}


#ifdef GPU
void get_nn_totalsum_gpu(int M, int N, int K,
                       int8_t *A, int lda,
                       int8_t *B, int ldb,
                       int8_t *C, int ldc,
                       int32_t *C32,
                       int32_t *biases, float *QS, int8_t *QZ)
{
    const auto RM = quantization::MapOrder::RowMajor;
    int rows = M, cols = N, depth = K;

    quantization::MatrixWithStorage<int8_t, RM> int8_lhs(A, rows, depth);
    quantization::MatrixWithStorage<int8_t, RM> int8_rhs(B, depth, cols);
    quantization::MatrixWithStorage<int32_t, RM> int32_biases(biases, rows, 1);
    quantization::MatrixWithStorage<int8_t, RM> int8_result(rows, cols);
    quantization::MatrixWithStorage<int32_t, RM> int32_result(C32, rows, cols);

    const quantization::quantizationParams lhs_qparams = {.scale=QS[1], .zero_point=QZ[1]};
    const quantization::quantizationParams rhs_qparams = {.scale=QS[0], .zero_point=QZ[0]};
    const quantization::quantizationParams biases_qparams = {.scale=QS[0]*QS[1], .zero_point=0};
    const quantization::quantizationParams result_qparams = {.scale=QS[2], .zero_point=QZ[2]};

    int8_lhs.setQParams(lhs_qparams);
    int8_rhs.setQParams(rhs_qparams);
    int32_biases.setQParams(biases_qparams);
    int8_result.setQParams(result_qparams);

    const float real_M = int8_lhs.getQParams().scale * int8_rhs.getQParams().scale / int8_result.getQParams().scale;
    int32_t quantized_M;
    int right_shift;
    quantization::QuantizeMultiplierSmallerThanOne(real_M, &quantized_M, &right_shift);

    quantization::OutputStageQuantizeDownInt32ByFixedPoint quantize_down_stage;
    quantize_down_stage.result_fixedpoint_multiplier = quantized_M;
    quantize_down_stage.result_shift = right_shift;

    auto int8_result_map = int8_result.Map();
    quantization::QuantizeGemm_nn_gpu(int8_lhs.ConstMap(),
                                   int8_rhs.ConstMap(),
                                   &int8_result_map,
                                   int32_result.ConstMap(),
                                   int8_lhs.getQParams(),
                                   int8_rhs.getQParams(),
                                   int8_result.getQParams(),
                                   quantize_down_stage,
                                   int32_biases.ConstMap());
    int8_result.toMatrix(&C);
}

void get_nt_totalsum_gpu(int M, int N, int K,
                       int8_t *A, int lda,
                       int8_t *B, int ldb,
                       int8_t *C, int ldc,
                       int32_t *C32,
                       int32_t *biases, float *QS, int8_t *QZ)
{
    const auto RM = quantization::MapOrder::RowMajor;
    const auto CM = quantization::MapOrder::ColMajor;
    int rows = M, cols = N, depth = K;

    quantization::MatrixWithStorage<int8_t, RM> int8_lhs(A, rows, depth);
    quantization::MatrixWithStorage<int8_t, CM> int8_rhs(B, depth, cols);
    quantization::MatrixWithStorage<int32_t, RM> int32_biases(biases, 1, cols);
    quantization::MatrixWithStorage<int8_t, RM> int8_result(rows, cols);
    quantization::MatrixWithStorage<int32_t, RM> int32_result(C32, rows, cols);

    const quantization::quantizationParams lhs_qparams = {.scale=QS[0], .zero_point=QZ[0]};
    const quantization::quantizationParams rhs_qparams = {.scale=QS[1], .zero_point=QZ[1]};
    const quantization::quantizationParams biases_qparams = {.scale=QS[0]*QS[1], .zero_point=0};
    const quantization::quantizationParams result_qparams = {.scale=QS[2], .zero_point=QZ[2]};

    int8_lhs.setQParams(lhs_qparams);
    int8_rhs.setQParams(rhs_qparams);
    int32_biases.setQParams(biases_qparams);
    int8_result.setQParams(result_qparams);

    const float real_M = int8_lhs.getQParams().scale * int8_rhs.getQParams().scale / int8_result.getQParams().scale;
    int32_t quantized_M;
    int right_shift;
    quantization::QuantizeMultiplierSmallerThanOne(real_M, &quantized_M, &right_shift);

    quantization::OutputStageQuantizeDownInt32ByFixedPoint quantize_down_stage;
    quantize_down_stage.result_fixedpoint_multiplier = quantized_M;
    quantize_down_stage.result_shift = right_shift;

    auto int8_result_map = int8_result.Map();
    quantization::QuantizeGemm_nt_gpu(int8_lhs.ConstMap(),
                                   int8_rhs.ConstMap(),
                                   &int8_result_map,
                                   int32_result.ConstMap(),
                                   int8_lhs.getQParams(),
                                   int8_rhs.getQParams(),
                                   int8_result.getQParams(),
                                   quantize_down_stage,
                                   int32_biases.ConstMap());
    int8_result.toMatrix(&C);
}

void gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA,
        float *A_gpu, int lda,
        float *B_gpu, int ldb,
        float BETA,
        float *C_gpu, int ldc)
{
    cublasHandle_t handle = blas_handle();
    cudaError_t status = static_cast<cudaError_t>(cublasSgemm(handle, (TB ? CUBLAS_OP_T : CUBLAS_OP_N),
            (TA ? CUBLAS_OP_T : CUBLAS_OP_N), N, M, K, &ALPHA, B_gpu, ldb, A_gpu, lda, &BETA, C_gpu, ldc));

    check_error(status);
}

void gemm_gpu_cublasGemmEx(int TA, int TB, int M, int N, int K, float ALPHA,
                           float *A_gpu, int lda,
                           float *B_gpu, int ldb,
                           float BETA,
                           float *C_gpu, int ldc)
{
    cublasHandle_t handle = blas_handle();
    cudaError_t status = static_cast<cudaError_t>(cublasGemmEx(
            handle,
            (TB ? CUBLAS_OP_T : CUBLAS_OP_N),
            (TA ? CUBLAS_OP_T : CUBLAS_OP_N),
            N, M, K,
            &ALPHA,
            B_gpu, CUDA_R_32F, ldb,
            A_gpu, CUDA_R_32F, lda,
            &BETA,
            C_gpu, CUDA_R_32F, ldc,
            CUDA_R_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

void gemm_gpu_cublasGemmEx_int(int TA, int TB,
        int M, int N, int K,
        float ALPHA,
        int *A_gpu, int lda,
        int *B_gpu, int ldb,
        float BETA,
        int *C_gpu, int ldc)
{
    cublasHandle_t handle = blas_handle();
    cudaError_t status = static_cast<cudaError_t>(cublasGemmEx(
            handle,
            (TB ? CUBLAS_OP_T : CUBLAS_OP_N),
            (TA ? CUBLAS_OP_T : CUBLAS_OP_N),
            N, M, K,
            &ALPHA,
            B_gpu, CUDA_R_8I, ldb,
            A_gpu, CUDA_R_8I, lda,
            &BETA,
            C_gpu, CUDA_R_32I, ldc,
            CUDA_R_32I,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

void gemm_gpu_cublasGemmEx_int32(int TA, int TB,
                                 int M, int N, int K,
                                 int ALPHA,
                                 int8_t *A_gpu, int lda,
                                 int8_t *B_gpu, int ldb,
                                 int BETA,
                                 int32_t *C_gpu, int ldc)
{
    cublasHandle_t handle = blas_handle();
    cudaError_t status = static_cast<cudaError_t>(cublasGemmEx(
            handle,
            (TB ? CUBLAS_OP_T : CUBLAS_OP_N),
            (TA ? CUBLAS_OP_T : CUBLAS_OP_N),
            N, M, K,
            &ALPHA,
            B_gpu, CUDA_R_8I, ldb,
            A_gpu, CUDA_R_8I, lda,
            &BETA,
            C_gpu, CUDA_R_32I, ldc,
            CUDA_R_32I,
            CUBLAS_GEMM_DEFAULT));
}

void gemm_nn_cublasGemmEx_int32(int TA, int TB,
                                 int M, int N, int K,
                                 int ALPHA,
                                 int8_t *A_gpu, int lda,
                                 int8_t *B_gpu, int ldb,
                                 int BETA,
                                 int32_t *C_gpu, int ldc)
{
    cublasHandle_t handle = blas_handle();
    cudaError_t status = static_cast<cudaError_t>(cublasGemmEx(
            handle,
            (TA ? CUBLAS_OP_T : CUBLAS_OP_N),
            (TB ? CUBLAS_OP_T : CUBLAS_OP_N),
            N, M, K,
            &ALPHA,
            A_gpu, CUDA_R_8I, lda,
            B_gpu, CUDA_R_8I, ldb,
            &BETA,
            C_gpu, CUDA_R_32I, ldc,
            CUDA_R_32I,
            CUBLAS_GEMM_DEFAULT));
}

void gemm_quant_gpu(int TA, int TB, int M, int N, int K, float ALPHA,
        int8_t *A_gpu, int lda,
        int8_t *B_gpu, int ldb,
        float BETA,
        int32_t *C_gpu, int ldc)
{
    //void* voidAlpha = (void*)(&ALPHA);
    //void* voidBeta = (void*)(&BETA);

    cublasHandle_t handle = blas_handle();
//    cudaError_t status = static_cast<cudaError_t>(cublasGemmEx(
//            handle,
//            (TB ? CUBLAS_OP_T : CUBLAS_OP_N),
//            (TA ? CUBLAS_OP_T : CUBLAS_OP_N),
//            N, M, K,
//            &ALPHA,
//            B_gpu, CUDA_R_8I, ldb,
//            A_gpu, CUDA_R_8I, lda,
//            &BETA,
//            C_gpu, CUDA_R_32I, ldc,
//            CUDA_R_32I,
//            CUBLAS_GEMM_DFALT_TENSOR_OP));
    cudaError_t status = static_cast<cudaError_t>(cublasSgemmEx(
            handle,
            (TB ? CUBLAS_OP_T : CUBLAS_OP_N),
            (TA ? CUBLAS_OP_T : CUBLAS_OP_N),
            N, M, K,
            &ALPHA,
            B_gpu, CUDA_R_32F, ldb,
            A_gpu, CUDA_R_32F, lda,
            &BETA,
            C_gpu, CUDA_R_32F, ldc));

    check_error(status);
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "cuda.h"

void time_gpu_random_matrix(int TA, int TB, int m, int k, int n)
{
    float *a;
    if(!TA) a = random_matrix(m,k);
    else a = random_matrix(k,m);
    int lda = (!TA)?k:m;
    float *b;
    if(!TB) b = random_matrix(k,n);
    else b = random_matrix(n,k);
    int ldb = (!TB)?n:k;

    float *c = random_matrix(m,n);
    int i;
    clock_t start = clock(), end;
    for(i = 0; i<32; ++i){
        gemm_gpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c,n);
    }
    end = clock();
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %lf s\n",m,k,k,n, TA, TB, (float)(end-start)/CLOCKS_PER_SEC);
    free(a);
    free(b);
    free(c);
}

void time_gpu(int TA, int TB, int m, int k, int n)
{
    int iter = 10;
    float *a = random_matrix(m,k);
    float *b = random_matrix(k,n);

    int lda = (!TA)?k:m;
    int ldb = (!TB)?n:k;

    float *c = random_matrix(m,n);

    float *a_cl = cuda_make_array(a, m*k);
    float *b_cl = cuda_make_array(b, k*n);
    float *c_cl = cuda_make_array(c, m*n);

    int i;
    clock_t start = clock(), end;
    for(i = 0; i<iter; ++i){
        gemm_gpu(TA,TB,m,n,k,1,a_cl,lda,b_cl,ldb,1,c_cl,n);
        cudaThreadSynchronize();
    }
    double flop = ((double)m)*n*(2.*k + 2.)*iter;
    double gflop = flop/pow(10., 9);
    end = clock();
    double seconds = sec(end-start);
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %lf s, %lf GFLOPS\n",m,k,k,n, TA, TB, seconds, gflop/seconds);
    cuda_free(a_cl);
    cuda_free(b_cl);
    cuda_free(c_cl);
    free(a);
    free(b);
    free(c);
}


void test_gpu_accuracy(int TA, int TB, int m, int k, int n)
{
    srand(0);
    float *a;
    if(!TA) a = random_matrix(m,k);
    else a = random_matrix(k,m);
    int lda = (!TA)?k:m;
    float *b;
    if(!TB) b = random_matrix(k,n);
    else b = random_matrix(n,k);
    int ldb = (!TB)?n:k;

    float *c = random_matrix(m,n);
    float *c_gpu = random_matrix(m,n);
    memset(c, 0, m*n*sizeof(float));
    memset(c_gpu, 0, m*n*sizeof(float));
    int i;
    //pm(m,k,b);
    gemm_gpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c_gpu,n);
    //printf("GPU\n");
    //pm(m, n, c_gpu);

    gemm_cpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c,n);
    //printf("\n\nCPU\n");
    //pm(m, n, c);
    double sse = 0;
    for(i = 0; i < m*n; ++i) {
        //printf("%f %f\n", c[i], c_gpu[i]);
        sse += pow(c[i]-c_gpu[i], 2);
    }
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %g SSE\n",m,k,k,n, TA, TB, sse/(m*n));
    free(a);
    free(b);
    free(c);
    free(c_gpu);
}

void printprint(float *C, int m, int n) {
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < n; col++) {
            printf("%.f ", C[row * n + col]);
        }
        printf("\n");
    }
    printf("\n\n");
}

void printprint_int8(int8_t *C, int m, int n) {
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < n; col++) {
            printf("%d ", C[row * n + col]);
        }
        printf("\n");
    }
    printf("\n\n");
}

void printprint_int(int *C, int m, int n) {
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < n; col++) {
            printf("%d ", C[row * n + col]);
        }
        printf("\n");
    }
    printf("\n\n");
}

void printprint_int32(int32_t *C, int m, int n) {
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < n; col++) {
            printf("%d ", C[row * n + col]);
        }
        printf("\n");
    }
    printf("\n\n");
}

// check
void test_gpu_accuracy_float()
{
    int TA = 0;
    int TB = 0;

    // host matrix ======================================
    float *A;  // (4 x 8)
    float *B;  // (8 x 4)
    float *C;  // (4 x 4)

    int m = 4;
    int k = 8;
    int n = 4;

    // A
    if(!TA) A = random_matrix(m,k);
    else A = random_matrix(k,m);
    int lda = (!TA) ? k : m;

    // B
    if(!TB) B = random_matrix(k,n);
    else B = random_matrix(n,k);
    int ldb = (!TB)?n:k;

    // C
    C = random_matrix_zero(m, n);

    /*
    gemm_cpu(TA, TB, m, n, k, 1, A, lda, B, ldb, 1, C, n);
    printf("cpu gemm\n");
    printprint(A, m, k);
    printprint(B, k, n);
    printprint(C, m, n);
    free(A);
    free(B);
    free(C);
    */
    // =================================================


    // device matrix -----------------------------------
    float *A_GPU;
    float *B_GPU;
    float *C_GPU;

    A_GPU = cuda_make_array(A, m*k);
    B_GPU = cuda_make_array(B, k*n);
    C_GPU = cuda_make_array(C, m*n);


    // A
    if(!TA) A = random_matrix(m,k);
    else A = random_matrix(k,m);
    lda = (!TA) ? k : m;

    // B
    if(!TB) B = random_matrix(k,n);
    else B = random_matrix(n,k);
    ldb = (!TB)?n:k;

    // C
    C = random_matrix_zero(m, n);

    // allocate to device
    cuda_push_array(A_GPU, A, m*k);
    cuda_push_array(B_GPU, B, k*n);
    cuda_push_array(C_GPU, C, m*n);

    //gemm_gpu(TA, TB, m, n, k, 1, A_GPU, lda, B_GPU, ldb, 1, C_GPU, n);
    gemm_gpu_cublasGemmEx(TA, TB, m, n, k, 1, A_GPU, lda, B_GPU, ldb, 1, C_GPU, n);

    memset(A, 0, m*k*sizeof(float));
    memset(B, 0, k*n*sizeof(float));
    memset(C, 0, m*n*sizeof(float));

    printf("reset check\n");
    printprint(C, m, n);

    cuda_pull_array(A_GPU, A, m*k);
    cuda_pull_array(B_GPU, B, k*n);
    cuda_pull_array(C_GPU, C, m*n);

    printf("gpu gemm\n");
    printprint(C, m, n);

    // free
    cuda_free(A_GPU);
    cuda_free(B_GPU);
    cuda_free(C_GPU);

    free(A);
    free(B);
    free(C);
}

void test_gpu_accuracy_int()
{
    int TA = 0;
    int TB = 0;

    // host matrix ======================================
    int *A;  // (4 x 8)
    int *B;  // (8 x 4)
    int *C;  // (4 x 4)

    int m = 4;
    int k = 8;
    int n = 4;


    // device matrix -----------------------------------
    int *A_GPU;
    int *B_GPU;
    int *C_GPU;

    A_GPU = cuda_make_array_int(A, m*k);
    B_GPU = cuda_make_array_int(B, k*n);
    C_GPU = cuda_make_array_int(C, m*n);


    // A
    if(!TA) A = random_matrix_int(m,k);
    else A = random_matrix_int(k,m);
    int lda = (!TA) ? k : m;

    // B
    if(!TB) B = random_matrix_int(k,n);
    else B = random_matrix_int(n,k);
    int ldb = (!TB)?n:k;

    // C
    C = random_matrix_zero_int(m, n);

    // allocate to device
    cuda_push_array_int(A_GPU, A, m*k);
    cuda_push_array_int(B_GPU, B, k*n);
    cuda_push_array_int(C_GPU, C, m*n);

    gemm_gpu_cublasGemmEx_int(TA, TB, m, n, k, 1, A_GPU, lda, B_GPU, ldb, 1, C_GPU, n);


    memset(A, 0, m*k*sizeof(int));
    memset(B, 0, k*n*sizeof(int));
    memset(C, 0, m*n*sizeof(int));

    printf("reset check\n");
    printprint_int(C, m, n);

    cuda_pull_array_int(A_GPU, A, m*k);
    cuda_pull_array_int(B_GPU, B, k*n);
    cuda_pull_array_int(C_GPU, C, m*n);

    printf("gpu gemm\n");
    printprint_int(C, m, n);

    // free
    cuda_free_int(A_GPU);
    cuda_free_int(B_GPU);
    cuda_free_int(C_GPU);

    free(A);
    free(B);
    free(C);
}

// check
void test_gpu_accuracy_int32()
{
    int TA = 0;
    int TB = 0;

    // host matrix ======================================
    int8_t *A = NULL;  // (4 x 8)
    int8_t *B = NULL;  // (8 x 4)
    int32_t *C = NULL;  // (4 x 4)

    //int m = 96;
    //int k = 75;
    //int n = 1024;
    int m = 4;
    int k = 15;
    int n = 4;


    // device matrix -----------------------------------
    int8_t *A_GPU;
    int8_t *B_GPU;
    int32_t *C_GPU;

    A_GPU = cuda_make_array_int8(A, m*k);
    B_GPU = cuda_make_array_int8(B, k*n);
    C_GPU = cuda_make_array_int32(C, m*n);


    // A
    if(!TA) A = random_matrix_int8(m,k);
    else A = random_matrix_int8(k,m);
    int lda = (!TA) ? k : m;

    // B
    if(!TB) B = random_matrix_int8(k,n);
    else B = random_matrix_int8(n,k);
    int ldb = (!TB)?n:k;

    // C
    C = random_matrix_zero_int32(m, n);

    // allocate to device
    cuda_push_array_int8(A_GPU, A, m*k);
    cuda_push_array_int8(B_GPU, B, k*n);
    cuda_push_array_int32(C_GPU, C, m*n);

    printprint_int8(A, m, k);
    printprint_int8(B, k, n);

    gemm_gpu_cublasGemmEx_int32(TA, TB, m, n, k, 1, A_GPU, lda, B_GPU, ldb, 1, C_GPU, n);

    memset(A, 0, m*k*sizeof(int8_t));
    memset(B, 0, k*n*sizeof(int8_t));
    memset(C, 0, m*n*sizeof(int32_t));

    printf("reset check\n");
    printprint_int32(C, m, n);

    cuda_pull_array_int8(A_GPU, A, m*k);
    cuda_pull_array_int8(B_GPU, B, k*n);
    cuda_pull_array_int32(C_GPU, C, m*n);

    printf("gpu gemm\n");
    printprint_int32(C, m, n);

    // free
    cuda_free_int8(A_GPU);
    cuda_free_int8(B_GPU);
    cuda_free_int32(C_GPU);

    free(A);
    free(B);
    free(C);
    exit(0);
}


int test_gpu_blas()
{
    /*
       test_gpu_accuracy(0,0,10,576,75);
       test_gpu_accuracy(0,0,17,10,10);
       test_gpu_accuracy(1,0,17,10,10);
       test_gpu_accuracy(0,1,17,10,10);
       test_gpu_accuracy(1,1,17,10,10);
       test_gpu_accuracy(0,0,1000,10,100);
       test_gpu_accuracy(1,0,1000,10,100);
       test_gpu_accuracy(0,1,1000,10,100);
       test_gpu_accuracy(1,1,1000,10,100);
       test_gpu_accuracy(0,0,10,10,10);
       time_gpu(0,0,64,2916,363);
       time_gpu(0,0,64,2916,363);
       time_gpu(0,0,64,2916,363);
       time_gpu(0,0,192,729,1600);
       time_gpu(0,0,384,196,1728);
       time_gpu(0,0,256,196,3456);
       time_gpu(0,0,256,196,2304);
       time_gpu(0,0,128,4096,12544);
       time_gpu(0,0,128,4096,4096);
     */
    time_gpu(0,0,64,75,12544);
    time_gpu(0,0,64,75,12544);
    time_gpu(0,0,64,75,12544);
    time_gpu(0,0,64,576,12544);
    time_gpu(0,0,256,2304,784);
    time_gpu(1,1,2304,256,784);
    time_gpu(0,0,512,4608,196);
    time_gpu(1,1,4608,512,196);

    return 0;
}
#endif

