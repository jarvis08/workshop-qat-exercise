void fake_quantize_int8_cpu(float* input, const int n, const float QS, const int8_t QZ)
{
    int i;
    #pragma omp parallel for
    for (i = 0; i < n; i++) {
        const float quantized = round(fmax(-128.f, fmin(127.f, QZ + input[i] / QS)));
        input[i] = QS * (quantized - QZ);
    }
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
    // mode = 0 :: nn (lhs, rhs: RowMajor)
    // mode = 1 :: nt (lhs: Rowmajor, rhs: ColMajor)
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

    // 아래 Floating Point로 표현되는 M 계산을 0이 아닌, 올바른 계산으로 수정하세요.
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
        gemm_nt(M, N, K, 1, lhs, K, rhs, K, C32, N); 

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

void totalsum_int8_cpu(const int M, const int N, const int K,
                       const int8_t *lhs,
                       const int8_t *rhs,
                       int8_t *C,
                       const int32_t *C32,
                       const int32_t *biases,
                       const float *QS, const int8_t *QZ,
                       const int mode)
{
    // mode = 0 :: nn (lhs, rhs: RowMajor)
    // mode = 1 :: nt (lhs: Rowmajor, rhs: ColMajor)
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
        int _col = 0;
        #pragma omp parallel for
        for (col = 0; col < N; col++) {
            _col = col * r_stride;
            for (depth = 0; depth < K; depth++) {
                a2col[col] += rhs[_col + depth];
            }
        }

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
                subSum += biases[row];

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

void ema_cpu(const float *mat, float *pmin, float *pmax, const int n, const float smooth_param, const int is_relu)
{
    int i;
    float _max = mat[0];
    if(is_relu) {
        for(i = 0; i < n; ++i)
        {
           if(mat[i] > _max) _max = mat[i]; 
        }
        *pmin = 0;
        *pmax = _max * (1 - smooth_param) + *pmax * smooth_param;
    } else {
        float _min = mat[0];
        for(i = 0; i < n; ++i)
        {
           if(mat[i] > _max) _max = mat[i]; 
           else if(mat[i] < _min) _min = mat[i]; 
        }
        *pmin = _min * (1 - smooth_param) + *pmin * smooth_param;
        *pmax = _max * (1 - smooth_param) + *pmax * smooth_param;
    }
}
