#ifndef QUANTIZATION_QUANTIZE_H
#define QUANTIZATION_QUANTIZE_H

#include <iostream>
#include <limits>
#include <cassert>
#include "map.h"
#include "fixedpoint.h"
#include "MatrixMap.h"

namespace quantization {

    template <typename quantization::MapOrder tOrder>
    void FindMinMax(const quantization::MatrixMap<float, tOrder>& m, float* min, float* max) {
        *min = *max = m(0, 0);

        for (int i = 0; i < m.rows(); i++) {
            for (int j = 0; j < m.cols(); j++) {
                const float val = m(i, j);
                *min = std::min(*min, val);
                *max = std::max(*max, val);
            }
        }
        //std::cout << "input's min=" << *min << " | max=" << *max << std::endl;
    }

    quantizationParams ChooseQuantizationParams(float min, float max) {
        min = std::min(min, 0.f);
        max = std::max(max, 0.f);

        const float qmin = -128;
        const float qmax = 127;

        const double scale = (max - min) / (qmax - qmin);
        const double initial_zero_point = qmin - (min / scale);

        std::int8_t nudged_zero_point = 0;
        if (initial_zero_point < qmin) {
            nudged_zero_point = qmin;
        } else if (initial_zero_point > qmax) {
            nudged_zero_point = qmax;
        } else {
            nudged_zero_point = static_cast<std::int8_t>(std::round(initial_zero_point));
        }

        quantizationParams result;
        result.scale = scale;
        result.zero_point = nudged_zero_point;

        return result;
    }

    void quantize(const quantizationParams& qparams,
                  const std::vector<float>& src,
                  std::vector<std::int8_t>* dst) {
        assert(src.size() == dst->size());

        #pragma omp parallel for
        for (std::size_t i = 0; i < src.size(); i++) {
            const float real_val = src[i];
            const float transformed_val = qparams.zero_point + real_val / qparams.scale;
            const float clamped_Val = std::max(-128.f, std::min(127.f, transformed_val));
            (*dst)[i] = static_cast<std::int8_t>(std::round(clamped_Val));
        }
    }

    void quantize_bias(const quantizationParams& qparams,
                       const std::vector<float>& src,
                       std::vector<std::int32_t>* dst) {
        assert(src.size() == dst->size());

        #pragma omp parallel for
        for (std::size_t i = 0; i < src.size(); i++) {
            const float real_val = src[i];
            const float transformed_val = qparams.zero_point + real_val / qparams.scale;
            const float clamped_Val = std::max(static_cast<float>(std::numeric_limits<int32_t>::min()),
                                      std::min(static_cast<float>(std::numeric_limits<int32_t>::max()), transformed_val));
            (*dst)[i] = static_cast<std::int32_t>(std::round(clamped_Val));
        }
    }

    void dequantize(const quantizationParams& qparams,
                    const std::vector<std::int8_t>& src,
                    std::vector<float>* dst) {
        assert(src.size() == dst->size());

        for (std::size_t i = 0; i < src.size(); i++) {
            const std::int8_t quantized_val = src[i];
            (*dst)[i] = qparams.scale * (quantized_val - qparams.zero_point);
        }
    }

    void QuantizeMultiplierSmallerThanOne(float real_multiplier,
                                          std::int32_t* quantized_multiplier,
                                          int* right_shift) {
        assert(real_multiplier > 0.f);
        assert(real_multiplier < 1.f);

        int s = 0;

        while(real_multiplier < 0.5f) {
            real_multiplier *= 2.0f;
            s++;
        }

        std::int64_t q = static_cast<std::int64_t>(std::round(real_multiplier * (1ll << 31)));
        assert(q <= (1ll << 31));

        if (q == (1ll << 31)) {
            q /= 2;
            s--;
        }
        assert(s >= 0);
        assert(q <= std::numeric_limits<std::int32_t>::max());
        *quantized_multiplier = static_cast<std::int32_t>(q);
        *right_shift = s;
    }

    std::int32_t SaturatingRoundingDoublingHighMul(std::int32_t subSum, std::int32_t multiplier) {
        bool overflow = subSum == multiplier && subSum == std::numeric_limits<std::int32_t>::min();
        std::int64_t subSum_64(subSum);
        std::int64_t multiplier_64(multiplier);
        std::int64_t subSumMultiplier = subSum_64 * multiplier_64;
        std::int32_t nudge = subSumMultiplier >= 0 ? (1 << 30) : (1 - (1 << 30));
        std::int32_t subSumMultiplier_high = static_cast<std::int32_t>((subSumMultiplier + nudge) / (1ll << 31));

        return overflow ? std::numeric_limits<std::int32_t>::max() : subSumMultiplier_high;
    }

    template <typename IntegerType, typename ExponentType>
    IntegerType RoundingDivideByPot(IntegerType x, ExponentType exponent) {
        assert(exponent >= 0);
        assert(exponent <= 31);

        const IntegerType mask = (1ll << exponent) - 1;
        const IntegerType zero = 0;
        const IntegerType one = 1;
        const IntegerType remainder = BitAnd(x, mask);
        const IntegerType threshold = ShiftRight(mask, 1) + BitAnd(MaskIfLessThan(x, zero), one);

        return ShiftRight(x, exponent) + BitAnd(MaskIfGreaterThan(remainder, threshold), one);
    }

    template <quantization::MapOrder tlhsOrder,
              quantization::MapOrder trhsOrder,
              quantization::MapOrder tResultOrder>
    void QuantizeGemm(const quantization::MatrixMap<const std::int8_t, tlhsOrder>& lhs,
                      const quantization::MatrixMap<const std::int8_t, trhsOrder>& rhs,
                      quantization::MatrixMap<std::int8_t, tResultOrder>* result,
                      const quantizationParams& qlparams,
                      const quantizationParams& qrparams,
                      const quantizationParams& qresultparams,
                      OutputStageQuantizeDownInt32ByFixedPoint quantize_down,
                      const quantization::MatrixMap<const std::int32_t, tResultOrder>& biases) {
        assert(lhs.cols() == rhs.rows());
        assert(lhs.rows() == result->rows());
        assert(rhs.cols() == result->cols());

        int32_t NZ1Z2 = lhs.cols() * qlparams.zero_point * qrparams.zero_point;
        int32_t* a1row = new int32_t[lhs.rows()]();
        int32_t* a2col = new int32_t[rhs.cols()]();
        int32_t sumQ1Q2 = 0;

        int32_t subSum = 0;
        int32_t totalSum = 0;
        int32_t total = 0;

        #pragma omp parallel for
        for (int row = 0; row < lhs.rows(); row++) {
            for (int depth = 0; depth < lhs.cols(); depth++) {
                a1row[row] += lhs(row, depth);
            }
        }

        #pragma omp parallel for
        for (int col = 0; col < rhs.cols(); col++) {
            for (int depth = 0; depth < lhs.cols(); depth++) {
                a2col[col] += rhs(depth, col);
            }
        }

        #pragma omp parallel for
        for (int row = 0; row < lhs.rows(); row++) {
            for (int col = 0; col <rhs.cols(); col++) {
                (*result)(row, col) = 0;
                sumQ1Q2 = 0;

                for (int depth = 0; depth < lhs.cols(); depth++) {
                    sumQ1Q2 += lhs(row, depth) * rhs(depth, col);
                }
                subSum = (NZ1Z2 - qlparams.zero_point * a2col[col] - qrparams.zero_point * a1row[row] + sumQ1Q2);
                subSum = SaturatingRoundingDoublingHighMul(subSum, quantize_down.result_fixedpoint_multiplier);
                subSum += biases(row, 0);
                total = RoundingDivideByPot<std::int32_t, std::int32_t>(subSum, quantize_down.result_shift);
                totalSum = qresultparams.zero_point + total;
                totalSum = totalSum > 127 ? 127 : totalSum < -128 ? -128 : totalSum;
                (*result)(row, col) = totalSum;
            }
        }
        delete[] a1row;
        delete[] a2col;
    }

    template <quantization::MapOrder tlhsOrder,
              quantization::MapOrder trhsOrder,
              quantization::MapOrder tResultOrder>
    void QuantizeGemm_nn(const quantization::MatrixMap<const std::int8_t, tlhsOrder>& lhs,
                         const quantization::MatrixMap<const std::int8_t, trhsOrder>& rhs,
                         quantization::MatrixMap<std::int8_t, tResultOrder>* result,
                         const quantizationParams& qlparams,
                         const quantizationParams& qrparams,
                         const quantizationParams& qresultparams,
                         OutputStageQuantizeDownInt32ByFixedPoint quantize_down,
                         const quantization::MatrixMap<const std::int32_t, tResultOrder>& biases) {
        assert(lhs.cols() == rhs.rows());
        assert(lhs.rows() == result->rows());
        assert(rhs.cols() == result->cols());

        int32_t NZ1Z2 = lhs.cols() * qlparams.zero_point * qrparams.zero_point;
        int32_t* a1row = new int32_t[lhs.rows()]();
        int32_t* a2col = new int32_t[rhs.cols()]();
        int32_t sumQ1Q2 = 0;

        int32_t subSum = 0;
        int32_t totalSum = 0;
        int32_t total = 0;

        #pragma omp parallel for
        for (int row = 0; row < lhs.rows(); row++) {
            for (int depth = 0; depth < lhs.cols(); depth++) {
                a1row[row] += lhs(row, depth);
            }
        }

        #pragma omp parallel for
        for (int col = 0; col < rhs.cols(); col++) {
            for (int depth = 0; depth < lhs.cols(); depth++) {
                a2col[col] += rhs(depth, col);
            }
        }

        #pragma omp parallel for
        for (int row = 0; row < lhs.rows(); row++) {
            for (int col = 0; col <rhs.cols(); col++) {
                (*result)(row, col) = 0;
                sumQ1Q2 = 0;

                for (int depth = 0; depth < lhs.cols(); depth++) {
                    sumQ1Q2 += lhs(row, depth) * rhs(depth, col);
                }
                subSum = (NZ1Z2 - qlparams.zero_point * a2col[col] - qrparams.zero_point * a1row[row] + sumQ1Q2);
                subSum = SaturatingRoundingDoublingHighMul(subSum, quantize_down.result_fixedpoint_multiplier);
                subSum += biases(row, 0);
                total = RoundingDivideByPot<std::int32_t, std::int32_t>(subSum, quantize_down.result_shift);
                totalSum = qresultparams.zero_point + total;
                totalSum = totalSum > 127 ? 127 : totalSum < -128 ? -128 : totalSum;
                (*result)(row, col) = totalSum;
            }
        }
        delete[] a1row;
        delete[] a2col;
    }

    template <quantization::MapOrder tlhsOrder,
              quantization::MapOrder trhsOrder,
              quantization::MapOrder tResultOrder>
    void QuantizeGemm_nt(const quantization::MatrixMap<const std::int8_t, tlhsOrder>& lhs,
                         const quantization::MatrixMap<const std::int8_t, trhsOrder>& rhs,
                         quantization::MatrixMap<std::int8_t, tResultOrder>* result,
                         const quantizationParams& qlparams,
                         const quantizationParams& qrparams,
                         const quantizationParams& qresultparams,
                         OutputStageQuantizeDownInt32ByFixedPoint quantize_down,
                         const quantization::MatrixMap<const std::int32_t, tResultOrder>& biases) {
        assert(lhs.cols() == rhs.rows());
        assert(lhs.rows() == result->rows());
        assert(rhs.cols() == result->cols());

        int32_t NZ1Z2 = lhs.cols() * qlparams.zero_point * qrparams.zero_point;
        int32_t* a1row = new int32_t[lhs.rows()]();
        int32_t* a2col = new int32_t[rhs.cols()]();
        int32_t sumQ1Q2 = 0;

        int32_t subSum = 0;
        int32_t totalSum = 0;
        int32_t total = 0;

        #pragma omp parallel for
        for (int row = 0; row < lhs.rows(); row++) {
            for (int depth = 0; depth < lhs.cols(); depth++) {
                a1row[row] += lhs(row, depth);
            }
        }

        #pragma omp parallel for
        for (int col = 0; col < rhs.cols(); col++) {
            for (int depth = 0; depth < lhs.cols(); depth++) {
                a2col[col] += rhs(depth, col);
            }
        }

        #pragma omp parallel for
        for (int row = 0; row < lhs.rows(); row++) {
            for (int col = 0; col <rhs.cols(); col++) {
                (*result)(row, col) = 0;
                sumQ1Q2 = 0;

                for (int depth = 0; depth < lhs.cols(); depth++) {
                    sumQ1Q2 += lhs(row, depth) * rhs(depth, col);
                }
                subSum = (NZ1Z2 - qlparams.zero_point * a2col[col] - qrparams.zero_point * a1row[row] + sumQ1Q2);
                subSum = SaturatingRoundingDoublingHighMul(subSum, quantize_down.result_fixedpoint_multiplier);
                subSum += biases(0, col);
                total = RoundingDivideByPot<std::int32_t, std::int32_t>(subSum, quantize_down.result_shift);
                totalSum = qresultparams.zero_point + total;
                totalSum = totalSum > 127 ? 127 : totalSum < -128 ? -128 : totalSum;
                (*result)(row, col) = totalSum;
            }
        }
        delete[] a1row;
        delete[] a2col;
    }

    template <quantization::MapOrder tlhsOrder,
              quantization::MapOrder trhsOrder,
              quantization::MapOrder tResultOrder>
    void QuantizeGemm_without_bias(const quantization::MatrixMap<const std::int8_t, tlhsOrder>& lhs,
                      const quantization::MatrixMap<const std::int8_t, trhsOrder>& rhs,
                      quantization::MatrixMap<std::int8_t, tResultOrder>* result,
                      const quantizationParams& qlparams,
                      const quantizationParams& qrparams,
                      const quantizationParams& qresultparams,
                      OutputStageQuantizeDownInt32ByFixedPoint quantize_down)
        {
        assert(lhs.cols() == rhs.rows());
        assert(lhs.rows() == result->rows());
        assert(rhs.cols() == result->cols());

        int32_t NZ1Z2 = lhs.cols() * qlparams.zero_point * qrparams.zero_point;
        int32_t* a1row = new int32_t[lhs.rows()]();
        int32_t* a2col = new int32_t[rhs.cols()]();
        int32_t sumQ1Q2 = 0;

        int32_t subSum = 0;
        int32_t totalSum = 0;
        int32_t total = 0;

        #pragma omp parallel for
        for (int row = 0; row < lhs.rows(); row++) {
            for (int depth = 0; depth < lhs.cols(); depth++) {
                a1row[row] += lhs(row, depth);
            }
        }

        #pragma omp parallel for
        for (int col = 0; col < rhs.cols(); col++) {
            for (int depth = 0; depth < lhs.cols(); depth++) {
                a2col[col] += rhs(depth, col);
            }
        }

        #pragma omp parallel for
        for (int row = 0; row < lhs.rows(); row++) {
            for (int col = 0; col <rhs.cols(); col++) {
                (*result)(row, col) = 0;
                sumQ1Q2 = 0;

                for (int depth = 0; depth < lhs.cols(); depth++) {
                    sumQ1Q2 += lhs(row, depth) * rhs(depth, col);
                }
                subSum = (NZ1Z2 - qlparams.zero_point * a2col[col] - qrparams.zero_point * a1row[row] + sumQ1Q2);
                subSum = SaturatingRoundingDoublingHighMul(subSum, quantize_down.result_fixedpoint_multiplier);
                total = RoundingDivideByPot<std::int32_t, std::int32_t>(subSum, quantize_down.result_shift);
                totalSum = qresultparams.zero_point + total;
                totalSum = totalSum > 127 ? 127 : totalSum < -128 ? -128 : totalSum;
                (*result)(row, col) = totalSum;
            }
        }
        delete[] a1row;
        delete[] a2col;
    }

    template <quantization::MapOrder tlhsOrder,
              quantization::MapOrder trhsOrder,
              quantization::MapOrder tResultOrder>
    void QuantizeGemm_nn_gpu(const quantization::MatrixMap<const std::int8_t, tlhsOrder>& lhs,
                          const quantization::MatrixMap<const std::int8_t, trhsOrder>& rhs,
                          quantization::MatrixMap<std::int8_t, tResultOrder>* result,
                          const quantization::MatrixMap<const std::int32_t, tResultOrder>& result32,
                          const quantizationParams& qlparams,
                          const quantizationParams& qrparams,
                          const quantizationParams& qresultparams,
                          OutputStageQuantizeDownInt32ByFixedPoint quantize_down,
                          const quantization::MatrixMap<const std::int32_t, tResultOrder>& biases) {
        assert(lhs.cols() == rhs.rows());
        assert(lhs.rows() == result->rows());
        assert(rhs.cols() == result->cols());

        int32_t NZ1Z2 = lhs.cols() * qlparams.zero_point * qrparams.zero_point;
        int32_t* a1row = new int32_t[lhs.rows()]();
        int32_t* a2col = new int32_t[rhs.cols()]();
        int32_t sumQ1Q2 = 0;

        int32_t subSum = 0;
        int32_t totalSum = 0;
        int32_t total = 0;

        #pragma omp parallel for
        for (int row = 0; row < lhs.rows(); row++) {
            for (int depth = 0; depth < lhs.cols(); depth++) {
                a1row[row] += lhs(row, depth);
            }
        }

        #pragma omp parallel for
        for (int col = 0; col < rhs.cols(); col++) {
            for (int depth = 0; depth < lhs.cols(); depth++) {
                a2col[col] += rhs(depth, col);
            }
        }

        #pragma omp parallel for
        for (int row = 0; row < lhs.rows(); row++) {
            for (int col = 0; col <rhs.cols(); col++) {
                sumQ1Q2 = result32(row, col);

                subSum = (NZ1Z2 - qlparams.zero_point * a2col[col] - qrparams.zero_point * a1row[row] + sumQ1Q2);
                subSum = SaturatingRoundingDoublingHighMul(subSum, quantize_down.result_fixedpoint_multiplier);
                subSum += biases(row, 0);
                total = RoundingDivideByPot<std::int32_t, std::int32_t>(subSum, quantize_down.result_shift);
                totalSum = qresultparams.zero_point + total;
                totalSum = totalSum > 127 ? 127 : totalSum < -128 ? -128 : totalSum;

                (*result)(row, col) = totalSum;
            }
        }
        delete[] a1row;
        delete[] a2col;
    }

    template <quantization::MapOrder tlhsOrder,
              quantization::MapOrder trhsOrder,
              quantization::MapOrder tResultOrder>
    void QuantizeGemm_nt_gpu(const quantization::MatrixMap<const std::int8_t, tlhsOrder>& lhs,
                          const quantization::MatrixMap<const std::int8_t, trhsOrder>& rhs,
                          quantization::MatrixMap<std::int8_t, tResultOrder>* result,
                          const quantization::MatrixMap<const std::int32_t, tResultOrder>& result32,
                          const quantizationParams& qlparams,
                          const quantizationParams& qrparams,
                          const quantizationParams& qresultparams,
                          OutputStageQuantizeDownInt32ByFixedPoint quantize_down,
                          const quantization::MatrixMap<const std::int32_t, tResultOrder>& biases) {
        assert(lhs.cols() == rhs.rows());
        assert(lhs.rows() == result->rows());
        assert(rhs.cols() == result->cols());

        int32_t NZ1Z2 = lhs.cols() * qlparams.zero_point * qrparams.zero_point;
        int32_t* a1row = new int32_t[lhs.rows()]();
        int32_t* a2col = new int32_t[rhs.cols()]();
        int32_t sumQ1Q2 = 0;

        int32_t subSum = 0;
        int32_t totalSum = 0;
        int32_t total = 0;

        #pragma omp parallel for
        for (int row = 0; row < lhs.rows(); row++) {
            for (int depth = 0; depth < lhs.cols(); depth++) {
                a1row[row] += lhs(row, depth);
            }
        }

        #pragma omp parallel for
        for (int col = 0; col < rhs.cols(); col++) {
            for (int depth = 0; depth < lhs.cols(); depth++) {
                a2col[col] += rhs(depth, col);
            }
        }

        #pragma omp parallel for
        for (int row = 0; row < lhs.rows(); row++) {
            for (int col = 0; col <rhs.cols(); col++) {
                sumQ1Q2 = result32(row, col);

                subSum = (NZ1Z2 - qlparams.zero_point * a2col[col] - qrparams.zero_point * a1row[row] + sumQ1Q2);
                subSum = SaturatingRoundingDoublingHighMul(subSum, quantize_down.result_fixedpoint_multiplier);
                subSum += biases(0, col);
                total = RoundingDivideByPot<std::int32_t, std::int32_t>(subSum, quantize_down.result_shift);
                totalSum = qresultparams.zero_point + total;
                totalSum = totalSum > 127 ? 127 : totalSum < -128 ? -128 : totalSum;

                (*result)(row, col) = totalSum;
            }
        }
        delete[] a1row;
        delete[] a2col;
    }
}
#endif //QUANTIZATION_QUANTIZE_H
