#ifndef QUANTIZATION_MULTIPLICATION_H
#define QUANTIZATION_MULTIPLICATION_H

#include <cassert>
#include "MatrixMap.h"

template <quantization::MapOrder tLhsOrder,
        quantization::MapOrder tRhsOrder,
        quantization::MapOrder tResultOrder,
        quantization::MapOrder tBiasOrder>
void FloatMatrixMultiplication(
        const quantization::MatrixMap<const float, tLhsOrder>& lhs,
        const quantization::MatrixMap<const float, tRhsOrder>& rhs,
        quantization::MatrixMap<float, tResultOrder>* result, 
        const quantization::MatrixMap<const float, tBiasOrder>& bias) {

    assert(lhs.cols() == rhs.rows());
    assert(lhs.rows() == result->rows());
    assert(rhs.cols() == result->cols());

    for (int i = 0; i < lhs.rows(); i++) {
        for (int k = 0; k < rhs.cols(); k++) {
            (*result)(i, k) = 0;
            for (int j = 0; j < lhs.cols(); j++) {
                (*result)(i, k) += lhs(i, j) * rhs(j, k);
            }
        (*result)(i, k) += bias(k, k);
        }
    }
}

template <quantization::MapOrder oriOrder,
          quantization::MapOrder quanOrder,
          quantization::MapOrder diffOrder>
void Verification(const quantization::MatrixMap<const float, oriOrder>& oriMatrix,
                  const quantization::MatrixMap<const float, quanOrder>& quanMatrix,
                  quantization::MatrixMap<float, diffOrder>* diffMatrix) {
    for (int row = 0; row < oriMatrix.rows(); row++) {
        for (int col = 0; col < oriMatrix.cols(); col++) {
            (*diffMatrix)(row, col) = oriMatrix(row, col) - quanMatrix(row, col);
        }
    }
}

#endif //QUANTIZATION_MULTIPLICATION_H
