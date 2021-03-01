#ifndef QUANTIZATION_FIXEDPOINT_H
#define QUANTIZATION_FIXEDPOINT_H

#include "MatrixMap.h"

namespace quantization {
    template<typename IntegerType>
    IntegerType BitAnd(IntegerType a, IntegerType b) {
        return a & b;
    }

    template<typename IntegerType>
    IntegerType ShiftRight(IntegerType a, int offset) {
        return a >> offset;
    }

    template<typename IntegerType>
    IntegerType MaskIfNonZero(IntegerType a) {
        static constexpr IntegerType zero = 0;
        return a ? ~zero : zero;
    }

    template<typename IntegerType>
    IntegerType MaskIfLessThan(IntegerType a, IntegerType b) {
        return MaskIfNonZero(a < b);
    }

    template<typename IntegerType>
    IntegerType MaskIfGreaterThan(IntegerType a, IntegerType b) {
        return MaskIfNonZero(a > b);
    }
}

#endif //QUANTIZATION_FIXEDPOINT_H
