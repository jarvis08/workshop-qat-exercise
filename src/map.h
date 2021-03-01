#ifndef MAP_H
#define MAP_H

#include <cstdint>

namespace quantization {
    enum class MapOrder {ColMajor, RowMajor};

    struct quantizationParams {
        float scale;
        std::int8_t zero_point;
    };

    struct OutputStageQuantizeDownInt32ByFixedPoint {
        std::int32_t result_fixedpoint_multiplier;
        std::int32_t result_shift;
    };

//    struct OutputStageSaturatingCastToint8 {};
}

#endif //MAP_H
