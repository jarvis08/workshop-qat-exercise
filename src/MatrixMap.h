#ifndef QUANTIZATION_MATRIXMAP_H
#define QUANTIZATION_MATRIXMAP_H

#include <iostream>
#include "map.h"

namespace quantization {

    template <typename tScalar, MapOrder tOrder>
    class MatrixMap {
    public:
        typedef tScalar Scalar;
        static const MapOrder kOrder = tOrder;

    protected:
        Scalar* data_;
        int rows_, cols_, stride_;

    public:
        MatrixMap()
            : data_(nullptr), rows_(0), cols_(0), stride_(0) {}
        MatrixMap(Scalar* data, int rows, int cols)
            : data_(data), rows_(rows), cols_(cols), stride_(kOrder == MapOrder::ColMajor ? rows : cols) {}
        MatrixMap(const MatrixMap& other)
            : data_(other.data_), rows_(other.rows_), cols_(other.cols_), stride_(other.stride_) {}

        int rows() const { return rows_; }
        int cols() const { return cols_; }
        int stride() const { return stride_; }
        int rows_stride() const { return kOrder == MapOrder::ColMajor ? 1 : stride_; }
        int cols_stride() const { return kOrder == MapOrder::RowMajor ? 1 : stride_; }

        Scalar* data() const { return data_; }
        Scalar* data(int row, int col) const {
            return data_ + row * rows_stride() + col * cols_stride();
        }
        Scalar& operator()(int row, int col) const { return *data(row, col); }
    };

    template <typename tScalar, MapOrder tOrder>
    std::ostream& operator<<(std::ostream& s, const MatrixMap<tScalar, tOrder>& m) {
        for (int i = 0; i < m.rows(); i++) {
            for (int j = 0; j < m.cols(); j++) {
                if (j) {
                    s << '\t';
                }
                s << static_cast<float>(m(i, j));
            }
            s << '\n';
        }
        return s;
    }
}

#endif //QUANTIZATION_MATRIXMAP_H
