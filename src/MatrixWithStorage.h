#ifndef MATRIXWITHSTORAGE_H
#define MATRIXWITHSTORAGE_H

#include <iostream>
#include <random>
#include <vector>

#include "map.h"
#include "MatrixMap.h"

namespace quantization {

    template <typename tScalar, MapOrder tOrder>
    class MatrixWithStorage {
    public:
        MatrixWithStorage(int rows, int cols)
            : storage(rows * cols), matrix_map(storage.data(), rows, cols) {}
        MatrixWithStorage(tScalar* mat, int rows, int cols)
            : storage(rows * cols), matrix_map(storage.data(), rows, cols) {
            storage.assign(mat, mat+(rows*cols));
        }
        void MakeRandom() {
            static std::mt19937 random_engine;
            std::uniform_real_distribution<float> distribution(-1, 1);
            for (auto& x: storage) {
                x = static_cast<tScalar>(distribution(random_engine));
	         }
        }

        void lhs() {
            storage[0] = 1.1; storage[1] = 2.2; storage[2] = 3.3;
            storage[3] = 4.4; storage[4] = 5.5; storage[5] = 6.6;
        }

        void rhs() {
            storage[0] = 1.1; storage[1] = 2.2; storage[2] = 3.3;
            storage[3] = 4.4; storage[4] = 5.5; storage[5] = 6.6;
        }

        void toMatrix(tScalar** matrix){
            std::copy(storage.begin(), storage.end(), *matrix);
	    }

        MatrixMap<const tScalar, tOrder> ConstMap() const {
            return MatrixMap<const tScalar, tOrder>(storage.data(), matrix_map.rows(), matrix_map.cols());
        }
        MatrixMap<tScalar, tOrder> Map() {
            return MatrixMap<tScalar, tOrder>(storage.data(), matrix_map.rows(), matrix_map.cols());
        }

        const std::vector<tScalar>& Storage() const { return storage; }
        std::vector<tScalar>& Storage() { return storage; }

    public:
        tScalar* setMin() { return &min_; }
        tScalar* setMax() { return &max_; }

        tScalar Min() { return min_; }
        tScalar Max() { return max_; }

        void setQParams(quantizationParams qp) { QParams = qp; }
        quantizationParams getQParams() const { return QParams; }

    private:
        std::vector<tScalar> storage;
        MatrixMap<tScalar, tOrder> matrix_map;

        tScalar min_;
        tScalar max_;

        quantizationParams QParams;
    };

    template <typename tScalar, MapOrder tOrder>
    std::ostream& operator<<(std::ostream& s, const MatrixWithStorage<tScalar, tOrder>& m) {
        return s << m.ConstMap();
    }

}
#endif
