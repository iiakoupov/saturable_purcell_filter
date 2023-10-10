/*
 * Copyright (c) 2021 Ivan Iakoupov
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef CSRMATRIX_H
#define CSRMATRIX_H

#include "Eigen/SparseCore"
#include <complex>
#include <vector>
#include <omp.h>

struct CSRMatrix
{
    int rows;
    int cols;
    std::vector<std::complex<double>> val;
    std::vector<int> col;
    std::vector<int> ptr;
    void mul_vec(Eigen::VectorXcd &ret, const Eigen::VectorXcd &v, double factor, double factor_ret = 0) const
    {
        assert(cols == v.rows() && "Matrix dimensions do not match!");
        assert(rows == ret.rows() && "Matrix dimensions do not match!");
        const int threads = omp_get_max_threads();
        #pragma omp parallel for if(val.size()>20000) schedule(dynamic,(rows+threads*4-1)/(threads*4)) num_threads(threads)
        for(int i = 0; i < rows; ++i) {
            std::complex<double> sum(0);
            for(int j = ptr[i]; j < ptr[i + 1]; ++j) {
                sum += factor * val[j] * v[col[j]];
            }
            ret[i] = factor_ret * ret[i] + sum;
        }
    }
};

inline void sortTripletsCSR(
        std::vector<Eigen::Triplet<std::complex<double>>> &triplets, int64_t rows)
{
    auto rowMajorSort = [rows] (const Eigen::Triplet<std::complex<double>> &a, const Eigen::Triplet<std::complex<double>> &b) -> bool
    {
        const int64_t index_a = a.row()*rows+a.col();
        const int64_t index_b = b.row()*rows+b.col();
        return index_a < index_b;
    };
    std::sort(triplets.begin(), triplets.end(), rowMajorSort);
}

inline void CSRMatrixFromSortedTriplets(
        CSRMatrix &csrMat,
        const Eigen::Triplet<std::complex<double>> *triplets,
        int64_t numTriplets,
        int rows, int cols)
{
    csrMat.rows = rows;
    csrMat.cols = cols;
    csrMat.val.reserve(numTriplets);
    csrMat.col.reserve(numTriplets);
    csrMat.ptr.reserve(csrMat.rows+1);
    // Since we have sorted triplets, we can
    // detect duplicates by checking whether
    // the previous triplet has the same row and column.
    // Initialize to -1 to handle the first triplet.
    int previous_row = -1;
    int previous_col = -1;
    for (int i = 0; i < numTriplets; ++i) {
        if (triplets[i].row() == previous_row
                && triplets[i].col() == previous_col) {
            // Sum duplicates
            csrMat.val.back() += triplets[i].value();
        } else {
            if (triplets[i].row() != previous_row) {
                // Row starts
                csrMat.ptr.push_back(csrMat.val.size());
            }
            csrMat.val.push_back(triplets[i].value());
            csrMat.col.push_back(triplets[i].col());
        }
        previous_row = triplets[i].row();
        previous_col = triplets[i].col();
    }
    // End of the last row
    csrMat.ptr.push_back(csrMat.val.size());
}

#endif // CSRMATRIX_H

