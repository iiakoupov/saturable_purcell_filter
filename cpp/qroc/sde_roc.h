// Copyright (c) 2020-2021 Ivan Iakoupov
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom
// the Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
// OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
// USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef SDE_ROC_H
#define SDE_ROC_H

#include <complex>
#include <functional>
#include <vector>
#include "sparse_matrix_data.h"

struct SDEData
{
    std::vector<double> time;
    std::vector<double> excitation;
};

SDEData sme_roc(int basis_size,
                int batch_size,
                const SparseMatrixData &L,
                const SparseMatrixData &RR,
                const std::complex<double> *rho_vec_0,
                const std::complex<double> *rho_vec_1,
                const double *filter,
                const double *DeltaW_matrix,
                const double *integration_times,
                int num_integration_times,
                int N_t_max);

#endif // SDE_ROC_H

