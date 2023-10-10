// Copyright (c) 2019-2022 Ivan Iakoupov
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

#ifndef MASTER_EQUATION_ROC_H
#define MASTER_EQUATION_ROC_H

#include <complex>
#include <functional>
#include <vector>
#include "sparse_matrix_data.h"

struct MasterEquationData
{
    std::vector<double> time;
    std::vector<std::vector<std::complex<double>>> M_values;
    std::vector<std::vector<std::complex<double>>> M_diag_values;
    std::vector<std::vector<std::complex<double>>> fidelities;
};

MasterEquationData evolve_master_equation_roc(
        int basis_size, const SparseMatrixData &L,
        const std::complex<double> *rho0,
        const std::vector<SparseMatrixData> &M_operators,
        const std::vector<const double*> &M_diag_operators,
        const std::vector<const std::complex<double>*> &fidelity_rhos,
        double dt, int64_t N_t, int64_t iterationsBetweenDeviceSynchronize);

MasterEquationData evolve_time_dependent_master_equation_roc(
        int basis_size, const SparseMatrixData &L,
        const SparseMatrixData &H_t_Re,
        const SparseMatrixData &H_t_Im,
        std::function<std::complex<double>(double)> H_t_factor,
        const std::complex<double> *rho0,
        const std::vector<SparseMatrixData> &M_operators,
        const std::vector<const double*> &M_diag_operators,
        const std::vector<const std::complex<double>*> &fidelity_rhos,
        double dt, int64_t N_t, int64_t iterationsBetweenDeviceSynchronize);

void adrk4_master_equation_roc(
        int N_amplitudes, const double *x, double *grad,
        double Omega_max, double t_final,
        double sigmaFilter, double sigmaWindow,
        const double *f1, const double *f2,
        std::function<void(double, double, double)> addToGradient,
        std::complex<double> *trace_final,
        int basis_size, const SparseMatrixData &L,
        const SparseMatrixData &L_adjoint,
        const SparseMatrixData &H_t_Re,
        const SparseMatrixData &H_t_Im,
        std::function<std::complex<double>(double)> calculateOmega,
        const std::complex<double> *rho_initial_vec,
        const std::complex<double> *rho_target_vec,
        double dt, int64_t N_t, int64_t iterationsBetweenDeviceSynchronize,
        int64_t cacheSizeTimeSteps, int64_t cacheSizeTimeStepsGPU,
        int64_t timeStepsPerStoredState);

#endif // MASTER_EQUATION_ROC_H

