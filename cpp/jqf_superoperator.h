// Copyright (c) 2020-2022 Ivan Iakoupov
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.

#ifndef JQF_SUPEROPERATOR_H
#define JQF_SUPEROPERATOR_H

#include <vector>
#include "operator.h"
#include "csrmatrix.h"

#define JQF_SUPEROPERATOR_OMIT_OMEGA_TERMS     (1 << 0)
#define JQF_SUPEROPERATOR_OMIT_DELTA_TERMS     (1 << 1)

inline std::vector<Eigen::Triplet<std::complex<double>>>
make_annihilation_operator_triplets(int n)
{
    std::vector<Eigen::Triplet<std::complex<double>>> ret;
    for (int i = 1; i < n; ++i) {
        ret.emplace_back(i-1, i, std::sqrt(i));
    }
    return ret;
}

inline std::vector<Eigen::Triplet<std::complex<double>>>
make_number_operator_triplets(int n)
{
    std::vector<Eigen::Triplet<std::complex<double>>> ret;
    for (int i = 0; i < n; ++i) {
        ret.emplace_back(i, i, i);
    }
    return ret;
}

struct JQFSuperoperatorData
{
    SpMat L;
    CSRMatrix L_csr;
    std::vector<BasisVector> basis;
    std::vector<Eigen::VectorXd> subsystem_eigenvalues;
    std::vector<Eigen::VectorXd> subsystem_eigenvalue_shifts;
    std::vector<Eigen::VectorXcd> psi_up;
    std::vector<Eigen::VectorXcd> psi_down;
    std::vector<SpMat> M_psi_up;
    std::vector<SpMat> M_psi_down;
    std::vector<SpMat> M_O;
    std::vector<SpMat> M_O_adjoint;
    std::vector<SpMat> M_a;
    std::vector<SpMat> M_a_adjoint;
    std::vector<SpMat> M_b;
    std::vector<SpMat> M_b_adjoint;
    std::vector<SpMat> M_b_adjoint_a;
    std::vector<SpMat> M_b_a_adjoint;
    std::vector<SpMat> M_n_res;
    std::vector<SpMat> M_n_atom;
    std::vector<std::vector<SpMat>> M_sigma;
    SpMat M_r;
    std::complex<double> factor_r;
    std::vector<int> psi_up_eigenstate_indices;
    std::vector<int> psi_down_eigenstate_indices;
    std::vector<double> Omega_factors;
    bool replaced_negative_frequencies_with_positive;
};

JQFSuperoperatorData generate_superoperator(
        const std::vector<double> &kappa,
        const std::vector<double> &kappaInternal,
        const std::vector<double> &gamma,
        const std::vector<double> &gammaInternal,
        const std::vector<double> &gammaDephasing,
        double g, double Omega, double J_x, double omega_d,
        const std::vector<double> &Delta_r,
        const std::vector<double> &Delta,
        const std::vector<double> &transmon_anharmonicity,
        const std::vector<double> &k0x_r,
        const std::vector<double> &k0x_a,
        double k0x_out,
        int num_excitations,
        const std::vector<int> &transmon_excitations,
        int flags);

JQFSuperoperatorData generate_superoperator_diag(
        const std::vector<double> &kappa,
        const std::vector<double> &kappaInternal,
        const std::vector<double> &gamma,
        const std::vector<double> &gammaInternal,
        const std::vector<double> &gammaDephasing,
        double g, double Omega, double J_x, double omega_d,
        const std::vector<double> &Delta_r,
        const std::vector<double> &Delta,
        const std::vector<double> &transmon_anharmonicity,
        const std::vector<double> &k0x_r,
        const std::vector<double> &k0x_a,
        double k0x_out,
        int num_excitations,
        const std::vector<int> &transmon_excitations,
        int flags);

#endif // JQF_SUPEROPERATOR_H
