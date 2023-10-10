// Copyright (c) 2020-2021 Ivan Iakoupov
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

#ifndef DIAGONALIZE_ATOM_RESONATOR_SYSTEM_H
#define DIAGONALIZE_ATOM_RESONATOR_SYSTEM_H

#include "operator.h"
#include "Eigen/Eigenvalues"

inline Eigen::Vector2cd get_up_eigenstate(double g, double Delta_r, double Delta_1)
{
    Eigen::Matrix2cd M2 = Eigen::Matrix2cd::Zero();
    M2(0,0) = -Delta_r;
    M2(0,1) = g;
    M2(1,0) = g;
    M2(1,1) = -Delta_1;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2cd> es;
    es.compute(M2);
    // Find the eigenvector that has most of the excitation
    // in the qubit
    int qubit_excited_eigenvector = 0;
    double max_qubit_population = 0;
    for (int i = 0; i < 2; ++i) {
        const double qubit_population = std::norm(es.eigenvectors().col(i)(1));
        if (max_qubit_population < qubit_population) {
            max_qubit_population = qubit_population;
            qubit_excited_eigenvector = i;
        }
    }
    Eigen::Vector2cd ret = es.eigenvectors().col(qubit_excited_eigenvector);
    return ret;
}

inline Eigen::Vector2cd get_down_eigenstate(double g, double Delta_r, double Delta_1)
{
    Eigen::Matrix2cd M2 = Eigen::Matrix2cd::Zero();
    M2(0,0) = -Delta_r;
    M2(0,1) = g;
    M2(1,0) = g;
    M2(1,1) = -Delta_1;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2cd> es;
    es.compute(M2);
    // Find the eigenvector that has least of the excitation
    // in the qubit
    int qubit_down_eigenvector = 0;
    double min_qubit_population = 1;
    for (int i = 0; i < 2; ++i) {
        const double qubit_population = std::norm(es.eigenvectors().col(i)(1));
        if (min_qubit_population > qubit_population) {
            min_qubit_population = qubit_population;
            qubit_down_eigenvector = i;
        }
    }
    Eigen::Vector2cd ret = es.eigenvectors().col(qubit_down_eigenvector);
    return ret;
}

// This function operates in the separate bases of artificial
// atom basis and the harmonic oscillator and returns the
// state where the data artificial atom and harmonic oscillator
// are in the eigenstate of their isolated system and the
// filter artificial atom is in the ground state
inline Eigen::VectorXcd get_psi_for_eigenstate(const Eigen::Vector2cd up_eigenstate,
                                       const std::vector<BasisVector> &aa_basis,
                                       int harmonic_basis_size)
{
    const int aa_basis_size = aa_basis.size();
    Eigen::VectorXcd psi_ref = Eigen::VectorXcd::Zero(aa_basis_size*harmonic_basis_size);
    std::vector<sigma_state_t> spec00(aa_basis[0].size(), 0);
    std::vector<sigma_state_t> spec01(aa_basis[0].size(), 0);
    spec01[0] = 1;
    BasisVector vec_00(spec00);
    BasisVector vec_01(spec01);
    for (int i = 0; i < aa_basis_size; ++i) {
        for (int j = 0; j < harmonic_basis_size; ++j) {
            if (aa_basis[i] == vec_00 && j == 1) {
                psi_ref(i*harmonic_basis_size+j) = up_eigenstate(0);
            } else if (aa_basis[i] == vec_01 && j == 0) {
                psi_ref(i*harmonic_basis_size+j) = up_eigenstate(1);
            }
        }
    }
    return psi_ref;
}

// This function operates in the combined basis of artificial
// atom basis and the harmonic oscillator and returns the
// state where the data artificial atom and harmonic oscillator
// are in the eigenstate of their isolated system and the
// filter artificial atom is in the ground state
inline Eigen::VectorXcd get_psi_for_eigenstate_non_tensor(
        const Eigen::Vector2cd up_eigenstate,
        const std::vector<BasisVector> &basis)
{
    const int basis_size = basis.size();
    Eigen::VectorXcd psi_ref = Eigen::VectorXcd::Zero(basis_size);
    std::vector<sigma_state_t> spec100(basis[0].size(), 0);
    std::vector<sigma_state_t> spec001(basis[0].size(), 0);
    // In both cases, the filter artificial atom is in the ground state
    spec100[2] = 1; // harmonic oscillator mode is excited
    spec001[0] = 1; // data artificial atom is excited
    BasisVector vec_100(spec100);
    BasisVector vec_001(spec001);
    for (int i = 0; i < basis_size; ++i) {
        if (basis[i] == vec_100) {
            psi_ref(i) = up_eigenstate(0);
        } else if (basis[i] == vec_001) {
            psi_ref(i) = up_eigenstate(1);
        }
    }
    return psi_ref;
}

// This function operates in the reduced basis of artificial
// atom basis and the harmonic oscillator and returns the
// state where the data artificial atom and harmonic oscillator
// are in the eigenstate (and the state of the filter artificial
// is irrelevant because it is assumed to be traced out).
inline Eigen::VectorXcd get_psi_for_eigenstate_2(const Eigen::Vector2cd up_eigenstate,
                                          const std::vector<BasisVector> &basis)
{
    const int basis_size = basis.size();
    Eigen::VectorXcd psi_ref = Eigen::VectorXcd::Zero(basis_size);
    std::vector<sigma_state_t> spec10(basis[0].size(), 0);
    std::vector<sigma_state_t> spec01(basis[0].size(), 0);
    spec01[0] = 1;
    spec10[1] = 1;
    BasisVector vec_10(spec10);
    BasisVector vec_01(spec01);
    for (int i = 0; i < basis_size; ++i) {
        if (basis[i] == vec_10) {
            psi_ref(i) = up_eigenstate(0);
        } else if (basis[i] == vec_01) {
            psi_ref(i) = up_eigenstate(1);
        }
    }
    return psi_ref;
}

#endif // DIAGONALIZE_ATOM_RESONATOR_SYSTEM_H
