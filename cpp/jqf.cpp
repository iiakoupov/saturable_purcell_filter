// Copyright (c) 2020-2023 Ivan Iakoupov
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

#include "jqf.h"

#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include <nlopt.hpp>

#include "findroot.h"
#include "operator.h"
#include "diagonalize_atom_resonator_system.h"
#include "get_psi_for_excited_atoms.h"
#include "jqf_superoperator.h"
#ifdef EIGEN_USE_MKL_ALL
#include "mkl_support.h"
#endif // EIGEN_USE_MKL_ALL
#ifdef USE_ROCM
#include "qroc/master_equation_roc.h"
#include "smd_from_spmat.h"
#endif // USE_ROCM
#include "master_equation.h"
#include "quadrature/quad.h"
#include "rk4.h"
#include "jqf_adrk4.h"

#define DO_NOT_DO_PARTIAL_TRACE

#ifndef DO_NOT_DO_PARTIAL_TRACE
#include "partial_trace.h"
#endif // DO_NOT_DO_PARTIAL_TRACE

void print_sparse_matrix(const SpMat &M)
{
    for (int k = 0; k < M.outerSize(); ++k) {
        for (SpMat::InnerIterator it(M, k); it; ++it) {
            std::cout << it.row() << ", " << it.col() << ", " << it.value() << std::endl;
        }
    }
}

Eigen::VectorXcd sparse_matrix_extract_diagonal(const SpMat &M)
{
    assert(M.rows() == M.cols() && "Sparse matrix is not square!");
    const int size = M.rows();
    Eigen::VectorXcd ret = Eigen::VectorXcd::Zero(size);
    for (int k = 0; k < M.outerSize(); ++k) {
        for (SpMat::InnerIterator it(M, k); it; ++it) {
            assert(it.row() == it.col() && "Sparse matrix is not diagonal!");
            ret(it.row()) += it.value();
        }
    }
    return ret;
}

JQFData jqf_data_qubit_decay_master_equation_manual(
        double kappa, double gamma2, double g, double omega_d, double omega_r,
        double omega_1, double omega_2, double k0x_r, double k0x_2,
        bool startInEigenstate, double t, int64_t N_t,
        int64_t data_reduce_factor)
{
    assert(N_t > 0 && "Number of steps needs to be a positive integer!");
    assert(startInEigenstate && "Only starting in eigenstate is supported!");

    JQFData ret;
    const std::complex<double> I(0,1);

    Eigen::Matrix2cd M2 = Eigen::Matrix2cd::Zero();
    M2(0,0) = omega_r;
    M2(0,1) = g;
    M2(1,0) = g;
    M2(1,1) = omega_1;
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
    //std::cout << "es.eigenvalues() = " << es.eigenvalues() << std::endl;
    //std::cout << "es.eigenvectors() = " << std::endl << es.eigenvectors() << std::endl;

    const double common_factor_12 = 0.5*std::sqrt(kappa*gamma2);
    const double common_factor_21 = common_factor_12;
    const double common_factor_11 = 0.5*std::sqrt(kappa*kappa);
    const double common_factor_22 = 0.5*std::sqrt(gamma2*gamma2);

    const double eigenvalue0 = 0.5*(omega_1+omega_r)+std::sqrt(std::pow(0.5*(omega_r-omega_1),2)+std::pow(g,2));
    const double eigenvalue1 = 0.5*(omega_1+omega_r)-std::sqrt(std::pow(0.5*(omega_r-omega_1),2)+std::pow(g,2));
    //std::cout << "eigenvalue0 = " << eigenvalue0 << std::endl;
    //std::cout << "eigenvalue1 = " << eigenvalue1 << std::endl;

    const double omega_1_1 = es.eigenvalues()(qubit_excited_eigenvector);
    const double omega_1_2 = es.eigenvalues()(qubit_down_eigenvector);
    const double omega_2_1 = omega_2;
    //std::cout << "omega_1_1 = " << omega_1_1 << std::endl;
    //std::cout << "omega_1_2 = " << omega_1_2 << std::endl;

    const double omega_1_10 = omega_1_1; // omega_1_0 = 0
    const double omega_1_20 = omega_1_2; // omega_1_0 = 0
    const double omega_2_10 = omega_2_1; // omega_2_0 = 0

    const double delta_1_1 = omega_1_1 - omega_1_10;
    const double delta_1_2 = omega_1_2 - omega_1_10;
    const double delta_2_1 = omega_2_1 - omega_1_10;

    const double omega_ref = omega_1_10;

    const double k_factor_1_10 = omega_1_10/omega_ref;
    const double k_factor_1_20 = omega_1_20/omega_ref;
    const double k_factor_2_10 = omega_2_10/omega_ref;

    const std::complex<double> xi_11_10 = common_factor_11*omega_1_10/std::sqrt(omega_r*omega_r)*exp_factor(k_factor_1_10, k0x_r, k0x_r);
    const std::complex<double> xi_11_20 = common_factor_11*omega_1_20/std::sqrt(omega_r*omega_r)*exp_factor(k_factor_1_20, k0x_r, k0x_r);
    const std::complex<double> xi_21_10 = common_factor_21*omega_1_10/std::sqrt(omega_2*omega_r)*exp_factor(k_factor_1_10, k0x_2, k0x_r);
    const std::complex<double> xi_21_20 = common_factor_21*omega_1_20/std::sqrt(omega_2*omega_r)*exp_factor(k_factor_1_20, k0x_2, k0x_r);
    const std::complex<double> xi_12_10 = common_factor_12*omega_2_10/std::sqrt(omega_r*omega_2)*exp_factor(k_factor_2_10, k0x_r, k0x_2);
    const std::complex<double> xi_22_10 = common_factor_22*omega_2_10/std::sqrt(omega_2*omega_2)*exp_factor(k_factor_2_10, k0x_2, k0x_2);

    const double theta = 0.5*std::arg(0.5*(omega_r-omega_1)+I*g);
    const double st = std::sin(theta);
    const double ct = std::cos(theta);
    const double st2 = st*st;
    const double ct2 = ct*ct;

    //std::cout << "xi_11_10 = " << xi_11_10 << std::endl;
    //std::cout << "xi_11_20 = " << xi_11_20 << std::endl;
    //std::cout << "xi_21_10 = " << xi_21_10 << std::endl;
    //std::cout << "xi_21_20 = " << xi_21_20 << std::endl;
    //std::cout << "xi_12_10 = " << xi_12_10 << std::endl;
    //std::cout << "xi_22_10 = " << xi_22_10 << std::endl;
    //std::cout << "sin(theta) = " << st << std::endl;
    //std::cout << "cos(theta) = " << ct << std::endl;

    Eigen::MatrixXcd M = Eigen::MatrixXcd::Zero(16,16);
    M(0,5) = xi_11_10.real()*st2;
    M(0,6) = 0.5*(xi_11_10+std::conj(xi_11_20))*st*ct;
    M(0,7) = 0.5*(xi_21_10+std::conj(xi_12_10))*st;
    M(0,9) = 0.5*(xi_11_20+std::conj(xi_11_10))*st*ct;
    M(0,10) = xi_11_20.real()*ct2;
    M(0,11) = 0.5*(xi_21_20+std::conj(xi_12_10))*ct;
    M(0,13) = 0.5*(xi_12_10+std::conj(xi_21_10))*st;
    M(0,14) = 0.5*(xi_12_10+std::conj(xi_21_20))*ct;
    M(0,15) = xi_22_10.real();
    M(1,1) = I*delta_1_1-0.5*std::conj(xi_11_10)*st2;
    M(1,2) = -0.5*std::conj(xi_11_20)*st*ct;
    M(1,3) = -0.5*std::conj(xi_12_10)*st;
    M(2,1) = -0.5*std::conj(xi_11_10)*st*ct;
    M(2,2) = I*delta_1_2-0.5*std::conj(xi_11_20)*ct2;
    M(2,3) = -0.5*std::conj(xi_12_10)*ct;
    M(3,1) = -0.5*std::conj(xi_21_10)*st;
    M(3,2) = -0.5*std::conj(xi_21_20)*ct;
    M(3,3) = I*delta_2_1-0.5*std::conj(xi_22_10);
    M(4,4) = -I*delta_1_1-0.5*xi_11_10*st2;
    M(4,8) = -0.5*xi_11_20*st*ct;
    M(4,12) = -0.5*xi_12_10*st;
    M(5,5) = -xi_11_10.real()*st2;
    M(5,6) = -0.5*std::conj(xi_11_20)*st*ct;
    M(5,7) = -0.5*std::conj(xi_12_10)*st;
    M(5,9) = -0.5*xi_11_20*st*ct;
    M(5,13) = -0.5*xi_12_10*st;
    M(6,5) = -0.5*std::conj(xi_11_10)*st*ct;
    M(6,6) = -I*(delta_1_1-delta_1_2)-0.5*xi_11_10*st2-0.5*std::conj(xi_11_20)*ct2;
    M(6,7) = -0.5*std::conj(xi_12_10)*ct;
    M(6,10) = -0.5*xi_11_20*st*ct;
    M(6,14) = -0.5*xi_12_10*st;
    M(7,5) = -0.5*std::conj(xi_21_10)*st;
    M(7,6) = -0.5*std::conj(xi_21_20)*ct;
    M(7,7) = -I*(delta_1_1-delta_2_1)-0.5*xi_11_10*st2-0.5*std::conj(xi_22_10);
    M(7,11) = -0.5*xi_11_20*st*ct;
    M(7,15) = -0.5*xi_12_10*st;
    M(8,4) = -0.5*xi_11_10*st*ct;
    M(8,8) = -I*delta_1_2-0.5*xi_11_20*ct2;
    M(8,12) = -0.5*xi_12_10*ct;
    M(9,5) = -0.5*xi_11_10*st*ct;
    M(9,9) = -I*(delta_1_2-delta_1_1)-0.5*xi_11_10*st2-0.5*xi_11_20*ct2;
    M(9,10) = -0.5*std::conj(xi_11_20)*st*ct;
    M(9,11) = -0.5*std::conj(xi_12_10)*st;
    M(9,13) = -0.5*xi_12_10*ct;
    M(10,6) = -0.5*xi_11_10*st*ct;
    M(10,9) = -0.5*std::conj(xi_11_10)*st*ct;
    M(10,10) = -xi_11_20.real()*ct2;
    M(10,11) = -0.5*std::conj(xi_12_10)*ct;
    M(10,14) = -0.5*xi_12_10*ct;
    M(11,7) = -0.5*xi_11_10*st*ct;
    M(11,9) = -0.5*std::conj(xi_21_10)*st;
    M(11,10) = -0.5*std::conj(xi_21_20)*ct;
    M(11,11) = -I*(delta_1_2-delta_2_1)-0.5*xi_11_20*ct2-0.5*std::conj(xi_22_10);
    M(11,15) = -0.5*xi_12_10*ct;
    M(12,4) = -0.5*xi_21_10*st;
    M(12,8) = -0.5*xi_21_20*ct;
    M(12,12) = -I*delta_2_1-0.5*xi_22_10;
    M(13,5) = -0.5*xi_21_10*st;
    M(13,9) = -0.5*xi_21_20*ct;
    M(13,13) = -I*(delta_2_1-delta_1_1)-0.5*std::conj(xi_11_10)*st2-0.5*xi_22_10;
    M(13,14) = -0.5*std::conj(xi_11_20)*st*ct;
    M(13,15) = -0.5*std::conj(xi_12_10)*st;
    M(14,6) = -0.5*xi_21_10*st;
    M(14,10) = -0.5*xi_21_20*ct;
    M(14,13) = -0.5*std::conj(xi_11_10)*st*ct;
    M(14,14) = -I*(delta_2_1-delta_1_2)-0.5*std::conj(xi_11_20)*ct2-0.5*xi_22_10;
    M(14,15) = -0.5*std::conj(xi_12_10)*ct;
    M(15,7) = -0.5*xi_21_10*st;
    M(15,11) = -0.5*xi_21_20*ct;
    M(15,13) = -0.5*std::conj(xi_21_10)*st;
    M(15,14) = -0.5*std::conj(xi_21_20)*ct;
    M(15,15) = -xi_22_10.real();

    //const int flags = 0;
    //const double J_x = 0;
    //const double k0x_out = k0x_2;
    //const double kappaInternal = 0;
    //const double gamma2Internal = 0;
    //const std::vector<double> gammaDephasing = {0, 0};
    //const std::vector<double> transmon_anharmonicity = {-200, -200};
    //const std::vector<int> transmon_excitations = {1, 1};
    //const double Omega = 0;
    //const int num_excitations = 1;
    //JQFSuperoperatorData sd = generate_superoperator_diag(
    //        {kappa}, {kappaInternal}, {gamma2}, {0, gamma2Internal},
    //        gammaDephasing, g, Omega, J_x, omega_d, {omega_r},
    //        {omega_1, omega_2}, transmon_anharmonicity, {k0x_r}, {k0x_2},
    //        k0x_out, num_excitations, transmon_excitations, flags);
    //Eigen::MatrixXcd L_dense = sd.L;
    //Eigen::MatrixXcd M_diff = M-L_dense;
    //for (int i = 0; i < 16; ++i) {
    //    for (int j = 0; j < 16; ++j) {
    //        if (std::abs(M_diff(i,j)) < 1e-15) {
    //            M_diff(i,j) = 0;
    //        }
    //    }
    //}
    //std::cout << "M = " << std::endl << M << std::endl;
    //std::cout << "L_dense = " << std::endl << L_dense << std::endl;
    //std::cout << "M_diff = " << std::endl << M_diff << std::endl;

    const int vec_size = 16;
    Eigen::VectorXcd v = Eigen::VectorXcd::Zero(vec_size);
    const int up_eigenstate_population_index = 1+4*1;
    const int down_eigenstate_population_index = 2+4*2;
    const int jqf_population_index = 3+4*3;
    v(up_eigenstate_population_index) = 1;

    const double dt = t/N_t;

    const int64_t N_t_reduced = N_t/data_reduce_factor;
    ret.time = std::vector<double>(N_t_reduced, 0);
    ret.Omega_Re = std::vector<double>(N_t_reduced, 0);
    ret.Omega_Im = std::vector<double>(N_t_reduced, 0);
    ret.res_populations.resize(1);
    ret.res_populations[0] = std::vector<double>(N_t_reduced, 0);
    ret.aa_populations.resize(2);
    ret.aa_populations[0] = std::vector<double>(N_t_reduced, 0);
    ret.aa_populations[1] = std::vector<double>(N_t_reduced, 0);
    ret.F.resize(1);
    ret.F[0] = std::vector<double>(N_t_reduced, 0);
    ret.F_down.resize(1);
    ret.F_down[0] = std::vector<double>(N_t_reduced, 0);
    ret.tilde_F = std::vector<double>(N_t_reduced, 0);

    Eigen::VectorXcd k1 = Eigen::VectorXcd::Zero(vec_size);
    Eigen::VectorXcd k2 = Eigen::VectorXcd::Zero(vec_size);
    Eigen::VectorXcd k3 = Eigen::VectorXcd::Zero(vec_size);
    Eigen::VectorXcd k4 = Eigen::VectorXcd::Zero(vec_size);
    for (int64_t i = 0; i < N_t; ++i) {
        if (i % data_reduce_factor == 0) {
            const int64_t dataStoreIndex = i / data_reduce_factor;
            ret.time[dataStoreIndex] = dt*i;
            //ret.res_populations[0][dataStoreIndex] = std::norm(v(0));
            //ret.aa_populations[0][dataStoreIndex] = std::norm(v(1));
            ret.aa_populations[1][dataStoreIndex] = v(jqf_population_index).real();
            ret.F[0][dataStoreIndex] = v(up_eigenstate_population_index).real();
            ret.F_down[0][dataStoreIndex] = v(down_eigenstate_population_index).real();
        }
        k1 = M*v*dt;
        k2 = M*(v+k1*0.5)*dt;
        k3 = M*(v+k2*0.5)*dt;
        k4 = M*(v+k3)*dt;
        v = v + (k1/6) + (k2/3) + (k3/3) + (k4/6);
    }
    return ret;
}

JQFData jqf_data_qubit_decay_dde_arbitrary_state(
        double kappa, double gamma2, double g, double omega_d, double omega_r,
        double omega_1, double omega_2, double k0x_r, double k0x_2,
        const std::vector<std::complex<double>> &v0, double t, int64_t N_t,
        int64_t data_reduce_factor)
{
    assert(N_t > 0 && "Number of steps needs to be a positive integer!");
    assert(v0.size() == 3 && "v0 should have size 3!");
    JQFData ret;
    const std::complex<double> I(0,1);

    Eigen::Matrix2cd M2 = Eigen::Matrix2cd::Zero();
    M2(0,0) = omega_r;
    M2(0,1) = g;
    M2(1,0) = g;
    M2(1,1) = omega_1;
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
    //std::cout << "es.eigenvalues() = " << es.eigenvalues() << std::endl;
    //std::cout << "es.eigenvectors() = " << std::endl << es.eigenvectors() << std::endl;

    const double common_factor_12 = 0.5*std::sqrt(kappa*gamma2);
    const double common_factor_21 = common_factor_12;
    const double common_factor_11 = 0.5*std::sqrt(kappa*kappa);
    const double common_factor_22 = 0.5*std::sqrt(gamma2*gamma2);

    // Note that omega_1_1 == omega_1_10, and omega_1_2 = omega_1_20,
    // because omega_1_0 == 0. Below, we use the two different names
    // to distinguish between the numerically calculated eigenfrequencies
    // and the analytically calculated ones. Either one can be used.
    const double omega_1_2 = 0.5*(omega_1+omega_r)+std::sqrt(std::pow(0.5*(omega_r-omega_1),2)+std::pow(g,2));
    const double omega_1_1 = 0.5*(omega_1+omega_r)-std::sqrt(std::pow(0.5*(omega_r-omega_1),2)+std::pow(g,2));
    //std::cout << "omega_1_1 = " << omega_1_1 << std::endl;
    //std::cout << "omega_1_2 = " << omega_1_2 << std::endl;

    const double omega_1_10 = es.eigenvalues()(qubit_excited_eigenvector);
    const double omega_1_20 = es.eigenvalues()(qubit_down_eigenvector);
    const double omega_2_10 = omega_2;
    //std::cout << "omega_1_10 = " << omega_1_10 << std::endl;
    //std::cout << "omega_1_20 = " << omega_1_20 << std::endl;

    // Use the qubit frequency (omega_1_1) as the reference one
    // for the the wave vectors k.
    const double omega_ref = omega_1_1;

    const double theta = 0.5*std::arg(0.5*(omega_r-omega_1)+I*g);
    const double C_1_01 = std::sin(theta);
    const double C_1_02 = std::cos(theta);
    const double C_2_01 = 1;

    //std::cout << "C_1_01 = " << C_1_01 << std::endl;
    //std::cout << "C_1_02 = " << C_1_02 << std::endl;

    Eigen::Vector3cd v = Eigen::Vector3cd::Zero(3);
    for (int i = 0; i < 3; ++i) {
        v(i) = v0[i];
    }

    const double dt = t/N_t;

    const double delayTime = M_PI*k0x_2/omega_ref;
    // The maximum length that the signal is delayed by is 2*x_2
    const int64_t delayBufferSizeHalf = ceil(delayTime/dt);
    const int64_t delayBufferSize = 2*delayBufferSizeHalf;
    Eigen::ArrayXcd v0_buf = Eigen::ArrayXcd::Zero(delayBufferSize);
    Eigen::ArrayXcd v1_buf = Eigen::ArrayXcd::Zero(delayBufferSize);
    Eigen::ArrayXcd v2_buf = Eigen::ArrayXcd::Zero(delayBufferSize);

    const std::complex<double> Md00 = -(I*(omega_1_10-omega_ref)+0.5*std::pow(C_1_01,2)*kappa*(omega_1_10/omega_r));
    const std::complex<double> Md01 = -0.5*C_1_01*C_1_02*kappa*(omega_1_20/omega_r);
    const std::complex<double> Md02 = -0.5*C_1_01*C_2_01*std::sqrt(kappa*gamma2)*(omega_2_10/std::sqrt(omega_r*omega_2))*std::exp(I*M_PI*k0x_2);
    const std::complex<double> Md10 = -0.5*C_1_02*C_1_01*kappa*(omega_1_10/omega_r);
    const std::complex<double> Md11 = -(I*(omega_1_20-omega_ref)+0.5*std::pow(C_1_02,2)*kappa*(omega_1_20/omega_r));
    const std::complex<double> Md12 = -0.5*C_1_02*C_2_01*std::sqrt(kappa*gamma2)*(omega_2_10/std::sqrt(omega_r*omega_2))*std::exp(I*M_PI*k0x_2);
    const std::complex<double> Md20 = -0.5*C_2_01*C_1_01*std::sqrt(kappa*gamma2)*(omega_1_10/std::sqrt(omega_r*omega_2))*std::exp(I*M_PI*k0x_2);
    const std::complex<double> Md21 = -0.5*C_2_01*C_1_02*std::sqrt(kappa*gamma2)*(omega_1_20/std::sqrt(omega_r*omega_2))*std::exp(I*M_PI*k0x_2);
    const std::complex<double> Md22 = -(I*(omega_2_10-omega_ref)+0.25*std::pow(C_2_01,2)*gamma2*(omega_2_10/omega_2));
    const std::complex<double> Md22_delay = -0.25*std::pow(C_2_01,2)*gamma2*(omega_2_10/omega_2)*std::exp(2.0*I*M_PI*k0x_2);

    Eigen::Vector3cd k1 = v;
    const int64_t N_t_reduced = N_t/data_reduce_factor;
    ret.time = std::vector<double>(N_t_reduced, 0);
    ret.Omega_Re = std::vector<double>(N_t_reduced, 0);
    ret.Omega_Im = std::vector<double>(N_t_reduced, 0);
    ret.res_populations.resize(1);
    ret.res_populations[0] = std::vector<double>(N_t_reduced, 0);
    ret.aa_populations.resize(2);
    ret.aa_populations[0] = std::vector<double>(N_t_reduced, 0);
    ret.aa_populations[1] = std::vector<double>(N_t_reduced, 0);
    ret.F.resize(1);
    ret.F[0] = std::vector<double>(N_t_reduced, 0);
    ret.F_down.resize(1);
    ret.F_down[0] = std::vector<double>(N_t_reduced, 0);
    ret.tilde_F = std::vector<double>(N_t_reduced, 0);
    for (int64_t i = 0; i < delayBufferSizeHalf; ++i) {
        if (i % data_reduce_factor == 0) {
            const int64_t dataStoreIndex = i / data_reduce_factor;
            ret.time[dataStoreIndex] = dt*i;
            ret.aa_populations[1][dataStoreIndex] = std::norm(v(2));
            ret.F[0][dataStoreIndex] = std::norm(v(0));
            ret.F_down[0][dataStoreIndex] = std::norm(v(1));
        }
        v0_buf(i) = v(0);
        v1_buf(i) = v(1);
        v2_buf(i) = v(2);
        k1(0) = Md00*v(0)+Md01*v(1);
        k1(1) = Md10*v(0)+Md11*v(1);
        k1(2) = Md22*v(2);
        v += k1*dt;
    }
    for (int64_t i = delayBufferSizeHalf; i < delayBufferSize; ++i) {
        if (i % data_reduce_factor == 0) {
            const int64_t dataStoreIndex = i / data_reduce_factor;
            ret.time[dataStoreIndex] = dt*i;
            ret.aa_populations[1][dataStoreIndex] = std::norm(v(2));
            ret.F[0][dataStoreIndex] = std::norm(v(0));
            ret.F_down[0][dataStoreIndex] = std::norm(v(1));
        }
        v0_buf(i) = v(0);
        v1_buf(i) = v(1);
        v2_buf(i) = v(2);
        k1(0) = Md00*v(0)+Md01*v(1)+Md02*v2_buf(i-delayBufferSizeHalf);
        k1(1) = Md10*v(0)+Md11*v(1)+Md12*v2_buf(i-delayBufferSizeHalf);
        k1(2) = Md20*v0_buf(i-delayBufferSizeHalf)+Md21*v1_buf(i-delayBufferSizeHalf)+Md22*v(2);
        v += k1*dt;
    }
    // Below, we implement a ring buffer by using module operators '%'
    // The idea is to avoid moving the memory. As an example, consider
    // delayBufferSize = 4. Below, we use the following arrow to denote
    // loadIndex:
    //
    // ^
    // |
    //
    // and the following arrow to denote loadIndexHalf:
    //
    // ^
    // H
    //
    // Then the two different approaches can be written
    // in the following as in an abbreviated form
    // (the numbers 0, 1, etc. stand for the v*_buf elements)
    //
    // Iteration index | memmove | modulo
    // i = 4           | 0 1 2 3 | 0 1 2 3
    //                   ^   ^     ^   ^
    //                   |   H     |   H
    //
    // i = 5           | 1 2 3 4 | 4 1 2 3
    //                   ^   ^       ^   ^
    //                   |   H       |   H
    //
    // i = 6           | 2 3 4 5 | 4 5 2 3
    //                   ^   ^     ^   ^
    //                   |   H     H   |
    //
    // i = 7           | 3 4 4 5 | 4 5 2 3
    //                   ^   ^       ^   ^
    //                   |   H       H   |
    //
    // etc.
    for (int64_t i = delayBufferSize; i < N_t; ++i) {
        if (i % data_reduce_factor == 0) {
            const int64_t dataStoreIndex = i / data_reduce_factor;
            ret.time[dataStoreIndex] = dt*i;
            ret.aa_populations[1][dataStoreIndex] = std::norm(v(2));
            ret.F[0][dataStoreIndex] = std::norm(v(0));
            ret.F_down[0][dataStoreIndex] = std::norm(v(1));
        }
        const int64_t loadIndexHalf = (i-delayBufferSizeHalf)%delayBufferSize;
        const int64_t loadIndex = i%delayBufferSize;
        k1(0) = Md00*v(0)+Md01*v(1)+Md02*v2_buf(loadIndexHalf);
        k1(1) = Md10*v(0)+Md11*v(1)+Md12*v2_buf(loadIndexHalf);
        k1(2) = Md20*v0_buf(loadIndexHalf)+Md21*v1_buf(loadIndexHalf)+Md22*v(2)+Md22_delay*v2_buf(loadIndex);
        const int64_t storeIndex = loadIndex;
        v0_buf(storeIndex) = v(0);
        v1_buf(storeIndex) = v(1);
        v2_buf(storeIndex) = v(2);
        v += k1*dt;
    }
    return ret;
}

JQFData jqf_data_qubit_decay_dde(
        double kappa, double gamma2, double g, double omega_d, double omega_r,
        double omega_1, double omega_2, double k0x_r, double k0x_2,
        bool startInEigenstate, double t, int64_t N_t,
        int64_t data_reduce_factor)
{
    assert(N_t > 0 && "Number of steps needs to be a positive integer!");
    assert(startInEigenstate && "Only starting in eigenstate is supported!");

    std::vector<std::complex<double>> v(3, 0);
    // The first component represents the amplitude of the
    // eigenstate with the largest part in the atom
    v[0] = 1;
    v[1] = 0;
    v[2] = 0;

    return jqf_data_qubit_decay_dde_arbitrary_state(
            kappa, gamma2, g, omega_d, omega_r, omega_1, omega_2, k0x_r, k0x_2,
            v, t, N_t, data_reduce_factor);
}

JQFData jqf_data_qubit_decay_Hamiltonian_arbitrary_state(
        double kappa, double gamma2, double g, double omega_d, double omega_r,
        double omega_1, double omega_2, double k0x_r, double k0x_2,
        const std::vector<std::complex<double>> &v0, double t, int64_t N_t,
        int64_t data_reduce_factor)
{
    assert(N_t > 0 && "Number of steps needs to be a positive integer!");
    assert(v0.size() == 3 && "v0 should have size 3!");
    const double Delta_r = omega_d-omega_r;
    const double Delta_1 = omega_d-omega_1;
    const double Delta_2 = omega_d-omega_2;
    JQFData ret;
    const std::complex<double> I(0,1);

    Eigen::Matrix2cd M2 = Eigen::Matrix2cd::Zero();
    M2(0,0) = omega_r;
    M2(0,1) = g;
    M2(1,0) = g;
    M2(1,1) = omega_1;
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
    std::cout << "es.eigenvalues() = " << es.eigenvalues() << std::endl;
    std::cout << "es.eigenvectors() = " << std::endl << es.eigenvectors() << std::endl;

    const double common_factor_12 = 0.5*std::sqrt(kappa*gamma2);
    const double common_factor_21 = common_factor_12;
    const double common_factor_11 = 0.5*std::sqrt(kappa*kappa);
    const double common_factor_22 = 0.5*std::sqrt(gamma2*gamma2);

    const double eigenvalue0 = 0.5*(omega_1+omega_r)+std::sqrt(std::pow(0.5*(omega_r-omega_1),2)+std::pow(g,2));
    const double eigenvalue1 = 0.5*(omega_1+omega_r)-std::sqrt(std::pow(0.5*(omega_r-omega_1),2)+std::pow(g,2));
    std::cout << "eigenvalue0 = " << eigenvalue0 << std::endl;
    std::cout << "eigenvalue1 = " << eigenvalue1 << std::endl;

    const double omega_1_10 = es.eigenvalues()(qubit_excited_eigenvector);
    const double omega_1_20 = es.eigenvalues()(qubit_down_eigenvector);
    const double omega_2_10 = omega_2;
    //const double omega_1_10 = omega_1;
    //const double omega_1_20 = omega_1;
    //const double omega_2_10 = omega_1;
    std::cout << "omega_1_10 = " << omega_1_10 << std::endl;
    std::cout << "omega_1_20 = " << omega_1_20 << std::endl;

    const double omega_ref = omega_1;

    const double k_factor_1_10 = omega_1_10/omega_ref;
    const double k_factor_1_20 = omega_1_20/omega_ref;
    const double k_factor_2_10 = omega_2_10/omega_ref;

    const std::complex<double> xi_11_10 = common_factor_11*omega_1_10/std::sqrt(omega_r*omega_r)*exp_factor(k_factor_1_10, k0x_r, k0x_r);
    const std::complex<double> xi_11_20 = common_factor_11*omega_1_20/std::sqrt(omega_r*omega_r)*exp_factor(k_factor_1_20, k0x_r, k0x_r);
    const std::complex<double> xi_21_10 = common_factor_21*omega_1_10/std::sqrt(omega_2*omega_r)*exp_factor(k_factor_1_10, k0x_2, k0x_r);
    const std::complex<double> xi_21_20 = common_factor_21*omega_1_20/std::sqrt(omega_2*omega_r)*exp_factor(k_factor_1_20, k0x_2, k0x_r);
    const std::complex<double> xi_12_10 = common_factor_12*omega_2_10/std::sqrt(omega_r*omega_2)*exp_factor(k_factor_2_10, k0x_r, k0x_2);
    const std::complex<double> xi_22_10 = common_factor_22*omega_2_10/std::sqrt(omega_2*omega_2)*exp_factor(k_factor_2_10, k0x_2, k0x_2);

    const double theta = 0.5*std::arg(0.5*(omega_r-omega_1)+I*g);
    const double C_1_01 = std::sin(theta);
    const double C_1_02 = std::cos(theta);
    const double C_2_01 = 1;

    std::cout << "xi_11_10 = " << xi_11_10 << std::endl;
    std::cout << "xi_11_20 = " << xi_11_20 << std::endl;
    std::cout << "xi_21_10 = " << xi_21_10 << std::endl;
    std::cout << "xi_21_20 = " << xi_21_20 << std::endl;
    std::cout << "xi_12_10 = " << xi_12_10 << std::endl;
    std::cout << "xi_22_10 = " << xi_22_10 << std::endl;
    std::cout << "C_1_01 = " << C_1_01 << std::endl;
    std::cout << "C_1_02 = " << C_1_02 << std::endl;

    Eigen::Vector2cd down_eigenstate = get_down_eigenstate(g, Delta_r, Delta_1);

    Eigen::Matrix3cd M = Eigen::Matrix3cd::Zero();
    M(0,0) = -(I*(omega_1_10-omega_d)+0.5*std::pow(C_1_01,2)*xi_11_10);
    M(0,1) = -0.5*C_1_01*C_1_02*xi_11_20;
    M(0,2) = -0.5*C_1_01*C_2_01*xi_12_10;
    M(1,0) = -0.5*C_1_02*C_1_01*xi_11_10;
    M(1,1) = -(I*(omega_1_20-omega_d)+0.5*std::pow(C_1_02,2)*xi_11_20);
    M(1,2) = -0.5*C_1_02*C_2_01*xi_12_10;
    M(2,0) = -0.5*C_2_01*C_1_01*xi_21_10;
    M(2,1) = -0.5*C_2_01*C_1_02*xi_21_20;
    M(2,2) = -(I*(omega_2_10-omega_d)+0.5*std::pow(C_2_01,2)*xi_22_10);

    Eigen::Vector3cd v = Eigen::Vector3cd::Zero(3);
    for (int i = 0; i < 3; ++i) {
        v(i) = v0[i];
    }

    const double dt = t/N_t;

    const int64_t N_t_reduced = N_t/data_reduce_factor;
    ret.time = std::vector<double>(N_t_reduced, 0);
    ret.Omega_Re = std::vector<double>(N_t_reduced, 0);
    ret.Omega_Im = std::vector<double>(N_t_reduced, 0);
    ret.res_populations.resize(1);
    ret.res_populations[0] = std::vector<double>(N_t_reduced, 0);
    ret.aa_populations.resize(2);
    ret.aa_populations[0] = std::vector<double>(N_t_reduced, 0);
    ret.aa_populations[1] = std::vector<double>(N_t_reduced, 0);
    ret.F.resize(1);
    ret.F[0] = std::vector<double>(N_t_reduced, 0);
    ret.F_down.resize(1);
    ret.F_down[0] = std::vector<double>(N_t_reduced, 0);
    ret.tilde_F = std::vector<double>(N_t_reduced, 0);
    for (int64_t i = 0; i < N_t; ++i) {
        if (i % data_reduce_factor == 0) {
            const int64_t dataStoreIndex = i / data_reduce_factor;
            ret.time[dataStoreIndex] = dt*i;
            //ret.res_populations[0][dataStoreIndex] = std::norm(v(0));
            //ret.aa_populations[0][dataStoreIndex] = std::norm(v(1));
            ret.aa_populations[1][dataStoreIndex] = std::norm(v(2));
            ret.F[0][dataStoreIndex] = std::norm(v(0));
            ret.F_down[0][dataStoreIndex] = std::norm(v(1));
        }
        Eigen::Vector3cd k1 = M*v*dt;
        Eigen::Vector3cd k2 = M*(v+k1*0.5)*dt;
        Eigen::Vector3cd k3 = M*(v+k2*0.5)*dt;
        Eigen::Vector3cd k4 = M*(v+k3)*dt;
        v = v + (k1/6) + (k2/3) + (k3/3) + (k4/6);
    }
    return ret;
}

JQFData jqf_data_qubit_decay_Hamiltonian(
        double kappa, double gamma2, double g, double omega_d, double omega_r,
        double omega_1, double omega_2, double k0x_r, double k0x_2,
        bool startInEigenstate, double t, int64_t N_t,
        int64_t data_reduce_factor)
{
    assert(N_t > 0 && "Number of steps needs to be a positive integer!");
    assert(startInEigenstate && "Only starting in eigenstate is supported!");

    std::vector<std::complex<double>> v(3, 0);
    // The first component represents the amplitude of the
    // eigenstate with the largest part in the atom
    v[0] = 1;
    v[1] = 0;
    v[2] = 0;

    return jqf_data_qubit_decay_Hamiltonian_arbitrary_state(
            kappa, gamma2, g, omega_d, omega_r, omega_1, omega_2, k0x_r, k0x_2,
            v, t, N_t, data_reduce_factor);
}

JQFData jqf_simulation(
        AtomResonatorInitialState initialState,
        double kappa, double gamma2,
        const std::vector<double> &gammaInternal,
        const std::vector<double> &gammaDephasing,
        double g, double Omega,
        double omega_d, double omega_r, double omega_1, double omega_2,
        const std::vector<double> &transmon_anharmonicity,
        double k0x_r, double k0x_2, int num_excitations,
        const std::vector<int> &transmon_excitations, double t, int64_t N_t,
        int64_t data_reduce_factor)
{
    assert(N_t > 0 && "Number of steps needs to be a positive integer!");
    JQFData ret;
    const int flags = 0;
    const double J_x = 0;
    const double k0x_out = k0x_2;
    const double kappaInternal = 0;
    JQFSuperoperatorData sd = generate_superoperator_diag(
            {kappa}, {kappaInternal}, {gamma2}, gammaInternal,
            gammaDephasing, g, Omega, J_x, omega_d, {omega_r},
            {omega_1, omega_2}, transmon_anharmonicity, {k0x_r}, {k0x_2},
            k0x_out, num_excitations, transmon_excitations, flags);
    std::vector<BasisVector> basis = std::move(sd.basis);
    const int basis_size = basis.size();
    const int basis_size_squared = basis_size*basis_size;
    Eigen::MatrixXcd rho;
    switch (initialState) {
    case AtomResonatorInitialState::AllDown:
    {
        Eigen::VectorXcd psi = Eigen::VectorXcd::Zero(basis_size);
        psi(0) = 1; // atoms and the resonator are in their ground states
        rho = psi*psi.adjoint();
    }
    break;
    case AtomResonatorInitialState::AtomUp:
    {
        const std::vector<int> excited_atom_indices = {0};
        Eigen::VectorXcd psi = get_psi_for_excited_atoms(
                excited_atom_indices, basis);
        rho = psi*psi.adjoint();
    }
    break;
    case AtomResonatorInitialState::EigenstateUp:
    {
        rho = sd.psi_up[0]*sd.psi_up[0].adjoint();
    }
    break;
    case AtomResonatorInitialState::EigenstateDown:
    {
        rho = sd.psi_down[0]*sd.psi_down[0].adjoint();
    }
    break;
    default:
        assert(0 && "Unknown initial state specification!");
    }

    Eigen::VectorXcd rho_vec(basis_size*basis_size);
    for (int i = 0; i < basis_size; ++i) {
        for (int j = 0; j < basis_size; ++j) {
            rho_vec(i*basis_size+j) = rho(i,j);
        }
    }
    Eigen::MatrixXcd rho_target = sd.psi_up[0]*sd.psi_up[0].adjoint();
    Eigen::VectorXcd rho_vec_target(basis_size*basis_size);
    for (int i = 0; i < basis_size; ++i) {
        for (int j = 0; j < basis_size; ++j) {
            rho_vec_target(i*basis_size+j) = rho_target(i,j);
        }
    }

#ifndef DO_NOT_DO_PARTIAL_TRACE
    const double Delta_r = omega_d-omega_r;
    const double Delta_1 = omega_d-omega_1;
    const double Delta_2 = omega_d-omega_2;
    Eigen::VectorXcd up_eigenstate = get_up_eigenstate(g, Delta_r, Delta_1);
    Eigen::VectorXcd down_eigenstate = get_down_eigenstate(g, Delta_r, Delta_1);
    // Trace out the second atom (index 1)
    PartialTraceSpecification pts = make_partial_trace_specification(basis, 1);
    Eigen::VectorXcd psi_up = get_psi_for_eigenstate_2(up_eigenstate, pts.reduced_basis);
    Eigen::VectorXcd psi_down = get_psi_for_eigenstate_2(down_eigenstate, pts.reduced_basis);
    // Additionally trace out the resonator (index 1 in the reduced basis)
    PartialTraceSpecification pts2 = make_partial_trace_specification(pts.reduced_basis, 1);
    const int reduced_size = pts.reduced_basis.size();
    Eigen::MatrixXcd rho_reduced
            = Eigen::MatrixXcd::Zero(reduced_size, reduced_size);
    Eigen::MatrixXcd rho_reduced_squared
            = Eigen::MatrixXcd::Zero(reduced_size, reduced_size);
    const int aa_basis_size = pts2.reduced_basis.size();
    Eigen::MatrixXcd rho_aa
            = Eigen::MatrixXcd::Zero(aa_basis_size, aa_basis_size);
    Eigen::MatrixXcd rho_aa_squared
            = Eigen::MatrixXcd::Zero(aa_basis_size, aa_basis_size);
#endif // DO_NOT_DO_PARTIAL_TRACE

    const double dt = t/N_t;

    ret.time = std::vector<double>(N_t, 0);
    ret.Omega_Re = std::vector<double>(N_t, Omega);
    ret.Omega_Im = std::vector<double>(N_t, 0);
    ret.res_populations.resize(1);
    ret.res_populations[0] = std::vector<double>(N_t, 0);
    ret.aa_populations.resize(2);
    ret.aa_populations[0] = std::vector<double>(N_t, 0);
    ret.aa_populations[1] = std::vector<double>(N_t, 0);
    ret.purity_1 = std::vector<double>(N_t, 0);
    ret.purity_1r = std::vector<double>(N_t, 0);
    ret.F.resize(1);
    ret.F[0] = std::vector<double>(N_t, 0);
    ret.F_down.resize(1);
    ret.F_down[0] = std::vector<double>(N_t, 0);
    ret.tilde_F = std::vector<double>(N_t, 0);
#ifdef USE_ROCM
    SparseMatrixData rocL = smd_from_spmat(sd.L);
    std::vector<SparseMatrixData> M_operators;
    M_operators.push_back(smd_from_spmat(sd.M_n_res[0]));
    M_operators.push_back(smd_from_spmat(sd.M_n_atom[0]));
    M_operators.push_back(smd_from_spmat(sd.M_n_atom[1]));
    M_operators.push_back(smd_from_spmat(sd.M_psi_up[0]));
    M_operators.push_back(smd_from_spmat(sd.M_psi_down[0]));
    std::vector<const double*> M_diag_operators;
    std::vector<const std::complex<double>*> fidelity_rhos;
    fidelity_rhos.push_back(rho_vec_target.data());
    const int64_t iterationsBetweenDeviceSynchronize = 1e4;
    MasterEquationData data = evolve_master_equation_roc(
            basis_size, rocL, rho_vec.data(), M_operators, M_diag_operators,
            fidelity_rhos, dt, N_t, iterationsBetweenDeviceSynchronize);
    for (int64_t i = 0; i < N_t; ++i) {
        ret.time[i] = data.time[i];
        ret.res_populations[0][i] = data.M_values[0][i].real();
        ret.aa_populations[0][i] = data.M_values[1][i].real();
        ret.aa_populations[1][i] = data.M_values[2][i].real();
        ret.F[0][i] = data.M_values[3][i].real();
        ret.F_down[0][i] = data.M_values[4][i].real();
        ret.tilde_F[i] = data.fidelities[0][i].real();
    }
#else // USE_ROCM
#ifdef EIGEN_USE_MKL_ALL
    MKLSparseMatrix mklL(sd.L);
#endif // EIGEN_USE_MKL_ALL
    Eigen::VectorXcd k1 = Eigen::VectorXcd::Zero(basis_size_squared);
    Eigen::VectorXcd k2 = Eigen::VectorXcd::Zero(basis_size_squared);
    Eigen::VectorXcd k3 = Eigen::VectorXcd::Zero(basis_size_squared);
    Eigen::VectorXcd k4 = Eigen::VectorXcd::Zero(basis_size_squared);
    Eigen::VectorXcd temp = Eigen::VectorXcd::Zero(basis_size_squared);
    for (int64_t i = 0; i < N_t; ++i) {
        Eigen::Map<MatrixXcdRowMajor> rho_map(rho_vec.data(), basis_size, basis_size);
        const std::complex<double> res_population = trace_of_product(sd.M_n_res[0],rho_map);
        const std::complex<double> population_1 = trace_of_product(sd.M_n_atom[0],rho_map);
        const std::complex<double> population_2 = trace_of_product(sd.M_n_atom[1],rho_map);
#ifdef DO_NOT_DO_PARTIAL_TRACE
        const std::complex<double> F = trace_of_product(sd.M_psi_up[0],rho_map);
        const std::complex<double> F_down = trace_of_product(sd.M_psi_down[0],rho_map);
        const std::complex<double> tilde_F = sd.psi_up[0].adjoint()*rho_map*sd.psi_up[0];
#else // DO_NOT_DO_PARTIAL_TRACE
        rho_reduced = partial_trace(rho_map, pts);
        const std::complex<double> F = psi_up.adjoint()*rho_reduced*psi_up;
        const std::complex<double> F_down = psi_down.adjoint()*rho_reduced*psi_down;
        rho_reduced_squared = rho_reduced*rho_reduced;
        const std::complex<double> purity_1r = rho_reduced_squared.trace();
        rho_aa = partial_trace(rho_reduced, pts2);
        rho_aa_squared = rho_aa*rho_aa;
        const std::complex<double> purity_1 = rho_aa_squared.trace();
#endif // DO_NOT_DO_PARTIAL_TRACE
        ret.time[i] = dt*(i+1);
        ret.res_populations[0][i] = res_population.real();
        ret.aa_populations[0][i] = population_1.real();
        ret.aa_populations[1][i] = population_2.real();
#ifndef DO_NOT_DO_PARTIAL_TRACE
        ret.purity_1[i] = purity_1.real();
        ret.purity_1r[i] = purity_1r.real();
#endif // DO_NOT_DO_PARTIAL_TRACE
        ret.F[0][i] = F.real();
        ret.F_down[0][i] = F_down.real();
        ret.tilde_F[i] = tilde_F.real();
#ifdef EIGEN_USE_MKL_ALL
        rk4_step_mkl(rho_vec, mklL, k1, k2, k3, k4, temp, dt);
#else // EIGEN_USE_MKL_ALL
        rk4_step(rho_vec, sd.L, k1, k2, k3, k4, temp, dt);
#endif // EIGEN_USE_MKL_ALL
    }
#endif // USE_ROCM
    return ret;
}

JQFData control_with_jqf_time_dependent_Omega(
        std::complex<double> initial_excited_state_amplitude,
        std::complex<double> target_excited_state_amplitude,
        double kappa, double gamma2, const std::vector<double> &gammaInternal,
        const std::vector<double> &gammaDephasing, double g,
        std::function<std::complex<double>(double)> Omega,
        double omega_d, double omega_r, double omega_1, double omega_2,
        const std::vector<double> &transmon_anharmonicity,
        double k0x_r, double k0x_2, int num_excitations,
        const std::vector<int> &transmon_excitations, double t_final,
        int64_t N_t, int64_t data_reduce_factor)
{
    const double J_x = 0;
    InitialFinalStateSpec stateSpec;
    stateSpec.initial = initial_excited_state_amplitude;
    stateSpec.target = target_excited_state_amplitude;
    return jqf_time_dependent_Omega(
                stateSpec, {kappa}, {gamma2}, gammaInternal, gammaDephasing, g,
                J_x, Omega, omega_d, {omega_r}, {omega_1, omega_2},
                transmon_anharmonicity, {k0x_r}, {k0x_2}, num_excitations,
                transmon_excitations, t_final, N_t, data_reduce_factor);
}

double control_with_jqf_time_dependent_Omega_avg_fidelity(
        double kappa, double gamma2, const std::vector<double> &gammaInternal,
        const std::vector<double> &gammaDephasing, double g,
        std::function<std::complex<double>(double)> Omega,
        double omega_d, double omega_r, double omega_1, double omega_2,
        const std::vector<double> &transmon_anharmonicity,
        double k0x_r, double k0x_2, int num_excitations,
        const std::vector<int> &transmon_excitations, double t_final,
        int64_t N_t, int64_t data_reduce_factor)
{
    const double J_x = 0;
    std::vector<InitialFinalStateSpec> stateSpec = initializeStateSpecToAverageSigmaX();
    const int num_states = stateSpec.size();
    const bool useRhoSpec = isUsingRhoSpec(stateSpec);
    double fidelity_sum = 0;
    for (int i = 0; i < num_states; ++i) {
        JQFData data = jqf_time_dependent_Omega(
                    stateSpec[i], {kappa}, {gamma2}, gammaInternal,
                    gammaDephasing, g, J_x, Omega, omega_d, {omega_r},
                    {omega_1, omega_2}, transmon_anharmonicity, {k0x_r},
                    {k0x_2}, num_excitations, transmon_excitations, t_final,
                    N_t, data_reduce_factor);
        double last_fidelity = data.tilde_F[N_t-1];
        if (useRhoSpec && i == 0) {
            // See Eq. (7) of Bowdrey et al. [Phys. Lett. A 294 (2002) 258]).
            // The term where the identity (sigma_0) is propagated, is 3 times larger than
            // the other terms. Also note that we cannot make the exact replacement of
            // this term with 1/2, because the "identity" extended to a larger Hilbert
            // space by filling in zeros in the other elements is not an exact identity
            // anymore.
            fidelity_sum += 3*last_fidelity;
        } else {
            fidelity_sum += last_fidelity;
        }
    }
    if (useRhoSpec) {
        assert(num_states == 4 && "Only the case of 3 Pauli matrices plus identity is supported!");
        return fidelity_sum/12;
    } else {
        return fidelity_sum/num_states;
    }
}

JQFData control_with_jqf_time_dependent_omegad_Omega(
        std::complex<double> initial_excited_state_amplitude,
        std::complex<double> target_excited_state_amplitude,
        double kappa, double gamma2, const std::vector<double> &gammaInternal,
        const std::vector<double> &gammaDephasing, double g,
        std::function<std::complex<double>(double)> Omega,
        std::function<double(double)> omega_d, double omega_r, double omega_1,
        double omega_2, const std::vector<double> &transmon_anharmonicity,
        double k0x_r, double k0x_2, int num_excitations,
        const std::vector<int> &transmon_excitations, double t_final,
        int64_t N_t, int64_t data_reduce_factor)
{
    const double J_x = 0;
    InitialFinalStateSpec stateSpec;
    stateSpec.initial = initial_excited_state_amplitude;
    stateSpec.target = target_excited_state_amplitude;
    return jqf_time_dependent_omegad_Omega(
                stateSpec, {kappa}, {gamma2}, gammaInternal, gammaDephasing, g,
                J_x, Omega, omega_d, {omega_r}, {omega_1, omega_2},
                transmon_anharmonicity, {k0x_r}, {k0x_2}, num_excitations,
                transmon_excitations, t_final, N_t, data_reduce_factor);
}

double control_with_jqf_time_dependent_omegad_Omega_avg_fidelity(
        double kappa, double gamma2, const std::vector<double> &gammaInternal,
        const std::vector<double> &gammaDephasing, double g,
        std::function<std::complex<double>(double)> Omega,
        std::function<double(double)> omega_d, double omega_r, double omega_1,
        double omega_2, const std::vector<double> &transmon_anharmonicity,
        double k0x_r, double k0x_2, int num_excitations,
        const std::vector<int> &transmon_excitations, double t_final,
        int64_t N_t, int64_t data_reduce_factor)
{
    const double J_x = 0;
    std::vector<InitialFinalStateSpec> stateSpec = initializeStateSpecToAverageSigmaX();
    const int num_states = stateSpec.size();
    const bool useRhoSpec = isUsingRhoSpec(stateSpec);
    double fidelity_sum = 0;
    for (int i = 0; i < num_states; ++i) {
        JQFData data = jqf_time_dependent_omegad_Omega(
                    stateSpec[i], {kappa}, {gamma2}, gammaInternal,
                    gammaDephasing, g, J_x, Omega, omega_d, {omega_r},
                    {omega_1, omega_2}, transmon_anharmonicity, {k0x_r},
                    {k0x_2}, num_excitations, transmon_excitations, t_final,
                    N_t, data_reduce_factor);
        double last_fidelity = data.tilde_F[N_t-1];
        if (useRhoSpec && i == 0) {
            // See Eq. (7) of Bowdrey et al. [Phys. Lett. A 294 (2002) 258]).
            // The term where the identity (sigma_0) is propagated, is 3 times larger than
            // the other terms. Also note that we cannot make the exact replacement of
            // this term with 1/2, because the "identity" extended to a larger Hilbert
            // space by filling in zeros in the other elements is not an exact identity
            // anymore.
            fidelity_sum += 3*last_fidelity;
        } else {
            fidelity_sum += last_fidelity;
        }
    }
    if (useRhoSpec) {
        assert(num_states == 4 && "Only the case of 3 Pauli matrices plus identity is supported!");
        return fidelity_sum/12;
    } else {
        return fidelity_sum/num_states;
    }
}

inline double f_sigmoid(double x)
{
    const double e = std::exp(-x);
    return 1.0/(1.0+e);
}

//#define OPTIMIZE_WITH_BOUNDED_OMEGA

JQFData control_with_jqf_array_complex_Omega(
        double kappa, double gamma2, double g,
        const std::vector<std::complex<double>> &Omega,
        double omega_d, double omega_r, double omega_1, double omega_2,
        const std::vector<double> &transmon_anharmonicity,
        double k0x_r, double k0x_2, int num_excitations,
        const std::vector<int> &transmon_excitations, double t, int64_t N_t,
        int64_t data_reduce_factor)
{
    assert(N_t > 0 && "Number of steps needs to be a positive integer!");
    assert(int(Omega.size()) == N_t && "Delta_drive array has wrong size!");
    JQFData ret;
    int flags = 0;
    flags |= JQF_SUPEROPERATOR_OMIT_OMEGA_TERMS;
    const double unused_Omega = 0;
    const double J_x = 0;
    const double k0x_out = k0x_2;
    const double kappaInternal = 0;
    const double gamma2Internal = 0;
    const double gamma2Dephasing = 0;
    JQFSuperoperatorData sd = generate_superoperator_diag(
            {kappa}, {kappaInternal}, {gamma2}, {0, gamma2Internal},
            {0, gamma2Dephasing}, g, unused_Omega, J_x, omega_d, {omega_r},
            {omega_1, omega_2}, transmon_anharmonicity, {k0x_r}, {k0x_2},
            k0x_out, num_excitations, transmon_excitations, flags);
    std::vector<BasisVector> basis = std::move(sd.basis);
    const int basis_size = basis.size();
    const int basis_size_squared = basis_size*basis_size;

    Eigen::VectorXcd psi = Eigen::VectorXcd::Zero(basis_size);
    psi(0) = 1; // atoms and the resonator are in their ground states
    Eigen::MatrixXcd rho = psi*psi.adjoint();
    //Eigen::MatrixXcd rho = sd.psi_up[0]*sd.psi_up[0].adjoint();

    Eigen::VectorXcd rho_vec(basis_size*basis_size);
    for (int i = 0; i < basis_size; ++i) {
        for (int j = 0; j < basis_size; ++j) {
            rho_vec(i*basis_size+j) = rho(i,j);
        }
    }

#ifndef DO_NOT_DO_PARTIAL_TRACE
    const double Delta_r = omega_d-omega_r;
    const double Delta_1 = omega_d-omega_1;
    const double Delta_2 = omega_d-omega_2;
    Eigen::VectorXcd up_eigenstate = get_up_eigenstate(g, Delta_r, Delta_1);
    Eigen::VectorXcd down_eigenstate = get_down_eigenstate(g, Delta_r, Delta_1);
    // Trace out the second atom (index 1)
    PartialTraceSpecification pts = make_partial_trace_specification(basis, 1);
    Eigen::VectorXcd psi_up = get_psi_for_eigenstate_2(up_eigenstate, pts.reduced_basis);
    Eigen::VectorXcd psi_down = get_psi_for_eigenstate_2(down_eigenstate, pts.reduced_basis);
    // Additionally trace out the resonator (index 1 in the reduced basis)
    PartialTraceSpecification pts2 = make_partial_trace_specification(pts.reduced_basis, 1);
    const int reduced_size = pts.reduced_basis.size();
    Eigen::MatrixXcd rho_reduced
            = Eigen::MatrixXcd::Zero(reduced_size, reduced_size);
    Eigen::MatrixXcd rho_reduced_squared
            = Eigen::MatrixXcd::Zero(reduced_size, reduced_size);
    const int aa_basis_size = pts2.reduced_basis.size();
    Eigen::MatrixXcd rho_aa
            = Eigen::MatrixXcd::Zero(aa_basis_size, aa_basis_size);
    Eigen::MatrixXcd rho_aa_squared
            = Eigen::MatrixXcd::Zero(aa_basis_size, aa_basis_size);
#endif // DO_NOT_DO_PARTIAL_TRACE

    const double dt = t/N_t;

    SpMat Identity(basis_size, basis_size);
    Identity.setIdentity();
#ifdef EIGEN_USE_MKL_ALL
    MKLSparseMatrix mklL(sd.L);
#endif // EIGEN_USE_MKL_ALL

    SpMat H_Omega(basis_size, basis_size);
    const double Omega_factor_r = sd.Omega_factors[0];
    const double Omega_factor_2 = sd.Omega_factors[1];
    Eigen::VectorXcd k1 = Eigen::VectorXcd::Zero(basis_size_squared);
    Eigen::VectorXcd k2 = Eigen::VectorXcd::Zero(basis_size_squared);
    Eigen::VectorXcd k3 = Eigen::VectorXcd::Zero(basis_size_squared);
    Eigen::VectorXcd k4 = Eigen::VectorXcd::Zero(basis_size_squared);
    Eigen::VectorXcd temp = Eigen::VectorXcd::Zero(basis_size_squared);
    ret.time = std::vector<double>(N_t, 0);
    ret.Omega_Re = std::vector<double>(N_t, 0);
    ret.Omega_Im = std::vector<double>(N_t, 0);
    ret.res_populations.resize(1);
    ret.res_populations[0] = std::vector<double>(N_t, 0);
    ret.aa_populations.resize(2);
    ret.aa_populations[0] = std::vector<double>(N_t, 0);
    ret.aa_populations[1] = std::vector<double>(N_t, 0);
    ret.purity_1 = std::vector<double>(N_t, 0);
    ret.purity_1r = std::vector<double>(N_t, 0);
    ret.F.resize(1);
    ret.F[0] = std::vector<double>(N_t, 0);
    ret.F_down.resize(1);
    ret.F_down[0] = std::vector<double>(N_t, 0);
    ret.tilde_F = std::vector<double>(N_t, 0);
    for (int64_t i = 0; i < N_t; ++i) {
        H_Omega.setZero();
        const SpMat temp_2 = Omega_factor_2*(Omega[i]*sd.M_b_adjoint[1]+std::conj(Omega[i])*sd.M_b[1]);
        H_Omega += temp_2;
        const SpMat temp_r = Omega_factor_r*(Omega[i]*sd.M_a_adjoint[0]+std::conj(Omega[i])*sd.M_a[0]);
        H_Omega += temp_r;
#ifdef EIGEN_USE_MKL_ALL
        rk4_step_t_mkl(rho_vec, mklL, H_Omega, 1, k1, k2, k3, k4, temp, dt);
#else // EIGEN_USE_MKL_ALL
        rk4_step_t(rho_vec, sd.L, H_Omega, 1, k1, k2, k3, k4, temp, dt);
#endif // EIGEN_USE_MKL_ALL

        Eigen::Map<MatrixXcdRowMajor> rho_map(rho_vec.data(), basis_size, basis_size);
        const std::complex<double> res_population = trace_of_product(sd.M_n_res[0],rho_map);
        const std::complex<double> population_1 = trace_of_product(sd.M_n_atom[0],rho_map);
        const std::complex<double> population_2 = trace_of_product(sd.M_n_atom[1],rho_map);
#ifdef DO_NOT_DO_PARTIAL_TRACE
        const std::complex<double> F = trace_of_product(sd.M_psi_up[0],rho_map);
        const std::complex<double> F_down = trace_of_product(sd.M_psi_down[0],rho_map);
        const std::complex<double> tilde_F = sd.psi_up[0].adjoint()*rho_map*sd.psi_up[0];
#else // DO_NOT_DO_PARTIAL_TRACE
        rho_reduced = partial_trace(rho_map, pts);
        const std::complex<double> overlap_with_up_eigenstate = psi_up.adjoint()*rho_reduced*psi_up;
        const std::complex<double> overlap_with_down_eigenstate = psi_down.adjoint()*rho_reduced*psi_down;
        rho_reduced_squared = rho_reduced*rho_reduced;
        const std::complex<double> purity_1r = rho_reduced_squared.trace();
        rho_aa = partial_trace(rho_reduced, pts2);
        rho_aa_squared = rho_aa*rho_aa;
        const std::complex<double> purity_1 = rho_aa_squared.trace();
#endif // DO_NOT_DO_PARTIAL_TRACE
        ret.time[i] = dt*(i+1);
        ret.res_populations[0][i] = res_population.real();
        ret.aa_populations[0][i] = population_1.real();
        ret.aa_populations[1][i] = population_2.real();
#ifndef DO_NOT_DO_PARTIAL_TRACE
        ret.purity_1[i] = purity_1.real();
        ret.purity_1r[i] = purity_1r.real();
#endif // DO_NOT_DO_PARTIAL_TRACE
        ret.F[0][i] = F.real();
        ret.F_down[0][i] = F_down.real();
        ret.tilde_F[i] = tilde_F.real();
        ret.Omega_Re[i] = Omega[i].real();
        ret.Omega_Im[i] = Omega[i].imag();
    }
    return ret;
}

JQFData control_with_jqf_optimize_array_complex_Omega(
        double kappa, double gamma2, double g,
        const std::vector<std::complex<double>> &Omega,
        const std::vector<double> &learning_rate, int N_iterations,
        double omega_d, double omega_r, double omega_1, double omega_2,
        const std::vector<double> &transmon_anharmonicity,
        double k0x_r, double k0x_2, int num_excitations,
        const std::vector<int> &transmon_excitations, double t, int64_t N_t,
        int64_t data_reduce_factor)
{
    assert(N_t > 0 && "Number of steps needs to be a positive integer!");
    assert(int(Omega.size()) == N_t && "Delta_drive array has wrong size!");

    // Because generate_superoperator_diag() selects different approximations
    // depending on whether Omega is zero or non-zero, the value has to be
    // non-zero here. Which non-zero value is unimportant, since we also
    // pass the flag JQF_SUPEROPERATOR_OMIT_OMEGA_TERMS that prevents the
    // drive terms being added to the Hamiltonian (we add them manually later).
    const double Omega0 = 1;

    const std::complex<double> I(0,1);
    int flags = 0;
    flags |= JQF_SUPEROPERATOR_OMIT_OMEGA_TERMS;
    const double J_x = 0;
    const double k0x_out = k0x_2;
    const double kappaInternal = 0;
    const double gamma2Internal = 0;
    const double gamma2Dephasing = 0;
    JQFSuperoperatorData sd = generate_superoperator_diag(
            {kappa}, {kappaInternal}, {gamma2}, {0, gamma2Internal},
            {0, gamma2Dephasing}, g, Omega0, J_x, omega_d, {omega_r},
            {omega_1, omega_2}, transmon_anharmonicity, {k0x_r}, {k0x_2},
            k0x_out, num_excitations, transmon_excitations, flags);
    std::vector<BasisVector> basis = std::move(sd.basis);
    SpMat L_adjoint = -sd.L.adjoint();
#ifdef OPTIMIZE_WITH_BOUNDED_OMEGA
    const double Omega_max = 200;
#endif// OPTIMIZE_WITH_BOUNDED_OMEGA
    const int basis_size = basis.size();
    const int basis_size_squared = basis_size*basis_size;

    Eigen::VectorXcd psi = Eigen::VectorXcd::Zero(basis_size);
    psi(0) = 1; // atoms and the resonator are in their ground states
    Eigen::MatrixXcd rho_target = sd.M_psi_up[0];
    Eigen::MatrixXcd rho_initial = psi*psi.adjoint();

    Eigen::VectorXcd rho_vec(basis_size*basis_size);
    for (int i = 0; i < basis_size; ++i) {
        for (int j = 0; j < basis_size; ++j) {
            rho_vec(i*basis_size+j) = rho_initial(i,j);
        }
    }

    const double dt = t/N_t;

    SpMat Identity(basis_size, basis_size);
    Identity.setIdentity();
#ifdef EIGEN_USE_MKL_ALL
    MKLSparseMatrix mklL(sd.L);
    MKLSparseMatrix mklL_adjoint(L_adjoint);
#endif // EIGEN_USE_MKL_ALL

    std::vector<double> Omega_opt_Re(N_t, 0);
    std::vector<double> Omega_opt_Im(N_t, 0);
    for (int64_t i = 0; i < N_t; ++i) {
        Omega_opt_Re[i] = Omega[i].real();
        Omega_opt_Im[i] = Omega[i].imag();
    }
    std::vector<std::complex<double>> Omega_opt_2 = Omega;
#ifdef OPTIMIZE_WITH_BOUNDED_OMEGA
    for (int64_t i = 0; i < N_t; ++i) {
        // Inverse sigmoid function ("logit")
        const double p = Omega[i]/Omega_max;
        Omega_opt[i] = std::log(p/(1.0-p));
    }
#endif // OPTIMIZE_WITH_BOUNDED_OMEGA
    const double Omega_factor_r = sd.Omega_factors[0];
    const double Omega_factor_2 = sd.Omega_factors[1];
    SpMat H_Omega(basis_size, basis_size);
    SpMat H_Delta(basis_size, basis_size);
    SpMat dH_Omega_d_Omega(basis_size, basis_size);

    dH_Omega_d_Omega += Omega_factor_2*(sd.M_b_adjoint[1]+sd.M_b[1]);
    dH_Omega_d_Omega += Omega_factor_r*(sd.M_a_adjoint[0]+sd.M_a[0]);
    SpMat L_dH_Omega_d_Omega
            = -I*(Eigen::kroneckerProduct(dH_Omega_d_Omega, Identity)
                  -Eigen::kroneckerProduct(Identity,
                                           dH_Omega_d_Omega.transpose()));
    SpMat dH_Omega_d_Im_Omega(basis_size, basis_size);
    dH_Omega_d_Im_Omega += I*Omega_factor_2*(sd.M_b_adjoint[1]-sd.M_b[1]);
    dH_Omega_d_Im_Omega += I*Omega_factor_r*(sd.M_a_adjoint[0]-sd.M_a[0]);
    SpMat L_dH_Omega_d_Im_Omega
            = -I*(Eigen::kroneckerProduct(dH_Omega_d_Im_Omega, Identity)
                  -Eigen::kroneckerProduct(Identity,
                                           dH_Omega_d_Im_Omega.transpose()));
    std::vector<double> fidelity_under_optimization(N_iterations, 0);
    double old_trace_final = 0;
    Eigen::VectorXcd k1 = Eigen::VectorXcd::Zero(basis_size_squared);
    Eigen::VectorXcd k2 = Eigen::VectorXcd::Zero(basis_size_squared);
    Eigen::VectorXcd k3 = Eigen::VectorXcd::Zero(basis_size_squared);
    Eigen::VectorXcd k4 = Eigen::VectorXcd::Zero(basis_size_squared);
    Eigen::VectorXcd temp = Eigen::VectorXcd::Zero(basis_size_squared);
    Eigen::MatrixXcd rho_vec_all_t = Eigen::MatrixXcd::Zero(basis_size_squared,N_t);
    for (int n = 0; n < N_iterations; ++n) {
        // Backward propagation
        for (int i = 0; i < basis_size; ++i) {
            for (int j = 0; j < basis_size; ++j) {
                rho_vec(i*basis_size+j) = rho_target(i,j);
            }
        }
        auto start_back = std::chrono::steady_clock::now();
        for (int i = N_t-1; i >= 0; --i) {
#ifdef OPTIMIZE_WITH_BOUNDED_OMEGA
            const double Omega_cur = Omega_max*f_sigmoid(Omega_opt[i]);
#else // OPTIMIZE_WITH_BOUNDED_OMEGA
            const std::complex<double> Omega_cur = std::complex<double>(Omega_opt_Re[i], Omega_opt_Im[i]);
#endif // OPTIMIZE_WITH_BOUNDED_OMEGA
            H_Omega.setZero();
            const SpMat temp_2 = Omega_factor_2*(Omega_cur*sd.M_b_adjoint[1]+std::conj(Omega_cur)*sd.M_b[1]);
            H_Omega += temp_2;
            const SpMat temp_r = Omega_factor_r*(Omega_cur*sd.M_a_adjoint[0]+std::conj(Omega_cur)*sd.M_a[0]);
            H_Omega += temp_r;
#ifdef EIGEN_USE_MKL_ALL
            rk4_step_t_mkl(rho_vec, mklL_adjoint, H_Omega, 1, k1, k2, k3, k4, temp, -dt);
#else // EIGEN_USE_MKL_ALL
            rk4_step_t(rho_vec, L_adjoint, H_Omega, 1, k1, k2, k3, k4, temp, -dt);
#endif // EIGEN_USE_MKL_ALL
            rho_vec_all_t.col(i) = rho_vec;
        }
        auto end_back = std::chrono::steady_clock::now();
        std::chrono::duration<double> diff_back = end_back-start_back;
        //std::cout << "Backward propagation: " << diff_back.count() << " s" << std::endl;
        // Forward propagation
        for (int i = 0; i < basis_size; ++i) {
            for (int j = 0; j < basis_size; ++j) {
                rho_vec(i*basis_size+j) = rho_initial(i,j);
            }
        }
        auto start_forward = std::chrono::steady_clock::now();
        for (int64_t i = 0; i < N_t; ++i) {
#ifdef OPTIMIZE_WITH_BOUNDED_OMEGA
            const double f_cur = f_sigmoid(Omega_opt[i]);
            Eigen::VectorXcd rho_vec_temp = Omega_max*L_dH_Omega_d_Omega*f_cur*(f_cur+1)*rho_vec;
#else // OPTIMIZE_WITH_BOUNDED_OMEGA
            Eigen::VectorXcd rho_vec_temp = L_dH_Omega_d_Omega*rho_vec;
            Eigen::VectorXcd rho_vec_temp_Im = L_dH_Omega_d_Im_Omega*rho_vec;
#endif // OPTIMIZE_WITH_BOUNDED_OMEGA
            Eigen::VectorXcd rho_vec_temp2 = rho_vec_all_t.col(i);
            Eigen::Map<MatrixXcdRowMajor> rho_temp_map(rho_vec_temp.data(), basis_size, basis_size);
            Eigen::Map<MatrixXcdRowMajor> rho_temp_Im_map(rho_vec_temp_Im.data(), basis_size, basis_size);
            Eigen::Map<MatrixXcdRowMajor> rho_temp2_map(rho_vec_temp2.data(), basis_size, basis_size);
            std::complex<double> trace_Re = (rho_temp2_map*rho_temp_map).trace();
            std::complex<double> trace_Im = (rho_temp2_map*rho_temp_Im_map).trace();
            Omega_opt_Re[i] += learning_rate[i]*trace_Re.real();
            Omega_opt_Im[i] += learning_rate[i]*trace_Im.real();
#ifdef OPTIMIZE_WITH_BOUNDED_OMEGA
            const double Omega_cur = Omega_max*f_cur;
#else // OPTIMIZE_WITH_BOUNDED_OMEGA
            const std::complex<double> Omega_cur = std::complex<double>(Omega_opt_Re[i], Omega_opt_Im[i]);
#endif // OPTIMIZE_WITH_BOUNDED_OMEGA

            H_Omega.setZero();
            const SpMat temp_2 = Omega_factor_2*(Omega_cur*sd.M_b_adjoint[1]+std::conj(Omega_cur)*sd.M_b[1]);
            H_Omega += temp_2;
            const SpMat temp_r = Omega_factor_r*(Omega_cur*sd.M_a_adjoint[0]+std::conj(Omega_cur)*sd.M_a[0]);
            H_Omega += temp_r;
#ifdef EIGEN_USE_MKL_ALL
            rk4_step_t_mkl(rho_vec, mklL, H_Omega, 1, k1, k2, k3, k4, temp, dt);
#else // EIGEN_USE_MKL_ALL
            rk4_step_t(rho_vec, sd.L, H_Omega, 1, k1, k2, k3, k4, temp, dt);
#endif // EIGEN_USE_MKL_ALL
        }
        auto end_forward = std::chrono::steady_clock::now();
        std::chrono::duration<double> diff_forward = end_forward-start_forward;
        //std::cout << "Forward propagation: " << diff_forward.count() << " s" << std::endl;
#ifdef OPTIMIZE_WITH_BOUNDED_OMEGA
        for (int64_t i = 0; i < N_t; ++i) {
            Omega_opt_2[i] = Omega_max*f_sigmoid(Omega_opt[i]);
        }
#else // OPTIMIZE_WITH_BOUNDED_OMEGA
        for (int64_t i = 0; i < N_t; ++i) {
            Omega_opt_2[i] = std::complex<double>(Omega_opt_Re[i], Omega_opt_Im[i]);
        }
#endif // OPTIMIZE_WITH_BOUNDED_OMEGA
        double max_Omega_Re = Omega_opt_2[0].real();
        double max_Omega_Im = Omega_opt_2[0].imag();
        double Omega_abs_integral = std::abs(Omega_opt_2[0])*dt;
        double d_Omega_abs_integral = 0;
        for (int i = 1; i < N_t; ++i) {
            Omega_abs_integral += std::abs(Omega_opt_2[i])*dt;
            d_Omega_abs_integral += std::abs(Omega_opt_2[i]-Omega_opt_2[i-1]);
            if (max_Omega_Re < Omega_opt_2[i].real()) {
                max_Omega_Re = Omega_opt_2[i].real();
            }
            if (max_Omega_Im < Omega_opt_2[i].imag()) {
                max_Omega_Im = Omega_opt_2[i].imag();
            }
        }
        Eigen::Map<MatrixXcdRowMajor> rho_map(rho_vec.data(), basis_size, basis_size);
        std::complex<double> trace_final = (rho_target*rho_map).trace();
        fidelity_under_optimization[n] = trace_final.real();
        const double diff_trace_final = trace_final.real() - old_trace_final;
        std::cout << "n = " << n << ", trace_final = " << trace_final
                  << ", diff_trace_final = " << diff_trace_final
                  << ", Omega_abs_int = " << Omega_abs_integral
                  << ", d_Omega_abs_int = " << d_Omega_abs_integral
                  << ", max_Omega_Re = " << max_Omega_Re
                  << ", max_Omega_Im = " << max_Omega_Im
                  << std::endl;
        if (diff_trace_final < 0) {
            std::cout << "Non-monotonic convergence detected, stopping optimization." << std::endl;
            break;
        }
        old_trace_final = trace_final.real();
        //TODO: Check for convergence
    }
    JQFData ret = control_with_jqf_array_complex_Omega(
            kappa, gamma2, g, Omega_opt_2, omega_d, omega_r, omega_1, omega_2,
            transmon_anharmonicity, k0x_r, k0x_2, num_excitations,
            transmon_excitations, t, N_t, data_reduce_factor);
    ret.fidelity_under_optimization
        = std::move(fidelity_under_optimization);
    return ret;
}

JQFData control_with_jqf_optimize_array_complex_Omega_adrk4(
        double kappa, double gamma2, const std::vector<double> &gammaInternal,
        const std::vector<double> &gammaDephasing, double g,
        const std::vector<std::complex<double>> &Omega, double Omega_max,
        double sigmaFilter, double sigmaWindow, int N_amplitudes,
        double omega_d, double omega_r, double omega_1, double omega_2,
        const std::vector<double> &transmon_anharmonicity,
        double k0x_r, double k0x_2, int num_excitations,
        const std::vector<int> &transmon_excitations, double t_final,
        int64_t N_t, int64_t data_reduce_factor,
        const std::vector<double> &x0, int max_f_evaluations,
        std::function<void(double, const std::vector<double>&)> saveDataToFile)
{
    const double target_excited_state_amplitude = 1;
    const double J_x = 0;
    // Empty stateSpec means calculating the (approximate) average fidelity
    std::vector<InitialFinalStateSpec> stateSpec;
    return jqf_optimize_adrk4(
            stateSpec, {kappa}, {gamma2}, gammaInternal, gammaDephasing, g, J_x,
            Omega, Omega_max, sigmaFilter, sigmaWindow, N_amplitudes, {omega_d},
            {omega_r}, {omega_1, omega_2}, transmon_anharmonicity, {k0x_r},
            {k0x_2}, num_excitations, transmon_excitations, t_final, N_t,
            data_reduce_factor, x0, max_f_evaluations, saveDataToFile);
}

struct find_pulse_start_stop_params
{
    int Omega_index;
    double kappa;
    double gamma2;
    std::vector<double> gammaInternal;
    std::vector<double> gammaDephasing;
    double g;
    double Omega_max;
    double omega_d;
    double omega_r;
    double omega_1;
    double omega_2;
    std::vector<double> transmon_anharmonicity;
    double k0x_r;
    double k0x_2;
    double sigmaFilter;
    int num_excitations;
    std::vector<int> transmon_excitations;
    double t_final;
    int N_t;
    std::vector<double> fidelity_under_optimization;
    std::vector<std::vector<double>> x_array;
};

double find_pulse_start_stop_f(unsigned n, const double *x, double *grad, void *params)
{
    find_pulse_start_stop_params *p
            = (find_pulse_start_stop_params *) params;
    const double b = x[0];
    double dragFactor = 0;
    if (n >= 2) {
        dragFactor = x[1];
    }
    const double kappa = p->kappa;
    const double gamma2 = p->gamma2;
    const double g = p->g;
    double Omega_max = p->Omega_max;
    if (Omega_max == 0) {
        Omega_max = x[p->Omega_index];
    }
    const double omega_d = p->omega_d;
    double Delta_d_factor = 0;
    if (n >= 3) {
        Delta_d_factor = x[2];
    }
    const double omega_r = p->omega_r;
    const double omega_1 = p->omega_1;
    const double omega_2 = p->omega_2;
    const double k0x_r = p->k0x_r;
    const double k0x_2 = p->k0x_2;
    const double sigmaFilter = p->sigmaFilter;
    const int num_excitations = p->num_excitations;
    const double t_final = p->t_final;
    const int64_t N_t = p->N_t;
    const int64_t data_reduce_factor = 1;
    const double a_guess1 = 2*sigmaFilter;
    const double a_guess2 = 2.5*sigmaFilter;
    const double a_lower_bound_tolerance = 1e-2;
    const double pulse_start_upper_bound = 0.1;
    const double a = find_root_secant<double>([=] (double a) -> double {
            return Omega_max*filteredSquarePulse(0, a, t_final, sigmaFilter)
                    -pulse_start_upper_bound;
        }, a_guess1, a_guess2, a_lower_bound_tolerance);
    auto Omega_func = [=] (double t) -> std::complex<double> {
        return Omega_max*std::complex<double>(
                    filteredSquarePulse(t, a, b, sigmaFilter),
                    dragFactor*filteredSquarePulseDerivative(t, a, b, sigmaFilter));
    };
    auto omegad_func = [=] (double t) -> double {
        return omega_d+Delta_d_factor*std::pow(Omega_func(t).real(),2);
    };
    double fidelity = 0;
    if (n >= 3) {
        fidelity = control_with_jqf_time_dependent_omegad_Omega_avg_fidelity(
                kappa, gamma2, p->gammaInternal, p->gammaDephasing, g,
                Omega_func, omegad_func, omega_r, omega_1, omega_2,
                p->transmon_anharmonicity, k0x_r, k0x_2, num_excitations,
                p->transmon_excitations, t_final, N_t, data_reduce_factor);
    } else {
        fidelity = control_with_jqf_time_dependent_Omega_avg_fidelity(
                kappa, gamma2, p->gammaInternal, p->gammaDephasing, g,
                Omega_func, omega_d, omega_r, omega_1, omega_2,
                p->transmon_anharmonicity, k0x_r, k0x_2, num_excitations,
                p->transmon_excitations, t_final, N_t, data_reduce_factor);
    }
    p->fidelity_under_optimization.push_back(fidelity);
    std::vector<double> x_val(n);
    for (int i = 0; i < n; ++i) {
        x_val[i] = x[i];
    }
    p->x_array.push_back(x_val);
    std::cout << "  start = " << a << ", stop = " << b
              << ", dragFactor = " << dragFactor
              << ", Delta_d_factor = " << Delta_d_factor
              << ", Omega_max = " << Omega_max
              << ", fidelity = " << fidelity << std::endl;
    return -fidelity;
}

JQFData control_with_jqf_optimize_Omega_gaussian_pulse(
        double kappa, double gamma2, const std::vector<double> &gammaInternal,
        const std::vector<double> &gammaDephasing, double g,
        double Omega_max, double sigmaFilter, int numParams,
        double omega_d, double omega_r, double omega_1, double omega_2,
        const std::vector<double> &transmon_anharmonicity,
        double k0x_r, double k0x_2, int num_excitations,
        const std::vector<int> &transmon_excitations, double t_final,
        int64_t N_t, int64_t data_reduce_factor)
{
    assert(numParams > 0 && "The the should be at least one parameter!");
    assert(numParams <= 3 && "Up to 3 parameters are supported: b, dragFactor, Delta_d_factor");
    std::vector<InitialFinalStateSpec> stateSpec = initializeStateSpecToAverageSigmaX();

    find_pulse_start_stop_params params;
    params.Omega_index = 0;
    params.kappa = kappa;
    params.gamma2 = gamma2;
    params.gammaInternal = gammaInternal;
    params.gammaDephasing = gammaDephasing;
    params.g = g;
    params.Omega_max = Omega_max;
    params.omega_d = omega_d;
    params.omega_r = omega_r;
    params.omega_1 = omega_1;
    params.omega_2 = omega_2;
    params.transmon_anharmonicity = transmon_anharmonicity;
    params.k0x_r = k0x_r;
    params.k0x_2 = k0x_2;
    params.sigmaFilter = sigmaFilter;
    params.num_excitations = num_excitations;
    params.transmon_excitations = transmon_excitations;
    params.t_final = t_final;
    params.N_t = N_t;
    int extraParams = 0;
    if (Omega_max == 0) {
        extraParams = 1;
        params.Omega_index = numParams;
    }
    nlopt::opt opt(nlopt::LN_BOBYQA, numParams+extraParams);
    std::vector<double> x(numParams+extraParams);
    //x[0] = 0.5*t_final;
    x[0] = 0.342253;
    // The two values below are for the 1st order DRAG
    // See [Phys. Rev. Lett. 103, 110501 (2009)]
    if (numParams >= 2) {
        x[1] = -1.0/transmon_anharmonicity[0];
    }
    if (numParams >= 3) {
        //x[2] = -2.0/(4.0*transmon_anharmonicity[0]);
        // Zero is closer the the optimal than the above expression
        x[2] = 0;
        std::cout << "x[2] = " << x[2] << std::endl;
    }
    if (extraParams == 1) {
        x[params.Omega_index] = 100;
    }
    std::vector<double> step(numParams+extraParams);
    step[0] = 0.01*t_final;
    if (numParams >= 2) {
        step[1] = 0.1/transmon_anharmonicity[0];
    }
    if (numParams >= 3) {
        step[2] = 0.2/(4.0*transmon_anharmonicity[0]);
    }
    if (extraParams == 1) {
        step[params.Omega_index] = 1;
    }
    opt.set_initial_step(step);

    std::vector<double> lb(numParams+extraParams);
    lb[0] = 0;
    if (numParams >= 2) {
        lb[1] = -HUGE_VAL;
    }
    if (numParams >= 3) {
        lb[2] = -HUGE_VAL;
    }
    if (extraParams == 1) {
        lb[params.Omega_index] = 0;
    }
    opt.set_lower_bounds(lb);

    std::vector<double> ub(numParams+extraParams);
    ub[0] = HUGE_VAL;
    if (numParams >= 2) {
        ub[1] = HUGE_VAL;
    }
    if (numParams >= 3) {
        ub[2] = HUGE_VAL;
    }
    if (extraParams == 1) {
        ub[params.Omega_index] = HUGE_VAL;
    }
    opt.set_upper_bounds(ub);

    opt.set_min_objective(find_pulse_start_stop_f, &params);
    opt.set_xtol_abs(1e-14);
    //opt.set_maxeval(1);

    double minf;
    nlopt::result result = opt.optimize(x, minf);
    const double b_opt = x[0];
    double dragFactor_opt = 0;
    if (numParams >= 2) {
        dragFactor_opt = x[1];
    }
    double Delta_d_factor_opt = 0;
    if (numParams >= 3) {
        Delta_d_factor_opt = x[2];
    }
    double Omega_max_opt = Omega_max;
    if (Omega_max == 0) {
        Omega_max_opt = x[params.Omega_index];
    }
    const double a_guess1 = 2*sigmaFilter;
    const double a_guess2 = 2.5*sigmaFilter;
    const double a_lower_bound_tolerance = 1e-2;
    const double pulse_start_upper_bound = 0.1;
    const double a = find_root_secant<double>([=] (double a) -> double {
            return Omega_max_opt*filteredSquarePulse(0, a, t_final, sigmaFilter)
                    -pulse_start_upper_bound;
        }, a_guess1, a_guess2, a_lower_bound_tolerance);
    std::cout << "Pulse start lower bound = " << a << std::endl;
    auto Omega_func = [=] (double t) -> std::complex<double> {
        return Omega_max_opt*std::complex<double>(
                    filteredSquarePulse(t, a, b_opt, sigmaFilter),
                    dragFactor_opt*filteredSquarePulseDerivative(t, a, b_opt, sigmaFilter));
    };
    auto omegad_func = [=] (double t) -> double {
        return omega_d+Delta_d_factor_opt*std::pow(Omega_func(t).real(),2);
    };
    const double initial_excited_state_amplitude = 0;
    const double target_excited_state_amplitude = 1;
    JQFData data;
    if (numParams >= 3) {
        data = control_with_jqf_time_dependent_omegad_Omega(
            initial_excited_state_amplitude, target_excited_state_amplitude,
            kappa, gamma2, gammaInternal, gammaDephasing, g, Omega_func,
            omegad_func, omega_r, omega_1, omega_2, transmon_anharmonicity,
            k0x_r, k0x_2, num_excitations, transmon_excitations, t_final,
            N_t, data_reduce_factor);
    } else {
        data = control_with_jqf_time_dependent_Omega(
            initial_excited_state_amplitude, target_excited_state_amplitude,
            kappa, gamma2, gammaInternal, gammaDephasing, g, Omega_func,
            omega_d, omega_r, omega_1, omega_2, transmon_anharmonicity,
            k0x_r, k0x_2, num_excitations, transmon_excitations, t_final,
            N_t, data_reduce_factor);
    }
    data.fidelity_under_optimization
        = std::move(params.fidelity_under_optimization);
    data.x_array = std::move(params.x_array);
    return data;
}
