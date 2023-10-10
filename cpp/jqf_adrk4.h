// Copyright (c) 2021-2022 Ivan Iakoupov
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

#ifndef JQF_ADRK4_H
#define JQF_ADRK4_H

#include "confined_Gaussian.h"
#include "jqf_superoperator.h"
#include "jqf_data.h"

#include "Eigen/Dense"
#include <complex>
#include <functional>
#include <vector>

#define FILTERED_BASIS_FUNCTION_BY_QUADRATURE

#define ADRK4_FILTER_BASIS_FUNCTIONS             (1 << 0)
#define ADRK4_CLAMP_OMEGA                        (1 << 1)
#define ADRK4_PRECOMPUTE_BASIS_FUNCTION_VALUES   (1 << 2)

struct InitialFinalStateSpec
{
    std::complex<double> initial;
    std::complex<double> target;
    Eigen::MatrixXcd rho_initial;
    Eigen::MatrixXcd rho_target;
};

JQFData jqf_time_dependent_Omega(
        InitialFinalStateSpec stateSpec,
        const std::vector<double> &kappa, const std::vector<double> &gamma,
        const std::vector<double> &gammaInternal,
        const std::vector<double> &gammaDephasing,
        double g, double J_x, std::function<std::complex<double>(double)> Omega,
        double omega_d, const std::vector<double> &omega_r, const std::vector<double> &omega,
        const std::vector<double> &transmon_anharmonicity,
        const std::vector<double> &k0x_r, const std::vector<double> &k0x_a,
        int num_excitations, const std::vector<int> &transmon_excitations,
        double t_final, int64_t N_t, int64_t data_reduce_factor);

JQFData jqf_time_dependent_omegad_Omega(
        InitialFinalStateSpec stateSpec,
        const std::vector<double> &kappa, const std::vector<double> &gamma,
        const std::vector<double> &gammaInternal,
        const std::vector<double> &gammaDephasing,
        double g, double J_x, std::function<std::complex<double>(double)> Omega,
        std::function<double(double)> omega_d,
        const std::vector<double> &omega_r, const std::vector<double> &omega,
        const std::vector<double> &transmon_anharmonicity,
        const std::vector<double> &k0x_r, const std::vector<double> &k0x_a,
        int num_excitations, const std::vector<int> &transmon_excitations,
        double t_final, int64_t N_t, int64_t data_reduce_factor);

struct find_optimal_Omega_adrk4_params
{
    bool useRhoSpec;
    int adrk4_flags;
    int basis_size;
    int N_amplitudes;
    double NormalizationFactor;
    double omegaFactor;
    double sigmaFilter;
    double sigmaWindow;
    double t_final;
    int64_t start_N_t;
    int64_t N_t;
    double Omega_max;
    int rho_index;
    std::vector<Eigen::VectorXcd> rho_initial_vec;
    std::vector<Eigen::VectorXcd> rho_target_vec;
    std::unordered_map<int64_t,Eigen::MatrixXd> f1;
    std::unordered_map<int64_t,Eigen::MatrixXd> f2;
    std::function<void(int64_t)> precomputeBasisFunctionValues;
    JQFSuperoperatorData sd;
    std::vector<double> fidelity_under_optimization;
    std::vector<std::vector<double>> x_array;
    std::function<void(double, const std::vector<double>&)> saveDataToFile;
};

inline double basisFunction(int k, double t, double NormalizationFactor, double omegaFactor)
{
    return NormalizationFactor*std::sin(omegaFactor*(k+1)*t);
}

double filteredBasisFunction(int k, double t, double NormalizationFactor, double omegaFactor, double t_f, double sigma);

double find_optimal_Omega_adrk4_f(unsigned n, const double *x, double *grad, void *params);
double find_optimal_Omega_adrk4_auto_dt_f(unsigned n, const double *x, double *grad, void *params);

std::vector<InitialFinalStateSpec> initializeStateSpecToAverageSigmaX();

bool isUsingRhoSpec(const std::vector<InitialFinalStateSpec> &stateSpec);

void expandQubitStateToFullBasis(
        Eigen::VectorXcd &rho_initial_vec_n,
        Eigen::VectorXcd &rho_target_vec_n,
        const InitialFinalStateSpec &stateSpec_n,
        const std::vector<Eigen::VectorXcd> &psi_vec);

void expandQubitStateToFullBasisRhoSpec(
        Eigen::VectorXcd &rho_initial_vec_n,
        Eigen::VectorXcd &rho_target_vec_n,
        const InitialFinalStateSpec &stateSpec_n,
        const std::vector<Eigen::VectorXcd> &psi_vec);

JQFData jqf_optimize_adrk4(
        const std::vector<InitialFinalStateSpec> &stateSpec0,
        const std::vector<double> &kappa, const std::vector<double> &gamma,
        const std::vector<double> &gammaInternal,
        const std::vector<double> &gammaDephasing,
        double g, double J_x, const std::vector<std::complex<double>> &Omega,
        double Omega_max, double sigmaFilter, double sigmaWindow,
        int N_amplitudes, const std::vector<double> &omega_d,
        const std::vector<double> &omega_r, const std::vector<double> &omega,
        const std::vector<double> &transmon_anharmonicity,
        const std::vector<double> &k0x_r, const std::vector<double> &k0x_a,
        int num_excitations, const std::vector<int> &transmon_excitations,
        double t_final, int64_t N_t, int64_t data_reduce_factor,
        const std::vector<double> &x0, int max_f_evaluations,
        std::function<void(double, const std::vector<double>&)> saveDataToFile);

#endif // JQF_ADRK4_H
