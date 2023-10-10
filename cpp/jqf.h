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

#ifndef JQF_H
#define JQF_H

#include <vector>
#include <complex>
#include <functional>

#include "jqf_data.h"

enum class AtomResonatorInitialState
{
    AllDown,
    AtomUp,
    EigenstateUp,
    EigenstateDown
};

JQFData jqf_data_qubit_decay_master_equation_manual(
        double kappa, double gamma2, double g, double omega_d, double omega_r,
        double omega_1, double omega_2, double k0x_r, double k0x_2,
        bool startInEigenstate, double t, int64_t N_t,
        int64_t data_reduce_factor);

JQFData jqf_data_qubit_decay_dde_arbitrary_state(
        double kappa, double gamma2, double g, double omega_d, double omega_r,
        double omega_1, double omega_2, double k0x_r, double k0x_2,
        const std::vector<std::complex<double>> &v0, double t, int64_t N_t,
        int64_t data_reduce_factor);

JQFData jqf_data_qubit_decay_dde(
        double kappa, double gamma2, double g, double omega_d, double omega_r,
        double omega_1, double omega_2, double k0x_r, double k0x_2,
        bool startInEigenstate, double t, int64_t N_t,
        int64_t data_reduce_factor);

JQFData jqf_data_qubit_decay_Hamiltonian_arbitrary_state(
        double kappa, double gamma2, double g, double omega_d, double omega_r,
        double omega_1, double omega_2, double k0x_r, double k0x_2,
        const std::vector<std::complex<double>> &v0, double t, int64_t N_t,
        int64_t data_reduce_factor);

JQFData jqf_data_qubit_decay_Hamiltonian(
        double kappa, double gamma2, double g, double omega_d, double omega_r,
        double omega_1, double omega_2, double k0x_r, double k0x_2,
        bool startInEigenstate, double t, int64_t N_t,
        int64_t data_reduce_factor);

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
        int64_t data_reduce_factor);

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
        int64_t N_t, int64_t data_reduce_factor);

JQFData control_with_jqf_array_complex_Omega(
        double kappa, double gamma2, double g,
        const std::vector<std::complex<double>> &Omega,
        double omega_d, double omega_r, double omega_1, double omega_2,
        const std::vector<double> &transmon_anharmonicity,
        double k0x_r, double k0x_2, int num_excitations,
        const std::vector<int> &transmon_excitations, double t, int64_t N_t,
        int64_t data_reduce_factor);

JQFData control_with_jqf_optimize_array_complex_Omega(
        double kappa, double gamma2, double g,
        const std::vector<std::complex<double>> &Omega,
        const std::vector<double> &learning_rate, int N_iterations,
        double omega_d, double omega_r, double omega_1, double omega_2,
        const std::vector<double> &transmon_anharmonicity,
        double k0x_r, double k0x_2, int num_excitations,
        const std::vector<int> &transmon_excitations, double t, int64_t N_t,
        int64_t data_reduce_factor);

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
        std::function<void(double, const std::vector<double>&)> saveDataToFile);

inline double filteredSquarePulse(double t, double a, double b, double sigma)
{
    return 0.5*(std::erf(M_SQRT1_2*(t-a)/sigma)
                -std::erf(M_SQRT1_2*(t-b)/sigma));
};

inline double filteredSquarePulseDerivative(double t, double a, double b, double sigma)
{
    return (std::exp(-0.5*std::pow((a-t)/sigma,2))
            -std::exp(-0.5*std::pow((b-t)/sigma,2)))
            /(std::sqrt(2*M_PI)*sigma);
};

JQFData control_with_jqf_optimize_Omega_gaussian_pulse(
        double kappa, double gamma2, const std::vector<double> &gammaInternal,
        const std::vector<double> &gammaDephasing, double g,
        double Omega_max, double sigmaFilter, int N_parameters,
        double omega_d, double omega_r, double omega_1, double omega_2,
        const std::vector<double> &transmon_anharmonicity,
        double k0x_r, double k0x_2, int num_excitations,
        const std::vector<int> &transmon_excitations, double t_final,
        int64_t N_t, int64_t data_reduce_factor);

#endif // JQF_H
