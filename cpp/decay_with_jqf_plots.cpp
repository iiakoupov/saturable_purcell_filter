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

#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <cassert>
#include <cmath>

#include <nlopt.hpp>

#include "jqf.h"
#include "io.h"

void jqf_data_qubit_decay_specific_params(double k0x_2, double gamma2, double gamma1Internal, double gamma2Internal, bool useDDE)
{
    const bool startInEigenstate = true;
    const double kappa = 1;
    const std::vector<double> gammaInternal = {gamma1Internal, gamma2Internal};
    const std::vector<double> gammaDephasing = {0, 0};
    const double g = 54.772;
    const double Omega = 0;
    const double omega_r = 5000;
    const double omega_1 = 4000;
    const double omega_2 = 3997;
    //const double omega_2 = 4000;
    const double omega_d = omega_1;
    const std::vector<double> transmon_anharmonicity = {-200, -200};
    const double k0x_r = 0;
    const int num_excitations = 1;
    const std::vector<int> transmon_excitations = {1, 1};
    const double t = 10;
    int64_t N_t;
    int64_t data_reduce_factor;
    if (useDDE) {
        N_t = 4e11*t;
        data_reduce_factor = 4e6;
    } else {
        N_t = 1e4*t;
        data_reduce_factor = 1;
    }
    JQFData data;
    if (useDDE) {
        assert(Omega == 0
               && "Driving is not supported with DDE evolution");
        data = jqf_data_qubit_decay_dde(kappa, gamma2, g, omega_d, omega_r, omega_1, omega_2, k0x_r, k0x_2, startInEigenstate, t, N_t, data_reduce_factor);
    } else {
        AtomResonatorInitialState initialState;
        if (startInEigenstate) {
            initialState = AtomResonatorInitialState::EigenstateUp;
        } else {
            initialState = AtomResonatorInitialState::AtomUp;
        }
        data = jqf_simulation(initialState, kappa, gamma2, gammaInternal,
                              gammaDephasing, g, Omega, omega_d, omega_r,
                              omega_1, omega_2, transmon_anharmonicity,
                              k0x_r, k0x_2, num_excitations,
                              transmon_excitations, t, N_t,
                              data_reduce_factor);
    }

    std::stringstream fileNameStream;
    fileNameStream << "decay_with_jqf";
    if (startInEigenstate) {
        fileNameStream << "_startEigenstate";
    }
    if (useDDE) {
        fileNameStream << "_dde";
    }
    fileNameStream << "_kappa_" << kappa
                   << "_gamma2_" << gamma2
                   << "_gamma1I_" << gamma1Internal
                   << "_gamma2I_" << gamma2Internal
                   << "_g_" << g
                   << "_Omega_" << Omega
                   << "_omegad_" << omega_d
                   << "_omegar_" << omega_r
                   << "_omega1_" << omega_1
                   << "_omega2_" << omega_2
                   << "_kxr_" << k0x_r
                   << "_kx2_" << k0x_2;
    fileNameStream << ".csv";
    const std::string fileName = fileNameStream.str();
    savetxt(fileName,
            {data.time, data.aa_populations[0],
             data.aa_populations[1], data.res_populations[0],
             data.F[0], data.tilde_F},
            ';', "t;population_1;population_2;res_population;fidelity;tilde_F");
    std::cout << "Wrote to " << fileName << std::endl;
}

struct find_optimal_initial_state_params
{
    double kappa;
    double gamma2;
    double g;
    double Omega;
    double omega_d;
    double omega_r;
    double omega_1;
    double omega_2;
    double k0x_r;
    double k0x_2;
    double t;
    int64_t N_t;
};

double find_optimal_initial_state_constraint_f(unsigned n, const double *x, double *grad, void *params)
{
    find_optimal_initial_state_params *p
            = (find_optimal_initial_state_params *) params;
    double norm1r = 0;
    for (int i = 0; i < 2; ++i) {
        const std::complex<double> element_i(x[2*i], x[2*i+1]);
        norm1r += std::norm(element_i);
    }
    return norm1r-1;
}

void x_to_v(const double *x, std::vector<std::complex<double>> &v)
{
    double norm1r = 0;
    for (int i = 0; i < 2; ++i) {
        //v[i+1] = std::complex<double>(x[2*i], x[2*i+1]);
        v[i] = std::complex<double>(x[i], 0);
        norm1r += std::norm(v[i]);
    }
    v[2] = 1-std::sqrt(norm1r);
}

double find_optimal_initial_state_f(unsigned n, const double *x, double *grad, void *params)
{
    find_optimal_initial_state_params *p
            = (find_optimal_initial_state_params *) params;
    std::vector<std::complex<double>> v(3);
    x_to_v(x, v);
    const double kappa = p->kappa;
    const double gamma2 = p->gamma2;
    const double g = p->g;
    const double Omega = p->Omega;
    const double omega_d = p->omega_d;
    const double omega_r = p->omega_r;
    const double omega_1 = p->omega_1;
    const double omega_2 = p->omega_2;
    const double k0x_r = p->k0x_r;
    const double k0x_2 = p->k0x_2;
    const double t = p->t;
    const int64_t N_t = p->N_t;
    const int64_t data_reduce_factor = 1;
    JQFData data = jqf_data_qubit_decay_Hamiltonian_arbitrary_state(
            kappa, gamma2, g, omega_d, omega_r, omega_1, omega_2, k0x_r, k0x_2,
            v, t, N_t, data_reduce_factor);
    const double last_fidelity = data.F[0][N_t-1];
    std::cout << "fidelity = " << last_fidelity;
    std::cout << ",  x = { ";
    for (int k = 0; k < 3; ++k) {
        std::cout << v[k] << " ; ";
    }
    std::cout << std::endl;
    return -last_fidelity;
}

void jqf_data_qubit_decay_optimize_initial_state()
{
    const bool useHamiltonian = true;
    const bool startInEigenstate = true;
    const double kappa = 1;
    const double gamma2 = 50;
    const std::vector<double> gammaInternal = {0, 0};
    const std::vector<double> gammaDephasing = {0, 0};
    const double g = 54.772;
    const double Omega = 0;
    const double omega_r = 5000;
    const double omega_1 = 4000;
    const double omega_2 = 4000;
    const double omega_d = omega_1;
    const std::vector<double> transmon_anharmonicity = {-200, -200};
    const double k0x_r = 0;
    const double k0x_2 = 1;
    const int num_excitations = 1;
    const std::vector<int> transmon_excitations = {1, 1};
    const double t = 10;
    const int64_t N_t = 1e4*t;
    const int64_t data_reduce_factor = 1;
    JQFData data;
    if (useHamiltonian) {
        assert(Omega == 0
               && "Driving is not supported with Hamiltonian evolution");
        find_optimal_initial_state_params params;
        params.kappa = kappa;
        params.gamma2 = gamma2;
        params.g = g;
        params.Omega = Omega;
        params.omega_d = omega_d;
        params.omega_r = omega_r;
        params.omega_1 = omega_1;
        params.omega_2 = omega_2;
        params.k0x_r = k0x_r;
        params.k0x_2 = k0x_2;
        params.t = t;
        params.N_t = N_t;
        const int numParams = 2;
        //nlopt::opt opt(nlopt::LN_SBPLX, numParams);
        nlopt::opt opt(nlopt::LN_COBYLA, numParams);
        std::vector<double> x(numParams, 0);
        x[0] = 0;
        x[1] = 1;
        //x[2] = 1;
        //x[3] = 0;
        std::vector<double> step(numParams, 0.1);
        opt.set_initial_step(step);
        //std::vector<double> lb(numParams);
        //lb[0] = 0;
        //opt.set_lower_bounds(lb);
        //std::vector<double> ub(numParams);
        //ub[0] = 0;
        //opt.set_upper_bounds(ub);

        opt.set_min_objective(find_optimal_initial_state_f, &params);
        opt.add_inequality_constraint(find_optimal_initial_state_constraint_f,
                                      &params, 1e-10);
        opt.set_xtol_abs(1e-14);

        double minf;
        nlopt::result result = opt.optimize(x, minf);

        std::vector<std::complex<double>> v(3);
        x_to_v(x.data(), v);
        data = jqf_data_qubit_decay_Hamiltonian_arbitrary_state(kappa, gamma2, g, omega_d, omega_r, omega_1, omega_2, k0x_r, k0x_2, v, t, N_t, data_reduce_factor);
    } else {
        AtomResonatorInitialState initialState;
        if (startInEigenstate) {
            initialState = AtomResonatorInitialState::EigenstateUp;
        } else {
            initialState = AtomResonatorInitialState::AtomUp;
        }
        data = jqf_simulation(initialState, kappa, gamma2, gammaInternal,
                              gammaDephasing, g, Omega, omega_d, omega_r,
                              omega_1, omega_2, transmon_anharmonicity,
                              k0x_r, k0x_2, num_excitations,
                              transmon_excitations, t, N_t,
                              data_reduce_factor);
    }

    std::stringstream fileNameStream;
    fileNameStream << "decay_with_jqf_opt_state";
    if (startInEigenstate) {
        fileNameStream << "_startEigenstate";
    }
    if (useHamiltonian) {
        fileNameStream << "_H";
    }
    fileNameStream << "_kappa_" << kappa
                   << "_gamma2_" << gamma2
                   << "_g_" << g
                   << "_Omega_" << Omega
                   << "_omegad_" << omega_d
                   << "_omegar_" << omega_r
                   << "_omega1_" << omega_1
                   << "_omega2_" << omega_2
                   << "_kxr_" << k0x_r
                   << "_kx2_" << k0x_2;
    fileNameStream << ".csv";
    const std::string fileName = fileNameStream.str();
    savetxt(fileName,
            {data.time, data.aa_populations[0],
             data.aa_populations[1], data.res_populations[0],
             data.F[0]},
            ';', "t;population_1;population_2;res_population;fidelity");
    std::cout << "Wrote to " << fileName << std::endl;
}

void jqf_data_qubit_decay_omega_1()
{
    const bool useHamiltonian = true;
    const bool startInEigenstate = false;
    const double kappa = 1;
    const double gamma2 = 1000;
    const std::vector<double> gammaInternal = {0, 0};
    const std::vector<double> gammaDephasing = {0, 0};
    const double g = 10;
    const double Omega = 0;
    const double omega_r = 5000;
    const double omega_1_ref = 4000;
    const double omega_2 = 4000;
    const double omega_d = omega_1_ref;
    const std::vector<double> transmon_anharmonicity = {-200, -200};
    const double k0x_r = 0;
    const double k0x_2 = 0.5;
    const int num_excitations = 1;
    const std::vector<int> transmon_excitations = {1, 1};
    const double t = 10;
    const int64_t N_t = 100000;
    const int64_t data_reduce_factor = 1;

    const double min_omega_1 = omega_1_ref+2;
    const double max_omega_1 = omega_1_ref+20;
    const int num_omega_1 = 18*10+1;
    const double step_omega_1 = (max_omega_1 - min_omega_1) / (num_omega_1 - 1);

    std::vector<double> omega_1_array(num_omega_1, 0);
    std::vector<double> min_population_array(num_omega_1, 0);
    std::vector<double> time_for_min_population_array(num_omega_1, 0);
    #pragma omp parallel for
    for (int n = 0; n < num_omega_1; ++n) {
        const double omega_1 = min_omega_1 + n*step_omega_1;
        JQFData data;
        if (useHamiltonian) {
            assert(Omega == 0
                   && "Driving is not supported with Hamiltonian evolution");
            data = jqf_data_qubit_decay_Hamiltonian(kappa, gamma2, g, omega_d, omega_r, omega_1, omega_2, k0x_r, k0x_2, startInEigenstate, t, N_t, data_reduce_factor);
        } else {
            AtomResonatorInitialState initialState;
            if (startInEigenstate) {
                initialState = AtomResonatorInitialState::EigenstateUp;
            } else {
                initialState = AtomResonatorInitialState::AtomUp;
            }
            data = jqf_simulation(initialState, kappa, gamma2, gammaInternal,
                                  gammaDephasing, g, Omega, omega_d, omega_r,
                                  omega_1, omega_2, transmon_anharmonicity,
                                  k0x_r, k0x_2, num_excitations,
                                  transmon_excitations, t, N_t,
                                  data_reduce_factor);
        }
        const int N_t = data.time.size();
        double min_population = 1;
        double time_for_min_population = 0;
        for (int i = 0; i < N_t; ++i) {
            if (min_population > data.aa_populations[0][i]) {
                min_population = data.aa_populations[0][i];
                time_for_min_population = data.time[i];
            }
        }
        omega_1_array[n] = omega_1;
        min_population_array[n] = min_population;
        time_for_min_population_array[n] = time_for_min_population;
    }

    std::stringstream fileNameStream;
    fileNameStream << "decay_with_jqf";
    fileNameStream << "_omega1";
    if (useHamiltonian) {
        fileNameStream << "_H";
    }
    fileNameStream << "_kappa_" << kappa
                   << "_gamma2_" << gamma2
                   << "_g_" << g
                   << "_Omega_" << Omega
                   << "_omegad_" << omega_d
                   << "_omegar_" << omega_r
                   << "_omega2_" << omega_2
                   << "_kxr_" << k0x_r
                   << "_kx2_" << k0x_2;
    fileNameStream << ".csv";
    const std::string fileName = fileNameStream.str();
    savetxt(fileName,
            {omega_1_array, min_population_array,
            time_for_min_population_array}, ';',
            "omega_1;min_population;time_for_min_population");
    std::cout << "Wrote to " << fileName << std::endl;
}

void jqf_data_qubit_decay_omega_2(double gamma2, double gamma1Internal, double gamma2Internal)
{
    const bool useDDE = false;
    const bool startInEigenstate = true;
    const double kappa = 1;
    const std::vector<double> gammaInternal = {gamma1Internal, gamma2Internal};
    const std::vector<double> gammaDephasing = {0, 0};
    const double g = 54.772;
    const double Omega = 0;
    const double omega_r = 5000;
    const double omega_1 = 4000;
    const double omega_2_ref = 3997;
    const double omega_d = omega_1;
    const std::vector<double> transmon_anharmonicity = {-200, -200};
    const double k0x_r = 0;
    const double k0x_2 = 1;
    const int num_excitations = 1;
    const std::vector<int> transmon_excitations = {1, 1};
    const double t = 10;
    int64_t N_t;
    int64_t data_reduce_factor;
    if (useDDE) {
        N_t = 1e11*t;
        data_reduce_factor = 1e6;
    } else {
        N_t = 1e4*t;
        data_reduce_factor = 1;
    }

    const double min_omega_2 = omega_2_ref-5;
    const double max_omega_2 = omega_2_ref+5;
    int num_omega_2;
    if (useDDE) {
        num_omega_2 = 10+1;
    } else {
        num_omega_2 = 300+1;
    }
    const double step_omega_2 = (max_omega_2 - min_omega_2) / (num_omega_2 - 1);

    std::vector<double> omega_2_array(num_omega_2, 0);
    std::vector<double> min_population_array(num_omega_2, 0);
    std::vector<double> time_for_min_population_array(num_omega_2, 0);
    std::vector<double> min_fidelity_array(num_omega_2, 0);
    std::vector<double> time_for_min_fidelity_array(num_omega_2, 0);
    std::vector<double> max_fidelity_array(num_omega_2, 0);
    std::vector<double> time_for_max_fidelity_array(num_omega_2, 0);
    std::vector<double> last_fidelity_array(num_omega_2, 0);
    #pragma omp parallel for
    for (int n = 0; n < num_omega_2; ++n) {
        const double omega_2 = min_omega_2 + n*step_omega_2;
        JQFData data;
        if (useDDE) {
            assert(Omega == 0
                   && "Driving is not supported with DDE evolution");
            data = jqf_data_qubit_decay_dde(kappa, gamma2, g, omega_d, omega_r, omega_1, omega_2, k0x_r, k0x_2, startInEigenstate, t, N_t, data_reduce_factor);
        } else {
            AtomResonatorInitialState initialState;
            if (startInEigenstate) {
                initialState = AtomResonatorInitialState::EigenstateUp;
            } else {
                initialState = AtomResonatorInitialState::AtomUp;
            }
            data = jqf_simulation(initialState, kappa, gamma2, gammaInternal,
                                  gammaDephasing, g, Omega, omega_d, omega_r,
                                  omega_1, omega_2, transmon_anharmonicity,
                                  k0x_r, k0x_2, num_excitations,
                                  transmon_excitations, t, N_t,
                                  data_reduce_factor);
        }
        double min_population = 1;
        double time_for_min_population = 0;
        for (int i = 0; i < N_t; ++i) {
            if (min_population > data.aa_populations[0][i]) {
                min_population = data.aa_populations[0][i];
                time_for_min_population = data.time[i];
            }
        }
        double min_fidelity = 1;
        double time_for_min_fidelity = 0;
        double max_fidelity = 0;
        double time_for_max_fidelity = 0;
        for (int i = 0; i < N_t; ++i) {
            if (min_fidelity > data.F[0][i]) {
                min_fidelity = data.F[0][i];
                time_for_min_fidelity = data.time[i];
            }
            if (max_fidelity < data.F[0][i]) {
                max_fidelity = data.F[0][i];
                time_for_max_fidelity = data.time[i];
            }
        }
        omega_2_array[n] = omega_2;
        min_population_array[n] = min_population;
        time_for_min_population_array[n] = time_for_min_population;
        min_fidelity_array[n] = min_fidelity;
        time_for_min_fidelity_array[n] = time_for_min_fidelity;
        max_fidelity_array[n] = max_fidelity;
        time_for_max_fidelity_array[n] = time_for_max_fidelity;
        last_fidelity_array[n] = data.F[0][N_t-1];
        if (useDDE) {
            #pragma omp critical
            {
                std::cout << "n = " << n << ", omega_2 = " << omega_2 << std::endl;
            }
        }
    }

    std::stringstream fileNameStream;
    fileNameStream << "decay_with_jqf";
    fileNameStream << "_omega2";
    if (useDDE) {
        fileNameStream << "_dde";
    }
    fileNameStream << "_kappa_" << kappa
                   << "_gamma2_" << gamma2
                   << "_gamma1I_" << gamma1Internal
                   << "_gamma2I_" << gamma2Internal
                   << "_g_" << g
                   << "_Omega_" << Omega
                   << "_omegad_" << omega_d
                   << "_omegar_" << omega_r
                   << "_omega1_" << omega_1
                   << "_kxr_" << k0x_r
                   << "_kx2_" << k0x_2;
    fileNameStream << ".csv";
    const std::string fileName = fileNameStream.str();
    savetxt(fileName,
            {omega_2_array, min_population_array,
            time_for_min_population_array,
            min_fidelity_array,
            time_for_min_fidelity_array,
            max_fidelity_array,
            time_for_max_fidelity_array,
            last_fidelity_array
            }, ';',
            "omega_2;min_population;time_for_min_population;min_fidelity;time_for_min_fidelity;max_fidelity;time_for_max_fidelity;last_fidelity");
    std::cout << "Wrote to " << fileName << std::endl;
}

void jqf_data_qubit_decay_k0x_r()
{
    const bool useHamiltonian = false;
    const bool startInEigenstate = true;
    const double kappa = 1;
    const double gamma2 = 0;
    const std::vector<double> gammaInternal = {0, 0};
    const std::vector<double> gammaDephasing = {0, 0};
    const double g = 54.772;
    const double Omega = 0;
    const double omega_r = 5000;
    const double omega_1 = 4000;
    const double omega_2 = 3997;
    const double omega_d = omega_1;
    const std::vector<double> transmon_anharmonicity = {-200, -200};
    //const double k0x_r = 0;
    const double k0x_2 = 1;
    const int num_excitations = 1;
    const std::vector<int> transmon_excitations = {1, 1};
    const double t = 10;
    const int64_t N_t = 1e4*t;
    const int64_t data_reduce_factor = 1;

    const double min_k0x_r = 0.499;
    const double max_k0x_r = 0.501;
    const int num_k0x_r = 10*10+1;
    const double step_k0x_r = (max_k0x_r - min_k0x_r) / (num_k0x_r - 1);

    std::vector<double> k0x_r_array(num_k0x_r, 0);
    std::vector<double> min_population_array(num_k0x_r, 0);
    std::vector<double> time_for_min_population_array(num_k0x_r, 0);
    std::vector<double> min_fidelity_array(num_k0x_r, 0);
    std::vector<double> time_for_min_fidelity_array(num_k0x_r, 0);
    std::vector<double> max_fidelity_array(num_k0x_r, 0);
    std::vector<double> time_for_max_fidelity_array(num_k0x_r, 0);
    std::vector<double> last_fidelity_array(num_k0x_r, 0);
    #pragma omp parallel for
    for (int n = 0; n < num_k0x_r; ++n) {
        const double k0x_r = min_k0x_r + n*step_k0x_r;
        JQFData data;
        if (useHamiltonian) {
            assert(Omega == 0
                   && "Driving is not supported with Hamiltonian evolution");
            data = jqf_data_qubit_decay_Hamiltonian(kappa, gamma2, g, omega_d, omega_r, omega_1, omega_2, k0x_r, k0x_2, startInEigenstate, t, N_t, data_reduce_factor);
        } else {
            AtomResonatorInitialState initialState;
            if (startInEigenstate) {
                initialState = AtomResonatorInitialState::EigenstateUp;
            } else {
                initialState = AtomResonatorInitialState::AtomUp;
            }
            data = jqf_simulation(initialState, kappa, gamma2, gammaInternal,
                                  gammaDephasing, g, Omega, omega_d, omega_r,
                                  omega_1, omega_2, transmon_anharmonicity,
                                  k0x_r, k0x_2, num_excitations,
                                  transmon_excitations, t, N_t,
                                  data_reduce_factor);
        }
        const int N_t = data.time.size();
        double min_population = 1;
        double time_for_min_population = 0;
        for (int i = 0; i < N_t; ++i) {
            if (min_population > data.aa_populations[0][i]) {
                min_population = data.aa_populations[0][i];
                time_for_min_population = data.time[i];
            }
        }
        double min_fidelity = 1;
        double time_for_min_fidelity = 0;
        double max_fidelity = 0;
        double time_for_max_fidelity = 0;
        for (int i = 0; i < N_t; ++i) {
            if (min_fidelity > data.F[0][i]) {
                min_fidelity = data.F[0][i];
                time_for_min_fidelity = data.time[i];
            }
            if (max_fidelity < data.F[0][i]) {
                max_fidelity = data.F[0][i];
                time_for_max_fidelity = data.time[i];
            }
        }
        k0x_r_array[n] = k0x_r;
        min_population_array[n] = min_population;
        time_for_min_population_array[n] = time_for_min_population;
        min_fidelity_array[n] = min_fidelity;
        time_for_min_fidelity_array[n] = time_for_min_fidelity;
        max_fidelity_array[n] = max_fidelity;
        time_for_max_fidelity_array[n] = time_for_max_fidelity;
        last_fidelity_array[n] = data.F[0][N_t-1];
    }

    std::stringstream fileNameStream;
    fileNameStream << "decay_with_jqf";
    fileNameStream << "_k0xr";
    if (useHamiltonian) {
        fileNameStream << "_H";
    }
    fileNameStream << "_kappa_" << kappa
                   << "_gamma2_" << gamma2
                   << "_g_" << g
                   << "_Omega_" << Omega
                   << "_omegad_" << omega_d
                   << "_omegar_" << omega_r
                   << "_omega1_" << omega_1
                   << "_omega2_" << omega_2
                   << "_kx2_" << k0x_2;
    fileNameStream << ".csv";
    const std::string fileName = fileNameStream.str();
    savetxt(fileName,
            {k0x_r_array, min_population_array,
            time_for_min_population_array,
            min_fidelity_array,
            time_for_min_fidelity_array,
            max_fidelity_array,
            time_for_max_fidelity_array,
            last_fidelity_array
            }, ';',
            "k0x_r;min_population;time_for_min_population;min_fidelity;time_for_min_fidelity;max_fidelity;time_for_max_fidelity;last_fidelity");
    std::cout << "Wrote to " << fileName << std::endl;
}

void jqf_data_qubit_decay_k0x_2(double gamma2, double gamma1Internal, double gamma2Internal)
{
    const bool useDDE = false;
    const bool startInEigenstate = false;
    const double kappa = 1;
    const std::vector<double> gammaInternal = {gamma1Internal, gamma2Internal};
    const std::vector<double> gammaDephasing = {0, 0};
    const double g = 54.772;
    const double Omega = 0;
    const double omega_r = 5000;
    const double omega_1 = 4000;
    const double omega_2 = 3997;
    const double omega_d = omega_1;
    const std::vector<double> transmon_anharmonicity = {-200, -200};
    const double k0x_r = 0;
    //const double k0x_2 = 0.5;
    const int num_excitations = 1;
    const std::vector<int> transmon_excitations = {1, 1};
    const double t = 10;
    int64_t N_t;
    int64_t data_reduce_factor;
    if (useDDE) {
        N_t = 1e11*t;
        data_reduce_factor = 1e6;
    } else {
        N_t = 1e4*t;
        data_reduce_factor = 1;
    }

    const double min_k0x_2 = 0.7;
    const double max_k0x_2 = 1.3;
    int num_k0x_2;
    if (useDDE) {
        num_k0x_2 = 10+1;
    } else {
        num_k0x_2 = 100+1;
    }
    const double step_k0x_2 = (max_k0x_2 - min_k0x_2) / (num_k0x_2 - 1);

    std::vector<double> k0x_2_array(num_k0x_2, 0);
    std::vector<double> min_population_array(num_k0x_2, 0);
    std::vector<double> time_for_min_population_array(num_k0x_2, 0);
    std::vector<double> min_fidelity_array(num_k0x_2, 0);
    std::vector<double> time_for_min_fidelity_array(num_k0x_2, 0);
    std::vector<double> max_fidelity_array(num_k0x_2, 0);
    std::vector<double> time_for_max_fidelity_array(num_k0x_2, 0);
    std::vector<double> last_fidelity_array(num_k0x_2, 0);
    #pragma omp parallel for
    for (int n = 0; n < num_k0x_2; ++n) {
        const double k0x_2 = min_k0x_2 + n*step_k0x_2;
        JQFData data;
        if (useDDE) {
            assert(Omega == 0
                   && "Driving is not supported with DDE evolution");
            data = jqf_data_qubit_decay_dde(kappa, gamma2, g, omega_d, omega_r, omega_1, omega_2, k0x_r, k0x_2, startInEigenstate, t, N_t, data_reduce_factor);
        } else {
            AtomResonatorInitialState initialState;
            if (startInEigenstate) {
                initialState = AtomResonatorInitialState::EigenstateUp;
            } else {
                initialState = AtomResonatorInitialState::AtomUp;
            }
            data = jqf_simulation(initialState, kappa, gamma2, gammaInternal,
                                  gammaDephasing, g, Omega, omega_d, omega_r,
                                  omega_1, omega_2, transmon_anharmonicity,
                                  k0x_r, k0x_2, num_excitations,
                                  transmon_excitations, t, N_t,
                                  data_reduce_factor);
        }
        const int N_t = data.time.size();
        double min_population = 1;
        double time_for_min_population = 0;
        for (int i = 0; i < N_t; ++i) {
            if (min_population > data.aa_populations[0][i]) {
                min_population = data.aa_populations[0][i];
                time_for_min_population = data.time[i];
            }
        }
        double min_fidelity = 1;
        double time_for_min_fidelity = 0;
        double max_fidelity = 0;
        double time_for_max_fidelity = 0;
        for (int i = 0; i < N_t; ++i) {
            if (min_fidelity > data.F[0][i]) {
                min_fidelity = data.F[0][i];
                time_for_min_fidelity = data.time[i];
            }
            if (max_fidelity < data.F[0][i]) {
                max_fidelity = data.F[0][i];
                time_for_max_fidelity = data.time[i];
            }
        }
        k0x_2_array[n] = k0x_2;
        min_population_array[n] = min_population;
        time_for_min_population_array[n] = time_for_min_population;
        min_fidelity_array[n] = min_fidelity;
        time_for_min_fidelity_array[n] = time_for_min_fidelity;
        max_fidelity_array[n] = max_fidelity;
        time_for_max_fidelity_array[n] = time_for_max_fidelity;
        last_fidelity_array[n] = data.F[0][N_t-1];
        if (useDDE) {
            #pragma omp critical
            {
                std::cout << "n = " << n << ", k0x_2 = " << k0x_2 << std::endl;
            }
        }
    }

    std::stringstream fileNameStream;
    fileNameStream << "decay_with_jqf";
    fileNameStream << "_k0x2";
    if (useDDE) {
        fileNameStream << "_dde";
    }
    fileNameStream << "_kappa_" << kappa
                   << "_gamma2_" << gamma2
                   << "_gamma1I_" << gamma1Internal
                   << "_gamma2I_" << gamma2Internal
                   << "_g_" << g
                   << "_Omega_" << Omega
                   << "_omegad_" << omega_d
                   << "_omegar_" << omega_r
                   << "_omega1_" << omega_1
                   << "_omega2_" << omega_2
                   << "_kxr_" << k0x_r;
    fileNameStream << ".csv";
    const std::string fileName = fileNameStream.str();
    savetxt(fileName,
            {k0x_2_array, min_population_array,
            time_for_min_population_array,
            min_fidelity_array,
            time_for_min_fidelity_array,
            max_fidelity_array,
            time_for_max_fidelity_array,
            last_fidelity_array
            }, ';',
            "k0x_2;min_population;time_for_min_population;min_fidelity;time_for_min_fidelity;max_fidelity;time_for_max_fidelity;last_fidelity");
    std::cout << "Wrote to " << fileName << std::endl;
}

void jqf_data_qubit_decay_gamma1Internal(double gamma2, double gamma2Internal)
{
    const bool useDDE = false;
    const bool startInEigenstate = true;
    const double kappa = 1;
    const std::vector<double> gammaDephasing = {0, 0};
    const double g = 54.772;
    const double Omega = 0;
    const double omega_r = 5000;
    const double omega_1 = 4000;
    const double omega_2 = 3997;
    const double omega_d = omega_1;
    const std::vector<double> transmon_anharmonicity = {-200, -200};
    const double k0x_r = 0;
    const double k0x_2 = 1;
    const int num_excitations = 1;
    const std::vector<int> transmon_excitations = {1, 1};
    const double t = 10;
    int64_t N_t;
    int64_t data_reduce_factor;
    if (useDDE) {
        N_t = 1e11*t;
        data_reduce_factor = 1e6;
    } else {
        N_t = 1e4*t;
        data_reduce_factor = 1;
    }

    const double min_T1Internal = 1e-6; // 1 us
    const double max_T1Internal = 500e-6; // 500 us
    const double kappaHz = 2e6;

    const double min_gamma1Internal = 1.0/(2*M_PI*max_T1Internal*kappaHz);
    const double max_gamma1Internal = 1.0/(2*M_PI*min_T1Internal*kappaHz);;
    int num_gamma1Internal;
    if (useDDE) {
        num_gamma1Internal = 10+1;
    } else {
        num_gamma1Internal = 300+1;
    }
    const double step_gamma1Internal = (max_gamma1Internal - min_gamma1Internal) / (num_gamma1Internal - 1);

    std::vector<double> gamma1Internal_array(num_gamma1Internal, 0);
    std::vector<double> min_population_array(num_gamma1Internal, 0);
    std::vector<double> time_for_min_population_array(num_gamma1Internal, 0);
    std::vector<double> min_fidelity_array(num_gamma1Internal, 0);
    std::vector<double> time_for_min_fidelity_array(num_gamma1Internal, 0);
    std::vector<double> max_fidelity_array(num_gamma1Internal, 0);
    std::vector<double> time_for_max_fidelity_array(num_gamma1Internal, 0);
    std::vector<double> last_fidelity_array(num_gamma1Internal, 0);
    #pragma omp parallel for
    for (int n = 0; n < num_gamma1Internal; ++n) {
        const double gamma1Internal = min_gamma1Internal + n*step_gamma1Internal;
        const std::vector<double> gammaInternal = {gamma1Internal, gamma2Internal};
        JQFData data;
        if (useDDE) {
            assert(Omega == 0
                   && "Driving is not supported with DDE evolution");
            data = jqf_data_qubit_decay_dde(kappa, gamma2, g, omega_d, omega_r, omega_1, omega_2, k0x_r, k0x_2, startInEigenstate, t, N_t, data_reduce_factor);
        } else {
            AtomResonatorInitialState initialState;
            if (startInEigenstate) {
                initialState = AtomResonatorInitialState::EigenstateUp;
            } else {
                initialState = AtomResonatorInitialState::AtomUp;
            }
            data = jqf_simulation(initialState, kappa, gamma2, gammaInternal,
                                  gammaDephasing, g, Omega, omega_d, omega_r,
                                  omega_1, omega_2, transmon_anharmonicity,
                                  k0x_r, k0x_2, num_excitations,
                                  transmon_excitations, t, N_t,
                                  data_reduce_factor);
        }
        double min_population = 1;
        double time_for_min_population = 0;
        for (int i = 0; i < N_t; ++i) {
            if (min_population > data.aa_populations[0][i]) {
                min_population = data.aa_populations[0][i];
                time_for_min_population = data.time[i];
            }
        }
        double min_fidelity = 1;
        double time_for_min_fidelity = 0;
        double max_fidelity = 0;
        double time_for_max_fidelity = 0;
        for (int i = 0; i < N_t; ++i) {
            if (min_fidelity > data.F[0][i]) {
                min_fidelity = data.F[0][i];
                time_for_min_fidelity = data.time[i];
            }
            if (max_fidelity < data.F[0][i]) {
                max_fidelity = data.F[0][i];
                time_for_max_fidelity = data.time[i];
            }
        }
        gamma1Internal_array[n] = gamma1Internal;
        min_population_array[n] = min_population;
        time_for_min_population_array[n] = time_for_min_population;
        min_fidelity_array[n] = min_fidelity;
        time_for_min_fidelity_array[n] = time_for_min_fidelity;
        max_fidelity_array[n] = max_fidelity;
        time_for_max_fidelity_array[n] = time_for_max_fidelity;
        last_fidelity_array[n] = data.F[0][N_t-1];
        if (useDDE) {
            #pragma omp critical
            {
                std::cout << "n = " << n << ", gamma1Internal = " << gamma1Internal << std::endl;
            }
        }
    }

    std::stringstream fileNameStream;
    fileNameStream << "decay_with_jqf";
    fileNameStream << "_gamma1I";
    if (useDDE) {
        fileNameStream << "_dde";
    }
    fileNameStream << "_kappa_" << kappa
                   << "_gamma2_" << gamma2
                   << "_gamma2I_" << gamma2Internal
                   << "_g_" << g
                   << "_Omega_" << Omega
                   << "_omegad_" << omega_d
                   << "_omegar_" << omega_r
                   << "_omega1_" << omega_1
                   << "_omega2_" << omega_2
                   << "_kxr_" << k0x_r
                   << "_kx2_" << k0x_2;
    fileNameStream << ".csv";
    const std::string fileName = fileNameStream.str();
    savetxt(fileName,
            {gamma1Internal_array, min_population_array,
            time_for_min_population_array,
            min_fidelity_array,
            time_for_min_fidelity_array,
            max_fidelity_array,
            time_for_max_fidelity_array,
            last_fidelity_array
            }, ';',
            "gamma1Internal;min_population;time_for_min_population;min_fidelity;time_for_min_fidelity;max_fidelity;time_for_max_fidelity;last_fidelity");
    std::cout << "Wrote to " << fileName << std::endl;
}

void jqf_data_qubit_decay_omega_1_k0x_2()
{
    const bool useHamiltonian = true;
    const bool startInEigenstate = false;
    const double kappa = 1;
    const double gamma2 = 1000;
    const std::vector<double> gammaInternal = {0, 0};
    const std::vector<double> gammaDephasing = {0, 0};
    const double g = 1;
    const double Omega = 0;
    const double omega_r = 5000;
    const double omega_1_ref = 4000;
    const double omega_2 = 4000;
    const double omega_d = omega_1_ref;
    const std::vector<double> transmon_anharmonicity = {-200, -200};
    const double k0x_r = 0;
    //const double k0x_2 = 0.5;
    const int num_excitations = 1;
    const std::vector<int> transmon_excitations = {1, 1};
    const double t = 10;
    const int64_t N_t = 100000;
    const int64_t data_reduce_factor = 1;

    const double min_omega_1 = omega_1_ref+2;
    const double max_omega_1 = omega_1_ref+20;
    const int num_omega_1 = 18*10+1;
    const double step_omega_1 = (max_omega_1 - min_omega_1) / (num_omega_1 - 1);

    const double min_k0x_2 = 0;
    const double max_k0x_2 = 1;
    const int num_k0x_2 = 10*10+1;
    const double step_k0x_2 = (max_k0x_2 - min_k0x_2) / (num_k0x_2 - 1);

    const int num_data_points = num_omega_1*num_k0x_2;

    std::vector<double> omega_1_array(num_data_points, 0);
    std::vector<double> k0x_2_array(num_data_points, 0);
    std::vector<double> min_population_array(num_data_points, 0);
    std::vector<double> time_for_min_population_array(num_data_points, 0);
    std::vector<double> max_population_array(num_data_points, 0);
    std::vector<double> time_for_max_population_array(num_data_points, 0);
    #pragma omp parallel for
    for (int m = 0; m < num_k0x_2; ++m) {
        const double k0x_2 = min_k0x_2 + m*step_k0x_2;
        for (int n = 0; n < num_omega_1; ++n) {
            const double omega_1 = min_omega_1 + n*step_omega_1;
            JQFData data;
            if (useHamiltonian) {
                assert(Omega == 0
                       && "Driving is not supported with Hamiltonian evolution");
                data = jqf_data_qubit_decay_Hamiltonian(kappa, gamma2, g, omega_d, omega_r, omega_1, omega_2, k0x_r, k0x_2, startInEigenstate, t, N_t, data_reduce_factor);
            } else {
                AtomResonatorInitialState initialState;
                if (startInEigenstate) {
                    initialState = AtomResonatorInitialState::EigenstateUp;
                } else {
                    initialState = AtomResonatorInitialState::AtomUp;
                }
                data = jqf_simulation(initialState, kappa, gamma2,
                                      gammaInternal, gammaDephasing, g, Omega,
                                      omega_d, omega_r, omega_1, omega_2,
                                      transmon_anharmonicity, k0x_r, k0x_2,
                                      num_excitations, transmon_excitations, t,
                                      N_t, data_reduce_factor);
            }
            const int N_t = data.time.size();
            double min_population = 1;
            double time_for_min_population = 0;
            double max_population = 0;
            double time_for_max_population = 0;
            for (int i = 0; i < N_t; ++i) {
                if (min_population > data.aa_populations[0][i]) {
                    min_population = data.aa_populations[0][i];
                    time_for_min_population = data.time[i];
                }
                if (max_population < data.aa_populations[0][i]) {
                    max_population = data.aa_populations[0][i];
                    time_for_max_population = data.time[i];
                }
            }
            const int index = n + num_omega_1*m;
            k0x_2_array[index] = k0x_2;
            omega_1_array[index] = omega_1;
            min_population_array[index] = min_population;
            time_for_min_population_array[index] = time_for_min_population;
            max_population_array[index] = max_population;
            time_for_max_population_array[index] = time_for_max_population;
        }
    }

    std::stringstream fileNameStream;
    fileNameStream << "decay_with_jqf";
    fileNameStream << "_omega1_k0x2";
    if (useHamiltonian) {
        fileNameStream << "_H";
    }
    fileNameStream << "_kappa_" << kappa
                   << "_gamma2_" << gamma2
                   << "_g_" << g
                   << "_Omega_" << Omega
                   << "_omegad_" << omega_d
                   << "_omegar_" << omega_r
                   << "_omega2_" << omega_2
                   << "_kxr_" << k0x_r;
    fileNameStream << ".csv";
    const std::string fileName = fileNameStream.str();
    savetxt(fileName,
            {k0x_2_array, omega_1_array,
            min_population_array, time_for_min_population_array,
            max_population_array, time_for_max_population_array
            }, ';',
            "k0x_2;omega_1;min_population;time_for_min_population;max_population;time_for_max_population");
    std::cout << "Wrote to " << fileName << std::endl;
}

void jqf_data_qubit_decay_omega_2_k0x_2()
{
    const bool useHamiltonian = false;
    const bool startInEigenstate = true;
    const double kappa = 1;
    const double gamma2 = 50;
    const std::vector<double> gammaInternal = {0, 0};
    const std::vector<double> gammaDephasing = {0, 0};
    const double g = 54.772;
    const double Omega = 0;
    const double omega_r = 5000;
    const double omega_1 = 4000;
    const double omega_2_ref = 4000;
    const double omega_d = omega_1;
    const std::vector<double> transmon_anharmonicity = {-200, -200};
    const double k0x_r = 0;
    //const double k0x_2 = 0.5;
    const int num_excitations = 1;
    const std::vector<int> transmon_excitations = {1, 1};
    const double t = 20;
    const int64_t N_t = 1e4*t;
    const int64_t data_reduce_factor = 1;

    const double min_omega_2 = omega_2_ref-10;
    const double max_omega_2 = omega_2_ref+10;
    const int num_omega_2 = 20+1;
    const double step_omega_2 = (max_omega_2 - min_omega_2) / (num_omega_2 - 1);

    const double min_k0x_2 = 0.8;
    const double max_k0x_2 = 1.1;
    const int num_k0x_2 = 30+1;
    const double step_k0x_2 = (max_k0x_2 - min_k0x_2) / (num_k0x_2 - 1);

    const int num_data_points = num_omega_2*num_k0x_2;

    std::vector<double> omega_2_array(num_data_points, 0);
    std::vector<double> k0x_2_array(num_data_points, 0);
    std::vector<double> min_population_array(num_data_points, 0);
    std::vector<double> time_for_min_population_array(num_data_points, 0);
    std::vector<double> min_fidelity_array(num_data_points, 0);
    std::vector<double> time_for_min_fidelity_array(num_data_points, 0);
    std::vector<double> max_fidelity_array(num_data_points, 0);
    std::vector<double> time_for_max_fidelity_array(num_data_points, 0);
    std::vector<double> last_fidelity_array(num_data_points, 0);
    #pragma omp parallel for
    for (int m = 0; m < num_k0x_2; ++m) {
        const double k0x_2 = min_k0x_2 + m*step_k0x_2;
        for (int n = 0; n < num_omega_2; ++n) {
            const double omega_2 = min_omega_2 + n*step_omega_2;
            JQFData data;
            if (useHamiltonian) {
                assert(Omega == 0
                       && "Driving is not supported with Hamiltonian evolution");
                data = jqf_data_qubit_decay_Hamiltonian(kappa, gamma2, g, omega_d, omega_r, omega_1, omega_2, k0x_r, k0x_2, startInEigenstate, t, N_t, data_reduce_factor);
            } else {
                AtomResonatorInitialState initialState;
                if (startInEigenstate) {
                    initialState = AtomResonatorInitialState::EigenstateUp;
                } else {
                    initialState = AtomResonatorInitialState::AtomUp;
                }
                data = jqf_simulation(initialState, kappa, gamma2,
                                      gammaInternal, gammaDephasing, g, Omega,
                                      omega_d, omega_r, omega_1, omega_2,
                                      transmon_anharmonicity, k0x_r, k0x_2,
                                      num_excitations, transmon_excitations, t,
                                      N_t, data_reduce_factor);
            }
            const int N_t = data.time.size();
            double min_population = 1;
            double time_for_min_population = 0;
            for (int i = 0; i < N_t; ++i) {
                if (min_population > data.aa_populations[0][i]) {
                    min_population = data.aa_populations[0][i];
                    time_for_min_population = data.time[i];
                }
            }
            double min_fidelity = 1;
            double time_for_min_fidelity = 0;
            double max_fidelity = 0;
            double time_for_max_fidelity = 0;
            for (int i = 0; i < N_t; ++i) {
                if (min_fidelity > data.F[0][i]) {
                    min_fidelity = data.F[0][i];
                    time_for_min_fidelity = data.time[i];
                }
                if (max_fidelity < data.F[0][i]) {
                    max_fidelity = data.F[0][i];
                    time_for_max_fidelity = data.time[i];
                }
            }
            const int index = n + num_omega_2*m;
            k0x_2_array[index] = k0x_2;
            omega_2_array[index] = omega_2;
            min_population_array[index] = min_population;
            time_for_min_population_array[index] = time_for_min_population;
            min_fidelity_array[index] = min_fidelity;
            time_for_min_fidelity_array[index] = time_for_min_fidelity;
            max_fidelity_array[index] = max_fidelity;
            time_for_max_fidelity_array[index] = time_for_max_fidelity;
            last_fidelity_array[index] = data.F[0][N_t-1];
        }
    }

    std::stringstream fileNameStream;
    fileNameStream << "decay_with_jqf";
    fileNameStream << "_omega2_k0x2";
    if (useHamiltonian) {
        fileNameStream << "_H";
    }
    fileNameStream << "_kappa_" << kappa
                   << "_gamma2_" << gamma2
                   << "_g_" << g
                   << "_Omega_" << Omega
                   << "_omegad_" << omega_d
                   << "_omegar_" << omega_r
                   << "_omega1_" << omega_1
                   << "_kxr_" << k0x_r;
    fileNameStream << ".csv";
    const std::string fileName = fileNameStream.str();
    savetxt(fileName,
            {k0x_2_array, omega_2_array, min_population_array,
            time_for_min_population_array,
            min_fidelity_array,
            time_for_min_fidelity_array,
            max_fidelity_array,
            time_for_max_fidelity_array,
            last_fidelity_array
            }, ';',
            "k0x_2;omega_2;min_population;time_for_min_population;min_fidelity;time_for_min_fidelity;max_fidelity;time_for_max_fidelity;last_fidelity");
    std::cout << "Wrote to " << fileName << std::endl;
}

void decay_jqf_article_plots()
{
    const double k0x_2 = 1;
    // Assuming 1/\kappa with \kappa=2\pi*2 MHz is used as
    const double gamma2 = 50; // 2\pi*100 MHz
    const double gamma2Internal = 1.5; // 2\pi*3 MHz
    // Fig. 3
    jqf_data_qubit_decay_specific_params(k0x_2, 0, 0, 0, false);
    jqf_data_qubit_decay_specific_params(k0x_2, gamma2, 0, 0, false);
    // The commented out call below checks non-zero gamma2Internal,
    // but was not used in the paper. Instead we show sweeps
    // over different parameters in Fig. 4.
    //jqf_data_qubit_decay_specific_params(k0x_2, gamma2, 0, gamma2Internal, false);
    // Fig. 4 (a)
    jqf_data_qubit_decay_omega_2(0, 0, 0);
    jqf_data_qubit_decay_omega_2(gamma2, 0, 0);
    jqf_data_qubit_decay_omega_2(gamma2, 0, gamma2Internal);
    // Fig. 4 (b)
    jqf_data_qubit_decay_k0x_2(0, 0, 0);
    jqf_data_qubit_decay_k0x_2(gamma2, 0, 0);
    jqf_data_qubit_decay_k0x_2(gamma2, 0, gamma2Internal);
    // Fig. 4 (c)
    jqf_data_qubit_decay_gamma1Internal(0, 0);
    jqf_data_qubit_decay_gamma1Internal(gamma2, 0);
    jqf_data_qubit_decay_gamma1Internal(gamma2, gamma2Internal);
    // Fig. 9
    jqf_data_qubit_decay_specific_params(k0x_2, gamma2, 0, 0, true);
}

int main() {
    decay_jqf_article_plots();
    //jqf_data_qubit_decay_specific_params();
    //jqf_data_qubit_decay_optimize_initial_state();
    //jqf_data_qubit_decay_omega_1();
    //jqf_data_qubit_decay_omega_2();
    //jqf_data_qubit_decay_k0x_r();
    //jqf_data_qubit_decay_k0x_2();
    //jqf_data_qubit_decay_omega_1_k0x_2();
    //jqf_data_qubit_decay_omega_2_k0x_2();
    return 0;
}
