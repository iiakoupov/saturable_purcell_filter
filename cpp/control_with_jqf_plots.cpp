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

#include <chrono>
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
#include "x0_data.h"

inline double smooth_f(double x)
{
    if (x > 0) {
        return std::exp(-1.0/x);
    } else {
        return 0;
    }
};

enum class OptimalControlAlgorithm
{
    None,
    ADRK4,
    GaussianPulse,
    Krotov
};

std::string optimalControlAlgorithmToString(OptimalControlAlgorithm optAlg)
{
    switch (optAlg) {
    case OptimalControlAlgorithm::None:
        return std::string();
    case OptimalControlAlgorithm::ADRK4:
        return "adrk4";
    case OptimalControlAlgorithm::GaussianPulse:
        return "gauss";
    case OptimalControlAlgorithm::Krotov:
        return "krotov";
    default:
        assert(0 && "Unknown optimal control algorithm!");
        return std::string();
    }
}

void control_with_jqf_specific_params(
        OptimalControlAlgorithm optAlg, double t_final, double Omega_max,
        double sigmaFilter, double sigmaWindow, int N_parameters,
        int num_excitations, const std::vector<int> &transmon_excitations,
        const std::vector<double> x0, bool optimize,
        const std::string &optAlgSuffix)
{
    const bool adiabaticDrive = false;
    const double kappa = 1;
    const double gamma2 = 50;
    const std::vector<double> gammaInternal = {0, 0};
    const std::vector<double> gammaDephasing = {0, 0};
    const double g = 54.772;
    const double Omega = 200;
    const double Omega_initial = 50;
    const double omega_r = 5000;
    const double omega_1 = 4000;
    //const double omega_2 = 4000;
    const double omega_2 = 3997;
    const double omega_d = omega_2;
    const std::vector<double> transmon_anharmonicity = {-200, -200};
    const double k0x_r = 0;
    const double k0x_2 = 1;
    const double rise_time = 0.05;
    const double rise_time_offset = 3;
    const int64_t N_t = 200000*t_final;
    const int64_t data_reduce_factor = 1;
    // Negative means no limit
    int max_f_evaluations = -1;
    if (!optimize) {
        max_f_evaluations = 0;
    }

    const int N_iterations = 40000;

    std::stringstream OmegaStream;
    std::string optAlgString = optimalControlAlgorithmToString(optAlg);
    if (optAlgString.empty()) {
        OmegaStream << Omega;
    } else {
        OmegaStream << optAlgString;
    }
    if (x0.size() == 2*N_parameters && !optAlgSuffix.empty()) {
        OmegaStream << optAlgSuffix;
    }

    auto generateOutputPath = [&](const std::string &baseName, const std::string &suffix = std::string()) -> std::string
    {
        std::stringstream fileNameStream;
        fileNameStream << baseName
                       << "_" << OmegaStream.str()
                       << "_kappa_" << kappa
                       << "_gamma2_" << gamma2
                       << "_g_" << g
                       << "_omegad_" << omega_d
                       << "_omegar_" << omega_r
                       << "_omega1_" << omega_1
                       << "_omega2_" << omega_2
                       << "_kxr_" << k0x_r
                       << "_kx2_" << k0x_2
                       << "_t_" << t_final;
        if (optAlg == OptimalControlAlgorithm::ADRK4) {
            if (sigmaFilter > 0 && sigmaWindow > 0) {
                fileNameStream << "_sigmaF_" << sigmaFilter;
            }
            if (Omega_max > 0) {
                fileNameStream << "_OmegaMax_" << Omega_max;
            }
            fileNameStream << "_Nc_" << N_parameters;
        } else if (optAlg == OptimalControlAlgorithm::GaussianPulse) {
            fileNameStream << "_sigmaF_" << sigmaFilter;
            if (Omega_max > 0) {
                fileNameStream << "_OmegaMax_" << Omega_max;
            }
            fileNameStream << "_Nc_" << N_parameters;
        }
        fileNameStream << ".csv";
        if (!suffix.empty()) {
            fileNameStream << suffix;
        }
        return fileNameStream.str();
    };
    const std::string fileNameOptDataNew
        = generateOutputPath("control_with_jqf_opt_data", ".new");
    const std::string fileNameAmplitudesDataNew
        = generateOutputPath("control_with_jqf_amplitudes_data", ".new");
    if (optAlg == OptimalControlAlgorithm::ADRK4 && max_f_evaluations != 0) {
        std::cout << "Writing during optimization to:" << std::endl;
        {
            std::cout << fileNameOptDataNew << std::endl;
            std::ofstream file(fileNameOptDataNew);
            file << "# fidelity\n";
        }
        {
            std::cout << fileNameAmplitudesDataNew << std::endl;
            std::ofstream file(fileNameAmplitudesDataNew);
            file << "# amplitudes\n";
        }
    }
    auto saveDataToFile = [&](double fidelity_under_optimization, const std::vector<double> &x) -> void
    {
        {
            std::ofstream file(fileNameOptDataNew, std::ios_base::app);
            file.precision(17);
            file << fidelity_under_optimization
                 << '\n';
        }
        {
            const char delimiter = ';';
            std::ofstream file(fileNameAmplitudesDataNew, std::ios_base::app);
            file.precision(17);
            const int cols = x.size();
            for (int j = 0; j < cols-1; ++j) {
                file << x[j] << delimiter;
            }
            file << x[cols-1] << '\n';
        }
    };

    JQFData data;
    switch (optAlg) {
    case OptimalControlAlgorithm::None:
    {
        if (adiabaticDrive) {
            //auto Omega_func = [rise_time,Omega] (double x) -> double {
            //    if (x > 0) {
            //        return Omega*smooth_f(x)/(smooth_f(x)+smooth_f(rise_time-x));
            //    } else {
            //        return 0;
            //    }
            //};
            auto Omega_func = [=] (double t) -> std::complex<double> {
                return Omega*filteredSquarePulse(
                        t, rise_time_offset*rise_time,
                        t_final-rise_time_offset*rise_time, rise_time);
            };
            const double initial_excited_state_amplitude = 0;
            const double target_excited_state_amplitude = 1;
            data = control_with_jqf_time_dependent_Omega(
                    initial_excited_state_amplitude,
                    target_excited_state_amplitude,
                    kappa, gamma2, gammaInternal, gammaDephasing, g, Omega_func,
                    omega_d, omega_r, omega_1, omega_2, transmon_anharmonicity,
                    k0x_r, k0x_2, num_excitations, transmon_excitations,
                    t_final, N_t, data_reduce_factor);
        } else {
            data = jqf_simulation(AtomResonatorInitialState::AllDown, kappa,
                                  gamma2, gammaInternal, gammaDephasing, g,
                                  Omega, omega_d, omega_r, omega_1, omega_2,
                                  transmon_anharmonicity, k0x_r, k0x_2,
                                  num_excitations, transmon_excitations,
                                  t_final, N_t, data_reduce_factor);
        }
    }
    break;
    case OptimalControlAlgorithm::GaussianPulse:
    {
        data = control_with_jqf_optimize_Omega_gaussian_pulse(
                kappa, gamma2, gammaInternal, gammaDephasing, g, Omega_max,
                sigmaFilter, N_parameters, omega_d, omega_r, omega_1, omega_2,
                transmon_anharmonicity, k0x_r, k0x_2,
                num_excitations, transmon_excitations, t_final, N_t,
                data_reduce_factor);
    }
    break;
    case OptimalControlAlgorithm::ADRK4:
    {
        std::vector<std::complex<double>> Omega_array(N_t, 0);
        if (x0.size() != 2*N_parameters) {
            const double Omega_max_Gauss = 100;
            const double sigmaFilterGauss = 0.02;
            const int N_parameters_Gauss = 1;
            std::cout << "First, we optimize a Gaussian pulse to use as the input to ADRK4" << std::endl;
            auto start_Gauss = std::chrono::steady_clock::now();
            data = control_with_jqf_optimize_Omega_gaussian_pulse(
                    kappa, gamma2, gammaInternal, gammaDephasing, g, Omega_max_Gauss,
                    sigmaFilterGauss, N_parameters_Gauss, omega_d, omega_r, omega_1,
                    omega_2, transmon_anharmonicity, k0x_r, k0x_2,
                    num_excitations, transmon_excitations, t_final, N_t,
                    data_reduce_factor);
            auto end_Gauss = std::chrono::steady_clock::now();
            std::chrono::duration<double> diff_Gauss = end_Gauss-start_Gauss;
            std::cout << "Gaussian pulse optimization took "
                      << diff_Gauss.count() << " s"
                      << std::endl;
            for (int n = 0; n < N_t; ++n) {
                Omega_array[n] = std::complex<double>(data.Omega_Re[n], data.Omega_Im[n]);
            }
            std::cout << "Now we further optimize the pulse shape with ADRK4" << std::endl;
        } else {
            std::cout << "Skipping the preliminary Gaussian pulse optimization,\n"
                      << "because the initial Fourier coefficients were passed."
                      << std::endl;
        }
        data = control_with_jqf_optimize_array_complex_Omega_adrk4(
                kappa, gamma2, gammaInternal, gammaDephasing, g, Omega_array,
                Omega_max, sigmaFilter, sigmaWindow, N_parameters, omega_d,
                omega_r, omega_1, omega_2, transmon_anharmonicity, k0x_r, k0x_2,
                num_excitations, transmon_excitations, t_final, N_t,
                data_reduce_factor, x0, max_f_evaluations, saveDataToFile);
    }
    break;
    case OptimalControlAlgorithm::Krotov:
    {
        auto Omega_func = [rise_time,t_final,Omega] (double t) -> double {
            if (t > 0) {
                if (t < rise_time) {
                    return Omega*smooth_f(t)/(smooth_f(t)+smooth_f(rise_time-t));
                    //return (Omega/rise_time)*x;
                } else {
                    return Omega*(1-smooth_f(t-(t_final-rise_time))/(smooth_f(t-(t_final-rise_time))+smooth_f(rise_time-(t-(t_final-rise_time)))));
                }
            } else {
                return 0;
            }
        };
        std::vector<std::complex<double>> Omega_array(N_t, 0);
        std::vector<double> learning_rate(N_t, 0);
        const double dt = t_final/N_t;
        //Only allow non-zero Omega and its change
        //during the gate time. The rest is the
        //relaxation time.
        for (int n = 0; n < N_t; ++n) {
            //const double Omega_func_n = Omega_func(dt*n);
            //Omega_array[n] = std::complex<double>(Omega_func_n, Omega_func_n);
            Omega_array[n] = std::complex<double>(Omega_max/2, Omega_max/2);
            //learning_rate[n] = 0.001;
            learning_rate[n] = 100;
        }
        data = control_with_jqf_optimize_array_complex_Omega(kappa, gamma2, g, Omega_array, learning_rate, N_iterations, omega_d, omega_r, omega_1, omega_2, transmon_anharmonicity, k0x_r, k0x_2, num_excitations, transmon_excitations, t_final, N_t, data_reduce_factor);
    }
    break;
    default:
        assert(0 && "Unknown optimal control algorithm!");
    }

    std::stringstream fileNameStream;
    fileNameStream << "control_with_jqf";
    if (adiabaticDrive && optAlg == OptimalControlAlgorithm::None) {
        fileNameStream << "_adiabatic";
    }
    fileNameStream << "_kappa_" << kappa
                   << "_gamma2_" << gamma2
                   << "_g_" << g
                   << "_Omega_" << OmegaStream.str()
                   << "_omegad_" << omega_d
                   << "_omegar_" << omega_r
                   << "_omega1_" << omega_1
                   << "_omega2_" << omega_2
                   << "_kxr_" << k0x_r
                   << "_kx2_" << k0x_2
                   << "_t_" << t_final;
    if (optAlg == OptimalControlAlgorithm::ADRK4) {
        if (sigmaFilter > 0 && sigmaWindow > 0) {
            fileNameStream << "_sigmaF_" << sigmaFilter;
        }
        if (Omega_max > 0) {
            fileNameStream << "_OmegaMax_" << Omega_max;
        }
        fileNameStream << "_Nc_" << N_parameters;
    } else if (optAlg == OptimalControlAlgorithm::GaussianPulse) {
        fileNameStream << "_sigmaF_" << sigmaFilter;
        if (Omega_max > 0) {
            fileNameStream << "_OmegaMax_" << Omega_max;
        }
        fileNameStream << "_Nc_" << N_parameters;
    }
    fileNameStream << ".csv";
    const std::string fileName = fileNameStream.str();
    std::vector<std::vector<double>> dataToSave = {
        data.time,
        data.Omega_Re, data.Omega_Im,
        data.aa_populations[0],
        data.aa_populations[1], data.res_populations[0],
        data.F[0],
        data.tilde_F,
        data.purity_1, data.purity_1r};
    std::stringstream headerStream;
    headerStream << "t;Omega_Re;Omega_Im;population_1;population_2;"
                 << "res_population;fidelity;tilde_F;purity_1;purity_1r";
    for (int m = 0; m < data.aa_level_populations.size(); ++m) {
        for (int j = 0; j < data.aa_level_populations[m].size(); ++j) {
            dataToSave.push_back(data.aa_level_populations[m][j]);
            headerStream << ";s_" << m << "_" << j;
        }
    }
    savetxt(fileName,dataToSave, ';', headerStream.str());
    std::cout << "Wrote to " << fileName << std::endl;
    if (!data.fidelity_under_optimization.empty()) {
        const std::string fileNameOptData
            = generateOutputPath("control_with_jqf_opt_data");
        savetxt(fileNameOptData,
                {data.fidelity_under_optimization},
                ';', "fidelity");
        std::cout << "Wrote to " << fileNameOptData << std::endl;
    }
    if (!data.window_array.empty() && sigmaFilter > 0 && sigmaWindow > 0) {
        const std::string fileNameWindowData
            = generateOutputPath("control_with_jqf_window_data");
        savetxt(fileNameWindowData,
                {data.time,data.window_array},
                ';', "t;windowFunction");
        std::cout << "Wrote to " << fileNameWindowData << std::endl;
    }
    if (!data.x_array.empty()) {
        const std::string fileNameAmplitudesData
            = generateOutputPath("control_with_jqf_amplitudes_data");
        const char delimiter = ';';
        std::ofstream file(fileNameAmplitudesData);
        file.precision(17);
        file << "# amplitudes\n";
        const int rows = data.x_array.size();
        for (int i = 0; i < rows; ++i) {
            const int cols = data.x_array[i].size();
            for (int j = 0; j < cols-1; ++j) {
                file << data.x_array[i][j] << delimiter;
            }
            file << data.x_array[i][cols-1] << '\n';
        }
        std::cout << "Wrote to " << fileNameAmplitudesData << std::endl;
    }
}


void control_with_jqf_article_plots()
{
    const std::vector<double> x0;
    const std::vector<double> x0_iter13 = {ITER13_X};
    const std::vector<double> x0_iter2000 = {ITER2000_X};
    const double Omega_max = 100;
    const int N_amplitudes = 100;
    // The definitions of sigmas are different for the
    // parametrizations of the ADRK4 optimal control and
    // a simple Gaussian filtered rectangle pulse.
    const double sigmaFilterGauss = 0.02;
    const double sigmaWindowGauss = 0.1;
    const int numParamsGauss = 3;
    const int num_excitations = 4;
    const std::vector<int> transmon_excitations = {2, 10};
    // Assuming 1/\kappa with \kappa=2*\pi*2 MHz is used as
    // the time unit.
    const double t_30ns = 0.377; // 30 ns
    const double t_40ns = 0.503; // 40 ns
    const double t_50ns = 0.628; // 50 ns
    // Fig. 6
    control_with_jqf_specific_params(
            OptimalControlAlgorithm::GaussianPulse, t_50ns, Omega_max,
            sigmaFilterGauss, sigmaWindowGauss, numParamsGauss, num_excitations,
            transmon_excitations, x0, true, "");
    // The two calls below will skip the optimization.
    // These should finish quickly.
    // Fig. 6 (c)
    control_with_jqf_specific_params(
            OptimalControlAlgorithm::ADRK4, t_50ns, 0, 0, 0, N_amplitudes,
            num_excitations, transmon_excitations, x0_iter13, false, "iter13");
    // Fig. 7
    control_with_jqf_specific_params(
            OptimalControlAlgorithm::ADRK4, t_50ns, 0, 0, 0, N_amplitudes,
            num_excitations, transmon_excitations, x0_iter2000, false, "");
    // The below call will actually do the optimization.
    // This will take a very long time to finish.
    // It has never actually completed in testing. Its intermediate
    // results were used above as "x0_iter13" and "x0_iter2000".
    //control_with_jqf_specific_params(
    //        OptimalControlAlgorithm::ADRK4, t_50ns, 0, 0, 0, N_amplitudes,
    //        num_excitations, transmon_excitations, x0, true, "");
}

void control_with_jqf_omega_d()
{
    const bool adiabaticDrive = false;
    //const OptimalControlAlgorithm optAlg = OptimalControlAlgorithm::ADRK4;
    //const OptimalControlAlgorithm optAlg = OptimalControlAlgorithm::None;
    const OptimalControlAlgorithm optAlg = OptimalControlAlgorithm::GaussianPulse;
    const double kappa = 1;
    const double gamma2 = 50;
    const std::vector<double> gammaInternal = {0, 0};
    const std::vector<double> gammaDephasing = {0, 0};
    const double g = 54.772;
    const double Omega = 200;
    const double Omega_max = 100;
    const int N_amplitudes = 10;
    const double omega_r = 5000;
    const double omega_1 = 4000;
    const double omega_2 = 4000;
    // Reference value for omega_d
    const double omega_d_ref = omega_1;
    const std::vector<double> transmon_anharmonicity = {-200, -200};
    const double k0x_r = 0;
    const double k0x_2 = 1;
    const double rise_time = 0.05;
    const double rise_time_offset = 3;
    const int num_excitations = 16;
    const std::vector<int> transmon_excitations = {1, 1};
    //const double t_final = 10;
    //const int N_t = 100000;
    //const double t_final = 1*0.1785;
    const double t_final = 0.628;
    const int64_t N_t = 100000*t_final;
    const int64_t data_reduce_factor = 1;
    const std::vector<double> x0;
    // Negative means no limit
    const int max_f_evaluations = -1;

    const double sigmaFilter = 0.2;
    const double sigmaWindow = 0.1;
    const int N_iterations = 2000;

    auto saveDataToFile = [](double fidelity_under_optimization, const std::vector<double> &x) -> void
    {};

    const double min_omega_d = omega_d_ref-10;
    const double max_omega_d = omega_d_ref+10;
    const int num_omega_d = 40+1;
    const double step_omega_d = (max_omega_d - min_omega_d) / (num_omega_d - 1);

    std::vector<double> omega_d_array(num_omega_d, 0);
    std::vector<double> max_population_array(num_omega_d, 0);
    std::vector<double> time_for_max_population_array(num_omega_d, 0);
    std::vector<double> max_fidelity_array(num_omega_d, 0);
    std::vector<double> time_for_max_fidelity_array(num_omega_d, 0);
#ifndef USE_ROCM
    #pragma omp parallel for schedule(dynamic,1)
    for (int n = 0; n < num_omega_d; ++n) {
#else // USE_ROCM
    for (int n = 0; n < num_omega_d; ++n) {
#endif // USE_ROCM
        const double omega_d = min_omega_d + n*step_omega_d;
        JQFData data;
        switch (optAlg) {
        case OptimalControlAlgorithm::None:
        {
            if (adiabaticDrive) {
                //auto Omega_func = [rise_time,Omega] (double x) -> double {
                //    if (x > 0) {
                //        return Omega*smooth_f(x)/(smooth_f(x)+smooth_f(rise_time-x));
                //    } else {
                //        return 0;
                //    }
                //};
                auto Omega_func = [=] (double t) -> std::complex<double> {
                    return Omega*filteredSquarePulse(
                            t, rise_time_offset*rise_time,
                            t_final-rise_time_offset*rise_time, rise_time);
                };
                const double initial_excited_state_amplitude = 0;
                const double target_excited_state_amplitude = 1;
                data = control_with_jqf_time_dependent_Omega(
                        initial_excited_state_amplitude,
                        target_excited_state_amplitude,
                        kappa, gamma2, gammaInternal, gammaDephasing, g,
                        Omega_func, omega_d, omega_r, omega_1, omega_2,
                        transmon_anharmonicity, k0x_r, k0x_2, num_excitations,
                        transmon_excitations, t_final, N_t, data_reduce_factor);
            } else {
                data = jqf_simulation(AtomResonatorInitialState::AllDown, kappa,
                                      gamma2, gammaInternal, gammaDephasing, g,
                                      Omega, omega_d, omega_r, omega_1, omega_2,
                                      transmon_anharmonicity, k0x_r, k0x_2,
                                      num_excitations, transmon_excitations, t_final,
                                      N_t, data_reduce_factor);
            }
        }
        break;
        case OptimalControlAlgorithm::GaussianPulse:
        {
            const double Omega_initial = 0.8*Omega;
            const double Omega_max = Omega;
            const double numParams = 1;
            data = control_with_jqf_optimize_Omega_gaussian_pulse(
                    kappa, gamma2, gammaInternal, gammaDephasing, g, Omega_max,
                    sigmaFilter, numParams, omega_d, omega_r, omega_1, omega_2,
                    transmon_anharmonicity, k0x_r, k0x_2,
                    num_excitations, transmon_excitations, t_final, N_t,
                    data_reduce_factor);
        }
        break;
        case OptimalControlAlgorithm::ADRK4:
        {
            std::vector<std::complex<double>> Omega_array(N_t, 0);
            for (int n = 0; n < N_t; ++n) {
                Omega_array[n] = std::complex<double>(Omega_max/2, Omega_max/2);
            }
            data = control_with_jqf_optimize_array_complex_Omega_adrk4(
                    kappa, gamma2, gammaInternal, gammaDephasing, g,
                    Omega_array, Omega_max, sigmaFilter, sigmaWindow,
                    N_amplitudes, omega_d, omega_r, omega_1, omega_2,
                    transmon_anharmonicity, k0x_r, k0x_2, num_excitations,
                    transmon_excitations, t_final, N_t, data_reduce_factor, x0,
                    max_f_evaluations, saveDataToFile);
        }
        break;
        case OptimalControlAlgorithm::Krotov:
        {
            std::vector<std::complex<double>> Omega_array(N_t, std::complex<double>(Omega_max/2, Omega_max/2));
            std::vector<double> learning_rate(N_t, 100);
            data = control_with_jqf_optimize_array_complex_Omega(kappa, gamma2, g, Omega_array, learning_rate, N_iterations, omega_d, omega_r, omega_1, omega_2, transmon_anharmonicity, k0x_r, k0x_2, num_excitations, transmon_excitations, t_final, N_t, data_reduce_factor);
        }
        break;
        default:
            assert(0 && "Unknown optimal control algorithm!");
        }
        const int64_t N_t = data.time.size();
        double max_population = 0;
        double time_for_max_population = 0;
        for (int i = 0; i < N_t; ++i) {
            if (max_population < data.aa_populations[0][i]) {
                max_population = data.aa_populations[0][i];
                time_for_max_population = data.time[i];
            }
        }
        double max_fidelity = 0;
        double time_for_max_fidelity = 0;
        for (int i = 0; i < N_t; ++i) {
            if (max_fidelity < data.F[0][i]) {
                max_fidelity = data.F[0][i];
                time_for_max_fidelity = data.time[i];
            }
        }
        omega_d_array[n] = omega_d;
        max_population_array[n] = max_population;
        time_for_max_population_array[n] = time_for_max_population;
        max_fidelity_array[n] = max_fidelity;
        time_for_max_fidelity_array[n] = time_for_max_fidelity;
        #pragma omp critical
        {
            std::cout << "omega_d = " << omega_d << ", max_fidelity = " << max_fidelity << std::endl;
        }
    }

    std::stringstream OmegaStream;
    std::string optAlgString = optimalControlAlgorithmToString(optAlg);
    if (optAlgString.empty()) {
        OmegaStream << Omega;
    } else {
        OmegaStream << optAlgString;
    }
    std::stringstream fileNameStream;
    fileNameStream << "control_with_jqf_omegad";
    if (adiabaticDrive && optAlg == OptimalControlAlgorithm::None) {
        fileNameStream << "_adiabatic";
    }
    fileNameStream << "_kappa_" << kappa
                   << "_gamma2_" << gamma2
                   << "_g_" << g
                   << "_Omega_" << OmegaStream.str()
                   << "_omegar_" << omega_r
                   << "_omega1_" << omega_1
                   << "_omega2_" << omega_2
                   << "_kxr_" << k0x_r
                   << "_kx2_" << k0x_2;
    fileNameStream << ".csv";
    const std::string fileName = fileNameStream.str();
    savetxt(fileName,
            {omega_d_array, max_population_array,
             time_for_max_population_array, max_fidelity_array,
             time_for_max_fidelity_array},
            ';', "omega_d;max_population;time_for_max_population;max_fidelity;time_for_max_fidelity");
    std::cout << "Wrote to " << fileName << std::endl;
}

struct find_optimal_omega_d_params
{
    double kappa;
    double gamma2;
    std::vector<double> gammaInternal;
    std::vector<double> gammaDephasing;
    double g;
    double Omega;
    double omega_r;
    double omega_1;
    double omega_2;
    std::vector<double> transmon_anharmonicity;
    double k0x_r;
    double k0x_2;
    int num_excitations;
    std::vector<int> transmon_excitations;
    double t;
    int64_t N_t;
};

double find_optimal_omega_d_f(unsigned n, const double *x, double *grad, void *params)
{
    find_optimal_omega_d_params *p
            = (find_optimal_omega_d_params *) params;
    const double omega_d = x[0];
    const double kappa = p->kappa;
    const double gamma2 = p->gamma2;
    const double g = p->g;
    const double Omega = p->Omega;
    const double omega_r = p->omega_r;
    const double omega_1 = p->omega_1;
    const double omega_2 = p->omega_2;
    const double k0x_r = p->k0x_r;
    const double k0x_2 = p->k0x_2;
    const int num_excitations = p->num_excitations;
    const double t = p->t;
    const int64_t N_t = p->N_t;
    const int64_t data_reduce_factor = 1;
    JQFData data = jqf_simulation(AtomResonatorInitialState::AllDown, kappa,
                                  gamma2, p->gammaInternal, p->gammaDephasing,
                                  g, Omega, omega_d, omega_r, omega_1, omega_2,
                                  p->transmon_anharmonicity, k0x_r, k0x_2,
                                  num_excitations, p->transmon_excitations, t,
                                  N_t, data_reduce_factor);
    double max_fidelity = 0;
    for (int i = 0; i < N_t; ++i) {
        if (max_fidelity < data.F[0][i]) {
            max_fidelity = data.F[0][i];
        }
    }
    //std::cout << "  Omega = " << Omega << ", omega_d = " << omega_d << ", max_fidelity = " << max_fidelity << std::endl;
    return -max_fidelity;
}

double find_optimal_omega_d(find_optimal_omega_d_params *params, double omega_d_guess)
{
    const double omega_d_step = 1;
    const int numParams = 1;
    nlopt::opt opt(nlopt::LN_SBPLX, numParams);
    std::vector<double> x(numParams);
    x[0] = omega_d_guess;
    std::vector<double> step(numParams);
    step[0] = omega_d_step;
    opt.set_initial_step(step);

    //std::vector<double> lb(numParams);
    //lb[0] = 0;
    //opt.set_lower_bounds(lb);

    std::vector<double> ub(numParams);

    ub[0] = 0;
    opt.set_upper_bounds(ub);

    opt.set_min_objective(find_optimal_omega_d_f, params);
    opt.set_xtol_abs(1e-14);

    double minf;
    nlopt::result result = opt.optimize(x, minf);
    const double omega_d_opt = x[0];
    return omega_d_opt;
}

void control_with_jqf_Omega()
{
    const bool adiabaticDrive = false;
    const bool optimize_omega_d = true;
    const double kappa = 1;
    const double gamma2 = 0;
    const std::vector<double> gammaInternal = {0, 0};
    const std::vector<double> gammaDephasing = {0, 0};
    const double g = 10;
    //const double Omega = 100;
    const double omega_r = 5000;
    const double omega_1 = 4900;
    const double omega_2 = 4900;
    const double omega_d_ref = omega_1-3;
    const std::vector<double> transmon_anharmonicity = {-200, -200};
    const double k0x_r = 0;
    const double k0x_2 = 1;
    const double rise_time = 1;
    const int num_excitations = 16;
    const std::vector<int> transmon_excitations = {1, 1};
    const double t = 2;
    const int64_t N_t = 20000;
    const int64_t data_reduce_factor = 1;

    const double min_Omega = 10;
    const double max_Omega = 160;
    const int num_Omega = 150+1;
    const double step_Omega = (max_Omega - min_Omega) / (num_Omega - 1);

    std::vector<double> Omega_array(num_Omega, 0);
    std::vector<double> omega_d_array(num_Omega, 0);
    std::vector<double> max_population_array(num_Omega, 0);
    std::vector<double> time_for_max_population_array(num_Omega, 0);
    std::vector<double> max_fidelity_array(num_Omega, 0);
    std::vector<double> time_for_max_fidelity_array(num_Omega, 0);
    //#pragma omp parallel for schedule(dynamic)
    for (int n = 0; n < num_Omega; ++n) {
        const double Omega = min_Omega + n*step_Omega;
        double omega_d_opt = omega_d_ref;
        if (optimize_omega_d) {
            find_optimal_omega_d_params params;
            params.kappa = kappa;
            params.gamma2 = gamma2;
            params.g = g;
            params.gammaInternal = gammaInternal;
            params.gammaDephasing = gammaDephasing;
            params.Omega = Omega;
            params.omega_r = omega_r;
            params.omega_1 = omega_1;
            params.omega_2 = omega_2;
            params.transmon_anharmonicity = transmon_anharmonicity;
            params.k0x_r = k0x_r;
            params.k0x_2 = k0x_2;
            params.num_excitations = num_excitations;
            params.transmon_excitations = transmon_excitations;
            params.t = t;
            params.N_t = N_t;
            //TODO: handle adiabaticDrive = true
            omega_d_opt = find_optimal_omega_d(&params, omega_d_ref);
        }
        JQFData data;
        if (adiabaticDrive) {
            auto Omega_func = [rise_time,Omega] (double x) -> std::complex<double> {
                if (x > 0) {
                    return Omega*smooth_f(x)/(smooth_f(x)+smooth_f(rise_time-x));
                } else {
                    return 0;
                }
            };
            const double initial_excited_state_amplitude = 0;
            const double target_excited_state_amplitude = 1;
            data = control_with_jqf_time_dependent_Omega(
                    initial_excited_state_amplitude,
                    target_excited_state_amplitude,
                    kappa, gamma2, gammaInternal, gammaDephasing, g, Omega_func,
                    omega_d_opt, omega_r, omega_1, omega_2,
                    transmon_anharmonicity, k0x_r, k0x_2, num_excitations,
                    transmon_excitations, t, N_t, data_reduce_factor);
        } else {
            data = jqf_simulation(AtomResonatorInitialState::AllDown, kappa,
                                  gamma2, gammaInternal, gammaDephasing, g,
                                  Omega, omega_d_opt, omega_r, omega_1, omega_2,
                                  transmon_anharmonicity, k0x_r, k0x_2,
                                  num_excitations, transmon_excitations, t, N_t,
                                  data_reduce_factor);
        }
        const int64_t N_t = data.time.size();
        double max_population = 0;
        double time_for_max_population = 0;
        for (int i = 0; i < N_t; ++i) {
            if (max_population < data.aa_populations[0][i]) {
                max_population = data.aa_populations[0][i];
                time_for_max_population = data.time[i];
            }
        }
        double max_fidelity = 0;
        double time_for_max_fidelity = 0;
        for (int i = 0; i < N_t; ++i) {
            if (max_fidelity < data.F[0][i]) {
                max_fidelity = data.F[0][i];
                time_for_max_fidelity = data.time[i];
            }
        }
        Omega_array[n] = Omega;
        omega_d_array[n] = omega_d_opt;
        max_population_array[n] = max_population;
        time_for_max_population_array[n] = time_for_max_population;
        max_fidelity_array[n] = max_fidelity;
        time_for_max_fidelity_array[n] = time_for_max_fidelity;
    }

    std::stringstream fileNameStream;
    fileNameStream << "control_with_jqf_Omega";
    if (adiabaticDrive) {
        fileNameStream << "_adiabatic";
    }
    fileNameStream << "_kappa_" << kappa
                   << "_gamma2_" << gamma2
                   << "_g_" << g;
    if (optimize_omega_d) {
        fileNameStream << "_omegad_opt";
    } else {
        fileNameStream << "_omegad_" << omega_d_ref;
    }
    fileNameStream << "_omegar_" << omega_r
                   << "_omega1_" << omega_1
                   << "_omega2_" << omega_2
                   << "_kxr_" << k0x_r
                   << "_kx2_" << k0x_2;
    fileNameStream << ".csv";
    const std::string fileName = fileNameStream.str();
    savetxt(fileName,
            {Omega_array, omega_d_array, max_population_array,
             time_for_max_population_array, max_fidelity_array,
             time_for_max_fidelity_array},
            ';', "Omega;omega_d;max_population;time_for_max_population;max_fidelity;time_for_max_fidelity");
    std::cout << "Wrote to " << fileName << std::endl;
}

void control_with_jqf_omega_d_Omega()
{
    const double kappa = 1;
    const double gamma2 = 0;
    const std::vector<double> gammaInternal = {0, 0};
    const std::vector<double> gammaDephasing = {0, 0};
    const double g = 10;
    //const double Omega = 100;
    const double omega_r = 5000;
    const double omega_1 = 4900;
    const double omega_2 = 4900;
    // Reference value for omega_d
    const double omega_d_ref = omega_1;
    const std::vector<double> transmon_anharmonicity = {-200, -200};
    const double k0x_r = 0;
    const double k0x_2 = 1;
    const int num_excitations = 16;
    const std::vector<int> transmon_excitations = {1, 1};
    const double t = 1;
    const int64_t N_t = 10000;
    const int64_t data_reduce_factor = 1;

    const double min_omega_d = omega_d_ref-20;
    const double max_omega_d = omega_d_ref;
    const int num_omega_d = 20+1;
    const double step_omega_d = (max_omega_d - min_omega_d) / (num_omega_d - 1);

    const double min_Omega = 80;
    const double max_Omega = 160;
    const int num_Omega = 80+1;
    const double step_Omega = (max_Omega - min_Omega) / (num_Omega - 1);

    const int num_data_points = num_omega_d*num_Omega;

    std::vector<double> omega_d_array(num_data_points, 0);
    std::vector<double> Omega_array(num_data_points, 0);
    std::vector<double> max_population_array(num_data_points, 0);
    std::vector<double> time_for_max_population_array(num_data_points, 0);
    std::vector<double> max_fidelity_array(num_data_points, 0);
    std::vector<double> time_for_max_fidelity_array(num_data_points, 0);
    #pragma omp parallel for
    for (int m = 0; m < num_Omega; ++m) {
        #pragma omp critical
        {
            std::cout << "m = " << m << std::endl;
        }
        const double Omega = min_Omega + m*step_Omega;
        for (int n = 0; n < num_omega_d; ++n) {
            const double omega_d = min_omega_d + n*step_omega_d;
            JQFData data = jqf_simulation(AtomResonatorInitialState::AllDown,
                                          kappa, gamma2, gammaInternal,
                                          gammaDephasing, g, Omega, omega_d,
                                          omega_r, omega_1, omega_2,
                                          transmon_anharmonicity, k0x_r, k0x_2,
                                          num_excitations, transmon_excitations,
                                          t, N_t, data_reduce_factor);
            const int N_t = data.time.size();
            double max_population = 0;
            double time_for_max_population = 0;
            for (int i = 0; i < N_t; ++i) {
                if (max_population < data.aa_populations[0][i]) {
                    max_population = data.aa_populations[0][i];
                    time_for_max_population = data.time[i];
                }
            }
            double max_fidelity = 0;
            double time_for_max_fidelity = 0;
            for (int i = 0; i < N_t; ++i) {
                if (max_fidelity < data.F[0][i]) {
                    max_fidelity = data.F[0][i];
                    time_for_max_fidelity = data.time[i];
                }
            }
            const int index = n + num_omega_d*m;
            Omega_array[index] = Omega;
            omega_d_array[index] = omega_d;
            max_population_array[index] = max_population;
            time_for_max_population_array[index] = time_for_max_population;
            max_fidelity_array[index] = max_fidelity;
            time_for_max_fidelity_array[index] = time_for_max_fidelity;
        }
    }

    std::stringstream fileNameStream;
    fileNameStream << "control_with_jqf";
    fileNameStream << "_omegad_Omega";
    fileNameStream << "_kappa_" << kappa
                   << "_gamma2_" << gamma2
                   << "_g_" << g
                   << "_omegar_" << omega_r
                   << "_omega1_" << omega_1
                   << "_omega2_" << omega_2
                   << "_kxr_" << k0x_r;
    fileNameStream << ".csv";
    const std::string fileName = fileNameStream.str();
    savetxt(fileName,
            {Omega_array, omega_d_array, max_population_array,
            time_for_max_population_array, max_fidelity_array,
            time_for_max_fidelity_array}, ';',
            "Omega;omega_d;max_population;time_for_max_population;max_fidelity;time_for_max_fidelity");
    std::cout << "Wrote to " << fileName << std::endl;
}

int main() {
    control_with_jqf_article_plots();
    //control_with_jqf_omega_d();
    //control_with_jqf_Omega();
    //control_with_jqf_omega_d_Omega();
    return 0;
}
