#include <atomic>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <cassert>
#include <cmath>

#include <nlopt.hpp>

#include "measure_with_jqf.h"
#include "io.h"

void measure_with_jqf_specific_params()
{
    int flags = 0;
    //flags |= MEASURE_WITH_JQF_SSE;
    const double kappa = 1;
    const double kappaInternal = 0;
    const double gamma2 = 50;
    const double gamma2Internal = 0;
    const double gamma2Dephasing = 0;
    const double g = 54.772;
    //const double Omega = 0.1;
    const double Omega = 0.5;
    //const double Omega = 8;
    //const double Omega = 5;
    const double omega_r = 5000;
    const double omega_1 = 4000;
    const double omega_2 = 3997;
    const double omega_d = 5000;
    const std::vector<double> transmon_anharmonicity = {-200, -200};
    const double k0x_r = 0;
    const double k0x_2 = 1;
    const double phi = 0;
    //const int num_excitations = 4;
    const int num_excitations = 8;
    //const int num_excitations = 500;
    //const int num_excitations = 200;
    const std::vector<int> transmon_excitations = {4, 1};
    //const int N_trajectories = 0; // zero means only calculating deterministic evolution
    const int N_trajectories = ONLY_ONE_DETERMINISTIC_TRAJECTORY_RK4;
    const double t = 40;
    const int batch_size = 1;
    const int64_t N_t_max = 1e4*t;
    std::vector<double> integration_times = { 1, 2, 5, t };
    MeasureWithJQFData data = measure_with_jqf(
                kappa, kappaInternal, gamma2,
                gamma2Internal, gamma2Dephasing, g, Omega, omega_d,
                omega_r, omega_1, omega_2, transmon_anharmonicity,
                k0x_r, k0x_2, phi, num_excitations,
                transmon_excitations, N_trajectories, batch_size,
                integration_times, N_t_max, flags);

    std::stringstream fileNameStream;
    fileNameStream << "measure_with_jqf";
    fileNameStream << "_kappa_" << kappa
                   << "_kappaI_" << kappaInternal
                   << "_gamma2_" << gamma2
                   << "_gamma2I_" << gamma2Internal
                   << "_gamma2D_" << gamma2Dephasing
                   << "_g_" << g
                   << "_Omega_" << Omega
                   << "_omegad_" << omega_d
                   << "_omegar_" << omega_r
                   << "_omega1_" << omega_1
                   << "_omega2_" << omega_2
                   << "_kxr_" << k0x_r
                   << "_kx2_" << k0x_2
                   << "_phi_" << phi;
    fileNameStream << ".csv";
    const std::string fileName = fileNameStream.str();
    savetxt(fileName,
            {data.time_array,
             data.I_deterministic[0], data.population_1[0],
             data.population_2[0], data.res_population[0],
             data.reflection_coefficient_re[0],
             data.reflection_coefficient_im[0],
             data.I_deterministic[1], data.population_1[1],
             data.population_2[1], data.res_population[1],
             data.reflection_coefficient_re[1],
             data.reflection_coefficient_im[1],
             }, ';',
             "t;"
             "I_deterministic_0;population_1_0;"
             "population_2_0;res_population_0;"
             "r_re_0;"
             "r_im_0;"
             "I_deterministic_1;population_1_1;"
             "population_2_1;res_population_1;"
             "r_re_1;"
             "r_im_1");
    std::cout << "Wrote to " << fileName << std::endl;
    if (data.S_arrays.empty()) {
        return;
    }
    assert(data.S_arrays[0].size() == data.S_arrays[1].size()
            && "S arrays are not of the same size!");
    const int S_arrays_size = data.S_arrays[0].size();
    for (int i = 0; i < S_arrays_size; ++i) {
        std::stringstream fileNameStream;
        fileNameStream << "measure_with_jqf_S";
        if (flags & MEASURE_WITH_JQF_SSE) {
            fileNameStream << "_sse";
        }
        fileNameStream << "_kappa_" << kappa
                       << "_kappaI_" << kappaInternal
                       << "_gamma2_" << gamma2
                       << "_gamma2I_" << gamma2Internal
                       << "_gamma2D_" << gamma2Dephasing
                       << "_g_" << g
                       << "_Omega_" << Omega
                       << "_omegad_" << omega_d
                       << "_omegar_" << omega_r
                       << "_omega1_" << omega_1
                       << "_omega2_" << omega_2
                       << "_kxr_" << k0x_r
                       << "_kx2_" << k0x_2
                       << "_phi_" << phi
                       << "_t_" << data.integration_times_sorted[i];
        fileNameStream << ".csv";
        const std::string fileName = fileNameStream.str();
        savetxt(fileName,
                {data.S_arrays[0][i], data.S_arrays[1][i]},
                ';', "S_0;S_1");
        std::cout << "Wrote to " << fileName << std::endl;
    }
}

void measure_with_jqf_omega_d(double gamma2, double Omega, int num_excitations)
{
    int flags = 0;
    //flags |= MEASURE_WITH_JQF_SSE;
    const double kappa = 1;
    const double kappaInternal = 0;
    const double gamma2Internal = 0;
    const double gamma2Dephasing = 0;
    const double g = 54.772;
    const double omega_r = 5000;
    const double omega_1 = 4000;
    //const double omega_2 = 4000;
    const double omega_2 = 3997;
    // Reference value for omega_d
    const double omega_d_ref = omega_r;
    const std::vector<double> transmon_anharmonicity = {-200, -200};
    const double k0x_r = 0;
    const double k0x_2 = 1;
    const double phi = 0;
    const std::vector<int> transmon_excitations = {4, 1};
    //const int N_trajectories = ONLY_ONE_DETERMINISTIC_TRAJECTORY_EULER;
    const int N_trajectories = ONLY_ONE_DETERMINISTIC_TRAJECTORY_RK4;
    //const int N_trajectories = ONLY_ONE_DETERMINISTIC_TRAJECTORY_SS;
    const double t = 20;
    const int batch_size = 1;
    const int64_t N_t_max = 1e4*t;
    std::vector<double> integration_times = { t };

    const double min_omega_d = omega_d_ref+1.5;
    const double max_omega_d = omega_d_ref+3.5;
    const int num_omega_d = 401;
    const double step_omega_d = (max_omega_d - min_omega_d) / (num_omega_d - 1);

    std::vector<double> omega_d_array(num_omega_d, 0);
    std::vector<double> last_r_re_0_array(num_omega_d, 0);
    std::vector<double> last_r_im_0_array(num_omega_d, 0);
    std::vector<double> last_r_re_1_array(num_omega_d, 0);
    std::vector<double> last_r_im_1_array(num_omega_d, 0);
    std::vector<double> last_population_1_0_array(num_omega_d, 0);
    std::vector<double> last_population_1_1_array(num_omega_d, 0);
    std::vector<double> last_population_2_0_array(num_omega_d, 0);
    std::vector<double> last_population_2_1_array(num_omega_d, 0);
    std::vector<double> last_res_population_0_array(num_omega_d, 0);
    std::vector<double> last_res_population_1_array(num_omega_d, 0);
    std::atomic_flag replaced_negative_frequencies_with_positive = ATOMIC_FLAG_INIT;
#ifndef USE_ROCM
    #pragma omp parallel for schedule(dynamic,1)
    for (int n = 0; n < num_omega_d; ++n) {
#else // USE_ROCM
    for (int n = 0; n < num_omega_d; ++n) {
#endif // USE_ROCM
        const double omega_d = min_omega_d + n*step_omega_d;
        MeasureWithJQFData data = measure_with_jqf(
                    kappa, kappaInternal, gamma2,
                    gamma2Internal, gamma2Dephasing, g, Omega, omega_d,
                    omega_r, omega_1, omega_2, transmon_anharmonicity,
                    k0x_r, k0x_2, phi, num_excitations,
                    transmon_excitations, N_trajectories, batch_size,
                    integration_times, N_t_max, flags);
        // N_t_ret = 1 != N_t when ONLY_ONE_DETERMINISTIC_TRAJECTORY_SS is set
        const int64_t N_t_ret = data.time_array.size();
        omega_d_array[n] = omega_d;
        last_r_re_0_array[n] = data.reflection_coefficient_re[0][N_t_ret-1];
        last_r_im_0_array[n] = data.reflection_coefficient_im[0][N_t_ret-1];
        last_r_re_1_array[n] = data.reflection_coefficient_re[1][N_t_ret-1];
        last_r_im_1_array[n] = data.reflection_coefficient_im[1][N_t_ret-1];
        last_population_1_0_array[n] = data.population_1[0][N_t_ret-1];
        last_population_1_1_array[n] = data.population_1[1][N_t_ret-1];
        last_population_2_0_array[n] = data.population_2[0][N_t_ret-1];
        last_population_2_1_array[n] = data.population_2[1][N_t_ret-1];
        last_res_population_0_array[n] = data.res_population[0][N_t_ret-1];
        last_res_population_1_array[n] = data.res_population[1][N_t_ret-1];
        if (data.replaced_negative_frequencies_with_positive) {
            replaced_negative_frequencies_with_positive.test_and_set(
                    std::memory_order_relaxed);
        }
    }
    if (replaced_negative_frequencies_with_positive.test_and_set(
                    std::memory_order_relaxed)) {
        std::cout << "Warning: replaced negative eigenfrequencies with a "
                  << "positive drive frequency" << std::endl;
    }

    std::stringstream fileNameStream;
    fileNameStream << "measure_with_jqf_omegad";
    if (N_trajectories == ONLY_ONE_DETERMINISTIC_TRAJECTORY_SS) {
        fileNameStream << "_ss";
    }
    fileNameStream << "_kappa_" << kappa
                   << "_kappaI_" << kappaInternal
                   << "_gamma2_" << gamma2
                   << "_gamma2I_" << gamma2Internal
                   << "_gamma2D_" << gamma2Dephasing
                   << "_g_" << g
                   << "_Omega_" << Omega
                   << "_omegar_" << omega_r
                   << "_omega1_" << omega_1
                   << "_omega2_" << omega_2
                   << "_kxr_" << k0x_r
                   << "_kx2_" << k0x_2
                   << "_phi_" << phi;
    fileNameStream << ".csv";
    const std::string fileName = fileNameStream.str();
    savetxt(fileName,
            {omega_d_array,
             last_r_re_0_array, last_r_im_0_array,
             last_r_re_1_array, last_r_im_1_array,
             last_population_1_0_array, last_population_1_1_array,
             last_population_2_0_array, last_population_2_1_array,
             last_res_population_0_array, last_res_population_1_array
             }, ';',
             "omega_d;"
             "r_re_0;r_im_0;"
             "r_re_1;r_im_1;"
             "last_population_1_0;last_population_1_1;"
             "last_population_2_0;last_population_2_1;"
             "last_res_population_0;last_res_population_1"
             );
    std::cout << "Wrote to " << fileName << std::endl;
}

struct find_optimal_omega_d_params
{
    double max_angle;
    MeasureWithJQFData *data;
    double kappa;
    double kappaInternal;
    double gamma2;
    double gamma2Internal;
    double gamma2Dephasing;
    double g;
    double Omega;
    double omega_r;
    double omega_1;
    double omega_2;
    std::vector<double> transmon_anharmonicity;
    double k0x_r;
    double k0x_2;
    double phi;
    int num_excitations;
    std::vector<int> transmon_excitations;
    int N_trajectories;
    int batch_size;
    int64_t N_t_max;
    std::vector<double> integration_times;
    int flags;
};

double find_optimal_omega_d_f(unsigned n, const double *x, double *grad, void *params)
{
    find_optimal_omega_d_params *p
            = (find_optimal_omega_d_params *) params;
    const double kappa = p->kappa;
    const double kappaInternal = p->kappaInternal;
    const double gamma2 = p->gamma2;
    const double gamma2Internal = p->gamma2Internal;
    const double gamma2Dephasing = p->gamma2Dephasing;
    const double g = p->g;
    const double Omega = p->Omega;
    const double omega_r = p->omega_r;
    const double omega_1 = p->omega_1;
    const double omega_2 = p->omega_2;
    const double k0x_r = p->k0x_r;
    const double k0x_2 = p->k0x_2;
    const double phi = p->phi;
    const int num_excitations = p->num_excitations;
    const int N_trajectories = p->N_trajectories;
    const int batch_size = p->batch_size;
    const int64_t N_t_max = p->N_t_max;
    const int flags = p->flags;
    const double omega_d = x[0];
    MeasureWithJQFData data = measure_with_jqf(
                kappa, kappaInternal, gamma2,
                gamma2Internal, gamma2Dephasing, g, Omega, omega_d,
                omega_r, omega_1, omega_2, p->transmon_anharmonicity,
                k0x_r, k0x_2, phi, num_excitations,
                p->transmon_excitations, N_trajectories, batch_size,
                p->integration_times, N_t_max, flags);
    // N_t_ret = 1 != N_t when ONLY_ONE_DETERMINISTIC_TRAJECTORY_SS is set
    const int64_t N_t_ret = p->data->time_array.size();
    const double r0_re = data.reflection_coefficient_re[0][N_t_ret-1];
    const double r0_im = data.reflection_coefficient_im[0][N_t_ret-1];
    const double r1_re = data.reflection_coefficient_re[1][N_t_ret-1];
    const double r1_im = data.reflection_coefficient_im[1][N_t_ret-1];
    const double prod = r0_re*r1_re+r0_im*r1_im;
    const double mag0 = std::sqrt(r0_re*r0_re + r0_im*r0_im);
    const double mag1 = std::sqrt(r1_re*r1_re + r1_im*r1_im);
    const double cosTheta = prod/(mag0*mag1);
    const double angle = std::acos(cosTheta);
    if (p->max_angle < angle) {
        p->max_angle = angle;
        *(p->data) = data;
    }
    std::cout << "  omega_d = " << omega_d << ", angle/pi = " << angle/M_PI << std::endl;
    return -angle;
}


void measure_with_jqf_Omega(double gamma2, double omega_d, double tolerance)
{
    const bool optimize_omega_d = false;
    int flags = 0;
    //flags |= MEASURE_WITH_JQF_SSE;
    const double kappa = 1;
    const double kappaInternal = 0;
    const double gamma2Internal = 0;
    const double gamma2Dephasing = 0;
    const double g = 54.772;
    const double omega_r = 5000;
    const double omega_1 = 4000;
    const double omega_2 = 3997;
    const std::vector<double> transmon_anharmonicity = {-200, -200};
    const double k0x_r = 0;
    const double k0x_2 = 1;
    const double phi = 0;
    const std::vector<int> transmon_excitations = {4, 1};
    //const int N_trajectories = ONLY_ONE_DETERMINISTIC_TRAJECTORY_EULER;
    const int N_trajectories = ONLY_ONE_DETERMINISTIC_TRAJECTORY_RK4;
    //const int N_trajectories = ONLY_ONE_DETERMINISTIC_TRAJECTORY_SS;
    const double t = 20;
    const int batch_size = 1;
    const int64_t N_t_max = 1e4*t;
    std::vector<double> integration_times = { t };

    const double min_Omega = 0.1;
    const double max_Omega = 2;
    const int num_Omega = 40;
    const double step_Omega = (max_Omega - min_Omega) / (num_Omega - 1);

    std::vector<double> Omega_array(num_Omega, 0);
    std::vector<double> omega_d_array(num_Omega, 0);
    std::vector<double> last_r_re_0_array(num_Omega, 0);
    std::vector<double> last_r_im_0_array(num_Omega, 0);
    std::vector<double> last_r_re_1_array(num_Omega, 0);
    std::vector<double> last_r_im_1_array(num_Omega, 0);
    std::vector<double> last_population_1_0_array(num_Omega, 0);
    std::vector<double> last_population_1_1_array(num_Omega, 0);
    std::vector<double> last_population_2_0_array(num_Omega, 0);
    std::vector<double> last_population_2_1_array(num_Omega, 0);
    std::vector<double> last_res_population_0_array(num_Omega, 0);
    std::vector<double> last_res_population_1_array(num_Omega, 0);
    std::atomic_flag replaced_negative_frequencies_with_positive = ATOMIC_FLAG_INIT;
    // Make this loop serial, because the parameters make the simulation
    // so slow that is has to run on a GPU to complete in a
    // reasonable time. When running on GPU (USE_ROCM is defined),
    // it does not make sense to parallelize the CPU code (several
    // GPUs are not handled correctly).
    // TODO: If one tries to run on a CPU, it might help to parallelize
    // by uncommenting the #pragma below, but it was not tested.
    //#pragma omp parallel for schedule(dynamic,1)
    for (int n = 0; n < num_Omega; ++n) {
        int excitation_power = 2;
        const double Omega = min_Omega + n*step_Omega;
        int num_excitations = 1 << excitation_power;
        MeasureWithJQFData data = measure_with_jqf(
                    kappa, kappaInternal, gamma2,
                    gamma2Internal, gamma2Dephasing, g, Omega, omega_d,
                    omega_r, omega_1, omega_2, transmon_anharmonicity,
                    k0x_r, k0x_2, phi, num_excitations,
                    transmon_excitations, N_trajectories, batch_size,
                    integration_times, N_t_max, flags);
        // N_t_ret = 1 != N_t when ONLY_ONE_DETERMINISTIC_TRAJECTORY_SS is set
        int64_t N_t_ret = data.time_array.size();
        double last_r_re_0 = data.reflection_coefficient_re[0][N_t_ret-1];
        double last_r_im_0 = data.reflection_coefficient_im[0][N_t_ret-1];
        double last_r_re_1 = data.reflection_coefficient_re[1][N_t_ret-1];
        double last_r_im_1 = data.reflection_coefficient_im[1][N_t_ret-1];
        double max_diff = 0;
        ++excitation_power;
        for (; excitation_power < 10; ++excitation_power) {
            int num_excitations = 1 << excitation_power;
            data = measure_with_jqf(
                        kappa, kappaInternal, gamma2,
                        gamma2Internal, gamma2Dephasing, g, Omega, omega_d,
                        omega_r, omega_1, omega_2, transmon_anharmonicity,
                        k0x_r, k0x_2, phi, num_excitations,
                        transmon_excitations, N_trajectories, batch_size,
                        integration_times, N_t_max, flags);
            N_t_ret = data.time_array.size();
            const double new_r_re_0 = data.reflection_coefficient_re[0][N_t_ret-1];
            const double new_r_im_0 = data.reflection_coefficient_im[0][N_t_ret-1];
            const double new_r_re_1 = data.reflection_coefficient_re[1][N_t_ret-1];
            const double new_r_im_1 = data.reflection_coefficient_im[1][N_t_ret-1];
            const double r_re_0_diff = std::abs(last_r_re_0 - new_r_re_0);
            const double r_im_0_diff = std::abs(last_r_im_0 - new_r_im_0);
            const double r_re_1_diff = std::abs(last_r_re_1 - new_r_re_1);
            const double r_im_1_diff = std::abs(last_r_im_1 - new_r_im_1);
            max_diff = r_re_0_diff;
            if (max_diff < r_im_0_diff) {
                max_diff = r_im_0_diff;
            }
            if (max_diff < r_re_1_diff) {
                max_diff = r_re_1_diff;
            }
            if (max_diff < r_im_1_diff) {
                max_diff = r_im_1_diff;
            }
            if (max_diff < tolerance) {
                #pragma omp critical
                {
                    std::cout << "n = " << n << ", Omega = " << Omega
                              << ", max_diff = " << max_diff
                              << ", num_excitations = " << num_excitations
                              << std::endl;
                }
                break;
            }
            last_r_re_0 = new_r_re_0;
            last_r_im_0 = new_r_im_0;
            last_r_re_1 = new_r_re_1;
            last_r_im_1 = new_r_im_1;
        }
        double omega_d_opt = omega_d;
        if (optimize_omega_d) {
            int num_excitations = 1 << excitation_power;
            find_optimal_omega_d_params params;
            params.max_angle = 0;
            params.data = &data;
            params.kappa = kappa;
            params.kappaInternal = kappaInternal;
            params.gamma2 = gamma2;
            params.gamma2Internal = gamma2Internal;
            params.gamma2Dephasing = gamma2Dephasing;
            params.g = g;
            params.Omega = Omega;
            params.omega_r = omega_r;
            params.omega_1 = omega_1;
            params.omega_2 = omega_2;
            params.transmon_anharmonicity = transmon_anharmonicity;
            params.k0x_r = k0x_r;
            params.k0x_2 = k0x_2;
            params.phi = phi;
            params.num_excitations = num_excitations;
            params.transmon_excitations = transmon_excitations;
            params.N_trajectories = N_trajectories;
            params.batch_size = batch_size;
            params.N_t_max = N_t_max;
            params.integration_times = integration_times;
            params.flags = flags;
            const int numParams = 1;
            nlopt::opt opt(nlopt::LN_BOBYQA, numParams);
            std::vector<double> x(numParams, 0);
            x[0] = omega_d;
            std::vector<double> step(numParams, 0.01);
            opt.set_initial_step(step);
            opt.set_min_objective(find_optimal_omega_d_f, &params);
            //opt.set_xtol_abs(1e-14);
            opt.set_ftol_abs(1e-5);
            double minf;
            nlopt::result result = opt.optimize(x, minf);
            omega_d_opt = x[0];
        }
        Omega_array[n] = Omega;
        omega_d_array[n] = omega_d_opt;
        last_r_re_0_array[n] = data.reflection_coefficient_re[0][N_t_ret-1];
        last_r_im_0_array[n] = data.reflection_coefficient_im[0][N_t_ret-1];
        last_r_re_1_array[n] = data.reflection_coefficient_re[1][N_t_ret-1];
        last_r_im_1_array[n] = data.reflection_coefficient_im[1][N_t_ret-1];
        last_population_1_0_array[n] = data.population_1[0][N_t_ret-1];
        last_population_1_1_array[n] = data.population_1[1][N_t_ret-1];
        last_population_2_0_array[n] = data.population_2[0][N_t_ret-1];
        last_population_2_1_array[n] = data.population_2[1][N_t_ret-1];
        last_res_population_0_array[n] = data.res_population[0][N_t_ret-1];
        last_res_population_1_array[n] = data.res_population[1][N_t_ret-1];
        if (data.replaced_negative_frequencies_with_positive) {
            replaced_negative_frequencies_with_positive.test_and_set(
                    std::memory_order_relaxed);
        }
    }
    if (replaced_negative_frequencies_with_positive.test_and_set(
                    std::memory_order_relaxed)) {
        std::cout << "Warning: replaced negative eigenfrequencies with a "
                  << "positive drive frequency" << std::endl;
    }

    std::stringstream fileNameStream;
    fileNameStream << "measure_with_jqf_Omega";
    if (N_trajectories == ONLY_ONE_DETERMINISTIC_TRAJECTORY_SS) {
        fileNameStream << "_ss";
    }
    fileNameStream << "_kappa_" << kappa
                   << "_kappaI_" << kappaInternal
                   << "_gamma2_" << gamma2
                   << "_gamma2I_" << gamma2Internal
                   << "_gamma2D_" << gamma2Dephasing
                   << "_g_" << g
                   << "_omegar_" << omega_r
                   << "_omega1_" << omega_1
                   << "_omega2_" << omega_2;
    if (optimize_omega_d) {
        fileNameStream << "_omegad_opt";
    } else {
        fileNameStream << "_omegad_" << omega_d;
    }
    fileNameStream << "_kxr_" << k0x_r
                   << "_kx2_" << k0x_2
                   << "_phi_" << phi;
    fileNameStream << ".csv";
    const std::string fileName = fileNameStream.str();
    savetxt(fileName,
            {Omega_array, omega_d_array,
             last_r_re_0_array, last_r_im_0_array,
             last_r_re_1_array, last_r_im_1_array,
             last_population_1_0_array, last_population_1_1_array,
             last_population_2_0_array, last_population_2_1_array,
             last_res_population_0_array, last_res_population_1_array
             }, ';',
             "Omega;omega_d;"
             "r_re_0;r_im_0;"
             "r_re_1;r_im_1;"
             "last_population_1_0;last_population_1_1;"
             "last_population_2_0;last_population_2_1;"
             "last_res_population_0;last_res_population_1"
             );
    std::cout << "Wrote to " << fileName << std::endl;
}

void measure_with_jqf_g(double gamma2, double omega_d, double Omega, double tolerance)
{
    const bool optimize_omega_d = true;
    int flags = 0;
    //flags |= MEASURE_WITH_JQF_SSE;
    const double kappa = 1;
    const double kappaInternal = 0;
    const double gamma2Internal = 0;
    const double gamma2Dephasing = 0;
    const double omega_r = 5000;
    const double omega_1 = 4000;
    const double omega_2 = 3997;
    const std::vector<double> transmon_anharmonicity = {-200, -200};
    const double k0x_r = 0;
    const double k0x_2 = 1;
    const double phi = 0;
    const std::vector<int> transmon_excitations = {4, 1};
    //const int N_trajectories = ONLY_ONE_DETERMINISTIC_TRAJECTORY_EULER;
    const int N_trajectories = ONLY_ONE_DETERMINISTIC_TRAJECTORY_RK4;
    //const int N_trajectories = ONLY_ONE_DETERMINISTIC_TRAJECTORY_SS;
    const double t = 20;
    const int batch_size = 1;
    const int64_t N_t_max = 1e4*t;
    std::vector<double> integration_times = { t };

    const double min_g = 54.2;
    const double max_g = 56.2;
    const int num_g = 21;
    const double step_g = (max_g - min_g) / (num_g - 1);

    std::vector<double> g_array(num_g, 0);
    std::vector<double> omega_d_array(num_g, 0);
    std::vector<double> last_r_re_0_array(num_g, 0);
    std::vector<double> last_r_im_0_array(num_g, 0);
    std::vector<double> last_r_re_1_array(num_g, 0);
    std::vector<double> last_r_im_1_array(num_g, 0);
    std::vector<double> last_population_1_0_array(num_g, 0);
    std::vector<double> last_population_1_1_array(num_g, 0);
    std::vector<double> last_population_2_0_array(num_g, 0);
    std::vector<double> last_population_2_1_array(num_g, 0);
    std::vector<double> last_res_population_0_array(num_g, 0);
    std::vector<double> last_res_population_1_array(num_g, 0);
    std::atomic_flag replaced_negative_frequencies_with_positive = ATOMIC_FLAG_INIT;
    // Make this loop serial, because the parameters make the simulation
    // so slow that is has to run on a GPU to complete in a
    // reasonable time. When running on GPU (USE_ROCM is defined),
    // it does not make sense to parallelize the CPU code (several
    // GPUs are not handled correctly).
    // TODO: If one tries to run on a CPU, it might help to parallelize
    // by uncommenting the #pragma below, but it was not tested.
    //#pragma omp parallel for schedule(dynamic,1)
    for (int n = 0; n < num_g; ++n) {
        int excitation_power = 2;
        const double g = min_g + n*step_g;
        int num_excitations = 1 << excitation_power;
        MeasureWithJQFData data = measure_with_jqf(
                    kappa, kappaInternal, gamma2,
                    gamma2Internal, gamma2Dephasing, g, Omega, omega_d,
                    omega_r, omega_1, omega_2, transmon_anharmonicity,
                    k0x_r, k0x_2, phi, num_excitations,
                    transmon_excitations, N_trajectories, batch_size,
                    integration_times, N_t_max, flags);
        // N_t_ret = 1 != N_t when ONLY_ONE_DETERMINISTIC_TRAJECTORY_SS is set
        int64_t N_t_ret = data.time_array.size();
        double last_r_re_0 = data.reflection_coefficient_re[0][N_t_ret-1];
        double last_r_im_0 = data.reflection_coefficient_im[0][N_t_ret-1];
        double last_r_re_1 = data.reflection_coefficient_re[1][N_t_ret-1];
        double last_r_im_1 = data.reflection_coefficient_im[1][N_t_ret-1];
        double max_diff = 0;
        ++excitation_power;
        for (; excitation_power < 10; ++excitation_power) {
            int num_excitations = 1 << excitation_power;
            data = measure_with_jqf(
                        kappa, kappaInternal, gamma2,
                        gamma2Internal, gamma2Dephasing, g, Omega, omega_d,
                        omega_r, omega_1, omega_2, transmon_anharmonicity,
                        k0x_r, k0x_2, phi, num_excitations,
                        transmon_excitations, N_trajectories, batch_size,
                        integration_times, N_t_max, flags);
            N_t_ret = data.time_array.size();
            const double new_r_re_0 = data.reflection_coefficient_re[0][N_t_ret-1];
            const double new_r_im_0 = data.reflection_coefficient_im[0][N_t_ret-1];
            const double new_r_re_1 = data.reflection_coefficient_re[1][N_t_ret-1];
            const double new_r_im_1 = data.reflection_coefficient_im[1][N_t_ret-1];
            const double r_re_0_diff = std::abs(last_r_re_0 - new_r_re_0);
            const double r_im_0_diff = std::abs(last_r_im_0 - new_r_im_0);
            const double r_re_1_diff = std::abs(last_r_re_1 - new_r_re_1);
            const double r_im_1_diff = std::abs(last_r_im_1 - new_r_im_1);
            max_diff = r_re_0_diff;
            if (max_diff < r_im_0_diff) {
                max_diff = r_im_0_diff;
            }
            if (max_diff < r_re_1_diff) {
                max_diff = r_re_1_diff;
            }
            if (max_diff < r_im_1_diff) {
                max_diff = r_im_1_diff;
            }
            if (max_diff < tolerance) {
                #pragma omp critical
                {
                    std::cout << "n = " << n << ", g = " << g
                              << ", max_diff = " << max_diff
                              << ", num_excitations = " << num_excitations
                              << std::endl;
                }
                break;
            }
            last_r_re_0 = new_r_re_0;
            last_r_im_0 = new_r_im_0;
            last_r_re_1 = new_r_re_1;
            last_r_im_1 = new_r_im_1;
        }
        double omega_d_opt = omega_d;
        if (optimize_omega_d) {
            int num_excitations = 1 << excitation_power;
            find_optimal_omega_d_params params;
            params.max_angle = 0;
            params.data = &data;
            params.kappa = kappa;
            params.kappaInternal = kappaInternal;
            params.gamma2 = gamma2;
            params.gamma2Internal = gamma2Internal;
            params.gamma2Dephasing = gamma2Dephasing;
            params.g = g;
            params.Omega = Omega;
            params.omega_r = omega_r;
            params.omega_1 = omega_1;
            params.omega_2 = omega_2;
            params.transmon_anharmonicity = transmon_anharmonicity;
            params.k0x_r = k0x_r;
            params.k0x_2 = k0x_2;
            params.phi = phi;
            params.num_excitations = num_excitations;
            params.transmon_excitations = transmon_excitations;
            params.N_trajectories = N_trajectories;
            params.batch_size = batch_size;
            params.N_t_max = N_t_max;
            params.integration_times = integration_times;
            params.flags = flags;
            const int numParams = 1;
            nlopt::opt opt(nlopt::LN_BOBYQA, numParams);
            std::vector<double> x(numParams, 0);
            x[0] = omega_d;
            std::vector<double> step(numParams, 0.01);
            opt.set_initial_step(step);
            opt.set_min_objective(find_optimal_omega_d_f, &params);
            //opt.set_xtol_abs(1e-14);
            opt.set_ftol_abs(1e-5);
            double minf;
            nlopt::result result = opt.optimize(x, minf);
            omega_d_opt = x[0];
        }
        g_array[n] = g;
        omega_d_array[n] = omega_d_opt;
        last_r_re_0_array[n] = data.reflection_coefficient_re[0][N_t_ret-1];
        last_r_im_0_array[n] = data.reflection_coefficient_im[0][N_t_ret-1];
        last_r_re_1_array[n] = data.reflection_coefficient_re[1][N_t_ret-1];
        last_r_im_1_array[n] = data.reflection_coefficient_im[1][N_t_ret-1];
        last_population_1_0_array[n] = data.population_1[0][N_t_ret-1];
        last_population_1_1_array[n] = data.population_1[1][N_t_ret-1];
        last_population_2_0_array[n] = data.population_2[0][N_t_ret-1];
        last_population_2_1_array[n] = data.population_2[1][N_t_ret-1];
        last_res_population_0_array[n] = data.res_population[0][N_t_ret-1];
        last_res_population_1_array[n] = data.res_population[1][N_t_ret-1];
        if (data.replaced_negative_frequencies_with_positive) {
            replaced_negative_frequencies_with_positive.test_and_set(
                    std::memory_order_relaxed);
        }
    }
    if (replaced_negative_frequencies_with_positive.test_and_set(
                    std::memory_order_relaxed)) {
        std::cout << "Warning: replaced negative eigenfrequencies with a "
                  << "positive drive frequency" << std::endl;
    }

    std::stringstream fileNameStream;
    fileNameStream << "measure_with_jqf_g";
    if (N_trajectories == ONLY_ONE_DETERMINISTIC_TRAJECTORY_SS) {
        fileNameStream << "_ss";
    }
    fileNameStream << "_kappa_" << kappa
                   << "_kappaI_" << kappaInternal
                   << "_gamma2_" << gamma2
                   << "_gamma2I_" << gamma2Internal
                   << "_gamma2D_" << gamma2Dephasing
                   << "_Omega_" << Omega
                   << "_omegar_" << omega_r
                   << "_omega1_" << omega_1
                   << "_omega2_" << omega_2;
    if (optimize_omega_d) {
        fileNameStream << "_omegad_opt";
    } else {
        fileNameStream << "_omegad_" << omega_d;
    }
    fileNameStream << "_kxr_" << k0x_r
                   << "_kx2_" << k0x_2
                   << "_phi_" << phi;
    fileNameStream << ".csv";
    const std::string fileName = fileNameStream.str();
    savetxt(fileName,
            {g_array, omega_d_array,
             last_r_re_0_array, last_r_im_0_array,
             last_r_re_1_array, last_r_im_1_array,
             last_population_1_0_array, last_population_1_1_array,
             last_population_2_0_array, last_population_2_1_array,
             last_res_population_0_array, last_res_population_1_array
             }, ';',
             "g;omega_d;"
             "r_re_0;r_im_0;"
             "r_re_1;r_im_1;"
             "last_population_1_0;last_population_1_1;"
             "last_population_2_0;last_population_2_1;"
             "last_res_population_0;last_res_population_1"
             );
    std::cout << "Wrote to " << fileName << std::endl;
}

void measure_with_and_without_jqf_omega_d(double Omega, int num_excitations)
{
    const double gamma2 = 50;
    measure_with_jqf_omega_d(0, Omega, num_excitations);
    measure_with_jqf_omega_d(gamma2, Omega, num_excitations);
}

void measure_with_and_without_jqf_Omega(double omega_d, double tolerance)
{
    const double gamma2 = 50;
    measure_with_jqf_Omega(0, omega_d, tolerance);
    measure_with_jqf_Omega(gamma2, omega_d, tolerance);
}


void measure_with_and_without_jqf_g(double omega_d, double Omega, double tolerance)
{
    const double gamma2 = 50;
    measure_with_jqf_g(0, omega_d, Omega, tolerance);
    measure_with_jqf_g(gamma2, omega_d, Omega, tolerance);
}

void measure_with_jqf_omega_d_article_plots()
{
    const double Omega = 2; // Omega = 4 MHz (Omega/kappa = 2)
    const int num_excitations = 64;
    // Fig. 5(a)
    measure_with_and_without_jqf_omega_d(Omega, num_excitations);
    const double tolerance = 1e-2;
    const double omega_d = 5002.5;
    // Fig. 5(b)
    measure_with_and_without_jqf_Omega(omega_d, tolerance);
    // Fig. 5(c)
    const double Omega_low = 0.1; // Omega = 0.2 MHz (Omega/kappa = 0.1)
    measure_with_and_without_jqf_g(omega_d, Omega_low, tolerance);
    measure_with_and_without_jqf_g(omega_d, Omega, tolerance);
}

int main()
{
    //measure_with_jqf_specific_params();
    //measure_with_jqf_omega_d();
    measure_with_jqf_omega_d_article_plots();
    return 0;
}
