#ifndef MEASURE_WITH_JQF_H
#define MEASURE_WITH_JQF_H

#include <vector>
#include <complex>
#include <functional>

#define ONLY_ONE_DETERMINISTIC_TRAJECTORY_EULER   0
#define ONLY_ONE_DETERMINISTIC_TRAJECTORY_RK4    -1
#define ONLY_ONE_DETERMINISTIC_TRAJECTORY_SS     -2

#define MEASURE_WITH_JQF_SSE         (1 << 0)

struct MeasureWithJQFData
{
    std::vector<double> time_array;
    std::vector<std::vector<double>> I_deterministic;
    std::vector<std::vector<double>> res_population;
    std::vector<std::vector<double>> population_1;
    std::vector<std::vector<double>> population_2;
    std::vector<std::vector<double>> reflection_coefficient_re;
    std::vector<std::vector<double>> reflection_coefficient_im;
    std::vector<std::vector<std::vector<double>>> S_arrays;
    std::vector<double> integration_times_sorted;
    bool replaced_negative_frequencies_with_positive;
};

MeasureWithJQFData measure_with_jqf(
        double kappa, double kappaInternal, double gamma2,
        double gamma2Internal, double gamma2Dephasing, double g, double Omega,
        double omega_d, double omega_r, double omega_1, double omega_2,
        const std::vector<double> &transmon_anharmonicity,
        double k0x_r, double k0x_2, double phi, int num_excitations,
        const std::vector<int> &transmon_excitations, int N_trajectories,
        int batch_size, const std::vector<double> &integration_times,
        int64_t N_t_max, int flags);

#endif // MEASURE_WITH_JQF_H
