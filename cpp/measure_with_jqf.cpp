#include "measure_with_jqf.h"

#include <string>
#include <vector>
#include <random>

#include "operator.h"

#include "Eigen/SparseCore"
#include "Eigen/IterativeLinearSolvers"

#include "jqf_superoperator.h"
#ifdef EIGEN_USE_MKL_ALL
#include "mkl_support.h"
#endif // EIGEN_USE_MKL_ALL
#ifdef USE_ROCM
#include "qroc/master_equation_roc.h"
#include "smd_from_spmat.h"
#endif // USE_ROCM
#include "master_equation.h"
#include "rk4.h"

#define DEFAULT_SEED 7868759

#define USE_MILSTEIN_METHOD

//The difference between calculating the expectation
//value of the homodyne current operator before or
//after the time step becomes smaller with smaller
//time steps. This is because the expectation values
//will be integrated, i.e., multiplied by the time step and
//summed. The expectation value of the homodyne current
//operator calculated after the time step is the same
//one that is anyway calculated before the next time
//step (because it is needed to propagate the state).
//Hence, almost the same terms in the integral/sum
//are used either way, and the only difference is
//negligigle initial and final values (negligible
//because they are multiplied by the small time step).
//#define CALCULATE_I_OP_EXPECTATION_AFTER_TIME_STEP

#define ALTERNATIVE_RR

std::tuple<SpMat, Eigen::VectorXcd> impose_tr_one_condition(const SpMat &L)
{
    // Make a new L, where any reference to
    // rho(<last element>)
    // is replaced with 1 minus all the other populations. This will
    // effectively impose the trace(rho)=1 condition.
    const int basis_size_squared = L.rows();
    const int last_rho_element = basis_size_squared-1;
    const int basis_size = std::sqrt(basis_size_squared);
    Eigen::VectorXcd rhoLHSFull = Eigen::VectorXcd::Zero(L.rows());
    std::vector<Eigen::Triplet<std::complex<double>>> triplets;
    triplets.reserve(L.nonZeros()+basis_size);
    for (int k = 0; k < L.outerSize(); ++k) {
        for (SpMat::InnerIterator it(L, k); it; ++it) {
            if (it.col() == last_rho_element && it.row() != last_rho_element) {
                rhoLHSFull(it.row()) = -it.value();
                for (int k = 0; k < last_rho_element; k += basis_size+1) {
                    triplets.emplace_back(it.row(), k, -it.value());
                }
            } else if (it.row() != last_rho_element) {
                // One of the elements on the diagonal has been made
                // redundant by imposing the tr(rho)=1 requirement,
                // so we remove the last equation.
                triplets.emplace_back(it.row(), it.col(), it.value());
            }
        }
    }
    Eigen::VectorXcd rhoLHS = Eigen::VectorXcd::Zero(L.rows()-1);
    const int rhoLHS_size = rhoLHS.size();
    for (int i = 0; i < rhoLHS_size; ++i) {
        rhoLHS(i) = rhoLHSFull(i);
    }
    SpMat L_tr_rho(rhoLHS_size, rhoLHS_size);
    L_tr_rho.setFromTriplets(triplets.begin(), triplets.end());

    return std::make_tuple(L_tr_rho, rhoLHS);
}

void calculate_steady_state(
        MeasureWithJQFData &ret,
        const SpMat &L,
        int basis_size,
        const SpMat &I_Op,
        const SpMat &M_n_res,
        const SpMat &M_n_1,
        const SpMat &M_n_2,
        const SpMat &M_r,
        std::complex<double> factor_r)
{
    const int basis_size_squared = basis_size*basis_size;
    assert(L.cols() == basis_size_squared
            && "Superoperator L matrix has wrong size!");
    SpMat L_tr_rho;
    Eigen::VectorXcd rhoLHS;
    std::tie(L_tr_rho, rhoLHS) = impose_tr_one_condition(L);

    Eigen::BiCGSTAB<SpMat> solver;
    // The default number of 2 times number of columns
    // is not enough sometimes.
    // (number of columns = basis_size_squared-1)
    solver.setMaxIterations(10*basis_size_squared);
    solver.compute(L_tr_rho);
    Eigen::VectorXcd rhoTruncated = solver.solve(rhoLHS);

    Eigen::VectorXcd rho = Eigen::VectorXcd::Zero(basis_size_squared);
    const int last_rho_element = basis_size_squared - 1;

    for (int i = 0; i < last_rho_element; ++i) {
        rho(i) = rhoTruncated(i);
    }
    double other_elements_trace = 0;
    for (int k = 0; k < basis_size_squared; k += basis_size+1) {
        other_elements_trace += rho(k).real();
    }
    rho(last_rho_element) = 1.0-other_elements_trace;

    Eigen::Map<MatrixXcdRowMajor> rho_map(rho.data(), basis_size, basis_size);

    const std::complex<double> I(0,1);
    for (int m = 0; m < 2; ++m) {
        const std::complex<double> I_deterministic = trace_of_product(I_Op,rho_map);
        const std::complex<double> res_population = trace_of_product(M_n_res,rho_map);
        const std::complex<double> population_1 = trace_of_product(M_n_1,rho_map);
        const std::complex<double> population_2 = trace_of_product(M_n_2,rho_map);
        const std::complex<double> M_r_trace = trace_of_product(M_r,rho_map);
        const std::complex<double> r = factor_r+M_r_trace;
        ret.time_array[0] = HUGE_VAL;
        ret.res_population[m][0] = res_population.real();
        ret.population_1[m][0] = population_1.real();
        ret.population_2[m][0] = population_2.real();
        ret.I_deterministic[m][0] = I_deterministic.real();
        ret.reflection_coefficient_re[m][0] = r.real();
        ret.reflection_coefficient_im[m][0] = r.imag();
    }
}

void calculate_deterministic_evolution_euler(
        MeasureWithJQFData &ret,
        const SpMat &L,
        int basis_size,
        const std::vector<Eigen::MatrixXcd> &rho_initial,
        const SpMat &I_Op,
        const SpMat &M_n_res,
        const SpMat &M_n_1,
        const SpMat &M_n_2,
        const SpMat &M_r,
        std::complex<double> factor_r,
        int64_t N_t_max,
        double dt)
{
#ifdef EIGEN_USE_MKL_ALL
    MKLSparseMatrix mklL(L);
#endif // EIGEN_USE_MKL_ALL
    const std::complex<double> I(0,1);
    const int basis_size_squared = basis_size*basis_size;
    assert(L.cols() == basis_size_squared
            && "Superoperator L matrix has wrong size!");
    for (int m = 0; m < 2; ++m) {
        Eigen::VectorXcd rho_vec = Eigen::VectorXcd::Zero(basis_size_squared);
        for (int i = 0; i < basis_size; ++i) {
            for (int j = 0; j < basis_size; ++j) {
                rho_vec(i*basis_size+j) = rho_initial[m](i,j);
            }
        }
        Eigen::VectorXcd temp = Eigen::VectorXcd::Zero(basis_size_squared);
        for (int64_t i = 0; i < N_t_max; ++i) {
#ifdef EIGEN_USE_MKL_ALL
            mklL.mul_vector(temp, rho_vec, dt);
            rho_vec = rho_vec + temp;
#else // EIGEN_USE_MKL_ALL
            temp = L*rho_vec*dt;
            rho_vec = rho_vec + temp;
#endif // EIGEN_USE_MKL_ALL
            Eigen::Map<MatrixXcdRowMajor> rho_map(rho_vec.data(), basis_size, basis_size);
            const std::complex<double> I_deterministic = trace_of_product(I_Op,rho_map);
            const std::complex<double> res_population = trace_of_product(M_n_res,rho_map);
            const std::complex<double> population_1 = trace_of_product(M_n_1,rho_map);
            const std::complex<double> population_2 = trace_of_product(M_n_2,rho_map);
            const std::complex<double> M_r_trace = trace_of_product(M_r,rho_map);
            const std::complex<double> r = factor_r+M_r_trace;
            // ret.time_array is set on every loop iteration, but
            // with the same values, so there is no problem.
            ret.time_array[i] = dt*(i+1);
            ret.res_population[m][i] = res_population.real();
            ret.population_1[m][i] = population_1.real();
            ret.population_2[m][i] = population_2.real();
            ret.I_deterministic[m][i] = I_deterministic.real();
            ret.reflection_coefficient_re[m][i] = r.real();
            ret.reflection_coefficient_im[m][i] = r.imag();
        }
    }
}

void calculate_deterministic_evolution_rk4(
        MeasureWithJQFData &ret,
        const SpMat &L,
        int basis_size,
        const std::vector<Eigen::MatrixXcd> &rho_initial,
        const SpMat &I_Op,
        const SpMat &M_n_res,
        const SpMat &M_n_1,
        const SpMat &M_n_2,
        const SpMat &M_r,
        std::complex<double> factor_r,
        int64_t N_t_max,
        double dt)
{
#ifdef EIGEN_USE_MKL_ALL
    MKLSparseMatrix mklL(L);
#endif // EIGEN_USE_MKL_ALL
    const std::complex<double> I(0,1);
    const int basis_size_squared = basis_size*basis_size;
    assert(L.cols() == basis_size_squared
            && "Superoperator L matrix has wrong size!");
    for (int m = 0; m < 2; ++m) {
        Eigen::VectorXcd rho_vec = Eigen::VectorXcd::Zero(basis_size_squared);
        for (int i = 0; i < basis_size; ++i) {
            for (int j = 0; j < basis_size; ++j) {
                rho_vec(i*basis_size+j) = rho_initial[m](i,j);
            }
        }
#ifdef USE_ROCM
        SparseMatrixData rocL = smd_from_spmat(L);
        std::vector<SparseMatrixData> M_operators;
        M_operators.push_back(smd_from_spmat(I_Op));
        M_operators.push_back(smd_from_spmat(M_n_res));
        M_operators.push_back(smd_from_spmat(M_n_1));
        M_operators.push_back(smd_from_spmat(M_n_2));
        M_operators.push_back(smd_from_spmat(M_r));
        std::vector<const double*> M_diag_operators;
        std::vector<const std::complex<double>*> fidelity_rhos;
        const int64_t iterationsBetweenDeviceSynchronize = 1e4;
        MasterEquationData data = evolve_master_equation_roc(
                basis_size, rocL, rho_vec.data(), M_operators, M_diag_operators,
                fidelity_rhos, dt, N_t_max, iterationsBetweenDeviceSynchronize);
        for (int64_t i = 0; i < N_t_max; ++i) {
            ret.time_array[i] = data.time[i];
            ret.I_deterministic[m][i] = data.M_values[0][i].real();
            ret.res_population[m][i] = data.M_values[1][i].real();
            ret.population_1[m][i] = data.M_values[2][i].real();
            ret.population_2[m][i] = data.M_values[3][i].real();
            const std::complex<double> M_r_trace = data.M_values[4][i];
            const std::complex<double> r = factor_r+M_r_trace;
            ret.reflection_coefficient_re[m][i] = r.real();
            ret.reflection_coefficient_im[m][i] = r.imag();
        }
#else // USE_ROCM
        Eigen::VectorXcd k1 = Eigen::VectorXcd::Zero(basis_size_squared);
        Eigen::VectorXcd k2 = Eigen::VectorXcd::Zero(basis_size_squared);
        Eigen::VectorXcd k3 = Eigen::VectorXcd::Zero(basis_size_squared);
        Eigen::VectorXcd k4 = Eigen::VectorXcd::Zero(basis_size_squared);
        Eigen::VectorXcd temp = Eigen::VectorXcd::Zero(basis_size_squared);
        for (int64_t i = 0; i < N_t_max; ++i) {
#ifdef EIGEN_USE_MKL_ALL
            rk4_step_mkl(rho_vec, mklL, k1, k2, k3, k4, temp, dt);
#else // EIGEN_USE_MKL_ALL
            rk4_step(rho_vec, L, k1, k2, k3, k4, temp, dt);
#endif // EIGEN_USE_MKL_ALL
            Eigen::Map<MatrixXcdRowMajor> rho_map(rho_vec.data(), basis_size, basis_size);
            const std::complex<double> I_deterministic = trace_of_product(I_Op,rho_map);
            const std::complex<double> res_population = trace_of_product(M_n_res,rho_map);
            const std::complex<double> population_1 = trace_of_product(M_n_1,rho_map);
            const std::complex<double> population_2 = trace_of_product(M_n_2,rho_map);
            const std::complex<double> M_r_trace = trace_of_product(M_r,rho_map);
            const std::complex<double> r = factor_r+M_r_trace;
            // ret.time_array is set on every loop iteration, but
            // with the same values, so there is no problem.
            ret.time_array[i] = dt*(i+1);
            ret.res_population[m][i] = res_population.real();
            ret.population_1[m][i] = population_1.real();
            ret.population_2[m][i] = population_2.real();
            ret.I_deterministic[m][i] = I_deterministic.real();
            ret.reflection_coefficient_re[m][i] = r.real();
            ret.reflection_coefficient_im[m][i] = r.imag();
        }
#endif // USE_ROCM
    }
}

void allocate_output_data_arrays(MeasureWithJQFData &ret, int64_t N_t_max)
{
    ret.time_array = std::vector<double>(N_t_max, 0);
    ret.I_deterministic.resize(2);
    ret.res_population.resize(2);
    ret.population_1.resize(2);
    ret.population_2.resize(2);
    ret.reflection_coefficient_re.resize(2);
    ret.reflection_coefficient_im.resize(2);
    for (int m = 0; m < 2; ++m) {
        ret.I_deterministic[m] = std::vector<double>(N_t_max, 0);
        ret.res_population[m] = std::vector<double>(N_t_max, 0);
        ret.population_1[m] = std::vector<double>(N_t_max, 0);
        ret.population_2[m] = std::vector<double>(N_t_max, 0);
        ret.reflection_coefficient_re[m] = std::vector<double>(N_t_max, 0);
        ret.reflection_coefficient_im[m] = std::vector<double>(N_t_max, 0);
    }
}

MeasureWithJQFData measure_with_jqf_single(
        double kappa, double kappaInternal, double gamma2,
        double gamma2Internal, double gamma2Dephasing, double g, double Omega,
        double omega_d, double omega_r, double omega_1, double omega_2,
        const std::vector<double> &transmon_anharmonicity,
        double k0x_r, double k0x_2, double phi, int num_excitations,
        const std::vector<int> &transmon_excitations, int N_trajectories,
        const std::vector<double> &integration_times, int64_t N_t_max)
{
    MeasureWithJQFData ret;
    const int flags = 0;
    const double J_x = 0;
    const double k0x_out = k0x_2;
    JQFSuperoperatorData sd = generate_superoperator_diag(
            {kappa}, {kappaInternal}, {gamma2}, {0, gamma2Internal},
            {0, gamma2Dephasing}, g, Omega, J_x, omega_d, {omega_r},
            {omega_1, omega_2}, transmon_anharmonicity, {k0x_r}, {k0x_2},
            k0x_out, num_excitations, transmon_excitations, flags);
    ret.replaced_negative_frequencies_with_positive
        = sd.replaced_negative_frequencies_with_positive;
    std::vector<BasisVector> basis = std::move(sd.basis);
    const int basis_size = basis.size();
    const int basis_size_squared = basis_size*basis_size;
    assert(sd.L.cols() == basis_size_squared
            && "Superoperator L matrix has wrong size!");
    Eigen::VectorXcd psi = Eigen::VectorXcd::Zero(basis_size);
    psi(0) = 1; // atoms and the resonator are in their ground states
    std::vector<Eigen::MatrixXcd> rho_initial(2);
    rho_initial[0] = psi*psi.adjoint();
    rho_initial[1] = sd.psi_up[0]*sd.psi_up[0].adjoint();

    const std::complex<double> I(0,1);
    SpMat M_a_out = std::sqrt(kappa)*sd.M_a[0]*std::cos(M_PI*k0x_r);
    if (gamma2 != 0) {
        M_a_out += std::sqrt(gamma2)*sd.M_b[1]*std::cos(M_PI*k0x_2);
    }
    M_a_out *= std::exp(std::complex<double>(0,-M_PI*phi));
    //M_a_out = std::sqrt(kappa)*M_a*std::exp(std::complex<double>(0,-M_PI*phi)); // Old version, for comparison
    const SpMat M_a_out_adjoint = M_a_out.adjoint();
    const SpMat I_Op = M_a_out+M_a_out_adjoint;
#ifndef ALTERNATIVE_RR
    SpMat Identity(basis_size, basis_size);
    Identity.setIdentity();
    const SpMat RR = Eigen::kroneckerProduct(M_a_out, Identity)+Eigen::kroneckerProduct(Identity, M_a_out_adjoint.transpose());
#endif // ALTERNATIVE_RR

    ret.integration_times_sorted = integration_times;
    std::sort(ret.integration_times_sorted.begin(), ret.integration_times_sorted.end());
    const double t = ret.integration_times_sorted.back();
    const double dt = t/N_t_max;

#ifdef EIGEN_USE_MKL_ALL
    MKLSparseMatrix mklL(sd.L);
#ifdef ALTERNATIVE_RR
    MKLSparseMatrix mklM_a_out(M_a_out);
#else // ALTERNATIVE_RR
    MKLSparseMatrix mklRR(RR);
#endif // ALTERNATIVE_RR
#endif // EIGEN_USE_MKL_ALL

    if (N_trajectories == ONLY_ONE_DETERMINISTIC_TRAJECTORY_SS) {
        // Steady state only needs one of each expectation value
        allocate_output_data_arrays(ret, 1);
    } else {
        allocate_output_data_arrays(ret, N_t_max);
    }
    if (N_trajectories == ONLY_ONE_DETERMINISTIC_TRAJECTORY_SS) {
        calculate_steady_state(ret, sd.L, basis_size, I_Op, sd.M_n_res[0],
                sd.M_n_atom[0], sd.M_n_atom[1], sd.M_r, sd.factor_r);
    }
    else if (N_trajectories == ONLY_ONE_DETERMINISTIC_TRAJECTORY_RK4) {
        calculate_deterministic_evolution_rk4(ret, sd.L, basis_size,
                rho_initial, I_Op, sd.M_n_res[0], sd.M_n_atom[0],
                sd.M_n_atom[1], sd.M_r, sd.factor_r, N_t_max, dt);
    } else {
        calculate_deterministic_evolution_euler(ret, sd.L, basis_size,
                rho_initial, I_Op, sd.M_n_res[0], sd.M_n_atom[0],
                sd.M_n_atom[1], sd.M_r, sd.factor_r, N_t_max, dt);
    }
    if (N_trajectories < 1) {
        return ret;
    }
    const int integration_times_size = integration_times.size();
    ret.S_arrays.resize(2);
    ret.S_arrays[0].resize(integration_times_size);
    ret.S_arrays[1].resize(integration_times_size);
    for (int i = 0; i < integration_times_size; ++i) {
        ret.S_arrays[0][i] = std::vector<double>(N_trajectories, 0);
        ret.S_arrays[1][i] = std::vector<double>(N_trajectories, 0);
    }
    const int seed = DEFAULT_SEED;
    #pragma omp parallel for
    for (int n = 0; n < N_trajectories; ++n) {
        std::mt19937 generator(seed+n);
        std::normal_distribution<double> distribution(0, std::sqrt(dt));
        for (int m = 0; m < 2; ++m) {
            Eigen::VectorXcd rho_vec = Eigen::VectorXcd::Zero(basis_size_squared);
            for (int i = 0; i < basis_size; ++i) {
                for (int j = 0; j < basis_size; ++j) {
                    rho_vec(i*basis_size+j) = rho_initial[m](i,j);
                }
            }
            double S_m = 0;
            int current_S_index = 0;
            Eigen::VectorXcd temp = Eigen::VectorXcd::Zero(basis_size_squared);
            Eigen::VectorXcd rr_applied_rho
                    = Eigen::VectorXcd::Zero(basis_size_squared);
#ifdef USE_MILSTEIN_METHOD
            Eigen::VectorXcd rr_applied_twice_rho
                    = Eigen::VectorXcd::Zero(basis_size_squared);
#endif // USE_MILSTEIN_METHOD
            for (int64_t i = 0; i < N_t_max; ++i) {
#ifdef EIGEN_USE_MKL_ALL
                mklL.mul_vector(temp, rho_vec, dt);
#else // EIGEN_USE_MKL_ALL
                temp = sd.L*rho_vec*dt;
#endif // EIGEN_USE_MKL_ALL
                Eigen::Map<MatrixXcdRowMajor> rr_applied_rho_map(
                        rr_applied_rho.data(), basis_size, basis_size);
#ifdef ALTERNATIVE_RR
                // Since r*\rho+\rho*r^\dagger
                // = r*\rho+(r*\rho^\dagger)^\dagger
                // = r*\rho+(r*\rho)^\dagger
                // we can calculate r*\rho and then
                // add its adjoint (r*\rho)^\dagger
#ifdef EIGEN_USE_MKL_ALL
                mklM_a_out.mul_matrix_row_major(rr_applied_rho.data(), basis_size, basis_size, rho_vec.data(), basis_size, 1);
#else // EIGEN_USE_MKL_ALL
                Eigen::Map<MatrixXcdRowMajor> rho_map0(rho_vec.data(), basis_size, basis_size);
                rr_applied_rho_map = M_a_out*rho_map0;
#endif // EIGEN_USE_MKL_ALL
                addAdjointInPlace(rr_applied_rho_map);
#else // ALTERNATIVE_RR
#ifdef EIGEN_USE_MKL_ALL
                mklRR.mul_vector(rr_applied_rho, rho_vec, 1);
#else // EIGEN_USE_MKL_ALL
                rr_applied_rho = RR*rho_vec;
#endif // EIGEN_USE_MKL_ALL
#endif // ALTERNATIVE_RR
                const std::complex<double> rr_applied_rho_trace
                    = rr_applied_rho_map.trace();
#ifdef USE_MILSTEIN_METHOD
                Eigen::Map<MatrixXcdRowMajor> rr_applied_twice_rho_map(
                        rr_applied_twice_rho.data(), basis_size, basis_size);
#ifdef ALTERNATIVE_RR
#ifdef EIGEN_USE_MKL_ALL
                mklM_a_out.mul_matrix_row_major(rr_applied_twice_rho.data(), basis_size, basis_size, rr_applied_rho.data(), basis_size, 1);
#else // EIGEN_USE_MKL_ALL
                rr_applied_twice_rho_map = M_a_out*rr_applied_rho_map;
#endif // EIGEN_USE_MKL_ALL
                addAdjointInPlace(rr_applied_twice_rho_map);
#else // ALTERNATIVE_RR
#ifdef EIGEN_USE_MKL_ALL
                mklRR.mul_vector(rr_applied_twice_rho, rr_applied_rho, 1);
#else // EIGEN_USE_MKL_ALL
                rr_applied_twice_rho = RR*rr_applied_rho;
#endif // EIGEN_USE_MKL_ALL
#endif // ALTERNATIVE_RR
                const std::complex<double> rr_applied_twice_rho_trace
                    = rr_applied_twice_rho_map.trace();
#endif // USE_MILSTEIN_METHOD
                const double DeltaW = distribution(generator);
#ifndef CALCULATE_I_OP_EXPECTATION_AFTER_TIME_STEP
                //Note that trace_of_product(I_Op,rho_map) == rr_applied_rho_trace
                //so the commented out code here gives the same value of I_times_dt_0
                //but in a less efficient way (using an additional call to
                //trace_of_product(I_Op,rho_map)) and is included for reference:
                //Eigen::Map<MatrixXcdRowMajor> rho_map(rho_vec.data(), basis_size, basis_size);
                //const std::complex<double> I_times_dt = trace_of_product(I_Op,rho_map)*dt+DeltaW;
                const std::complex<double> I_times_dt = rr_applied_rho_trace*dt+DeltaW;
#endif // CALCULATE_I_OP_EXPECTATION_AFTER_TIME_STEP
                temp += DeltaW*(rr_applied_rho - rr_applied_rho_trace*rho_vec);
#ifdef USE_MILSTEIN_METHOD
                const double DeltaW2 = DeltaW*DeltaW;
                const double rr_applied_rho_trace2 = rr_applied_rho_trace.real()
                        *rr_applied_rho_trace.real();
                const double milstein_factor = 0.5*(DeltaW2-dt);
                const double milstein_factor2 = 2.0*rr_applied_rho_trace2
                        -rr_applied_twice_rho_trace.real();
                const double milstein_factor3 = 2.0*rr_applied_rho_trace.real();
                temp += milstein_factor*(milstein_factor2*rho_vec-milstein_factor3*rr_applied_rho+rr_applied_twice_rho);
#endif // USE_MILSTEIN_METHOD
                rho_vec += temp;
#ifdef CALCULATE_I_OP_EXPECTATION_AFTER_TIME_STEP
                Eigen::Map<MatrixXcdRowMajor> rho_map(rho_vec.data(), basis_size, basis_size);
                const std::complex<double> I_times_dt = trace_of_product(I_Op,rho_map)*dt+DeltaW;
#endif // CALCULATE_I_OP_EXPECTATION_AFTER_TIME_STEP
                const double filter_factor = std::abs(ret.I_deterministic[0][i]-ret.I_deterministic[1][i]);
                if (i*dt > ret.integration_times_sorted[current_S_index]) {
                    ret.S_arrays[m][current_S_index][n] = S_m;
                    ++current_S_index;
                }
                S_m += I_times_dt.real()*filter_factor;
            }
            // Store the last S_m value
            ret.S_arrays[m][current_S_index][n] = S_m;
        }
    }
    return ret;
}

MeasureWithJQFData measure_with_jqf_batched(
        double kappa, double kappaInternal, double gamma2,
        double gamma2Internal, double gamma2Dephasing, double g, double Omega,
        double omega_d, double omega_r, double omega_1, double omega_2,
        const std::vector<double> &transmon_anharmonicity,
        double k0x_r, double k0x_2, double phi, int num_excitations,
        const std::vector<int> &transmon_excitations, int N_trajectories,
        int batch_size, const std::vector<double> &integration_times,
        int64_t N_t_max)
{
    MeasureWithJQFData ret;
    const int flags = 0;
    const double J_x = 0;
    const double k0x_out = k0x_2;
    JQFSuperoperatorData sd = generate_superoperator_diag(
            {kappa}, {kappaInternal}, {gamma2}, {0, gamma2Internal},
            {0, gamma2Dephasing}, g, Omega, J_x, omega_d, {omega_r},
            {omega_1, omega_2}, transmon_anharmonicity, {k0x_r}, {k0x_2},
            k0x_out, num_excitations, transmon_excitations, flags);
    ret.replaced_negative_frequencies_with_positive
        = sd.replaced_negative_frequencies_with_positive;
    std::vector<BasisVector> basis = std::move(sd.basis);
    const int basis_size = basis.size();
    const int basis_size_squared = basis_size*basis_size;
    assert(sd.L.cols() == basis_size_squared
            && "Superoperator L matrix has wrong size!");
    Eigen::VectorXcd psi = Eigen::VectorXcd::Zero(basis_size);
    psi(0) = 1; // atoms and the resonator are in their ground states
    std::vector<Eigen::MatrixXcd> rho_initial(2);
    rho_initial[0] = psi*psi.adjoint();
    rho_initial[1] = sd.psi_up[0]*sd.psi_up[0].adjoint();

    const std::complex<double> I(0,1);
    SpMat M_a_out = std::sqrt(kappa)*sd.M_a[0]*std::cos(M_PI*k0x_r);
    if (gamma2 != 0) {
        M_a_out += std::sqrt(gamma2)*sd.M_b[1]*std::cos(M_PI*k0x_2);
    }
    M_a_out *= std::exp(std::complex<double>(0,-M_PI*phi));
    //M_a_out = std::sqrt(kappa)*M_a*std::exp(std::complex<double>(0,-M_PI*phi)); // Old version, for comparison
    const SpMat M_a_out_adjoint = M_a_out.adjoint();
    const SpMat I_Op = M_a_out+M_a_out_adjoint;
    SpMat Identity(basis_size, basis_size);
    Identity.setIdentity();
    const SpMat RR = Eigen::kroneckerProduct(M_a_out, Identity)+Eigen::kroneckerProduct(Identity, M_a_out_adjoint.transpose());

    ret.integration_times_sorted = integration_times;
    std::sort(ret.integration_times_sorted.begin(), ret.integration_times_sorted.end());
    const double t = ret.integration_times_sorted.back();
    const double dt = t/N_t_max;

#ifdef EIGEN_USE_MKL_ALL
    MKLSparseMatrix mklL(sd.L);
    MKLSparseMatrix mklRR(RR);
#endif // EIGEN_USE_MKL_ALL

    allocate_output_data_arrays(ret, N_t_max);
    if (N_trajectories == ONLY_ONE_DETERMINISTIC_TRAJECTORY_RK4) {
        calculate_deterministic_evolution_rk4(ret, sd.L, basis_size,
                rho_initial, I_Op, sd.M_n_res[0], sd.M_n_atom[0],
                sd.M_n_atom[1], sd.M_r, sd.factor_r, N_t_max, dt);
    } else {
        calculate_deterministic_evolution_euler(ret, sd.L, basis_size,
                rho_initial, I_Op, sd.M_n_res[0], sd.M_n_atom[0],
                sd.M_n_atom[1], sd.M_r, sd.factor_r, N_t_max, dt);
    }
    if (N_trajectories < 1) {
        return ret;
    }
    const int integration_times_size = integration_times.size();
    ret.S_arrays.resize(2);
    ret.S_arrays[0].resize(integration_times_size);
    ret.S_arrays[1].resize(integration_times_size);
    for (int i = 0; i < integration_times_size; ++i) {
        ret.S_arrays[0][i] = std::vector<double>(N_trajectories, 0);
        ret.S_arrays[1][i] = std::vector<double>(N_trajectories, 0);
    }
    const int N_batches = N_trajectories/batch_size;
    assert(N_trajectories % batch_size == 0
           && "batch size should divide number of trajectories without a remainder");
    const int seed = DEFAULT_SEED;
    //#pragma omp parallel for
    for (int n = 0; n < N_batches; ++n) {
        std::mt19937 generator;
        std::normal_distribution<double> distribution(0, std::sqrt(dt));
        Eigen::MatrixXd DeltaW_matrix(batch_size, 2*N_t_max);
        //#pragma omp critical
        for (int l = 0; l < batch_size; ++l) {
            generator.seed(seed+batch_size*n+l);
            const int DeltaW_matrix_cols = DeltaW_matrix.cols();
            for (int i = 0; i < DeltaW_matrix_cols; ++i) {
                DeltaW_matrix(l, i) = distribution(generator);
            }
        }
        for (int m = 0; m < 2; ++m) {
            Eigen::MatrixXcd rho_vec = Eigen::MatrixXcd::Zero(basis_size_squared, batch_size);
            Eigen::VectorXcd rho_column = Eigen::VectorXcd::Zero(basis_size_squared);
            for (int i = 0; i < basis_size; ++i) {
                for (int j = 0; j < basis_size; ++j) {
                    rho_column(i*basis_size+j) = rho_initial[m](i,j);
                }
            }
            for (int l = 0; l < batch_size; ++l) {
                rho_vec.col(l) = rho_column;
            }
            int current_S_index = 0;
            Eigen::MatrixXcd temp = Eigen::MatrixXcd::Zero(basis_size_squared, batch_size);
            Eigen::MatrixXcd rr_applied_rho
                    = Eigen::MatrixXcd::Zero(basis_size_squared, batch_size);
            Eigen::VectorXd DeltaW = Eigen::VectorXd::Zero(batch_size);
            Eigen::VectorXd I_times_dt = Eigen::VectorXd::Zero(batch_size);
            Eigen::VectorXd rr_applied_rho_trace = Eigen::VectorXd::Zero(batch_size);
            Eigen::VectorXd S_m = Eigen::VectorXd::Zero(batch_size);
#ifdef USE_MILSTEIN_METHOD
            Eigen::MatrixXcd rr_applied_twice_rho
                    = Eigen::MatrixXcd::Zero(basis_size_squared, batch_size);
#endif // USE_MILSTEIN_METHOD
            for (int64_t i = 0; i < N_t_max; ++i) {
#ifdef EIGEN_USE_MKL_ALL
                mklL.mul_matrix(temp, rho_vec, dt);
#else // EIGEN_USE_MKL_ALL
                temp = sd.L*rho_vec*dt;
#endif // EIGEN_USE_MKL_ALL
                // TODO: Use Eigen::Map to map rho_vec to a matrix here?
                //       Then we wouldn't need RR and just multiply M_a_out
                //       on the (mapped) matrix.
#ifdef EIGEN_USE_MKL_ALL
                mklRR.mul_matrix(rr_applied_rho, rho_vec, 1);
#else // EIGEN_USE_MKL_ALL
                rr_applied_rho = RR*rho_vec;
#endif // EIGEN_USE_MKL_ALL
                for (int l = 0; l < batch_size; ++l) {
                    double trace = 0;
                    for (int j = 0; j < basis_size; ++j) {
                        trace += rr_applied_rho(j*basis_size+j, l).real();
                    }
                    rr_applied_rho_trace(l) = trace;
                }
#ifdef USE_MILSTEIN_METHOD
#ifdef EIGEN_USE_MKL_ALL
                //mklRR.mul_vector(rr_applied_twice_rho, rr_applied_rho, 1);
#else // EIGEN_USE_MKL_ALL
                rr_applied_twice_rho = RR*rr_applied_rho;
#endif // EIGEN_USE_MKL_ALL
                Eigen::Map<MatrixXcdRowMajor> rr_applied_twice_rho_map(
                        rr_applied_twice_rho.data(), basis_size, basis_size);
                const std::complex<double> rr_applied_twice_rho_trace
                    = rr_applied_twice_rho_map.trace();
#endif // USE_MILSTEIN_METHOD
                //for (int l = 0; l < batch_size; ++l) {
                //    DeltaW(l) = distribution(*generator[l]);
                //}
                DeltaW = DeltaW_matrix.col(N_t_max*m+i);
#ifndef CALCULATE_I_OP_EXPECTATION_AFTER_TIME_STEP
                //Note that trace_of_product(I_Op,rho_map) == rr_applied_rho_trace
                //so the commented out code here gives the same value of I_times_dt_0
                //but in a less efficient way (using an additional call to
                //trace_of_product(I_Op,rho_map)) and is included for reference:
                //Eigen::Map<MatrixXcdRowMajor> rho_map(rho_vec.data(), basis_size, basis_size);
                //const std::complex<double> I_times_dt = trace_of_product(I_Op,rho_map)*dt+DeltaW;
                I_times_dt = rr_applied_rho_trace*dt+DeltaW;
#endif // CALCULATE_I_OP_EXPECTATION_AFTER_TIME_STEP
                temp += (rr_applied_rho - rho_vec*rr_applied_rho_trace.asDiagonal())*DeltaW.asDiagonal();
#ifdef USE_MILSTEIN_METHOD
                //const double DeltaW2 = DeltaW*DeltaW;
                //const double rr_applied_rho_trace2 = rr_applied_rho_trace.real()
                //        *rr_applied_rho_trace.real();
                //temp += 0.5*((2.0*rr_applied_rho_trace2 - rr_applied_twice_rho_trace)*rho_vec-2.0*rr_applied_rho_trace*rr_applied_rho+rr_applied_twice_rho)*(DeltaW2-dt);
#endif // USE_MILSTEIN_METHOD
                rho_vec += temp;
#ifdef CALCULATE_I_OP_EXPECTATION_AFTER_TIME_STEP
                Eigen::Map<MatrixXcdRowMajor> rho_map(rho_vec.data(), basis_size, basis_size);
                const std::complex<double> I_times_dt = trace_of_product(I_Op,rho_map)*dt+DeltaW;
#endif // CALCULATE_I_OP_EXPECTATION_AFTER_TIME_STEP
                const double filter_factor = std::abs(ret.I_deterministic[0][i]-ret.I_deterministic[1][i]);
                if (i*dt > ret.integration_times_sorted[current_S_index]) {
                    for (int l = 0; l < batch_size; ++l) {
                        ret.S_arrays[m][current_S_index][batch_size*n+l] = S_m(l);
                    }
                    ++current_S_index;
                }
                S_m += I_times_dt*filter_factor;
            }
            // Store the last S_m value
            for (int l = 0; l < batch_size; ++l) {
                ret.S_arrays[m][current_S_index][batch_size*n+l] = S_m(l);
            }
        }
    }
    return ret;
}

MeasureWithJQFData measure_with_jqf_sse(
        double kappa, double gamma2, double g, double Omega, double omega_d,
        double omega_r, double omega_1, double omega_2, double k0x_r,
        double k0x_2, double phi, int num_excitations, int N_trajectories,
        const std::vector<double> &integration_times, int64_t N_t_max)
{
    MeasureWithJQFData ret;
    const int flags = 0;
    const std::complex<double> I(0,1);
    const std::vector<double> kappaInternal;
    const std::vector<double> gammaInternal;
    const std::vector<double> gammaDephasing;
    const double J_x = 0;
    const double k0x_out = k0x_2;
    std::vector<double> transmon_anharmonicity = {0, 0};
    std::vector<int> transmon_excitations = {1, 1};
    JQFSuperoperatorData sd = generate_superoperator_diag(
            {kappa}, kappaInternal, {gamma2}, gammaInternal,
            gammaDephasing, g, Omega, J_x, omega_d, {omega_r},
            {omega_1, omega_2}, transmon_anharmonicity, {k0x_r}, {k0x_2},
            k0x_out, num_excitations, transmon_excitations, flags);
    ret.replaced_negative_frequencies_with_positive
        = sd.replaced_negative_frequencies_with_positive;
    std::vector<BasisVector> basis = std::move(sd.basis);
    const int basis_size = basis.size();
    const int basis_size_squared = basis_size*basis_size;
    assert(sd.L.cols() == basis_size_squared
            && "Superoperator L matrix has wrong size!");
    Eigen::VectorXcd psi = Eigen::VectorXcd::Zero(basis_size);
    psi(0) = 1; // atoms and the resonator are in their ground states
    std::vector<Eigen::VectorXcd> psi_initial(2);
    psi_initial[0] = psi;
    psi_initial[1] = sd.psi_up[0];
    std::vector<Eigen::MatrixXcd> rho_initial(2);
    rho_initial[0] = psi*psi.adjoint();
    rho_initial[1] = sd.psi_up[0]*sd.psi_up[0].adjoint();

    const double Delta_r = omega_d-omega_r;
    const double Delta_1 = omega_d-omega_1;
    const double Delta_2 = omega_d-omega_2;
    const std::vector<double> Delta = { Delta_1, Delta_2 };
    SpMat H(basis_size, basis_size);
    // -\sum_j \Delta_j b_j^\dagger*b_j
    H += -Delta[0]*sd.M_n_atom[0];
    H += -Delta[1]*sd.M_n_atom[1];
    // - \Delta_r a_r^\dagger*a_r
    H += -Delta_r*sd.M_n_res[0];

    const SpMat temp_g_r = g*(sd.M_b_adjoint_a[0]+sd.M_b_a_adjoint[0]);
    H += temp_g_r;
    // - (b_2^\dagger\Omega(z_j)+\Omega(z_j)^* b_2)
    const double Omega_factor_r = sd.Omega_factors[0];
    const double Omega_factor_2 = sd.Omega_factors[1];
    const SpMat temp_2 = Omega_factor_2*(Omega*sd.M_b_adjoint[1]+std::conj(Omega)*sd.M_b[1]);
    H += temp_2;
    const SpMat temp_r = Omega_factor_r*(Omega*sd.M_a_adjoint[0]+std::conj(Omega)*sd.M_a[0]);
    H += temp_r;

    const SpMat M_eff = -I*H-0.5*kappa*sd.M_n_res[0];

    //SpMat M_a_out = std::sqrt(kappa)*sd.M_a[0]*std::cos(M_PI*k0x_r);
    //if (gamma2 != 0) {
    //    M_a_out += std::sqrt(gamma2)*sd.M_b[1]*std::cos(M_PI*k0x_2);
    //}
    //M_a_out *= std::exp(std::complex<double>(0,-M_PI*phi));
    SpMat M_a_out = std::sqrt(kappa)*sd.M_a[0]*std::exp(std::complex<double>(0,-M_PI*phi)); // Old version, for comparison
    const SpMat M_a_out_adjoint = M_a_out.adjoint();
    const SpMat I_Op = M_a_out+M_a_out_adjoint;

    ret.integration_times_sorted = integration_times;
    std::sort(ret.integration_times_sorted.begin(), ret.integration_times_sorted.end());
    const double t = ret.integration_times_sorted.back();
    const double dt = t/N_t_max;

#ifdef EIGEN_USE_MKL_ALL
    MKLSparseMatrix mklL(sd.L);
#endif // EIGEN_USE_MKL_ALL

    allocate_output_data_arrays(ret, N_t_max);
    if (N_trajectories == ONLY_ONE_DETERMINISTIC_TRAJECTORY_RK4) {
        calculate_deterministic_evolution_rk4(ret, sd.L, basis_size,
                rho_initial, I_Op, sd.M_n_res[0], sd.M_n_atom[0],
                sd.M_n_atom[1], sd.M_r, sd.factor_r, N_t_max, dt);
    } else {
        calculate_deterministic_evolution_euler(ret, sd.L, basis_size,
                rho_initial, I_Op, sd.M_n_res[0], sd.M_n_atom[0],
                sd.M_n_atom[1], sd.M_r, sd.factor_r, N_t_max, dt);
    }
    if (N_trajectories < 1) {
        return ret;
    }
    const int integration_times_size = integration_times.size();
    ret.S_arrays.resize(2);
    ret.S_arrays[0].resize(integration_times_size);
    ret.S_arrays[1].resize(integration_times_size);
    for (int i = 0; i < integration_times_size; ++i) {
        ret.S_arrays[0][i] = std::vector<double>(N_trajectories, 0);
        ret.S_arrays[1][i] = std::vector<double>(N_trajectories, 0);
    }
    const int seed = DEFAULT_SEED;
    #pragma omp parallel for
    for (int n = 0; n < N_trajectories; ++n) {
        std::mt19937 generator(seed+n);
        std::normal_distribution<double> distribution(0, std::sqrt(dt));
        Eigen::VectorXcd temp = Eigen::VectorXcd::Zero(basis_size);
        Eigen::VectorXcd c_applied_psi
                = Eigen::VectorXcd::Zero(basis_size);
        Eigen::VectorXcd I_Op_applied_psi
                = Eigen::VectorXcd::Zero(basis_size);
#ifdef USE_MILSTEIN_METHOD
        Eigen::VectorXcd c_applied_twice_psi
                = Eigen::VectorXcd::Zero(basis_size);
#endif // USE_MILSTEIN_METHOD
        for (int m = 0; m < 2; ++m) {
            Eigen::VectorXcd psi_vec = psi_initial[m];
            double S_m = 0;
            int current_S_index = 0;
            for (int64_t i = 0; i < N_t_max; ++i) {
                temp = M_eff*psi_vec*dt;
                c_applied_psi = M_a_out*psi_vec;
                I_Op_applied_psi = I_Op*psi_vec;
                const std::complex<double> I_Op_E_complex
                    = psi_vec.adjoint()*I_Op_applied_psi;
                // I_Op is self-adjoint so its expectation value is real
                const double I_Op_E = I_Op_E_complex.real();
                const double I_Op_E_sq = I_Op_E*I_Op_E;
#ifdef USE_MILSTEIN_METHOD
                c_applied_twice_psi = M_a_out*c_applied_psi;
#endif // USE_MILSTEIN_METHOD
                const double DeltaW = distribution(generator);
#ifndef CALCULATE_I_OP_EXPECTATION_AFTER_TIME_STEP
                const double I_times_dt = I_Op_E*dt+DeltaW;
#endif // CALCULATE_I_OP_EXPECTATION_AFTER_TIME_STEP
                temp += dt*(0.5*I_Op_E*c_applied_psi-0.125*I_Op_E_sq*psi_vec);
                temp += DeltaW*(c_applied_psi - 0.5*I_Op_E*psi_vec);
#ifdef USE_MILSTEIN_METHOD
                const double DeltaW2 = DeltaW*DeltaW;
                const double milstein_factor = 0.5*(DeltaW2-dt);
                temp += milstein_factor*(I_Op_E_sq*psi_vec-1.5*I_Op_E*c_applied_psi+c_applied_twice_psi);
#endif // USE_MILSTEIN_METHOD
                psi_vec += temp;
#ifdef CALCULATE_I_OP_EXPECTATION_AFTER_TIME_STEP
                temp = I_Op*psi_vec;
                const std::complex<double> I_Op_E_after_complex
                    = psi_vec.adjoint()*temp;
                const double I_Op_E_after = I_Op_E_after_complex.real();
                const double I_times_dt = I_Op_E_after*dt+DeltaW;
#endif // CALCULATE_I_OP_EXPECTATION_AFTER_TIME_STEP
                const double filter_factor = std::abs(ret.I_deterministic[0][i]-ret.I_deterministic[1][i]);
                if (i*dt > ret.integration_times_sorted[current_S_index]) {
                    ret.S_arrays[m][current_S_index][n] = S_m;
                    ++current_S_index;
                }
                S_m += I_times_dt*filter_factor;
            }
            // Store the last S_m value
            ret.S_arrays[m][current_S_index][n] = S_m;
        }
    }
    return ret;
}

MeasureWithJQFData measure_with_jqf(
        double kappa, double kappaInternal, double gamma2,
        double gamma2Internal, double gamma2Dephasing, double g, double Omega,
        double omega_d, double omega_r, double omega_1, double omega_2,
        const std::vector<double> &transmon_anharmonicity,
        double k0x_r, double k0x_2, double phi, int num_excitations,
        const std::vector<int> &transmon_excitations, int N_trajectories,
        int batch_size, const std::vector<double> &integration_times,
        int64_t N_t_max, int flags)
{
    MeasureWithJQFData data;
    if (flags & MEASURE_WITH_JQF_SSE) {
        assert(kappaInternal == 0
                && "SSE only supports kappaInternal = 0");
        assert(gamma2Internal == 0
                && "SSE only supports gamma2Internal = 0");
        assert(gamma2Dephasing == 0
                && "SSE only supports gamma2Dephasing = 0");
        data = measure_with_jqf_sse(kappa, gamma2, g, Omega, omega_d, omega_r,
                omega_1, omega_2, k0x_r, k0x_2, phi, num_excitations,
                N_trajectories, integration_times, N_t_max);
    } else {
        if (batch_size > 1) {
            data = measure_with_jqf_batched(kappa, kappaInternal, gamma2,
                    gamma2Internal, gamma2Dephasing, g, Omega, omega_d, omega_r,
                    omega_1, omega_2, transmon_anharmonicity, k0x_r, k0x_2, phi,
                    num_excitations, transmon_excitations,
                    N_trajectories, batch_size, integration_times, N_t_max);
        } else {
            data = measure_with_jqf_single(kappa, kappaInternal, gamma2,
                    gamma2Internal, gamma2Dephasing, g, Omega, omega_d, omega_r,
                    omega_1, omega_2, transmon_anharmonicity, k0x_r, k0x_2, phi,
                    num_excitations, transmon_excitations,
                    N_trajectories, integration_times, N_t_max);
        }
    }
    return data;
}
