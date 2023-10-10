// Copyright (c) 2021-2023 Ivan Iakoupov
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

#include "jqf_adrk4.h"

#include <chrono>
#include <iostream>

#include <nlopt.hpp>

#include "quadrature/quad.h"
#include "Faddeeva.hh"
#ifdef EIGEN_USE_MKL_ALL
#include "mkl_support.h"
#endif // EIGEN_USE_MKL_ALL
#ifdef USE_ROCM
#include "qroc/master_equation_roc.h"
#include "smd_from_spmat.h"
#endif // USE_ROCM
#include "master_equation.h"
#include "rk4.h"

//#define DO_NOT_UPDATE_OMEGA_HAMILTONIAN
#define AVERAGE_FIDELITY_FROM_PAULI_MATRICES

Eigen::VectorXcd all_eigenstates_up_state(const JQFSuperoperatorData &sd)
{
    const int basis_size = sd.basis.size();
    std::vector<sigma_state_t> spec(sd.basis[0].size(), 0);
    const int NResonators = sd.psi_up_eigenstate_indices.size();
    for (int m = 0; m < NResonators; ++m) {
        spec[m] = sd.psi_up_eigenstate_indices[m];
    }
    BasisVector vec(spec);
    // psi_excited is a state where every AA&resonator system
    // is in the computational state |1>
    Eigen::VectorXcd psi_excited = Eigen::VectorXcd::Zero(basis_size);
    for (int i = 0; i < basis_size; ++i) {
        if (sd.basis[i] == vec) {
            psi_excited(i) = 1;
            break;
        }
    }
    return psi_excited;
}

Eigen::VectorXcd superposition_of_ground_and_all_eigenstates_up_state(
        const JQFSuperoperatorData &sd,
        const Eigen::VectorXcd &psi0,
        std::complex<double> target_excited_state_amplitude)
{
    Eigen::VectorXcd psi_excited = all_eigenstates_up_state(sd);
    // For one qubit (AA&resonator system), this is a superposition of
    // computational basis states |0> and |1>. For two qubits, it is
    // a Bell state if target_excited_state_amplitude=1/sqrt(2).
    // For more than two qubits, it is a GHZ state (if
    // target_excited_state_amplitude=1/sqrt(2)).
    Eigen::VectorXcd psi_superposition
        = target_excited_state_amplitude*psi_excited
        + std::sqrt(1-std::norm(target_excited_state_amplitude))*psi0;
    return psi_superposition;
}

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
        double t_final, int64_t N_t, int64_t data_reduce_factor)
{
    assert(N_t > 0 && "Number of steps needs to be a positive integer!");

    // Because generate_superoperator_diag() selects different approximations
    // depending on whether Omega is zero or non-zero, the value has to be
    // non-zero here. Which non-zero value is unimportant, since we also
    // pass the flag JQF_SUPEROPERATOR_OMIT_OMEGA_TERMS that prevents the
    // drive terms being added to the Hamiltonian (we add them manually later).
    const double Omega0 = 1;

    const bool useRhoSpec = isUsingRhoSpec({stateSpec});

    JQFData ret;
    int flags = 0;
    flags |= JQF_SUPEROPERATOR_OMIT_OMEGA_TERMS;
    double k0x_out = 0;
    if (!k0x_a.empty()) {
        k0x_out = k0x_a[k0x_a.size()-1];
    } else if (!k0x_r.empty()) {
        k0x_out = k0x_r[k0x_r.size()-1];
    }
    const std::vector<double> kappaInternal(omega_r.size(), 0);
    JQFSuperoperatorData sd = generate_superoperator_diag(
            kappa, kappaInternal, gamma, gammaInternal,
            gammaDephasing, g, Omega0, J_x, omega_d, omega_r,
            omega, transmon_anharmonicity, k0x_r, k0x_a,
            k0x_out, num_excitations, transmon_excitations, flags);
    if (sd.replaced_negative_frequencies_with_positive) {
        std::cout << "Warning: replaced negative eigenfrequencies with a "
                  << "positive drive frequency" << std::endl;
    }
    const int basis_size = sd.basis.size();
    const int basis_size_squared = basis_size*basis_size;

    Eigen::VectorXcd psi0 = Eigen::VectorXcd::Zero(basis_size);
    psi0(0) = 1; // atoms and the resonator are in their ground states
    Eigen::VectorXcd psi1 = all_eigenstates_up_state(sd);
    std::vector<Eigen::VectorXcd> psi_vec = {psi0, psi1};

    Eigen::VectorXcd rho_vec;
    Eigen::VectorXcd rho_target_vec;
    if (useRhoSpec) {
        expandQubitStateToFullBasisRhoSpec(
                rho_vec,
                rho_target_vec,
                stateSpec,
                psi_vec);
    } else {
        expandQubitStateToFullBasis(
                rho_vec,
                rho_target_vec,
                stateSpec,
                psi_vec);
    }

    const double dt = t_final/N_t;

    assert(sd.M_O.size() == sd.M_O_adjoint.size()
           && "Number of lowering and raising operators are not equal!");
    assert(sd.M_O.size() == sd.Omega_factors.size()
           && "Number of lowering operators and corresponding factors are not equal!");
    const int M_O_size = sd.M_O.size();
    const std::complex<double> I(0,1);
    SpMat dH_Omega_d_Re_Omega(basis_size, basis_size);
    SpMat dH_Omega_d_Im_Omega(basis_size, basis_size);
    for (int i = 0; i < M_O_size; ++i) {
        dH_Omega_d_Re_Omega += sd.Omega_factors[i]*(sd.M_O_adjoint[i]+sd.M_O[i]);
        dH_Omega_d_Im_Omega += I*sd.Omega_factors[i]*(sd.M_O_adjoint[i]-sd.M_O[i]);
    }
    dH_Omega_d_Re_Omega.makeCompressed();
    dH_Omega_d_Im_Omega.makeCompressed();
    SpMat H_Omega1(basis_size, basis_size);
    SpMat H_Omega2(basis_size, basis_size);
    SpMat H_Omega3(basis_size, basis_size);
#ifdef EIGEN_USE_MKL_ALL
    MKLSparseMatrix mklL(sd.L);
    MKLSparseMatrix mkldH_Omega_d_Re_Omega(dH_Omega_d_Re_Omega);
    MKLSparseMatrix mkldH_Omega_d_Im_Omega(dH_Omega_d_Im_Omega);
#endif // EIGEN_USE_MKL_ALL

    const int NResonators = sd.M_n_res.size();
    const int NAtoms = sd.M_n_atom.size();
    ret.time = std::vector<double>(N_t, 0);
    ret.Omega_Re = std::vector<double>(N_t, 0);
    ret.Omega_Im = std::vector<double>(N_t, 0);
    ret.res_populations.resize(NResonators);
    for (int m = 0; m < NResonators; ++m) {
        ret.res_populations[m] = std::vector<double>(N_t, 0);
    }
    ret.aa_populations.resize(NAtoms);
    for (int m = 0; m < NAtoms; ++m) {
        ret.aa_populations[m] = std::vector<double>(N_t, 0);
    }
    ret.aa_level_populations.resize(sd.M_sigma.size());
    for (int m = 0; m < ret.aa_level_populations.size(); ++m) {
        ret.aa_level_populations[m].resize(sd.M_sigma[m].size());
        for (int j = 0; j < ret.aa_level_populations[m].size(); ++j) {
            ret.aa_level_populations[m][j] = std::vector<double>(N_t, 0);
        }
    }
    ret.purity_1 = std::vector<double>(N_t, 0);
    ret.purity_1r = std::vector<double>(N_t, 0);
    ret.F.resize(NResonators);
    ret.F_down.resize(NResonators);
    for (int m = 0; m < NResonators; ++m) {
        ret.F[m] = std::vector<double>(N_t, 0);
        ret.F_down[m] = std::vector<double>(N_t, 0);
    }
    ret.tilde_F = std::vector<double>(N_t, 0);
#ifdef USE_ROCM
    SparseMatrixData rocL = smd_from_spmat(sd.L);
    SparseMatrixData rocH_t_Re = smd_from_spmat(dH_Omega_d_Re_Omega);
    SparseMatrixData rocH_t_Im = smd_from_spmat(dH_Omega_d_Im_Omega);
    std::vector<SparseMatrixData> M_operators;
    for (int i = 0; i < NResonators; ++i) {
        M_operators.push_back(smd_from_spmat(sd.M_n_res[i]));
    }
    for (int i = 0; i < NAtoms; ++i) {
        M_operators.push_back(smd_from_spmat(sd.M_n_atom[i]));
    }
    for (int i = 0; i < NResonators; ++i) {
        M_operators.push_back(smd_from_spmat(sd.M_psi_up[i]));
        M_operators.push_back(smd_from_spmat(sd.M_psi_down[i]));
    }
    // Save the indices so that we know which data.M_values
    // correspond to which sd.M_sigm[m][j] operator.
    std::vector<std::vector<int>> aa_level_population_indices(sd.M_sigma.size());
    for (int m = 0; m < ret.aa_level_populations.size(); ++m) {
        aa_level_population_indices[m].resize(sd.M_sigma[m].size());
        for (int j = 0; j < ret.aa_level_populations[m].size(); ++j) {
            M_operators.push_back(smd_from_spmat(sd.M_sigma[m][j]));
            aa_level_population_indices[m][j] = M_operators.size()-1;
        }
    }
    std::vector<const double*> M_diag_operators;
    std::vector<const std::complex<double>*> fidelity_rhos;
    fidelity_rhos.push_back(rho_target_vec.data());
    const int64_t iterationsBetweenDeviceSynchronize = 1e4;
    MasterEquationData data = evolve_time_dependent_master_equation_roc(
            basis_size, rocL, rocH_t_Re, rocH_t_Im, Omega, rho_vec.data(),
            M_operators, M_diag_operators, fidelity_rhos, dt, N_t,
            iterationsBetweenDeviceSynchronize);
    for (int64_t i = 0; i < N_t; ++i) {
        ret.time[i] = data.time[i];
        for (int m = 0; m < NResonators; ++m) {
            ret.res_populations[m][i] = data.M_values[m][i].real();
        }
        for (int m = 0; m < NAtoms; ++m) {
            ret.aa_populations[m][i] = data.M_values[NResonators+m][i].real();
        }
        for (int m = 0; m < NResonators; ++m) {
            ret.F[m][i] = data.M_values[NResonators+NAtoms+2*m][i].real();
            ret.F_down[m][i] = data.M_values[NResonators+NAtoms+2*m+1][i].real();
        }
        for (int m = 0; m < ret.aa_level_populations.size(); ++m) {
            for (int j = 0; j < ret.aa_level_populations[m].size(); ++j) {
                ret.aa_level_populations[m][j][i]
                        = data.M_values[
                        aa_level_population_indices[m][j]][i].real();
            }
        }
        ret.tilde_F[i] = data.fidelities[0][i].real();
        const std::complex<double> Omega_i = Omega(data.time[i]);
        ret.Omega_Re[i] = Omega_i.real();
        ret.Omega_Im[i] = Omega_i.imag();
    }
#else // USE_ROCM
    Eigen::VectorXcd k1 = Eigen::VectorXcd::Zero(basis_size_squared);
    Eigen::VectorXcd k2 = Eigen::VectorXcd::Zero(basis_size_squared);
    Eigen::VectorXcd k3 = Eigen::VectorXcd::Zero(basis_size_squared);
    Eigen::VectorXcd k4 = Eigen::VectorXcd::Zero(basis_size_squared);
    Eigen::VectorXcd temp = Eigen::VectorXcd::Zero(basis_size_squared);
    for (int64_t i = 0; i < N_t; ++i) {
        const std::complex<double> Omega1 = Omega(dt*i);
        const std::complex<double> Omega2 = Omega(dt*(i+0.5));
        const std::complex<double> Omega3 = Omega(dt*(i+1));
#ifdef DO_NOT_UPDATE_OMEGA_HAMILTONIAN
#ifdef EIGEN_USE_MKL_ALL
        rk4_step_t_mkl(rho_vec, mklL, mkldH_Omega_d_Re_Omega, mkldH_Omega_d_Im_Omega, Omega1, Omega2, Omega3, k1, k2, k3, k4, temp, dt);
#else // EIGEN_USE_MKL_ALL
        rk4_step_t(rho_vec, sd.L, dH_Omega_d_Re_Omega, dH_Omega_d_Im_Omega, Omega1, Omega2, Omega3, k1, k2, k3, k4, temp, dt);
#endif // EIGEN_USE_MKL_ALL
#else // DO_NOT_UPDATE_OMEGA_HAMILTONIAN
        H_Omega1.setZero();
        for (int i = 0; i < M_O_size; ++i) {
            H_Omega1 += sd.Omega_factors[i]*(Omega1*sd.M_O_adjoint[i]+std::conj(Omega1)*sd.M_O[i]);
        }
        H_Omega2.setZero();
        for (int i = 0; i < M_O_size; ++i) {
            H_Omega2 += sd.Omega_factors[i]*(Omega2*sd.M_O_adjoint[i]+std::conj(Omega2)*sd.M_O[i]);
        }
        H_Omega3.setZero();
        for (int i = 0; i < M_O_size; ++i) {
            H_Omega3 += sd.Omega_factors[i]*(Omega3*sd.M_O_adjoint[i]+std::conj(Omega3)*sd.M_O[i]);
        }
#ifdef EIGEN_USE_MKL_ALL
        rk4_step_t_mkl(rho_vec, mklL, H_Omega1, H_Omega2, H_Omega3, 1, k1, k2, k3, k4, temp, dt);
#else // EIGEN_USE_MKL_ALL
        rk4_step_t(rho_vec, sd.L, H_Omega1, H_Omega2, H_Omega3, 1, k1, k2, k3, k4, temp, dt);
#endif // EIGEN_USE_MKL_ALL
#endif // DO_NOT_UPDATE_OMEGA_HAMILTONIAN

        Eigen::Map<MatrixXcdRowMajor> rho_map(rho_vec.data(), basis_size, basis_size);
        for (int m = 0; m < NResonators; ++m) {
            const std::complex<double> res_population = trace_of_product(sd.M_n_res[m],rho_map);
            ret.res_populations[m][i] = res_population.real();
        }
        for (int m = 0; m < NAtoms; ++m) {
            const std::complex<double> aa_population = trace_of_product(sd.M_n_atom[m],rho_map);
            ret.aa_populations[m][i] = aa_population.real();
        }
        for (int m = 0; m < NResonators; ++m) {
            const std::complex<double> F = trace_of_product(sd.M_psi_up[m],rho_map);
            const std::complex<double> F_down = trace_of_product(sd.M_psi_down[m],rho_map);
            ret.F[m][i] = F.real();
            ret.F_down[m][i] = F_down.real();
        }
        for (int m = 0; m < ret.aa_level_populations.size(); ++m) {
            for (int j = 0; j < ret.aa_level_populations[m].size(); ++j) {
                const std::complex<double> sigma_m_j_population
                        = trace_of_product(sd.M_sigma[m][j],rho_map);
                ret.aa_level_populations[m][j][i]
                        = sigma_m_j_population.real();
            }
        }
        const std::complex<double> tilde_F = rho_target_vec.adjoint()*rho_vec;
        ret.tilde_F[i] = tilde_F.real();
        ret.time[i] = dt*(i+1);
        ret.Omega_Re[i] = Omega3.real();
        ret.Omega_Im[i] = Omega3.imag();
    }
#endif // USE_ROCM
    return ret;
}

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
        double t_final, int64_t N_t, int64_t data_reduce_factor)
{
    assert(N_t > 0 && "Number of steps needs to be a positive integer!");

    // This value is ignored
    const double omega_d0 = omega_d(0);
    // Because generate_superoperator_diag() selects different approximations
    // depending on whether Omega is zero or non-zero, the value has to be
    // non-zero here. Which non-zero value is unimportant, since we also
    // pass the flag JQF_SUPEROPERATOR_OMIT_OMEGA_TERMS that prevents the
    // drive terms being added to the Hamiltonian (we add them manually later).
    const double Omega0 = 1;

    const bool useRhoSpec = isUsingRhoSpec({stateSpec});

    JQFData ret;
    int flags = 0;
    flags |= JQF_SUPEROPERATOR_OMIT_OMEGA_TERMS;
    flags |= JQF_SUPEROPERATOR_OMIT_DELTA_TERMS;
    double k0x_out = 0;
    if (!k0x_a.empty()) {
        k0x_out = k0x_a[k0x_a.size()-1];
    } else if (!k0x_r.empty()) {
        k0x_out = k0x_r[k0x_r.size()-1];
    }
    const std::vector<double> kappaInternal(omega_r.size(), 0);
    JQFSuperoperatorData sd = generate_superoperator_diag(
            kappa, kappaInternal, gamma, gammaInternal,
            gammaDephasing, g, Omega0, J_x, omega_d0, omega_r,
            omega, transmon_anharmonicity, k0x_r, k0x_a,
            k0x_out, num_excitations, transmon_excitations, flags);
    if (sd.replaced_negative_frequencies_with_positive) {
        std::cout << "Warning: replaced negative eigenfrequencies with a "
                  << "positive drive frequency" << std::endl;
    }
    const int basis_size = sd.basis.size();
    const int basis_size_squared = basis_size*basis_size;

    Eigen::VectorXcd psi0 = Eigen::VectorXcd::Zero(basis_size);
    psi0(0) = 1; // atoms and the resonator are in their ground states
    Eigen::VectorXcd psi1 = all_eigenstates_up_state(sd);
    std::vector<Eigen::VectorXcd> psi_vec = {psi0, psi1};

    Eigen::VectorXcd rho_vec;
    Eigen::VectorXcd rho_target_vec;
    if (useRhoSpec) {
        expandQubitStateToFullBasisRhoSpec(
                rho_vec,
                rho_target_vec,
                stateSpec,
                psi_vec);
    } else {
        expandQubitStateToFullBasis(
                rho_vec,
                rho_target_vec,
                stateSpec,
                psi_vec);
    }

    const double dt = t_final/N_t;

    assert(sd.M_O.size() == sd.M_O_adjoint.size()
           && "Number of lowering and raising operators are not equal!");
    assert(sd.M_O.size() == sd.Omega_factors.size()
           && "Number of lowering operators and corresponding factors are not equal!");
    const int M_O_size = sd.M_O.size();
    const std::complex<double> I(0,1);
    SpMat dH_Omega_d_Re_Omega(basis_size, basis_size);
    SpMat dH_Omega_d_Im_Omega(basis_size, basis_size);
    for (int i = 0; i < M_O_size; ++i) {
        dH_Omega_d_Re_Omega += sd.Omega_factors[i]*(sd.M_O_adjoint[i]+sd.M_O[i]);
        dH_Omega_d_Im_Omega += I*sd.Omega_factors[i]*(sd.M_O_adjoint[i]-sd.M_O[i]);
    }
    dH_Omega_d_Re_Omega.makeCompressed();
    dH_Omega_d_Im_Omega.makeCompressed();
    SpMat H_Omega1(basis_size, basis_size);
    SpMat H_Omega2(basis_size, basis_size);
    SpMat H_Omega3(basis_size, basis_size);
#ifdef EIGEN_USE_MKL_ALL
    MKLSparseMatrix mklL(sd.L);
    MKLSparseMatrix mkldH_Omega_d_Re_Omega(dH_Omega_d_Re_Omega);
    MKLSparseMatrix mkldH_Omega_d_Im_Omega(dH_Omega_d_Im_Omega);
#endif // EIGEN_USE_MKL_ALL
    std::vector<std::vector<SpMat>> M_jj_m_array;
    M_jj_m_array.resize(sd.subsystem_eigenvalues.size());
    for (int m = 0; m < sd.subsystem_eigenvalues.size(); ++m) {
        const int num_eigenvalues_m = sd.subsystem_eigenvalues[m].size();
        M_jj_m_array[m].resize(num_eigenvalues_m);
        for (int j = 0; j < num_eigenvalues_m; ++j) {
            const Sigma s_jj_m(j, j, m);
            M_jj_m_array[m][j] = s_jj_m.matrix(sd.basis);
        }
    }

    const int NResonators = sd.M_n_res.size();
    const int NAtoms = sd.M_n_atom.size();
    ret.time = std::vector<double>(N_t, 0);
    ret.Omega_Re = std::vector<double>(N_t, 0);
    ret.Omega_Im = std::vector<double>(N_t, 0);
    ret.res_populations.resize(NResonators);
    for (int m = 0; m < NResonators; ++m) {
        ret.res_populations[m] = std::vector<double>(N_t, 0);
    }
    ret.aa_populations.resize(NAtoms);
    for (int m = 0; m < NAtoms; ++m) {
        ret.aa_populations[m] = std::vector<double>(N_t, 0);
    }
    ret.aa_level_populations.resize(sd.M_sigma.size());
    for (int m = 0; m < ret.aa_level_populations.size(); ++m) {
        ret.aa_level_populations[m].resize(sd.M_sigma[m].size());
        for (int j = 0; j < ret.aa_level_populations[m].size(); ++j) {
            ret.aa_level_populations[m][j] = std::vector<double>(N_t, 0);
        }
    }
    ret.purity_1 = std::vector<double>(N_t, 0);
    ret.purity_1r = std::vector<double>(N_t, 0);
    ret.F.resize(NResonators);
    ret.F_down.resize(NResonators);
    for (int m = 0; m < NResonators; ++m) {
        ret.F[m] = std::vector<double>(N_t, 0);
        ret.F_down[m] = std::vector<double>(N_t, 0);
    }
    ret.tilde_F = std::vector<double>(N_t, 0);
    Eigen::VectorXcd k1 = Eigen::VectorXcd::Zero(basis_size_squared);
    Eigen::VectorXcd k2 = Eigen::VectorXcd::Zero(basis_size_squared);
    Eigen::VectorXcd k3 = Eigen::VectorXcd::Zero(basis_size_squared);
    Eigen::VectorXcd k4 = Eigen::VectorXcd::Zero(basis_size_squared);
    Eigen::VectorXcd temp = Eigen::VectorXcd::Zero(basis_size_squared);
    for (int64_t i = 0; i < N_t; ++i) {
        const double omega_d1 = omega_d(dt*i);
        const double omega_d2 = omega_d(dt*(i+0.5));
        const double omega_d3 = omega_d(dt*(i+1));
        const std::complex<double> Omega1 = Omega(dt*i);
        const std::complex<double> Omega2 = Omega(dt*(i+0.5));
        const std::complex<double> Omega3 = Omega(dt*(i+1));
        H_Omega1.setZero();
        for (int m = 0; m < sd.subsystem_eigenvalues.size(); ++m) {
            const int num_eigenvalues_m = sd.subsystem_eigenvalues[m].size();
            for (int j = 0; j < num_eigenvalues_m; ++j) {
                const double omega_diff = sd.subsystem_eigenvalues[m][j]
                        -omega_d1*sd.subsystem_eigenvalue_shifts[m][j];
                H_Omega1 += omega_diff*M_jj_m_array[m][j];
            }
        }
        for (int i = 0; i < M_O_size; ++i) {
            H_Omega1 += sd.Omega_factors[i]*(Omega1*sd.M_O_adjoint[i]+std::conj(Omega1)*sd.M_O[i]);
        }
        H_Omega2.setZero();
        for (int m = 0; m < sd.subsystem_eigenvalues.size(); ++m) {
            const int num_eigenvalues_m = sd.subsystem_eigenvalues[m].size();
            for (int j = 0; j < num_eigenvalues_m; ++j) {
                const double omega_diff = sd.subsystem_eigenvalues[m][j]
                        -omega_d2*sd.subsystem_eigenvalue_shifts[m][j];
                H_Omega2 += omega_diff*M_jj_m_array[m][j];
            }
        }
        for (int i = 0; i < M_O_size; ++i) {
            H_Omega2 += sd.Omega_factors[i]*(Omega2*sd.M_O_adjoint[i]+std::conj(Omega2)*sd.M_O[i]);
        }
        H_Omega3.setZero();
        for (int m = 0; m < sd.subsystem_eigenvalues.size(); ++m) {
            const int num_eigenvalues_m = sd.subsystem_eigenvalues[m].size();
            for (int j = 0; j < num_eigenvalues_m; ++j) {
                const double omega_diff = sd.subsystem_eigenvalues[m][j]
                        -omega_d3*sd.subsystem_eigenvalue_shifts[m][j];
                H_Omega3 += omega_diff*M_jj_m_array[m][j];
            }
        }
        for (int i = 0; i < M_O_size; ++i) {
            H_Omega3 += sd.Omega_factors[i]*(Omega3*sd.M_O_adjoint[i]+std::conj(Omega3)*sd.M_O[i]);
        }
#ifdef EIGEN_USE_MKL_ALL
        rk4_step_t_mkl(rho_vec, mklL, H_Omega1, H_Omega2, H_Omega3, 1, k1, k2, k3, k4, temp, dt);
#else // EIGEN_USE_MKL_ALL
        rk4_step_t(rho_vec, sd.L, H_Omega1, H_Omega2, H_Omega3, 1, k1, k2, k3, k4, temp, dt);
#endif // EIGEN_USE_MKL_ALL

        Eigen::Map<MatrixXcdRowMajor> rho_map(rho_vec.data(), basis_size, basis_size);
        for (int m = 0; m < NResonators; ++m) {
            const std::complex<double> res_population = trace_of_product(sd.M_n_res[m],rho_map);
            ret.res_populations[m][i] = res_population.real();
        }
        for (int m = 0; m < NAtoms; ++m) {
            const std::complex<double> aa_population = trace_of_product(sd.M_n_atom[m],rho_map);
            ret.aa_populations[m][i] = aa_population.real();
        }
        for (int m = 0; m < NResonators; ++m) {
            const std::complex<double> F = trace_of_product(sd.M_psi_up[m],rho_map);
            const std::complex<double> F_down = trace_of_product(sd.M_psi_down[m],rho_map);
            ret.F[m][i] = F.real();
            ret.F_down[m][i] = F_down.real();
        }
        for (int m = 0; m < ret.aa_level_populations.size(); ++m) {
            for (int j = 0; j < ret.aa_level_populations[m].size(); ++j) {
                const std::complex<double> sigma_m_j_population
                        = trace_of_product(sd.M_sigma[m][j],rho_map);
                ret.aa_level_populations[m][j][i]
                        = sigma_m_j_population.real();
            }
        }
        const std::complex<double> tilde_F = rho_target_vec.adjoint()*rho_vec;
        ret.tilde_F[i] = tilde_F.real();
        ret.time[i] = dt*(i+1);
        ret.Omega_Re[i] = Omega3.real();
        ret.Omega_Im[i] = Omega3.imag();
    }
    return ret;
}

#ifdef FILTERED_BASIS_FUNCTION_BY_QUADRATURE
double filteredBasisFunction(int k, double t, double NormalizationFactor, double omegaFactor, double t_f, double sigma)
{
    const double GaussianNormalization = 1.0/(std::sqrt(2*M_PI)*sigma);
    auto integrand = [=] (double t_prime) -> double
    {
        return std::exp(-std::pow((t-t_prime)/sigma,2)/2)*basisFunction(k, t_prime, NormalizationFactor, omegaFactor);
    };
    const double absTol = 1e-10;
    const double relTol = 1e-3;
    double absErr = 0;
    double relErr = 0;
    const int max_degree = 15;
    const double filteredSine = quad_tanh_sinh<double>(integrand, 0, t_f, absTol, relTol, &absErr, &relErr,
                      QUAD_BOTH_BOUNDS_FINITE, max_degree);
    return GaussianNormalization*filteredSine;
}
#else // FILTERED_BASIS_FUNCTION_BY_QUADRATURE
double filteredBasisFunction(int k, double t, double NormalizationFactor, double omegaFactor, double t_f, double sigma)
{
    const std::complex<double> I(0,1);
    const double omega_k = omegaFactor*(k+1);
    const std::complex<double> exp_factor = std::exp(-I*omega_k*t);
    const double sigma_omega_term = std::pow(sigma,2)*omega_k;
    const double denominator = std::sqrt(2)*sigma;
    const std::complex<double> erfi_term_tf = exp_factor*Faddeeva::erfi((sigma_omega_term+I*(t-t_f))/denominator);
    const std::complex<double> erfi_term_t0 = exp_factor*Faddeeva::erfi((sigma_omega_term+I*t)/denominator);
    const double filteredSine = -0.5*std::exp(-std::pow(omega_k*sigma,2)/2)
                                *(erfi_term_tf.real()-erfi_term_t0.real());
    return NormalizationFactor*filteredSine;
}
#endif // FILTERED_BASIS_FUNCTION_BY_QUADRATURE

//#define DEBUG_STATE_REMATERIALIZATION

double find_optimal_Omega_adrk4_f_single_state(unsigned n, const double *x, double *grad, void *params)
{
    find_optimal_Omega_adrk4_params *p
            = (find_optimal_Omega_adrk4_params *) params;
    const int adrk4_flags = p->adrk4_flags;
    const int basis_size = p->basis_size;
    const int basis_size_squared = basis_size*basis_size;
    const int N_amplitudes = p->N_amplitudes;
    const double NormalizationFactor = p->NormalizationFactor;
    const double omegaFactor = p->omegaFactor;
    const double sigmaFilter = p->sigmaFilter;
    const double sigmaWindow = p->sigmaWindow;
    const double t_final = p->t_final;
    const int64_t N_t = p->N_t;
    const double dt = t_final/N_t;
    const double Omega_max = p->Omega_max;
    SpMat L_adjoint = p->sd.L.adjoint();
    Eigen::MatrixXd *f1 = nullptr;
    Eigen::MatrixXd *f2 = nullptr;
    if (p->f1.count(N_t) == 1) {
        f1 = &p->f1[N_t];
        f2 = &p->f2[N_t];
    }
    auto calculateOmega1 = [&](int64_t i) -> std::complex<double>
    {
        double Omega_Re = 0;
        double Omega_Im = 0;
        for (int k = 0; k < N_amplitudes; ++k) {
            Omega_Re += x[k]*(*f1)(k,i);
            Omega_Im += x[k+N_amplitudes]*(*f1)(k,i);
        }
        if (adrk4_flags & ADRK4_FILTER_BASIS_FUNCTIONS) {
            const double windowFactor = confinedGaussian(i, N_t, t_final, sigmaWindow);
            Omega_Re *= windowFactor;
            Omega_Im *= windowFactor;
        }
        if (adrk4_flags & ADRK4_CLAMP_OMEGA) {
            Omega_Re = std::tanh(Omega_Re);
            Omega_Im = std::tanh(Omega_Im);
        }
        return std::complex<double>(Omega_Re, Omega_Im);
    };
    auto calculateOmega2 = [&](int64_t i) -> std::complex<double>
    {
        double Omega_Re = 0;
        double Omega_Im = 0;
        for (int k = 0; k < N_amplitudes; ++k) {
            Omega_Re += x[k]*(*f2)(k,i);
            Omega_Im += x[k+N_amplitudes]*(*f2)(k,i);
        }
        if (adrk4_flags & ADRK4_FILTER_BASIS_FUNCTIONS) {
            const double windowFactor = confinedGaussian(i+0.5, N_t, t_final, sigmaWindow);
            Omega_Re *= windowFactor;
            Omega_Im *= windowFactor;
        }
        if (adrk4_flags & ADRK4_CLAMP_OMEGA) {
            Omega_Re = std::tanh(Omega_Re);
            Omega_Im = std::tanh(Omega_Im);
        }
        return std::complex<double>(Omega_Re, Omega_Im);
    };
    auto calculateOmega3 = [&](int64_t i) -> std::complex<double>
    {
        double Omega_Re = 0;
        double Omega_Im = 0;
        for (int k = 0; k < N_amplitudes; ++k) {
            Omega_Re += x[k]*(*f1)(k,i+1);
            Omega_Im += x[k+N_amplitudes]*(*f1)(k,i+1);
        }
        if (adrk4_flags & ADRK4_FILTER_BASIS_FUNCTIONS) {
            const double windowFactor = confinedGaussian(i+1, N_t, t_final, sigmaWindow);
            Omega_Re *= windowFactor;
            Omega_Im *= windowFactor;
        }
        if (adrk4_flags & ADRK4_CLAMP_OMEGA) {
            Omega_Re = std::tanh(Omega_Re);
            Omega_Im = std::tanh(Omega_Im);
        }
        return std::complex<double>(Omega_Re, Omega_Im);
    };
    auto addToGradient1 = [&](int64_t i, double factor_Re, double factor_Im) -> void
    {
        for (int k = 0; k < N_amplitudes; ++k) {
            // Factor of -1 is needed because we return -trace_final.real() below
            grad[k] += -factor_Re*(*f1)(k,i);
            grad[k+N_amplitudes] += -factor_Im*(*f1)(k,i);
        }
    };
    auto addToGradient2 = [&](int64_t i, double factor_Re, double factor_Im) -> void
    {
        for (int k = 0; k < N_amplitudes; ++k) {
            // Factor of -1 is needed because we return -trace_final.real() below
            grad[k] += -factor_Re*(*f2)(k,i);
            grad[k+N_amplitudes] += -factor_Im*(*f2)(k,i);
        }
    };
    auto addToGradient3 = [&](int64_t i, double factor_Re, double factor_Im) -> void
    {
        for (int k = 0; k < N_amplitudes; ++k) {
            // Factor of -1 is needed because we return -trace_final.real() below
            grad[k] += -factor_Re*(*f1)(k,i+1);
            grad[k+N_amplitudes] += -factor_Im*(*f1)(k,i+1);
        }
    };
    auto calculateOmega = [&](double t) -> std::complex<double>
    {
        double Omega_Re = 0;
        double Omega_Im = 0;
        if (adrk4_flags & ADRK4_FILTER_BASIS_FUNCTIONS) {
            for (int k = 0; k < N_amplitudes; ++k) {
                const double f = filteredBasisFunction(k, t, NormalizationFactor, omegaFactor, t_final, sigmaFilter);
                Omega_Re += x[k]*f;
                Omega_Im += x[k+N_amplitudes]*f;
            }
        } else {
            for (int k = 0; k < N_amplitudes; ++k) {
                const double f = basisFunction(k, t, NormalizationFactor, omegaFactor);
                Omega_Re += x[k]*f;
                Omega_Im += x[k+N_amplitudes]*f;
            }
        }
        if (adrk4_flags & ADRK4_FILTER_BASIS_FUNCTIONS) {
            const double windowFactor = confinedGaussian(t/dt, N_t, t_final, sigmaWindow);
            Omega_Re *= windowFactor;
            Omega_Im *= windowFactor;
        }
        if (adrk4_flags & ADRK4_CLAMP_OMEGA) {
            Omega_Re = std::tanh(Omega_Re);
            Omega_Im = std::tanh(Omega_Im);
        }
        return std::complex<double>(Omega_Re, Omega_Im);
    };
    auto addToGradient = [&](double t, double factor_Re, double factor_Im) -> void
    {
        if (adrk4_flags & ADRK4_FILTER_BASIS_FUNCTIONS) {
            for (int k = 0; k < N_amplitudes; ++k) {
                const double f = filteredBasisFunction(k, t, NormalizationFactor, omegaFactor, t_final, sigmaFilter);
                // Factor of -1 is needed because we return -trace_final.real() below
                grad[k] += -factor_Re*f;
                grad[k+N_amplitudes] += -factor_Im*f;
            }
        } else {
            for (int k = 0; k < N_amplitudes; ++k) {
                const double f = basisFunction(k, t, NormalizationFactor, omegaFactor);
                // Factor of -1 is needed because we return -trace_final.real() below
                grad[k] += -factor_Re*f;
                grad[k+N_amplitudes] += -factor_Im*f;
            }
        }
    };
    assert(p->sd.M_O.size() == p->sd.M_O_adjoint.size()
           && "Number of lowering and raising operators are not equal!");
    assert(p->sd.M_O.size() == p->sd.Omega_factors.size()
           && "Number of lowering operators and corresponding factors are not equal!");
    const int M_O_size = p->sd.M_O.size();
    const std::complex<double> I(0,1);
    SpMat Identity(basis_size, basis_size);
    Identity.setIdentity();
    SpMat dH_Omega_d_Re_Omega(basis_size, basis_size);
    SpMat dH_Omega_d_Im_Omega(basis_size, basis_size);
    for (int i = 0; i < M_O_size; ++i) {
        dH_Omega_d_Re_Omega += p->sd.Omega_factors[i]*(p->sd.M_O_adjoint[i]+p->sd.M_O[i]);
        dH_Omega_d_Im_Omega += I*p->sd.Omega_factors[i]*(p->sd.M_O_adjoint[i]-p->sd.M_O[i]);
    }

    const int64_t maxCacheSizeGiB = 30;
    const int64_t maxCacheSizeB = maxCacheSizeGiB*1024*1024*1024;
    const int64_t stateSizeB = basis_size*basis_size*sizeof(std::complex<double>);
    const int64_t maxStoredN_t = maxCacheSizeB/stateSizeB;
#ifdef DEBUG_STATE_REMATERIALIZATION
    const int64_t timeStepsPerStoredState = 2;
#else // DEBUG_STATE_REMATERIALIZATION
    const int64_t timeStepsPerStoredState = (N_t - 1)/maxStoredN_t + 1;
#endif // DEBUG_STATE_REMATERIALIZATION
    const int64_t cacheSizeTimeSteps = (N_t - 1)/timeStepsPerStoredState + 1;
    //std::cout << "maxCacheSizeB = " << maxCacheSizeB << std::endl;
    //std::cout << "stateSizeB = " << stateSizeB << std::endl;
    //std::cout << "maxStoredN_t = " << maxStoredN_t << std::endl;
    //std::cout << "timeStepsPerStoredState = " << timeStepsPerStoredState << std::endl;
    //std::cout << "cacheSizeTimeSteps = " << cacheSizeTimeSteps << std::endl;
#ifdef USE_ROCM
    int64_t cacheSizeTimeStepsGPU = cacheSizeTimeSteps/4; // TODO: Do not hardcode
    std::complex<double> trace_final;
    SparseMatrixData rocL = smd_from_spmat(p->sd.L);
    SparseMatrixData rocL_adjoint = smd_from_spmat(L_adjoint);
    SparseMatrixData rocH_t_Re = smd_from_spmat(dH_Omega_d_Re_Omega);
    SparseMatrixData rocH_t_Im = smd_from_spmat(dH_Omega_d_Im_Omega);
    double *f1_data = nullptr;
    double *f2_data = nullptr;
    if (adrk4_flags & ADRK4_PRECOMPUTE_BASIS_FUNCTION_VALUES) {
        f1_data = f1->data();
        f2_data = f2->data();
    }
    const int64_t iterationsBetweenDeviceSynchronize = 1e4;
    adrk4_master_equation_roc(
            N_amplitudes, x, grad, Omega_max, t_final,
            sigmaFilter, sigmaWindow, f1_data, f2_data,
            addToGradient, &trace_final,
            basis_size, rocL, rocL_adjoint,
            rocH_t_Re, rocH_t_Im, calculateOmega,
            p->rho_initial_vec[p->rho_index].data(),
            p->rho_target_vec[p->rho_index].data(),
            dt, N_t, iterationsBetweenDeviceSynchronize,
            cacheSizeTimeSteps, cacheSizeTimeStepsGPU,
            timeStepsPerStoredState);
#else // USE_ROCM
#ifdef EIGEN_USE_MKL_ALL
    MKLSparseMatrix mklL(p->sd.L);
    MKLSparseMatrix mklL_adjoint(L_adjoint);
    MKLSparseMatrix mkldH_Omega_d_Re_Omega(dH_Omega_d_Re_Omega);
    MKLSparseMatrix mkldH_Omega_d_Im_Omega(dH_Omega_d_Im_Omega);
#endif // EIGEN_USE_MKL_ALL
    SpMat H_t1(basis_size, basis_size);
    SpMat H_t2(basis_size, basis_size);
    SpMat H_t3(basis_size, basis_size);
    Eigen::VectorXcd rho_vec = Eigen::VectorXcd::Zero(basis_size_squared);
    Eigen::VectorXcd chi_vec = Eigen::VectorXcd::Zero(basis_size_squared);
    Eigen::VectorXcd k0 = Eigen::VectorXcd::Zero(basis_size_squared);
    Eigen::VectorXcd k1 = Eigen::VectorXcd::Zero(basis_size_squared);
    Eigen::VectorXcd k2 = Eigen::VectorXcd::Zero(basis_size_squared);
    Eigen::VectorXcd k3 = Eigen::VectorXcd::Zero(basis_size_squared);
    Eigen::VectorXcd k4 = Eigen::VectorXcd::Zero(basis_size_squared);
    Eigen::VectorXcd k5 = Eigen::VectorXcd::Zero(basis_size_squared);
    Eigen::VectorXcd k6 = Eigen::VectorXcd::Zero(basis_size_squared);
    Eigen::VectorXcd k7 = Eigen::VectorXcd::Zero(basis_size_squared);
    Eigen::VectorXcd k8 = Eigen::VectorXcd::Zero(basis_size_squared);
    Eigen::VectorXcd k9 = Eigen::VectorXcd::Zero(basis_size_squared);
    Eigen::VectorXcd temp = Eigen::VectorXcd::Zero(basis_size_squared);
    Eigen::VectorXcd another_temp = Eigen::VectorXcd::Zero(basis_size_squared);
    Eigen::MatrixXcd rho_vec_all_t;
    Eigen::MatrixXcd rho_vec_cached_t;
    if (grad != nullptr) {
        rho_vec_cached_t = Eigen::MatrixXcd::Zero(basis_size_squared,cacheSizeTimeSteps);
#ifdef DEBUG_STATE_REMATERIALIZATION
        rho_vec_all_t = Eigen::MatrixXcd::Zero(basis_size_squared,N_t);
#endif // DEBUG_STATE_REMATERIALIZATION
    }
    // Forward propagation
    rho_vec = p->rho_initial_vec[p->rho_index];
    auto start_forward = std::chrono::steady_clock::now();
#ifdef DO_NOT_UPDATE_OMEGA_HAMILTONIAN
    for (int64_t i = 0; i < N_t; ++i) {
#ifdef DEBUG_STATE_REMATERIALIZATION
        if (grad != nullptr) {
            rho_vec_all_t.col(i) = rho_vec;
        }
#endif // DEBUG_STATE_REMATERIALIZATION
        if (grad != nullptr && i % timeStepsPerStoredState == 0) {
            rho_vec_cached_t.col(i/timeStepsPerStoredState) = rho_vec;
        }
        const double t_i = i*dt;

        std::complex<double> Omega1 = 0
        std::complex<double> Omega2 = 0;
        std::complex<double> Omega3 = 0;
        if (adrk4_flags & ADRK4_PRECOMPUTE_BASIS_FUNCTION_VALUES) {
            Omega1 = calculateOmega1(i);
            Omega2 = calculateOmega2(i);
            Omega3 = calculateOmega3(i);
        } else {
            Omega1 = calculateOmega(t_i);
            Omega2 = calculateOmega(t_i+0.5*dt);
            Omega3 = calculateOmega(t_i+dt);
        }
        if (adrk4_flags & ADRK4_CLAMP_OMEGA) {
            Omega1 *= Omega_max;
            Omega2 *= Omega_max;
            Omega3 *= Omega_max;
        }
#ifdef EIGEN_USE_MKL_ALL
        rk4_step_t_mkl(rho_vec, mklL, mkldH_Omega_d_Re_Omega, mkldH_Omega_d_Im_Omega, Omega1, Omega2, Omega3, k1, k2, k3, k4, temp, dt);
#else // EIGEN_USE_MKL_ALL
        rk4_step_t(rho_vec, p->sd.L, dH_Omega_d_Re_Omega, dH_Omega_d_Im_Omega, Omega1, Omega2, Omega3, k1, k2, k3, k4, temp, dt);
#endif // EIGEN_USE_MKL_ALL
    }
#else // DO_NOT_UPDATE_OMEGA_HAMILTONIAN
    // H_t3 from one iteration is equal to H_t1 from
    // the following iteration. We swap H_t1 and H_t3
    // each iteration of the loop below. Therefore, we need
    // to initialize H_t3 here to be equal to H_t1 of
    // the first iteration.
    std::complex<double> Omega3 = 0;
    if (adrk4_flags & ADRK4_PRECOMPUTE_BASIS_FUNCTION_VALUES) {
        Omega3 = calculateOmega1(0);
    } else {
        Omega3 = calculateOmega(0);
    }
    if (adrk4_flags & ADRK4_CLAMP_OMEGA) {
        Omega3 *= Omega_max;
    }
    H_t3.setZero();
    for (int i = 0; i < M_O_size; ++i) {
        H_t3 += p->sd.Omega_factors[i]*(Omega3*p->sd.M_O_adjoint[i]+std::conj(Omega3)*p->sd.M_O[i]);
    }
    for (int64_t i = 0; i < N_t; ++i) {
#ifdef DEBUG_STATE_REMATERIALIZATION
        if (grad != nullptr) {
            rho_vec_all_t.col(i) = rho_vec;
        }
#endif // DEBUG_STATE_REMATERIALIZATION
        if (grad != nullptr && i % timeStepsPerStoredState == 0) {
            rho_vec_cached_t.col(i/timeStepsPerStoredState) = rho_vec;
        }
        const double t_i = i*dt;

        //H_t3 from the previous iteration is equal to H_t1 from
        //the current iteration.
        std::swap(H_t1, H_t3);
        std::complex<double> Omega2 = 0;
        if (adrk4_flags & ADRK4_PRECOMPUTE_BASIS_FUNCTION_VALUES) {
            Omega2 = calculateOmega2(i);
        } else {
            Omega2 = calculateOmega(t_i+0.5*dt);
        }
        if (adrk4_flags & ADRK4_CLAMP_OMEGA) {
            Omega2 *= Omega_max;
        }
        H_t2.setZero();
        for (int i = 0; i < M_O_size; ++i) {
            H_t2 += p->sd.Omega_factors[i]*(Omega2*p->sd.M_O_adjoint[i]+std::conj(Omega2)*p->sd.M_O[i]);
        }
        std::complex<double> Omega3 = 0;
        if (adrk4_flags & ADRK4_PRECOMPUTE_BASIS_FUNCTION_VALUES) {
            Omega3 = calculateOmega3(i);
        } else {
            Omega3 = calculateOmega(t_i+dt);
        }
        if (adrk4_flags & ADRK4_CLAMP_OMEGA) {
            Omega3 *= Omega_max;
        }
        H_t3.setZero();
        for (int i = 0; i < M_O_size; ++i) {
            H_t3 += p->sd.Omega_factors[i]*(Omega3*p->sd.M_O_adjoint[i]+std::conj(Omega3)*p->sd.M_O[i]);
        }
#ifdef EIGEN_USE_MKL_ALL
        rk4_step_t_mkl(rho_vec, mklL, H_t1, H_t2, H_t3, 1, k1, k2, k3, k4, temp, dt);
#else // EIGEN_USE_MKL_ALL
        rk4_step_t(rho_vec, p->sd.L, H_t1, H_t2, H_t3, 1, k1, k2, k3, k4, temp, dt);
#endif // EIGEN_USE_MKL_ALL
    }
#endif // DO_NOT_UPDATE_OMEGA_HAMILTONIAN
    auto end_forward = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff_forward = end_forward-start_forward;
    //std::cout << "Forward propagation: " << diff_forward.count() << " s" << std::endl;
    std::complex<double> trace_final = p->rho_target_vec[p->rho_index].adjoint()*rho_vec;
    if (grad != nullptr) {
        for (int k = 0; k < N_amplitudes; ++k) {
            grad[k] = 0;
            grad[k+N_amplitudes] = 0;
        }
        // Backward propagation
        chi_vec = p->rho_target_vec[p->rho_index];
        auto start_back = std::chrono::steady_clock::now();
        for (int i = N_t; i > 0; --i) {
            const double t_i = i*dt;
            std::complex<double> tanh1 = 0;
            std::complex<double> tanh2 = 0;
            std::complex<double> tanh3 = 0;
            if (adrk4_flags & ADRK4_PRECOMPUTE_BASIS_FUNCTION_VALUES) {
                tanh1 = calculateOmega1(i-1);
                tanh2 = calculateOmega2(i-1);
                tanh3 = calculateOmega3(i-1);
            } else {
                tanh1 = calculateOmega(t_i-dt);
                tanh2 = calculateOmega(t_i-0.5*dt);
                tanh3 = calculateOmega(t_i);
            }
            const double tanh1_Re = tanh1.real();
            const double tanh1_Im = tanh1.imag();
            const double tanh2_Re = tanh2.real();
            const double tanh2_Im = tanh2.imag();
            const double tanh3_Re = tanh3.real();
            const double tanh3_Im = tanh3.imag();

            std::complex<double> Omega1 = tanh1;
            std::complex<double> Omega2 = tanh2;
            std::complex<double> Omega3 = tanh3;
            if (adrk4_flags & ADRK4_CLAMP_OMEGA) {
                Omega1 *= Omega_max;
                Omega2 *= Omega_max;
                Omega3 *= Omega_max;
            }
            H_t1.setZero();
            for (int i = 0; i < M_O_size; ++i) {
                H_t1 += p->sd.Omega_factors[i]*(Omega1*p->sd.M_O_adjoint[i]+std::conj(Omega1)*p->sd.M_O[i]);
            }
            H_t2.setZero();
            for (int i = 0; i < M_O_size; ++i) {
                H_t2 += p->sd.Omega_factors[i]*(Omega2*p->sd.M_O_adjoint[i]+std::conj(Omega2)*p->sd.M_O[i]);
            }
            H_t3.setZero();
            for (int i = 0; i < M_O_size; ++i) {
                H_t3 += p->sd.Omega_factors[i]*(Omega3*p->sd.M_O_adjoint[i]+std::conj(Omega3)*p->sd.M_O[i]);
            }

            if ((i-1) % timeStepsPerStoredState == 0) {
                rho_vec = rho_vec_cached_t.col((i-1)/timeStepsPerStoredState);
            } else {
                // H_t1, H_t2, H_t3 are used in the reverse order here,
                // because this is propagation backwards in time
#ifdef EIGEN_USE_MKL_ALL
                rk4_step_t_mkl(rho_vec, mklL, H_t3, H_t2, H_t1, 1, k1, k2, k3, k4, temp, -dt);
#else // EIGEN_USE_MKL_ALL
                rk4_step_t(rho_vec, p->sd.L, H_t3, H_t2, H_t1, 1, k1, k2, k3, k4, temp, -dt);
#endif // EIGEN_USE_MKL_ALL
            }
#ifdef DEBUG_STATE_REMATERIALIZATION
            if (i > N_t-10) {
                const double diff_norm = (rho_vec-rho_vec_all_t.col(i-1)).norm();
                std::cout << "i = " << i << ", diff_norm = " << diff_norm << std::endl;
            } else if (i < 10) {
                const double diff_norm = (rho_vec-rho_vec_all_t.col(i-1)).norm();
                std::cout << "i = " << i << ", diff_norm = " << diff_norm << std::endl;
            }
#endif // DEBUG_STATE_REMATERIALIZATION

            const std::complex<double> H_factor = -I*dt;

#ifdef EIGEN_USE_MKL_ALL
            mulMatAddAdjoint(k0, dH_Omega_d_Re_Omega, rho_vec, H_factor);
            applyL_t_mkl(k1, mklL, H_t1, rho_vec, H_factor, dt);
            applyL_t_mkl(k2, mklL, H_t2, rho_vec, H_factor, dt);
            mulMatAddAdjoint(k3, dH_Omega_d_Re_Omega, k1, H_factor);
            mulMatAddAdjoint(k4, dH_Omega_d_Re_Omega, k2, H_factor);
            applyL_t_mkl(k5, mklL, H_t2, k0, H_factor, dt);
            applyL_t_mkl(k6, mklL, H_t2, k1, H_factor, dt);
            applyL_t_mkl(k7, mklL, H_t2, k3, H_factor, dt);
            applyL_t_mkl(k8, mklL, H_t2, k5, H_factor, dt);
            mulMatAddAdjoint(k9, dH_Omega_d_Re_Omega, k6, H_factor);

            applyL_t_mkl(another_temp, mklL, H_t3, k8, H_factor, dt);
            temp = (1.0/6)*k0+(1.0/6)*k5+(1.0/12)*k8+(1.0/24)*another_temp;
            const std::complex<double> trace1_Re = chi_vec.adjoint()*temp;

            temp = k0 + 0.5*k4 + 0.5*k5 + 0.25*k9 + 0.25*k7;
            applyL_t_mkl(another_temp, mklL, H_t3, temp, H_factor, dt);
            temp = (2.0/3)*k0+(1.0/6)*k3+(1.0/6)*k4+(1.0/6)*k5+(1.0/12)*k9+(1.0/12)*k7+(1.0/6)*another_temp;
            const std::complex<double> trace2_Re = chi_vec.adjoint()*temp;

            another_temp = k2+0.5*k6;
            applyL_t_mkl(temp, mklL, H_t2, another_temp, H_factor, dt);
            mulMatAddAdjoint(another_temp, dH_Omega_d_Re_Omega, temp, H_factor);
            temp = (1.0/6)*k0+(1.0/6)*k4+(1.0/12)*another_temp;
            const std::complex<double> trace3_Re = chi_vec.adjoint()*temp;

            mulMatAddAdjoint(k0, dH_Omega_d_Im_Omega, rho_vec, H_factor);
            applyL_t_mkl(k1, mklL, H_t1, rho_vec, H_factor, dt);
            applyL_t_mkl(k2, mklL, H_t2, rho_vec, H_factor, dt);
            mulMatAddAdjoint(k3, dH_Omega_d_Im_Omega, k1, H_factor);
            mulMatAddAdjoint(k4, dH_Omega_d_Im_Omega, k2, H_factor);
            applyL_t_mkl(k5, mklL, H_t2, k0, H_factor, dt);
            applyL_t_mkl(k6, mklL, H_t2, k1, H_factor, dt);
            applyL_t_mkl(k7, mklL, H_t2, k3, H_factor, dt);
            applyL_t_mkl(k8, mklL, H_t2, k5, H_factor, dt);
            mulMatAddAdjoint(k9, dH_Omega_d_Im_Omega, k6, H_factor);

            applyL_t_mkl(another_temp, mklL, H_t3, k8, H_factor, dt);
            temp = (1.0/6)*k0+(1.0/6)*k5+(1.0/12)*k8+(1.0/24)*another_temp;
            const std::complex<double> trace1_Im = chi_vec.adjoint()*temp;

            temp = k0 + 0.5*k4 + 0.5*k5 + 0.25*k9 + 0.25*k7;
            applyL_t_mkl(another_temp, mklL, H_t3, temp, H_factor, dt);
            temp = (2.0/3)*k0+(1.0/6)*k3+(1.0/6)*k4+(1.0/6)*k5+(1.0/12)*k9+(1.0/12)*k7+(1.0/6)*another_temp;
            const std::complex<double> trace2_Im = chi_vec.adjoint()*temp;

            another_temp = k2+0.5*k6;
            applyL_t_mkl(temp, mklL, H_t2, another_temp, H_factor, dt);
            mulMatAddAdjoint(another_temp, dH_Omega_d_Im_Omega, temp, H_factor);
            temp = (1.0/6)*k0+(1.0/6)*k4+(1.0/12)*another_temp;
            const std::complex<double> trace3_Im = chi_vec.adjoint()*temp;
#else // EIGEN_USE_MKL_ALL
            mulMatAddAdjoint(k0, dH_Omega_d_Re_Omega, rho_vec, H_factor);
            applyL_t(k1, p->sd.L, H_t1, rho_vec, H_factor, dt);
            applyL_t(k2, p->sd.L, H_t2, rho_vec, H_factor, dt);
            mulMatAddAdjoint(k3, dH_Omega_d_Re_Omega, k1, H_factor);
            mulMatAddAdjoint(k4, dH_Omega_d_Re_Omega, k2, H_factor);
            applyL_t(k5, p->sd.L, H_t2, k0, H_factor, dt);
            applyL_t(k6, p->sd.L, H_t2, k1, H_factor, dt);
            applyL_t(k7, p->sd.L, H_t2, k3, H_factor, dt);
            applyL_t(k8, p->sd.L, H_t2, k5, H_factor, dt);
            mulMatAddAdjoint(k9, dH_Omega_d_Re_Omega, k6, H_factor);

            applyL_t(another_temp, p->sd.L, H_t3, k8, H_factor, dt);
            temp = (1.0/6)*k0+(1.0/6)*k5+(1.0/12)*k8+(1.0/24)*another_temp;
            const std::complex<double> trace1_Re = chi_vec.adjoint()*temp;

            temp = k0 + 0.5*k4 + 0.5*k5 + 0.25*k9 + 0.25*k7;
            applyL_t(another_temp, p->sd.L, H_t3, temp, H_factor, dt);
            temp = (2.0/3)*k0+(1.0/6)*k3+(1.0/6)*k4+(1.0/6)*k5+(1.0/12)*k9+(1.0/12)*k7+(1.0/6)*another_temp;
            const std::complex<double> trace2_Re = chi_vec.adjoint()*temp;

            another_temp = k2+0.5*k6;
            applyL_t(temp, p->sd.L, H_t2, another_temp, H_factor, dt);
            mulMatAddAdjoint(another_temp, dH_Omega_d_Re_Omega, temp, H_factor);
            temp = (1.0/6)*k0+(1.0/6)*k4+(1.0/12)*another_temp;
            const std::complex<double> trace3_Re = chi_vec.adjoint()*temp;

            mulMatAddAdjoint(k0, dH_Omega_d_Im_Omega, rho_vec, H_factor);
            applyL_t(k1, p->sd.L, H_t1, rho_vec, H_factor, dt);
            applyL_t(k2, p->sd.L, H_t2, rho_vec, H_factor, dt);
            mulMatAddAdjoint(k3, dH_Omega_d_Im_Omega, k1, H_factor);
            mulMatAddAdjoint(k4, dH_Omega_d_Im_Omega, k2, H_factor);
            applyL_t(k5, p->sd.L, H_t2, k0, H_factor, dt);
            applyL_t(k6, p->sd.L, H_t2, k1, H_factor, dt);
            applyL_t(k7, p->sd.L, H_t2, k3, H_factor, dt);
            applyL_t(k8, p->sd.L, H_t2, k5, H_factor, dt);
            mulMatAddAdjoint(k9, dH_Omega_d_Im_Omega, k6, H_factor);

            applyL_t(another_temp, p->sd.L, H_t3, k8, H_factor, dt);
            temp = (1.0/6)*k0+(1.0/6)*k5+(1.0/12)*k8+(1.0/24)*another_temp;
            const std::complex<double> trace1_Im = chi_vec.adjoint()*temp;

            temp = k0 + 0.5*k4 + 0.5*k5 + 0.25*k9 + 0.25*k7;
            applyL_t(another_temp, p->sd.L, H_t3, temp, H_factor, dt);
            temp = (2.0/3)*k0+(1.0/6)*k3+(1.0/6)*k4+(1.0/6)*k5+(1.0/12)*k9+(1.0/12)*k7+(1.0/6)*another_temp;
            const std::complex<double> trace2_Im = chi_vec.adjoint()*temp;

            another_temp = k2+0.5*k6;
            applyL_t(temp, p->sd.L, H_t2, another_temp, H_factor, dt);
            mulMatAddAdjoint(another_temp, dH_Omega_d_Im_Omega, temp, H_factor);
            temp = (1.0/6)*k0+(1.0/6)*k4+(1.0/12)*another_temp;
            const std::complex<double> trace3_Im = chi_vec.adjoint()*temp;
#endif // EIGEN_USE_MKL_ALL

            double factor1_Re = trace1_Re.real();
            double factor1_Im = trace1_Im.real();
            double factor2_Re = trace2_Re.real();
            double factor2_Im = trace2_Im.real();
            double factor3_Re = trace3_Re.real();
            double factor3_Im = trace3_Im.real();
            if (adrk4_flags & ADRK4_FILTER_BASIS_FUNCTIONS) {
                const double windowFactor1 = confinedGaussian(i-1, N_t, t_final, sigmaWindow);
                factor1_Re *= windowFactor1;
                factor1_Im *= windowFactor1;
                const double windowFactor2 = confinedGaussian(i-0.5, N_t, t_final, sigmaWindow);
                factor2_Re *= windowFactor2;
                factor2_Im *= windowFactor2;
                const double windowFactor3 = confinedGaussian(i, N_t, t_final, sigmaWindow);
                factor3_Re *= windowFactor3;
                factor3_Im *= windowFactor3;
            }
            if (adrk4_flags & ADRK4_CLAMP_OMEGA) {
                const double clampFactor1_Re = Omega_max*(1-std::pow(tanh1_Re,2));
                const double clampFactor1_Im = Omega_max*(1-std::pow(tanh1_Im,2));
                factor1_Re *= clampFactor1_Re;
                factor1_Im *= clampFactor1_Im;
                const double clampFactor2_Re = Omega_max*(1-std::pow(tanh2_Re,2));
                const double clampFactor2_Im = Omega_max*(1-std::pow(tanh2_Im,2));
                factor2_Re *= clampFactor2_Re;
                factor2_Im *= clampFactor2_Im;
                const double clampFactor3_Re = Omega_max*(1-std::pow(tanh3_Re,2));
                const double clampFactor3_Im = Omega_max*(1-std::pow(tanh3_Im,2));
                factor3_Re *= clampFactor3_Re;
                factor3_Im *= clampFactor3_Im;
            }

            if (adrk4_flags & ADRK4_PRECOMPUTE_BASIS_FUNCTION_VALUES) {
                addToGradient1(i-1, factor1_Re, factor1_Im);
                addToGradient2(i-1, factor2_Re, factor2_Im);
                addToGradient3(i-1, factor3_Re, factor3_Im);
            } else {
                addToGradient(t_i-dt, factor1_Re, factor1_Im);
                addToGradient(t_i-0.5*dt, factor2_Re, factor2_Im);
                addToGradient(t_i, factor3_Re, factor3_Im);
            }
#ifdef EIGEN_USE_MKL_ALL
            adrk4_backward_step_t_mkl(chi_vec, mklL_adjoint, H_t1, H_t2, H_t3, 1, k1, k2, k3, k4, k5, temp, another_temp, dt);
#else // EIGEN_USE_MKL_ALL
            adrk4_backward_step_t(chi_vec, L_adjoint, H_t1, H_t2, H_t3, 1, k1, k2, k3, k4, k5, temp, another_temp, dt);
#endif // EIGEN_USE_MKL_ALL
        }
        auto end_back = std::chrono::steady_clock::now();
        std::chrono::duration<double> diff_back = end_back-start_back;
        //std::cout << "Backward propagation: " << diff_back.count() << " s" << std::endl;
    }
#endif // USE_ROCM
    //std::cout << "  trace_final = " << trace_final << std::endl;
    //std::cout << "  grad = { ";
    //for (int k = 0; k < N_amplitudes; ++k) {
    //    std::complex<double> cgrad(grad[k], grad[k+N_amplitudes]);
    //    std::cout << cgrad << " ; ";
    //}
    //std::cout << std::endl;
    //std::cout << "  x = { ";
    //for (int k = 0; k < N_amplitudes; ++k) {
    //    std::complex<double> cx(x[k], x[k+N_amplitudes]);
    //    std::cout << cx << " ; ";
    //}
    //std::cout << std::endl;
    return -trace_final.real();
}

double find_optimal_Omega_adrk4_f(unsigned n, const double *x, double *grad, void *params)
{
    find_optimal_Omega_adrk4_params *p
            = (find_optimal_Omega_adrk4_params *) params;
    assert(p->rho_initial_vec.size() == p->rho_target_vec.size()
           && "Number of initial and final states is not equal!");
    const int num_states = p->rho_initial_vec.size();

    // Calculate the average fidelity over the given states
    double f_sum = 0;
    if (grad != nullptr) {
        for (int i = 0; i < n; ++i) {
            grad[i] = 0;
        }
    }
    std::vector<double> grad_temp(n, 0);
    double *grad_temp_ptr = nullptr;
    if (grad != nullptr) {
        grad_temp_ptr = grad_temp.data();
    }
    for (int i = 0; i < num_states; ++i) {
        p->rho_index = i;
        if (p->useRhoSpec && i == 0) {
            // See Eq. (7) of Bowdrey et al. [Phys. Lett. A 294 (2002) 258]).
            // The term where the identity (sigma_0) is propagated, is 3 times larger than
            // the other terms. Also note that we cannot make the exact replacement of
            // this term with 1/2, because the "identity" extended to a larger Hilbert
            // space by filling in zeros in the other elements is not an exact identity
            // anymore.
            f_sum += 3*find_optimal_Omega_adrk4_f_single_state(n, x, grad_temp_ptr, params);
            if (grad != nullptr) {
                for (int i = 0; i < n; ++i) {
                    grad[i] += 3*grad_temp[i];
                }
            }
        } else {
            f_sum += find_optimal_Omega_adrk4_f_single_state(n, x, grad_temp_ptr, params);
            if (grad != nullptr) {
                for (int i = 0; i < n; ++i) {
                    grad[i] += grad_temp[i];
                }
            }
        }
    }
    if (p->useRhoSpec) {
        assert(num_states == 4 && "Only the case of 3 Pauli matrices plus identity is supported!");
        if (grad != nullptr) {
            for (int i = 0; i < n; ++i) {
                grad[i] /= 12;
            }
        }
        return f_sum/12;
    } else {
        if (grad != nullptr) {
            for (int i = 0; i < n; ++i) {
                grad[i] /= num_states;
            }
        }
        return f_sum/num_states;
    }
}

double find_optimal_Omega_adrk4_auto_dt_f(unsigned n, const double *x, double *grad, void *params)
{
    find_optimal_Omega_adrk4_params *p
            = (find_optimal_Omega_adrk4_params *) params;
    const int adrk4_flags = p->adrk4_flags;
    p->N_t = p->start_N_t;
    std::chrono::duration<double> diff;
    std::chrono::duration<double> diff_total;
    auto start_total = std::chrono::steady_clock::now();
    double ret = find_optimal_Omega_adrk4_f(n, x, grad, params);
    double ret_diff = 0;
    int i = 0;
    p->N_t *= 2;
    for (; i < 3; ++i) {
        if (adrk4_flags & ADRK4_PRECOMPUTE_BASIS_FUNCTION_VALUES) {
            if (adrk4_flags & ADRK4_FILTER_BASIS_FUNCTIONS
                    && p->f1.count(p->N_t) == 0) {
                std::cout << "    Caching basis function values for N_t = " << p->N_t << std::endl;
                p->precomputeBasisFunctionValues(p->N_t);
            }
        }
        auto start = std::chrono::steady_clock::now();
        const double ret_new = find_optimal_Omega_adrk4_f(n, x, grad, params);
        auto end = std::chrono::steady_clock::now();
        diff = end-start;
        diff_total = end-start_total;
        ret_diff = std::abs(ret-ret_new);
        ret = ret_new;
        if (ret_diff < 1e-6) {
            break;
        }
        p->N_t *= 2;
    }
    // We do not consider the failure to converge an error here.
    // This is motivated by the observation that a high number
    // of time steps N_t seems to only be needed for the initial
    // iterations of the optimization algorithm, where they are
    // least likely to have an influence. On the other hand,
    // close to the optimal parameters x, the number of time
    // steps N_t required to reach convergence is low and should
    // be within the permitted increase of the N_t in the loop above.
    std::cout << "  F = " << -ret << ", ret_diff = " << ret_diff
              << ", N_t = " << p->N_t
              << ", start_N_t = " << p->start_N_t
              << ", i = " << i
              << ", time = " << diff.count() << " s"
              << " (total time = " << diff_total.count() << " s)"
              << std::endl;
    p->fidelity_under_optimization.push_back(-ret);
    std::vector<double> x_val(2*p->N_amplitudes);
    for (int i = 0; i < 2*p->N_amplitudes; ++i) {
        x_val[i] = x[i];
    }
    p->x_array.push_back(x_val);
    p->saveDataToFile(p->fidelity_under_optimization.back(),
                      p->x_array.back());
    return ret;
}

std::vector<InitialFinalStateSpec> initializeStateSpecToAverageSigmaX()
{
    std::vector<InitialFinalStateSpec> stateSpec;
    // Below, we either use 3 Pauli matrices or 6 eigenvalues of the Pauli
    // matrices to calculate the average fidelity. Strictly speaking, these two
    // options are are valid ways to calculate the average fidelity only for a
    // qubit (see Bowdrey et al. [Phys. Lett. A 294 (2002) 258]), but we have a
    // transmon coupled to a resonator, i.e., a system with infinite number of
    // energy levels (which we truncate to make it possible to simulate) Hence,
    // both of these approaches are approximations to the average fidelity. For
    // an exact calculation, one has to use the generalized expressions from,
    // e.g., Nielsen [Phys. Lett. A 303 (2002) 249]. The latter approach will
    // need to propagate a large number of matrices that form a basis for
    // unitary operators on the space of a qudit. Hence, we appeal to physics
    // here and assume that in the ideal operation where higher states of the
    // transmon have small population and use the approach with 3 Pauli matrices
    // for efficiency. It is possible to check that the approach with 6
    // eigenvalues of the Pauli matrices gives the same results by undefining
    // AVERAGE_FIDELITY_FROM_PAULI_MATRICES. Note that in the case of the Pauli
    // matrices, we also have to propagate the "identity" matrix, precisely
    // because it is not an exact identity when extended to the entire Hilbert
    // space by filling other states to be zero (like we do in the simulations).
#ifdef AVERAGE_FIDELITY_FROM_PAULI_MATRICES
    Eigen::MatrixXcd Id = Eigen::MatrixXcd::Zero(2,2);
    Id(0,0) = 1;
    Id(1,1) = 1;
    Eigen::MatrixXcd sigma_x = Eigen::MatrixXcd::Zero(2,2);
    sigma_x(0,1) = 1;
    sigma_x(1,0) = 1;
    Eigen::MatrixXcd sigma_y = Eigen::MatrixXcd::Zero(2,2);
    sigma_y(0,1) = std::complex<double>(0,-1);
    sigma_y(1,0) = std::complex<double>(0,1);
    Eigen::MatrixXcd sigma_z = Eigen::MatrixXcd::Zero(2,2);
    sigma_z(0,0) = 1;
    sigma_z(1,1) = -1;
    InitialFinalStateSpec stateSpecId;
    stateSpecId.rho_initial = Id;
    stateSpecId.rho_target = Id;
    // Implementing sigma_x gate
    // sigma_x is unchanged
    InitialFinalStateSpec stateSpecX;
    stateSpecX.rho_initial = sigma_x;
    stateSpecX.rho_target = sigma_x;
    // sigma_y is flipped
    InitialFinalStateSpec stateSpecY;
    stateSpecY.rho_initial = sigma_y;
    stateSpecY.rho_target = -sigma_y;
    // sigma_z is flipped
    InitialFinalStateSpec stateSpecZ;
    stateSpecZ.rho_initial = sigma_z;
    stateSpecZ.rho_target = -sigma_z;
    stateSpec = {stateSpecId, stateSpecX, stateSpecY, stateSpecZ};
#else // AVERAGE_FIDELITY_FROM_PAULI_MATRICES
    InitialFinalStateSpec stateSpec01;
    stateSpec01.initial = 0;
    stateSpec01.target = 1;
    InitialFinalStateSpec stateSpec10;
    stateSpec10.initial = 1;
    stateSpec10.target = 0;
    // implementing sigma_x gate -- its eigenvalues are unchanged
    InitialFinalStateSpec stateSpecXplus;
    stateSpecXplus.initial = M_SQRT1_2;
    stateSpecXplus.target = M_SQRT1_2;
    InitialFinalStateSpec stateSpecXminus;
    stateSpecXminus.initial = -M_SQRT1_2;
    stateSpecXminus.target = -M_SQRT1_2;
    InitialFinalStateSpec stateSpecYplus;
    stateSpecYplus.initial = std::complex<double>(0, M_SQRT1_2);
    stateSpecYplus.target = std::complex<double>(0, -M_SQRT1_2);
    InitialFinalStateSpec stateSpecYminus;
    stateSpecYminus.initial = std::complex<double>(0, -M_SQRT1_2);
    stateSpecYminus.target = std::complex<double>(0, M_SQRT1_2);
    stateSpec = {stateSpec01, stateSpec10, stateSpecXplus, stateSpecXminus,
               stateSpecYplus, stateSpecYminus};
#endif // AVERAGE_FIDELITY_FROM_PAULI_MATRICES
    return stateSpec;
}

bool isUsingRhoSpec(const std::vector<InitialFinalStateSpec> &stateSpec)
{
    bool useRhoSpec = true;
    const int num_states = stateSpec.size();
    for (int n = 0; n < num_states; ++n) {
        const int rows_initial = stateSpec[n].rho_initial.rows();
        const int cols_initial = stateSpec[n].rho_initial.cols();
        if (rows_initial == 0 || rows_initial != cols_initial) {
            useRhoSpec = false;
            break;
        }
        const int rows_target = stateSpec[n].rho_target.rows();
        const int cols_target = stateSpec[n].rho_target.cols();
        if (rows_target == 0 || rows_target != cols_target) {
            useRhoSpec = false;
            break;
        }
    }
    return useRhoSpec;
}

void expandQubitStateToFullBasis(
        Eigen::VectorXcd &rho_initial_vec_n,
        Eigen::VectorXcd &rho_target_vec_n,
        const InitialFinalStateSpec &stateSpec_n,
        const std::vector<Eigen::VectorXcd> &psi_vec)
{
    assert(std::abs(stateSpec_n.initial) <= 1
           && "State amplitude is above unity!");
    assert(std::abs(stateSpec_n.target) <= 1
           && "State amplitude is above unity!");
    assert(psi_vec.size() == 2
           && "Only qubit (two-level) states are supported!");
    assert(psi_vec[0].size() == psi_vec[1].size()
            && "Basis state vectors have different size!");
    const int basis_size = psi_vec[0].size();
    const int basis_size_squared = basis_size*basis_size;
    // For one qubit (AA&resonator system), this is a superposition of
    // computational basis states |0> and |1>. For two qubits, it is
    // a superposition of states |00> and |11>. For more than two qubits,
    // |0...0> and |1...1>.
    Eigen::VectorXcd psi_initial = stateSpec_n.initial*psi_vec[1]
    + std::sqrt(1-std::norm(stateSpec_n.initial))*psi_vec[0];
    Eigen::MatrixXcd rho_initial = psi_initial*psi_initial.adjoint();
    Eigen::VectorXcd psi_target = stateSpec_n.target*psi_vec[1]
    + std::sqrt(1-std::norm(stateSpec_n.target))*psi_vec[0];
    Eigen::MatrixXcd rho_target = psi_target*psi_target.adjoint();
    rho_initial_vec_n = Eigen::VectorXcd::Zero(basis_size_squared);
    rho_target_vec_n = Eigen::VectorXcd::Zero(basis_size_squared);
    for (int i = 0; i < basis_size; ++i) {
        for (int j = 0; j < basis_size; ++j) {
            rho_initial_vec_n(i*basis_size+j) = rho_initial(i,j);
            rho_target_vec_n(i*basis_size+j) = rho_target(i,j);
        }
    }
}

void expandQubitStateToFullBasisRhoSpec(
        Eigen::VectorXcd &rho_initial_vec_n,
        Eigen::VectorXcd &rho_target_vec_n,
        const InitialFinalStateSpec &stateSpec_n,
        const std::vector<Eigen::VectorXcd> &psi_vec)
{
    assert(psi_vec.size() == 2
           && "Only qubit (two-level) states are supported!");
    assert(psi_vec[0].size() == psi_vec[1].size()
            && "Basis state vectors have different size!");
    const int basis_size = psi_vec[0].size();
    const int basis_size_squared = basis_size*basis_size;
    Eigen::MatrixXcd rho_initial = Eigen::MatrixXcd::Zero(basis_size, basis_size);
    Eigen::MatrixXcd rho_target = Eigen::MatrixXcd::Zero(basis_size, basis_size);
    // We have already checked that rows == cols above
    assert(stateSpec_n.rho_initial.rows()
           == stateSpec_n.rho_initial.cols()
           && "rho_initial matrix is not square!");
    assert(stateSpec_n.rho_target.rows()
           == stateSpec_n.rho_target.cols()
           && "rho_target matrix is not square!");
    const int m = stateSpec_n.rho_initial.rows();
    assert(m == 2 && "Only 2x2 density matrices are supported in the spec!");
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            rho_initial += stateSpec_n.rho_initial(i,j)*psi_vec[i]*psi_vec[j].adjoint();
            rho_target += stateSpec_n.rho_target(i,j)*psi_vec[i]*psi_vec[j].adjoint();
        }
    }
    rho_initial_vec_n = Eigen::VectorXcd::Zero(basis_size_squared);
    rho_target_vec_n = Eigen::VectorXcd::Zero(basis_size_squared);
    for (int i = 0; i < basis_size; ++i) {
        for (int j = 0; j < basis_size; ++j) {
            rho_initial_vec_n(i*basis_size+j) = rho_initial(i,j);
            rho_target_vec_n(i*basis_size+j) = rho_target(i,j);
        }
    }
}

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
        std::function<void(double, const std::vector<double>&)> saveDataToFile)
{
    assert(N_t > 0 && "Number of steps needs to be a positive integer!");
    assert(int(Omega.size()) == N_t && "Delta_drive array has wrong size!");
    assert(omega_d.size() == 1 && "Precisely one drive is supported at the moment!");
    std::vector<InitialFinalStateSpec> stateSpec;
    if (stateSpec0.empty()) {
        // Empty stateSpec0 means that the average fidelity needs to be
        // calculated.
        stateSpec = initializeStateSpecToAverageSigmaX();
    } else {
        stateSpec = stateSpec0;
    }
    const int num_states = stateSpec.size();
    const bool useRhoSpec = isUsingRhoSpec(stateSpec);

    // Because generate_superoperator_diag() selects different approximations
    // depending on whether Omega is zero or non-zero, the value has to be
    // non-zero here. Which non-zero value is unimportant, since we also
    // pass the flag JQF_SUPEROPERATOR_OMIT_OMEGA_TERMS that prevents the
    // drive terms being added to the Hamiltonian (we add them manually later).
    const double Omega0 = 1;

    const std::complex<double> I(0,1);
    find_optimal_Omega_adrk4_params params;
    int flags = 0;
    flags |= JQF_SUPEROPERATOR_OMIT_OMEGA_TERMS;
    double k0x_out = 0;
    if (!k0x_a.empty()) {
        k0x_out = k0x_a[k0x_a.size()-1];
    } else if (!k0x_r.empty()) {
        k0x_out = k0x_r[k0x_r.size()-1];
    }
    const std::vector<double> kappaInternal(omega_r.size(), 0);
    params.sd = generate_superoperator_diag(
            kappa, kappaInternal, gamma, gammaInternal,
            gammaDephasing, g, Omega0, J_x, omega_d[0], omega_r,
            omega, transmon_anharmonicity, k0x_r, k0x_a,
            k0x_out, num_excitations, transmon_excitations, flags);
    if (params.sd.replaced_negative_frequencies_with_positive) {
        std::cout << "Warning: replaced negative eigenfrequencies with a "
                  << "positive drive frequency" << std::endl;
    }
    const int basis_size = params.sd.basis.size();
    const int basis_size_squared = basis_size*basis_size;

    const double dt = t_final/N_t;

    SpMat Identity(basis_size, basis_size);
    Identity.setIdentity();

    int adrk4_flags = 0;
    if (Omega_max > 0) {
        adrk4_flags |= ADRK4_CLAMP_OMEGA;
    }
    if (sigmaFilter > 0 && sigmaWindow > 0) {
        adrk4_flags |= ADRK4_FILTER_BASIS_FUNCTIONS;
#ifdef FILTERED_BASIS_FUNCTION_BY_QUADRATURE
        // Otherwise it becomes too slow
        adrk4_flags |= ADRK4_PRECOMPUTE_BASIS_FUNCTION_VALUES;
#endif // FILTERED_BASIS_FUNCTION_BY_QUADRATURE
    }

    std::vector<double> Omega_opt_Re(N_t, 0);
    std::vector<double> Omega_opt_Im(N_t, 0);
    for (int64_t i = 0; i < N_t; ++i) {
        Omega_opt_Re[i] = Omega[i].real();
        Omega_opt_Im[i] = Omega[i].imag();
    }
    if (adrk4_flags & ADRK4_CLAMP_OMEGA) {
        for (int64_t i = 0; i < N_t; ++i) {
            assert(Omega_opt_Re[i] < Omega_max
                   && "Initial guess has Omega value above maximum!");
            assert(Omega_opt_Im[i] < Omega_max
                   && "Initial guess has Omega value above maximum!");
            Omega_opt_Re[i] = std::atanh(Omega_opt_Re[i]/Omega_max);
            Omega_opt_Im[i] = std::atanh(Omega_opt_Im[i]/Omega_max);
        }
    }
    std::vector<double> basisAmplitudes_Re(N_amplitudes);
    std::vector<double> basisAmplitudes_Im(N_amplitudes);
    const double NormalizationFactor = std::sqrt(2.0/t_final);
    const double omegaFactor = M_PI/t_final;
    for (int k = 0; k < N_amplitudes; ++k) {
        double amplitude_Re_k = 0;
        double amplitude_Im_k = 0;
        for (int64_t i = 0; i < N_t; ++i) {
            const double t_i = i*dt;
            const double f = basisFunction(k, t_i, NormalizationFactor, omegaFactor);
            amplitude_Re_k += Omega_opt_Re[i]*f*dt;
            amplitude_Im_k += Omega_opt_Im[i]*f*dt;
        }
        basisAmplitudes_Re[k] = amplitude_Re_k;
        basisAmplitudes_Im[k] = amplitude_Im_k;
    }
    auto calculateOmega = [&](double t) -> std::complex<double>
    {
        // N_t may be changed during optimization
        const double N_t = params.N_t;
        const double dt = t_final/N_t;
        double Omega_Re = 0;
        double Omega_Im = 0;
        if (adrk4_flags & ADRK4_FILTER_BASIS_FUNCTIONS) {
            for (int k = 0; k < N_amplitudes; ++k) {
                const double f = filteredBasisFunction(k, t, NormalizationFactor, omegaFactor, t_final, sigmaFilter);
                Omega_Re += basisAmplitudes_Re[k]*f;
                Omega_Im += basisAmplitudes_Im[k]*f;
            }
        } else {
            for (int k = 0; k < N_amplitudes; ++k) {
                const double f = basisFunction(k, t, NormalizationFactor, omegaFactor);
                Omega_Re += basisAmplitudes_Re[k]*f;
                Omega_Im += basisAmplitudes_Im[k]*f;
            }
        }
        if (adrk4_flags & ADRK4_FILTER_BASIS_FUNCTIONS) {
            const double windowFactor = confinedGaussian(t/dt, N_t, t_final, sigmaWindow);
            Omega_Re *= windowFactor;
            Omega_Im *= windowFactor;
        }
        if (adrk4_flags & ADRK4_CLAMP_OMEGA) {
            Omega_Re = Omega_max*std::tanh(Omega_Re);
            Omega_Im = Omega_max*std::tanh(Omega_Im);
        }
        return std::complex<double>(Omega_Re, Omega_Im);
    };
    params.useRhoSpec = useRhoSpec;
    params.adrk4_flags = adrk4_flags;
    params.basis_size = basis_size;
    params.N_amplitudes = N_amplitudes;
    params.NormalizationFactor = NormalizationFactor;
    params.omegaFactor = omegaFactor;
    params.sigmaFilter = sigmaFilter;
    params.sigmaWindow = sigmaWindow;
    params.t_final = t_final;
    params.start_N_t = N_t;
    params.N_t = N_t;
    params.Omega_max = Omega_max;
    params.rho_index = 0;
    params.rho_initial_vec.resize(num_states);
    params.rho_target_vec.resize(num_states);
    Eigen::VectorXcd psi0 = Eigen::VectorXcd::Zero(basis_size);
    psi0(0) = 1; // atoms and resonators are in their ground states
    Eigen::VectorXcd psi1 = all_eigenstates_up_state(params.sd);
    std::vector<Eigen::VectorXcd> psi_vec = {psi0, psi1};
    if (useRhoSpec) {
        for (int n = 0; n < num_states; ++n) {
            expandQubitStateToFullBasisRhoSpec(
                    params.rho_initial_vec[n],
                    params.rho_target_vec[n],
                    stateSpec[n],
                    psi_vec);
        }
    } else {
        for (int n = 0; n < num_states; ++n) {
            expandQubitStateToFullBasis(
                    params.rho_initial_vec[n],
                    params.rho_target_vec[n],
                    stateSpec[n],
                    psi_vec);
        }
    }
    auto precomputeBasisFunctionValues = [&](int64_t N_t) -> void
    {
        if (params.f1.count(N_t) == 0) {
            assert(params.f2.count(N_t) == 0
                   && "Cached basis function arrays are not consistent!");
            params.f1[N_t] = Eigen::MatrixXd::Zero(N_amplitudes, N_t+1);
            params.f2[N_t] = Eigen::MatrixXd::Zero(N_amplitudes, N_t);
        } else {
            return;
        }
        const double dt = t_final/N_t;
        Eigen::MatrixXd &f1 = params.f1[N_t];
        Eigen::MatrixXd &f2 = params.f2[N_t];
        if (adrk4_flags & ADRK4_FILTER_BASIS_FUNCTIONS) {
            #pragma omp parallel for schedule(dynamic,10)
            for (int64_t i = 0; i < N_t; ++i) {
                const double t1 = i*dt;
                const double t2 = i*dt+0.5*dt;
                for (int k = 0; k < N_amplitudes; ++k) {
                    f1(k,i) = filteredBasisFunction(k, t1, NormalizationFactor, omegaFactor, t_final, sigmaFilter);
                    f2(k,i) = filteredBasisFunction(k, t2, NormalizationFactor, omegaFactor, t_final, sigmaFilter);
                }
            }
            {
                const double t1 = N_t;
                for (int k = 0; k < N_amplitudes; ++k) {
                    f1(k,N_t) = filteredBasisFunction(k, t1, NormalizationFactor, omegaFactor, t_final, sigmaFilter);
                }
            }
        } else {
            #pragma omp parallel for schedule(dynamic,10)
            for (int64_t i = 0; i < N_t; ++i) {
                const double t1 = i*dt;
                const double t2 = i*dt+0.5*dt;
                for (int k = 0; k < N_amplitudes; ++k) {
                    f1(k,i) = basisFunction(k, t1, NormalizationFactor, omegaFactor);
                    f2(k,i) = basisFunction(k, t2, NormalizationFactor, omegaFactor);
                }
            }
            {
                const double t1 = N_t;
                for (int k = 0; k < N_amplitudes; ++k) {
                    f1(k,N_t) = basisFunction(k, t1, NormalizationFactor, omegaFactor);
                }
            }
        }
    };
    params.precomputeBasisFunctionValues = precomputeBasisFunctionValues;
    if ((adrk4_flags & ADRK4_PRECOMPUTE_BASIS_FUNCTION_VALUES)
                && (adrk4_flags & ADRK4_FILTER_BASIS_FUNCTIONS)
                && max_f_evaluations != 0) {
            std::cout << "    Caching basis function values for N_t = " << N_t << std::endl;
            precomputeBasisFunctionValues(N_t);
    }
    params.saveDataToFile = saveDataToFile;
    const int numParams = 2*N_amplitudes;

    nlopt::opt opt(nlopt::LD_LBFGS, numParams);
    //nlopt::opt opt(nlopt::LN_SBPLX, numParams);
    opt.set_min_objective(find_optimal_Omega_adrk4_auto_dt_f, &params);
    opt.set_xtol_abs(1e-10);

    std::vector<double> x(numParams);
    if (x0.size() == x.size()) {
        x = x0;
    } else {
        for (int k = 0; k < N_amplitudes; ++k) {
            x[k] = basisAmplitudes_Re[k];
            x[k+N_amplitudes] = basisAmplitudes_Im[k];
        }
    }
    //std::vector<double> f_grad(2*N_amplitudes);
    //const double f_val1 = find_optimal_Omega_adrk4_f(2*N_amplitudes, x.data(), f_grad.data(), &params);
    //for (int k = 0; k < 2*N_amplitudes; ++k) {
    //    std::vector<double> x2 = x;
    //    std::vector<double> f_grad2(2*N_amplitudes);
    //    const double dx = 0.00000001;
    //    x2[k] += dx;
    //    const double f_val2 = find_optimal_Omega_adrk4_f(2*N_amplitudes, x2.data(), f_grad2.data(), &params);
    //    const double f_diff_k = (f_val2-f_val1)/dx;
    //    std::cout << "f_grad[" << k << "] = " << f_grad[k] << ", f_diff_k = " << f_diff_k << std::endl;
    //}

    double minf;
    nlopt::result result;

    // The if statements below make it possible to
    // optimize without setting the maximum number of
    // function evaluations if max_f_evaluations is
    // negative.
    if (max_f_evaluations > 0) {
        opt.set_maxeval(max_f_evaluations);
    }
    if (max_f_evaluations != 0) {
        std::cout << "Optimizing" << std::endl;
        result = opt.optimize(x, minf);
    }
    for (int k = 0; k < N_amplitudes; ++k) {
        basisAmplitudes_Re[k] = x[k];
        basisAmplitudes_Im[k] = x[k+N_amplitudes];
    }
    InitialFinalStateSpec specForVisualizing;
    if (useRhoSpec) {
        // State transfer: |0> -> |1>
        specForVisualizing.initial = 0;
        specForVisualizing.target = 1;
    } else {
        specForVisualizing = stateSpec[0];
    }
    JQFData ret = jqf_time_dependent_Omega(
                specForVisualizing, kappa, gamma, gammaInternal, gammaDephasing,
                g, J_x, calculateOmega, omega_d[0], omega_r, omega,
                transmon_anharmonicity, k0x_r, k0x_a, num_excitations,
                transmon_excitations, t_final, params.N_t,
                data_reduce_factor*(params.N_t/N_t));
    ret.fidelity_under_optimization
        = std::move(params.fidelity_under_optimization);
    if (sigmaFilter > 0 && sigmaWindow > 0) {
        ret.window_array = std::vector<double>(params.N_t, 0);
        for (int64_t i = 0; i < params.N_t; ++i) {
            ret.window_array[i] = confinedGaussian(i+1, params.N_t, t_final, sigmaWindow);
        }
    }
    ret.x_array = std::move(params.x_array);
    return ret;
}
