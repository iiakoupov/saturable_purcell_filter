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

#include "jqf_superoperator.h"

#include "diagonalize_atom_resonator_system.h"
#include "master_equation.h"

#include "Eigen/Eigenvalues"

#include <fstream>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

// This function is useful to check whether
// the difference of two sparse matrices is zero
//inline void purge_zeros(SpMat &mat)
//{
//    std::vector<Eigen::Triplet<std::complex<double>>> triplets;
//    for (int k = 0; k < mat.outerSize(); ++k) {
//        for (SpMat::InnerIterator it(mat,k); it; ++it) {
//            if (std::abs(it.value()) > 1e-15) {
//                triplets.emplace_back(it.row(), it.col(), it.value());
//            }
//        }
//    }
//    mat.setFromTriplets(triplets.begin(), triplets.end());
//}

#define LADDER_OPERATORS_INSTEAD_OF_PAULI
#define TRANSMON_BASIS_INSTEAD_OF_TWO_LEVEL
#define EXP_TERMS_INSTEAD_OF_SIN_COS
#define ABSOLUTE_FREQUENCY_DEPENDENT_CORRECTIONS
#define ALTERNATIVE_DRIVEN_EVOLUTION_APPROXIMATION
#define COMPUTE_AND_USE_O_MN
//#define EXPENSIVE_CALCULATION_OF_O_MN_PRODUCTS
//#define LESS_RAM_FOR_L

SpMat rewrite_operator_in_basis(const SpMat &M_op_m, int m, const std::vector<BasisVector> &basis)
{
    const int basis_size = basis.size();
    SpMat M_op(basis_size, basis_size);
    std::vector<Eigen::Triplet<std::complex<double>>> M_op_triplets;
    for (int j = 0; j < M_op_m.outerSize(); ++j) {
        for (SpMat::InnerIterator it(M_op_m,j); it; ++it) {
            const Sigma s_jk_m(it.row(), it.col(), m);
            std::vector<Eigen::Triplet<std::complex<double>>> triplets
                = s_jk_m.triplets(basis);
            const int num_triplets = triplets.size();
            std::vector<Eigen::Triplet<std::complex<double>>> scaled_triplets;
            scaled_triplets.reserve(num_triplets);
            for (int l = 0; l < num_triplets; ++l) {
                scaled_triplets.emplace_back(triplets[l].row(),
                                             triplets[l].col(),
                                             triplets[l].value()*it.value());
            }
            M_op_triplets.insert(M_op_triplets.end(), scaled_triplets.cbegin(), scaled_triplets.cend());
        }
    }
    M_op.setFromTriplets(M_op_triplets.begin(), M_op_triplets.end());
    return M_op;
}

JQFSuperoperatorData generate_superoperator_diag(
        const std::vector<double> &kappa,
        const std::vector<double> &kappaInternal,
        const std::vector<double> &gamma,
        const std::vector<double> &gammaInternal,
        const std::vector<double> &gammaDephasing,
        double g, double Omega, double J_x, double omega_d,
        const std::vector<double> &omega_r,
        const std::vector<double> &omega,
        const std::vector<double> &transmon_anharmonicity,
        const std::vector<double> &k0x_r,
        const std::vector<double> &k0x_a,
        double k0x_out,
        int num_excitations,
        const std::vector<int> &transmon_excitations,
        int flags)
{
    const std::complex<double> I(0,1);

    const int NResonators = k0x_r.size();
    // Number of unpaired atoms
    const int NUnpaired = k0x_a.size();
    // Total number of atoms
    const int NAtoms = NResonators + NUnpaired;
    // Number of the atoms or resonators that are
    // directly attached to the waveguide is the
    // same as the total number of atoms. Some
    // of the atoms are attached through the
    // resonator, and some of them are attached
    // directly.
    const int NAttached = NAtoms;

    const int omega_size = omega.size();
    const int omega_r_size = omega_r.size();
    assert(omega_size == NAtoms
           && "omega array has wrong size!");
    assert(omega_r_size == NResonators
           && "omega_r array has wrong size!");
    assert(int(kappa.size()) == NResonators
           && "kappa array has wrong size!");
    assert(int(gamma.size()) == NUnpaired
           && "gamma array has wrong size!");

    const int numKappaInternal = kappaInternal.size();
    assert(numKappaInternal <= NResonators
            && "Too many kappaInternal values");
    const int numGammaInternal = gammaInternal.size();
    assert(numGammaInternal <= NAtoms
            && "Too many gammaInternal values");
    const int numGammaDephasing = gammaDephasing.size();
    assert(numGammaDephasing <= NAtoms
            && "Too many gammaDephasing values");

    const int transmon_anharmonicity_size = transmon_anharmonicity.size();
    assert(transmon_anharmonicity_size == NAtoms
            && "transmon_anharmonicity has wrong size!");
    const int transmon_excitations_size = transmon_excitations.size();
    assert(transmon_excitations_size == NAtoms
            && "transmon_excitations has wrong size!");
    std::vector<double> Delta_r(omega_r_size);
    for (int i = 0; i < omega_r_size; ++i) {
        Delta_r[i] = omega_d-omega_r[i];
    }
    std::vector<double> Delta(omega_size);
    for (int i = 0; i < omega_size; ++i) {
        Delta[i] = omega_d-omega[i];
    }
    JQFSuperoperatorData ret;

    const bool drivenEvolution = (Omega != 0);

    std::vector<Sigma> raising_operators;
    std::vector<Sigma> lowering_operators;
    std::vector<double> k0x_vals;
    std::vector<double> decay_factors;
    std::vector<double> omega_attached;
    raising_operators.reserve(NAttached);
    lowering_operators.reserve(NAttached);
    k0x_vals.reserve(NAttached);
    decay_factors.reserve(NAttached);
    omega_attached.reserve(NAttached);

    for (int i = 0; i < NResonators; ++i) {
        const Sigma a_i(NAtoms+i, AnnihilationOperatorType);
        const Sigma a_i_adjoint(NAtoms+i, CreationOperatorType);
        raising_operators.push_back(a_i_adjoint);
        lowering_operators.push_back(a_i);
        k0x_vals.push_back(k0x_r[i]);
        decay_factors.push_back(kappa[i]);
        omega_attached.push_back(omega_r[i]);
    }

    for (int i = 0; i < NUnpaired; ++i) {
        const Sigma b_i(NResonators+i, AnnihilationOperatorType);
        const Sigma b_i_adjoint(NResonators+i, CreationOperatorType);
        raising_operators.push_back(b_i_adjoint);
        lowering_operators.push_back(b_i);
        k0x_vals.push_back(k0x_a[i]);
        decay_factors.push_back(gamma[i]);
        omega_attached.push_back(omega[NResonators+i]);
    }

    std::vector<Eigen::VectorXd> subsystem_eigenvalues;
    std::vector<Eigen::VectorXd> subsystem_eigenvalue_shifts;
    std::vector<SpMat> lowering_operators_eigenbasis;
    std::vector<SpMat> a_operators_eigenbasis;
    std::vector<SpMat> b_operators_eigenbasis;
    std::vector<SpMat> n_res_operators_eigenbasis;
    std::vector<SpMat> n_atom_operators_eigenbasis;
    std::vector<SpMat> sqrt_n_atom_operators_eigenbasis;
    std::vector<int> psi_up_eigenstate_indices;
    std::vector<int> psi_down_eigenstate_indices;
    const double tolerance = 1e-15;
    for (int m = 0; m < NResonators; ++m) {
        std::vector<BasisVector> sub_basis;
        // Include the zero-excitation state, even though it introduces
        // a zero column and zero row in the Hamiltonian matrix. The
        // diagonalization routine does not seem to have issues with
        // it and just outputs a unit vector as the eigenstate corresponding
        // to the zero-excitation basis state.
        std::vector<int> sub_basis_num_excitations;
        for (int i = 0; i <= num_excitations; ++i) {
            std::vector<BasisVector> sub_basis_i
                    = generate_basis_vectors_combined_transmon_and_harmonic(
                        {transmon_excitations[m]}, 1, i);
            sub_basis.insert(sub_basis.end(),
                             sub_basis_i.cbegin(),
                             sub_basis_i.cend());
            const int sub_basis_i_size = sub_basis_i.size();
            std::vector<int> sub_basis_i_excitations(sub_basis_i_size, i);
            sub_basis_num_excitations.insert(sub_basis_num_excitations.end(),
                                             sub_basis_i_excitations.cbegin(),
                                             sub_basis_i_excitations.cend());
        }
        const int sub_basis_size = sub_basis.size();
        //std::cout << "Subsystem basis:" << std::endl;
        //for (int i = 0; i < sub_basis_size; ++i) {
        //    std::cout << stringify_spec(sub_basis[i].spec())
        //              << ", excitations = " << sub_basis_num_excitations[i]
        //              << std::endl;
        //}
        // Within every subsystem, the index of the atom is 0,
        // and the index of the resonator is 1
        const Sigma b_m(0, AnnihilationOperatorType);
        const Sigma b_m_adjoint(0, CreationOperatorType);
        const SigmaProduct numberOp_m({b_m_adjoint, b_m});
        const SpMat M_n_m = numberOp_m.matrix(sub_basis);
        std::vector<Eigen::Triplet<std::complex<double>>> n_triplets
            = numberOp_m.triplets(sub_basis);
        const int n_triplets_size = n_triplets.size();
        std::vector<Eigen::Triplet<std::complex<double>>> sqrt_n_triplets;
        sqrt_n_triplets.reserve(n_triplets_size);
        // Square root of a diagonal matrix is found by doing
        // elementwise square root of the diagonal elements.
        for (int j = 0; j < n_triplets_size; ++j) {
            assert(n_triplets[j].row() == n_triplets[j].col()
                    && "number operator is not diagonal!");
            sqrt_n_triplets.emplace_back(n_triplets[j].row(),
                    n_triplets[j].col(), std::sqrt(n_triplets[j].value()));
        }
        SpMat M_sqrt_n_atom_m(sub_basis_size, sub_basis_size);
        M_sqrt_n_atom_m.setFromTriplets(sqrt_n_triplets.begin(), sqrt_n_triplets.end());
        SpMat H_subsystem(sub_basis_size, sub_basis_size);
        // \omega b_m^\dagger*b_m
        H_subsystem += omega[m]*M_n_m;
        if (transmon_excitations[m] > 1) {
            const SigmaProduct anharmonicityOp_m({b_m_adjoint, b_m_adjoint, b_m, b_m});
            const SpMat M_alpha_m = anharmonicityOp_m.matrix(sub_basis);
            H_subsystem += 0.5*transmon_anharmonicity[m]*M_alpha_m;
        }
        const Sigma a_m(1, AnnihilationOperatorType);
        const Sigma a_m_adjoint(1, CreationOperatorType);
        const SigmaProduct numberOp_res_m({a_m_adjoint, a_m});
        const SpMat M_n_res_m = numberOp_res_m.matrix(sub_basis);
        // \omega_r a_m^\dagger*a_m
        H_subsystem += omega_r[m]*M_n_res_m;
        const SigmaProduct b_adjoint_a({b_m_adjoint, a_m});
        const SigmaProduct b_a_adjoint({b_m, a_m_adjoint});
        const SpMat M_b_adjoint_a = b_adjoint_a.matrix(sub_basis);
        const SpMat M_b_a_adjoint = b_a_adjoint.matrix(sub_basis);
        const SpMat temp_g_r = g*(M_b_adjoint_a+M_b_a_adjoint);
        H_subsystem += temp_g_r;
        //Additional terms if the rotating wave approximation is not made
        //const SigmaProduct b_a({b_m, a_m});
        //const SigmaProduct b_adjoint_a_adjoint({b_m_adjoint, a_m_adjoint});
        //const SpMat M_b_a = b_a.matrix(sub_basis);
        //const SpMat M_b_adjoint_a_adjoint = b_adjoint_a_adjoint.matrix(sub_basis);
        //const SpMat temp_g_r_non_rwa = g*(M_b_a+M_b_adjoint_a_adjoint);
        //H_subsystem += temp_g_r_non_rwa;
        Eigen::MatrixXcd H_subsystem_dense = H_subsystem;
        //std::cout << "H_subsystem_dense = \n" << H_subsystem_dense << std::endl;
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es;
        es.compute(H_subsystem_dense);
        Eigen::VectorXd eigenvalues = es.eigenvalues();
        Eigen::MatrixXcd eigenvectors = es.eigenvectors();
        // Check if the eigenvectors have unique excitation numbers.
        // This is only expected to occur within the rotating wave
        // approximation. If the number of excitations is unique, we use
        // it to sort the eigenvectors. Sorting according to the
        // excitation number is, in general, not the same as sorting
        // according to the eigenvalues, which Eigen does by default in
        // the diagonalization routine. Sorting according to the
        // excitation number is actually required for the following
        // code. There, the assumption is that the eigenbasis is such
        // that an eigenvector with a larger index has a larger or equal
        // excitation number compared to an eigenvector with a smaller
        // index.
        bool sortAccordingToExcitationNumber = true;
        Eigen::VectorXi eigenvector_num_excitations = Eigen::VectorXi::Constant(sub_basis_size, -1);
        for (int i = 0; i < sub_basis_size; ++i) {
            int eigenvector_excitations = -1;
            for (int j = 0; j < sub_basis_size; ++j) {
                if (std::abs(eigenvectors(j,i)) > tolerance) {
                    if (eigenvector_excitations < 0) {
                        eigenvector_excitations = sub_basis_num_excitations[j];
                    } else if (eigenvector_excitations != sub_basis_num_excitations[j]) {
                        sortAccordingToExcitationNumber = false;
                        break;
                    }
                }
            }
            if (!sortAccordingToExcitationNumber) {
                break;
            }
            eigenvector_num_excitations[i] = eigenvector_excitations;
        }
        Eigen::VectorXd eigenvalue_shifts = Eigen::VectorXd::Zero(sub_basis_size);
        if (sortAccordingToExcitationNumber) {
            for (int i = 0; i < sub_basis_size-1; ++i) {
                int k;
                eigenvector_num_excitations.segment(i,sub_basis_size-i).minCoeff(&k);
                if (k > 0) {
                    std::swap(eigenvector_num_excitations[i], eigenvector_num_excitations[k+i]);
                    std::swap(eigenvalues[i], eigenvalues[k+i]);
                    eigenvectors.col(i).swap(eigenvectors.col(k+i));
                    //std::cout << "Swapping " << i << " and " << k+i << std::endl;
                }
            }
            for (int i = 0; i < sub_basis_size; ++i) {
                eigenvalue_shifts(i) = eigenvector_num_excitations(i);
            }
        }
        //std::cout << "eigenvalue_shifts =  \n" << eigenvalue_shifts << std::endl;

        subsystem_eigenvalues.push_back(eigenvalues);
        assert(sub_basis[0].size() == 2
                && "Subsystem basis element size is not 2!");
        std::vector<sigma_state_t> spec_01(sub_basis[0].size(), 0);
        std::vector<sigma_state_t> spec_10(sub_basis[0].size(), 0);
        spec_01[0] = 1;
        spec_10[1] = 1;
        BasisVector vec_01(spec_01);
        BasisVector vec_10(spec_10);
        int vec_01_index = -1;
        int vec_10_index = -1;
        for (int i = 0; i < sub_basis_size; ++i) {
            if (sub_basis[i] == vec_01) {
                vec_01_index = i;
            } else if (sub_basis[i] == vec_10) {
                vec_10_index = i;
            }
        }
        //std::cout << "vec_01_index = " << vec_01_index << std::endl;
        //std::cout << "vec_10_index = " << vec_10_index << std::endl;
        double eigenstate_max_01_element = 0;
        double eigenstate_max_10_element = 0;
        int eigenstate_with_max_01_element_index = -1;
        int eigenstate_with_max_10_element_index = -1;
        for (int i = 0; i < sub_basis_size; ++i) {
            if (eigenstate_max_01_element < std::abs(eigenvectors(vec_01_index, i))) {
                eigenstate_max_01_element = std::abs(eigenvectors(vec_01_index, i));
                eigenstate_with_max_01_element_index = i;
            }
            if (eigenstate_max_10_element < std::abs(eigenvectors(vec_10_index, i))) {
                eigenstate_max_10_element = std::abs(eigenvectors(vec_10_index, i));
                eigenstate_with_max_10_element_index = i;
            }
        }
        psi_up_eigenstate_indices.push_back(eigenstate_with_max_01_element_index);
        psi_down_eigenstate_indices.push_back(eigenstate_with_max_10_element_index);
        //std::cout << "eigenstate_with_max_01_element_index = " << eigenstate_with_max_01_element_index << std::endl;
        //std::cout << "eigenstate_with_max_10_element_index = " << eigenstate_with_max_10_element_index << std::endl;
        //std::cout << "eigenvector_num_excitations = \n" << eigenvector_num_excitations << std::endl;
        //std::cout << "H_subsystem_dense eigenvalues = \n" << eigenvalues << std::endl;
        //std::cout << "H_subsystem_dense eigenvectors = \n" << eigenvectors << std::endl;
        Eigen::MatrixXcd M_a_m_dense = a_m.matrix(sub_basis);
        Eigen::MatrixXcd M_b_m_dense = b_m.matrix(sub_basis);
        Eigen::MatrixXcd M_n_atom_m_dense = M_n_m;
        Eigen::MatrixXcd M_sqrt_n_atom_m_dense = M_sqrt_n_atom_m;
        Eigen::MatrixXcd M_n_res_m_dense = M_n_res_m;
        //std::cout << "M_n_atom_m_dense = \n" << M_n_atom_m_dense << std::endl;
        //std::cout << "M_sqrt_n_atom_m_dense = \n" << M_sqrt_n_atom_m_dense << std::endl;
        //std::cout << "M_a_m_dense = \n" << M_a_m_dense << std::endl;
        Eigen::MatrixXcd M_a_m_transformed = eigenvectors.adjoint()*M_a_m_dense*eigenvectors;
        Eigen::MatrixXcd M_b_m_transformed = eigenvectors.adjoint()*M_b_m_dense*eigenvectors;
        Eigen::MatrixXcd M_n_res_m_transformed = eigenvectors.adjoint()*M_n_res_m_dense*eigenvectors;
        Eigen::MatrixXcd M_n_atom_m_transformed = eigenvectors.adjoint()*M_n_atom_m_dense*eigenvectors;
        Eigen::MatrixXcd M_sqrt_n_atom_m_transformed = eigenvectors.adjoint()*M_sqrt_n_atom_m_dense*eigenvectors;
        //std::cout << "M_a_m_transformed = \n" << M_a_m_transformed << std::endl;
        //std::cout << "M_a_m_adjoint_transformed = \n" << M_a_m_adjoint_transformed << std::endl;
        //std::cout << "M_a_m_transformed-M_a_m_adjoint_transformed = \n" << M_a_m_transformed-M_a_m_adjoint_transformed << std::endl;

        std::vector<Eigen::Triplet<std::complex<double>>> M_a_m_projected_triplets;
        std::vector<Eigen::Triplet<std::complex<double>>> M_b_m_projected_triplets;
        std::vector<Eigen::Triplet<std::complex<double>>> M_n_res_m_projected_triplets;
        std::vector<Eigen::Triplet<std::complex<double>>> M_n_atom_m_projected_triplets;
        std::vector<Eigen::Triplet<std::complex<double>>> M_sqrt_n_atom_m_projected_triplets;
        for (int j = 0; j < sub_basis_size; ++j) {
            for (int k = 0; k < sub_basis_size; ++k) {
                const std::complex<double> a_m_jk = M_a_m_transformed(j,k);
                if (std::abs(a_m_jk) > tolerance) {
                    assert(k > j && "a_m_jk is non-zero only for k > j");
                    M_a_m_projected_triplets.emplace_back(j, k, a_m_jk);
                }
                const std::complex<double> b_m_jk = M_b_m_transformed(j,k);
                if (std::abs(b_m_jk) > tolerance) {
                    M_b_m_projected_triplets.emplace_back(j, k, b_m_jk);
                }
                const std::complex<double> n_atom_m_jk = M_n_atom_m_transformed(j,k);
                if (std::abs(n_atom_m_jk) > tolerance) {
                    M_n_atom_m_projected_triplets.emplace_back(j, k, n_atom_m_jk);
                }
                const std::complex<double> sqrt_n_atom_m_jk = M_sqrt_n_atom_m_transformed(j,k);
                if (std::abs(sqrt_n_atom_m_jk) > tolerance) {
                    M_sqrt_n_atom_m_projected_triplets.emplace_back(j, k, sqrt_n_atom_m_jk);
                }
                const std::complex<double> n_res_m_jk = M_n_res_m_transformed(j,k);
                if (std::abs(n_res_m_jk) > tolerance) {
                    M_n_res_m_projected_triplets.emplace_back(j, k, n_res_m_jk);
                }
            }
        }
        SpMat M_a_m_projected_sparse(sub_basis_size, sub_basis_size);
        M_a_m_projected_sparse.setFromTriplets(M_a_m_projected_triplets.begin(),
                                               M_a_m_projected_triplets.end());
        lowering_operators_eigenbasis.push_back(M_a_m_projected_sparse);
        a_operators_eigenbasis.push_back(M_a_m_projected_sparse);
        subsystem_eigenvalue_shifts.push_back(eigenvalue_shifts);
        SpMat M_b_m_projected_sparse(sub_basis_size, sub_basis_size);
        M_b_m_projected_sparse.setFromTriplets(M_b_m_projected_triplets.begin(),
                                               M_b_m_projected_triplets.end());
        b_operators_eigenbasis.push_back(M_b_m_projected_sparse);
        SpMat M_n_res_m_projected_sparse(sub_basis_size, sub_basis_size);
        M_n_res_m_projected_sparse.setFromTriplets(M_n_res_m_projected_triplets.begin(),
                                                   M_n_res_m_projected_triplets.end());
        n_res_operators_eigenbasis.push_back(M_n_res_m_projected_sparse);
        SpMat M_n_atom_m_projected_sparse(sub_basis_size, sub_basis_size);
        M_n_atom_m_projected_sparse.setFromTriplets(M_n_atom_m_projected_triplets.begin(),
                                                    M_n_atom_m_projected_triplets.end());
        n_atom_operators_eigenbasis.push_back(M_n_atom_m_projected_sparse);
        SpMat M_sqrt_n_atom_m_projected_sparse(sub_basis_size, sub_basis_size);
        M_sqrt_n_atom_m_projected_sparse.setFromTriplets(M_sqrt_n_atom_m_projected_triplets.begin(),
                                                    M_sqrt_n_atom_m_projected_triplets.end());
        sqrt_n_atom_operators_eigenbasis.push_back(M_sqrt_n_atom_m_projected_sparse);
    }

    for (int m = NResonators; m < NAttached; ++m) {
        // Include the zero-excitation state, even though it introduces
        // a zero column and zero row in the Hamiltonian matrix. The
        // diagonalization routine does not seem to have issues with
        // it and just outputs a unit vector as the eigenstate corresponding
        // to the zero-excitation basis state.
        std::vector<BasisVector> sub_basis;
        std::vector<int> sub_basis_num_excitations;
        for (int i = 0; i <= transmon_excitations[m]; ++i) {
            std::vector<sigma_state_t> spec(1, 0);
            spec[0] = i;
            sub_basis.emplace_back(spec);
            sub_basis_num_excitations.push_back(i);
        }
        const int sub_basis_size = sub_basis.size();
        //std::cout << "Transmon subsystem basis:" << std::endl;
        //for (int i = 0; i < sub_basis_size; ++i) {
        //    std::cout << stringify_spec(sub_basis[i].spec())
        //              << ", excitations = " << sub_basis_num_excitations[i]
        //              << std::endl;
        //}
        const Sigma b_m(0, AnnihilationOperatorType);
        const Sigma b_m_adjoint(0, CreationOperatorType);
        const SigmaProduct numberOp_m({b_m_adjoint, b_m});
        const SpMat M_n_m = numberOp_m.matrix(sub_basis);
        std::vector<Eigen::Triplet<std::complex<double>>> n_triplets
            = numberOp_m.triplets(sub_basis);
        const int n_triplets_size = n_triplets.size();
        std::vector<Eigen::Triplet<std::complex<double>>> sqrt_n_triplets;
        sqrt_n_triplets.reserve(n_triplets_size);
        // Square root of a diagonal matrix is found by doing
        // elementwise square root of the diagonal elements.
        for (int j = 0; j < n_triplets_size; ++j) {
            assert(n_triplets[j].row() == n_triplets[j].col()
                    && "number operator is not diagonal!");
            sqrt_n_triplets.emplace_back(n_triplets[j].row(),
                    n_triplets[j].col(), std::sqrt(n_triplets[j].value()));
        }
        SpMat M_sqrt_n_atom_m(sub_basis_size, sub_basis_size);
        M_sqrt_n_atom_m.setFromTriplets(sqrt_n_triplets.begin(), sqrt_n_triplets.end());
        SpMat H_subsystem(sub_basis_size, sub_basis_size);
        H_subsystem += omega[m]*M_n_m;
        if (transmon_excitations[m] > 1) {
            const SigmaProduct anharmonicityOp_m({b_m_adjoint, b_m_adjoint, b_m, b_m});
            const SpMat M_alpha_m = anharmonicityOp_m.matrix(sub_basis);
            H_subsystem += 0.5*transmon_anharmonicity[m]*M_alpha_m;
        }
        Eigen::MatrixXcd H_subsystem_dense = H_subsystem;
        //std::cout << "H_subsystem_dense = \n" << H_subsystem_dense << std::endl;
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es;
        es.compute(H_subsystem_dense);
        Eigen::VectorXd eigenvalues = es.eigenvalues();
        Eigen::MatrixXcd eigenvectors = es.eigenvectors();
        // Sort eigenvalues and eigenvectors. In principle, it should not be
        // required, but unless the transmon model is adjusted to include the
        // higher order corrections, after some point, the higher excited states
        // will actually have lower energies. The code below has an assert which
        // checks for this unphysical situation and would need to be removed to
        // proceed.
        // Also see the comment above in the loop that deals with atoms coupled
        // to resonators.
        bool sortAccordingToExcitationNumber = true;
        Eigen::VectorXi eigenvector_num_excitations = Eigen::VectorXi::Constant(sub_basis_size, -1);
        for (int i = 0; i < sub_basis_size; ++i) {
            int eigenvector_excitations = -1;
            for (int j = 0; j < sub_basis_size; ++j) {
                if (std::abs(eigenvectors(j,i)) > tolerance) {
                    if (eigenvector_excitations < 0) {
                        eigenvector_excitations = sub_basis_num_excitations[j];
                    } else if (eigenvector_excitations != sub_basis_num_excitations[j]) {
                        sortAccordingToExcitationNumber = false;
                        break;
                    }
                }
            }
            if (!sortAccordingToExcitationNumber) {
                break;
            }
            eigenvector_num_excitations[i] = eigenvector_excitations;
        }
        Eigen::VectorXd eigenvalue_shifts = Eigen::VectorXd::Zero(sub_basis_size);
        if (sortAccordingToExcitationNumber) {
            for (int i = 0; i < sub_basis_size-1; ++i) {
                int k;
                eigenvector_num_excitations.segment(i,sub_basis_size-i).minCoeff(&k);
                if (k > 0) {
                    std::swap(eigenvector_num_excitations[i], eigenvector_num_excitations[k+i]);
                    std::swap(eigenvalues[i], eigenvalues[k+i]);
                    eigenvectors.col(i).swap(eigenvectors.col(k+i));
                    //std::cout << "Swapping " << i << " and " << k+i << std::endl;
                }
            }
            for (int i = 0; i < sub_basis_size; ++i) {
                eigenvalue_shifts(i) = eigenvector_num_excitations(i);
            }
        }
        //std::cout << "eigenvalue_shifts =  \n" << eigenvalue_shifts << std::endl;

        double last_eigenvalue = eigenvalues[0];
        for (int i = 1; i < sub_basis_size; ++i) {
            if (last_eigenvalue >= eigenvalues[i]) {
                // This assert could be removed, but then it is not clear
                // that the model can be trusted, because this is unphysical.
                assert(0 && "Transmon eigenvalues are not"
                            "monotonically increasing!");
            }
            last_eigenvalue = eigenvalues[i];
        }
        subsystem_eigenvalues.push_back(eigenvalues);
        //std::cout << "H_subsystem_dense eigenvalues = \n" << eigenvalues << std::endl;
        //std::cout << "H_subsystem_dense eigenvectors = \n" << eigenvectors << std::endl;
        Eigen::MatrixXcd M_b_m_dense = b_m.matrix(sub_basis);
        Eigen::MatrixXcd M_n_atom_m_dense = M_n_m;
        Eigen::MatrixXcd M_sqrt_n_atom_m_dense = M_sqrt_n_atom_m;
        //std::cout << "M_n_atom_m_dense = \n" << M_n_atom_m_dense << std::endl;
        //std::cout << "M_sqrt_n_atom_m_dense = \n" << M_sqrt_n_atom_m_dense << std::endl;
        //std::cout << "M_b_m_dense = \n" << M_b_m_dense << std::endl;
        Eigen::MatrixXcd M_b_m_transformed = eigenvectors.adjoint()*M_b_m_dense*eigenvectors;
        Eigen::MatrixXcd M_n_atom_m_transformed = eigenvectors.adjoint()*M_n_atom_m_dense*eigenvectors;
        Eigen::MatrixXcd M_sqrt_n_atom_m_transformed = eigenvectors.adjoint()*M_sqrt_n_atom_m_dense*eigenvectors;
        std::vector<Eigen::Triplet<std::complex<double>>> M_b_m_projected_triplets;
        std::vector<Eigen::Triplet<std::complex<double>>> M_n_atom_m_projected_triplets;
        std::vector<Eigen::Triplet<std::complex<double>>> M_sqrt_n_atom_m_projected_triplets;
        for (int j = 0; j < sub_basis_size; ++j) {
            for (int k = 0; k < sub_basis_size; ++k) {
                const std::complex<double> b_m_jk = M_b_m_transformed(j,k);
                if (std::abs(b_m_jk) > tolerance) {
                    assert(k > j && "b_m_jk is non-zero only for k > j");
                    M_b_m_projected_triplets.emplace_back(j, k, b_m_jk);
                }
                const std::complex<double> n_atom_m_jk = M_n_atom_m_transformed(j,k);
                if (std::abs(n_atom_m_jk) > tolerance) {
                    M_n_atom_m_projected_triplets.emplace_back(j, k, n_atom_m_jk);
                }
                const std::complex<double> sqrt_n_atom_m_jk = M_sqrt_n_atom_m_transformed(j,k);
                if (std::abs(sqrt_n_atom_m_jk) > tolerance) {
                    M_sqrt_n_atom_m_projected_triplets.emplace_back(j, k, sqrt_n_atom_m_jk);
                }
            }
        }
        SpMat M_b_m_projected_sparse(sub_basis_size, sub_basis_size);
        M_b_m_projected_sparse.setFromTriplets(M_b_m_projected_triplets.begin(),
                                               M_b_m_projected_triplets.end());
        lowering_operators_eigenbasis.push_back(M_b_m_projected_sparse);
        b_operators_eigenbasis.push_back(M_b_m_projected_sparse);
        subsystem_eigenvalue_shifts.push_back(eigenvalue_shifts);
        SpMat M_n_atom_m_projected_sparse(sub_basis_size, sub_basis_size);
        M_n_atom_m_projected_sparse.setFromTriplets(M_n_atom_m_projected_triplets.begin(),
                                                    M_n_atom_m_projected_triplets.end());
        n_atom_operators_eigenbasis.push_back(M_n_atom_m_projected_sparse);
        SpMat M_sqrt_n_atom_m_projected_sparse(sub_basis_size, sub_basis_size);
        M_sqrt_n_atom_m_projected_sparse.setFromTriplets(M_sqrt_n_atom_m_projected_triplets.begin(),
                                                    M_sqrt_n_atom_m_projected_triplets.end());
        sqrt_n_atom_operators_eigenbasis.push_back(M_sqrt_n_atom_m_projected_sparse);
    }

    const int num_subsystems = subsystem_eigenvalues.size();
    assert(num_subsystems == NAttached
            && "Number of attached subsystems is not unique!");
    std::vector<int> excitations;
    for (int m = 0; m < num_subsystems; ++m) {
        // Eigenvalue arrays always contain the zero-excitation
        // element, so the the maximum number of excitations is
        // the number of eigenvaues minus 1.
        const int max_excitation_m = subsystem_eigenvalues[m].size()-1;
        excitations.push_back(max_excitation_m);
    }
    const std::vector<BasisVector> basis
                = generate_tensor_product_basis(excitations);
    const std::unordered_map<BasisVector, int> basis_map = generate_basis_map(basis);
    //Single-excitation subspace basis for testing
    //std::vector<BasisVector> basis;
    //{
    //    std::vector<sigma_state_t> spec00 = {0, 0};
    //    basis.push_back(BasisVector(spec00));
    //    std::vector<sigma_state_t> spec10 = {1, 0};
    //    basis.push_back(BasisVector(spec10));
    //    std::vector<sigma_state_t> spec20 = {2, 0};
    //    basis.push_back(BasisVector(spec20));
    //    std::vector<sigma_state_t> spec01 = {0, 1};
    //    basis.push_back(BasisVector(spec01));
    //}
    const int basis_size = basis.size();
    const int64_t basis_size_squared = basis_size*basis_size;
    //std::cout << "Basis:" << std::endl;
    //for (int i = 0; i < basis_size; ++i) {
    //    std::cout << stringify_spec(basis[i].spec()) << std::endl;
    //}
    // Write the operators in the basis for the entire system
    // (consisting of the tensor product of all the subsystems)
    // The implicit assumption in the conversion loops below is
    // that the index "m" of the operator is the same as of the
    // corresponding subsystem.
    const int num_a_operators = a_operators_eigenbasis.size();
    assert(num_a_operators == NResonators
            && "Wrong number of a operators!");
    for (int m = 0; m < num_a_operators; ++m) {
        const SpMat M_a = rewrite_operator_in_basis(
                a_operators_eigenbasis[m], m, basis);
        ret.M_a.push_back(M_a);
        const SpMat M_a_adjoint = M_a.adjoint();
        ret.M_a_adjoint.push_back(M_a_adjoint);
        ret.M_O.push_back(M_a);
        ret.M_O_adjoint.push_back(M_a_adjoint);
    }
    const int num_b_operators = b_operators_eigenbasis.size();
    assert(num_b_operators == NAtoms
            && "Wrong number of b operators!");
    for (int m = 0; m < num_b_operators; ++m) {
        const SpMat M_b = rewrite_operator_in_basis(
                b_operators_eigenbasis[m], m, basis);
        ret.M_b.push_back(M_b);
        const SpMat M_b_adjoint = M_b.adjoint();
        ret.M_b_adjoint.push_back(M_b_adjoint);
        if (m >= NResonators) {
            ret.M_O.push_back(M_b);
            ret.M_O_adjoint.push_back(M_b_adjoint);
        }
    }
    const int num_n_res_operators = n_res_operators_eigenbasis.size();
    assert(num_n_res_operators == NResonators
            && "Wrong number of n_res operators!");
    for (int m = 0; m < num_n_res_operators; ++m) {
        const SpMat M_n_res = rewrite_operator_in_basis(
                n_res_operators_eigenbasis[m], m, basis);
        ret.M_n_res.push_back(M_n_res);
    }
    const int num_n_atom_operators = n_atom_operators_eigenbasis.size();
    assert(num_n_atom_operators == NAtoms
            && "Wrong number of n_atom operators!");
    for (int m = 0; m < num_n_atom_operators; ++m) {
        const SpMat M_n_atom = rewrite_operator_in_basis(
                n_atom_operators_eigenbasis[m], m, basis);
        ret.M_n_atom.push_back(M_n_atom);
    }
    std::vector<SpMat> M_sqrt_n_atom;
    const int num_sqrt_n_atom_operators = sqrt_n_atom_operators_eigenbasis.size();
    assert(num_sqrt_n_atom_operators == NAtoms
            && "Wrong number of sqrt_n_atom operators!");
    for (int m = 0; m < num_sqrt_n_atom_operators; ++m) {
        const SpMat M_sqrt_n_atom_m = rewrite_operator_in_basis(
                sqrt_n_atom_operators_eigenbasis[m], m, basis);
        M_sqrt_n_atom.push_back(M_sqrt_n_atom_m);
    }
    SpMat H(basis_size, basis_size);
    if (!(flags & JQF_SUPEROPERATOR_OMIT_DELTA_TERMS)) {
        for (int m = 0; m < NAttached; ++m) {
            const int num_eigenvalues_m = subsystem_eigenvalues[m].size();
            for (int j = 0; j < num_eigenvalues_m; ++j) {
                const Sigma s_jj_m(j, j, m);
                const SpMat M_jj_m = s_jj_m.matrix(basis);
                const double omega_diff = subsystem_eigenvalues[m][j]
                        -omega_d*subsystem_eigenvalue_shifts[m][j];
                H += omega_diff*M_jj_m;
            }
        }
    }
    if (J_x != 0) {
        for (int i = 0; i < NResonators-1; ++i) {
            SpMat temp = J_x*(ret.M_b_adjoint[i]*ret.M_b[i+1]+ret.M_b_adjoint[i+1]*ret.M_b[i]);
            H += temp;
        }
    }

    const double omega_1_2 = 0.5*(omega[0]+omega_r[0])+std::sqrt(std::pow(0.5*(omega_r[0]-omega[0]),2)+std::pow(g,2));
    const double omega_1_1 = 0.5*(omega[0]+omega_r[0])-std::sqrt(std::pow(0.5*(omega_r[0]-omega[0]),2)+std::pow(g,2));
    //std::cout << "omega_1_1 = " << omega_1_1 << std::endl;
    //std::cout << "omega_1_2 = " << omega_1_2 << std::endl;

    // Use the qubit frequency (omega_1_1) as the reference one
    // for the the wave vectors k.
    const double omega_ref = omega_1_1;
    // For the decay rates, we pick the decay rate of the first
    // resonator as the reference, and hence the reference frequency
    // is also the one of the resonator.
    double decay_factor_ref = kappa[0];
    const double omega_ref_for_decay = omega_r[0];
    if (decay_factor_ref == 0) {
        // Allow to turn off the coupling to the resonator.
        // (Prevent divide by zero in the code below.)
        // In this case, we assume that all the decay rates
        // are rescaled with respect to some reference.
        decay_factor_ref = 1;
    }
    const double k0x_val_ref_for_decay = k0x_vals[0];
    const double k_factor_d = omega_d/omega_ref;
    ret.Omega_factors = std::vector<double>(NAttached, 0);
    for (int m = 0; m < NAttached; ++m) {
#ifdef ABSOLUTE_FREQUENCY_DEPENDENT_CORRECTIONS
        ret.Omega_factors[m] = std::sqrt(omega_ref_for_decay*decay_factors[m]
                                         /(omega_attached[m]*decay_factor_ref))
                               *std::cos(k_factor_d*M_PI*k0x_vals[m])
                               /std::cos(k_factor_d*M_PI*k0x_val_ref_for_decay);
#else // ABSOLUTE_FREQUENCY_DEPENDENT_CORRECTIONS
        ret.Omega_factors[m] = std::sqrt(decay_factors[m])
                               *std::cos(M_PI*k0x_vals[m]);
#endif // ABSOLUTE_FREQUENCY_DEPENDENT_CORRECTIONS
    }

    if (!(flags & JQF_SUPEROPERATOR_OMIT_OMEGA_TERMS)) {
        const int M_O_size = ret.M_O.size();
        assert(M_O_size == NAttached && "Incorrect number of operators O!");
        for (int m = 0; m < M_O_size; ++m) {
            H += ret.Omega_factors[m]*(Omega*ret.M_O_adjoint[m]+std::conj(Omega)*ret.M_O[m]);
        }
    }

    ret.factor_r = 1;
    ret.M_r = SpMat(basis_size, basis_size);
    for (int m = 0; m < NAttached; ++m) {
        for (int j = 0; j < lowering_operators_eigenbasis[m].outerSize(); ++j) {
            for (SpMat::InnerIterator it(lowering_operators_eigenbasis[m],j); it; ++it) {
                const Sigma s_jk_m(it.row(), it.col(), m);
                const SpMat M_jk_m = s_jk_m.matrix(basis);
#ifdef ABSOLUTE_FREQUENCY_DEPENDENT_CORRECTIONS
#ifdef ALTERNATIVE_DRIVEN_EVOLUTION_APPROXIMATION
                const double omega_m_kj = subsystem_eigenvalues[m](it.col())
                                          -subsystem_eigenvalues[m](it.row());
                ret.M_r -= I*omega_m_kj/std::sqrt(omega_ref_for_decay*omega_attached[m])
                           *(std::sqrt(decay_factor_ref*decay_factors[m])/Omega)
                           *it.value()*M_jk_m*std::cos(k_factor_d*M_PI*k0x_vals[m]);
#else // ALTERNATIVE_DRIVEN_EVOLUTION_APPROXIMATION
                ret.M_r -= I*omega_d/std::sqrt(omega_ref_for_decay*omega_attached[m])
                           *(std::sqrt(decay_factor_ref*decay_factors[m])/Omega)
                           *it.value()*M_jk_m*std::cos(k_factor_d*M_PI*k0x_vals[m]);
#endif // ALTERNATIVE_DRIVEN_EVOLUTION_APPROXIMATION
#else // ABSOLUTE_FREQUENCY_DEPENDENT_CORRECTIONS
                ret.M_r -= I*(std::sqrt(decay_factor_ref*decay_factors[m])/Omega)
                           *it.value()*M_jk_m*std::cos(M_PI*k0x_vals[m]);
#endif // ABSOLUTE_FREQUENCY_DEPENDENT_CORRECTIONS
            }
        }
    }

    SpMat Identity(basis_size, basis_size);
    Identity.setIdentity();
#ifdef LESS_RAM_FOR_L
    std::vector<Eigen::Triplet<std::complex<double>>> L_triplets;
    {
        SpMat L_from_H = -I*(Eigen::kroneckerProduct(H, Identity)-Eigen::kroneckerProduct(Identity, H.transpose()));
        for (int j = 0; j < L_from_H.outerSize(); ++j) {
            for (SpMat::InnerIterator it(L_from_H,j); it; ++it) {
                L_triplets.emplace_back(it.row(), it.col(), it.value());
            }
        }
    }
#else // LESS_RAM_FOR_L
    ret.L = -I*(Eigen::kroneckerProduct(H, Identity)-Eigen::kroneckerProduct(Identity, H.transpose()));
#endif // LESS_RAM_FOR_L

    // When a higher-excitation eigenstate has lower energy than a
    // lower-excitation state, this will be set to true in the loops below.
    // It is unclear whether this is an issue, and is kept around as a reminder.
    bool replaced_negative_frequencies_with_positive = false;

    for (int m = 0; m < NAttached; ++m) {
        for (int n = 0; n < NAttached; ++n) {
            const double common_factor_mn = 0.5*std::sqrt(decay_factors[m]*decay_factors[n]);
#ifdef COMPUTE_AND_USE_O_MN
            std::vector<Eigen::Triplet<std::complex<double>>> O_mn_triplets;
            std::vector<Eigen::Triplet<std::complex<double>>> O_n_triplets;
            for (int j = 0; j < lowering_operators_eigenbasis[n].outerSize(); ++j) {
                for (SpMat::InnerIterator it(lowering_operators_eigenbasis[n],j); it; ++it) {
                    double omega_n_kj;
                    double k_factor_n;
                    if (drivenEvolution) {
#ifdef ALTERNATIVE_DRIVEN_EVOLUTION_APPROXIMATION
                        omega_n_kj = subsystem_eigenvalues[n](it.col())
                                     -subsystem_eigenvalues[n](it.row());
                        k_factor_n = omega_d/omega_ref;
                        if (omega_n_kj < 0) {
                            replaced_negative_frequencies_with_positive = true;
                        }
#else // ALTERNATIVE_DRIVEN_EVOLUTION_APPROXIMATION
                        omega_n_kj = omega_d;
                        k_factor_n = omega_d/omega_ref;
#endif // ALTERNATIVE_DRIVEN_EVOLUTION_APPROXIMATION
                    } else {
                        omega_n_kj = subsystem_eigenvalues[n](it.col())
                                     -subsystem_eigenvalues[n](it.row());
                        k_factor_n = omega_n_kj/omega_ref;
                    }
#ifdef ABSOLUTE_FREQUENCY_DEPENDENT_CORRECTIONS
                    std::complex<double> xi_mn_jk = common_factor_mn*omega_n_kj/std::sqrt(omega_attached[n]*omega_attached[m])*exp_factor(k_factor_n, k0x_vals[m], k0x_vals[n]);
#else // ABSOLUTE_FREQUENCY_DEPENDENT_CORRECTIONS
                    std::complex<double> xi_mn_jk = common_factor_mn*exp_factor(1, k0x_vals[m], k0x_vals[n]);
#endif // ABSOLUTE_FREQUENCY_DEPENDENT_CORRECTIONS
                    const Sigma s_n_jk(it.row(), it.col(), n);
                    std::vector<Eigen::Triplet<std::complex<double>>> triplets
                        = s_n_jk.triplets(basis);
                    const int num_triplets = triplets.size();
                    std::vector<Eigen::Triplet<std::complex<double>>> scaled_triplets;
                    scaled_triplets.reserve(num_triplets);
                    for (int l = 0; l < num_triplets; ++l) {
                        scaled_triplets.emplace_back(triplets[l].row(),
                                                     triplets[l].col(),
                                                     triplets[l].value()*it.value());
                    }
                    std::vector<Eigen::Triplet<std::complex<double>>> scaled_triplets_xi;
                    scaled_triplets_xi.reserve(num_triplets);
                    for (int l = 0; l < num_triplets; ++l) {
                        scaled_triplets_xi.emplace_back(triplets[l].row(),
                                                        triplets[l].col(),
                                                        triplets[l].value()*it.value()*xi_mn_jk);
                    }
                    O_mn_triplets.insert(O_mn_triplets.end(), scaled_triplets_xi.cbegin(), scaled_triplets_xi.cend());
                    O_n_triplets.insert(O_n_triplets.end(), scaled_triplets.cbegin(), scaled_triplets.cend());
                }
            }
            SpMat O_mn(basis_size, basis_size);
            O_mn.setFromTriplets(O_mn_triplets.begin(), O_mn_triplets.end());
            SpMat O_n(basis_size, basis_size);
            O_n.setFromTriplets(O_n_triplets.begin(), O_n_triplets.end());
            std::vector<Eigen::Triplet<std::complex<double>>> O_nm_adjoint_triplets;
            std::vector<Eigen::Triplet<std::complex<double>>> O_m_adjoint_triplets;
            for (int j = 0; j < lowering_operators_eigenbasis[m].outerSize(); ++j) {
                for (SpMat::InnerIterator it(lowering_operators_eigenbasis[m],j); it; ++it) {
                    double omega_m_kj;
                    double k_factor_m;
                    if (drivenEvolution) {
#ifdef ALTERNATIVE_DRIVEN_EVOLUTION_APPROXIMATION
                        omega_m_kj = subsystem_eigenvalues[m](it.col())
                                     -subsystem_eigenvalues[m](it.row());
                        k_factor_m = omega_d/omega_ref;
                        if (omega_m_kj < 0) {
                            replaced_negative_frequencies_with_positive = true;
                        }
#else // ALTERNATIVE_DRIVEN_EVOLUTION_APPROXIMATION
                        omega_m_kj = omega_d;
                        k_factor_m = omega_d/omega_ref;
#endif // ALTERNATIVE_DRIVEN_EVOLUTION_APPROXIMATION
                    } else {
                        omega_m_kj = subsystem_eigenvalues[m](it.col())
                                     -subsystem_eigenvalues[m](it.row());
                        k_factor_m = omega_m_kj/omega_ref;
                    }
                    //std::cout << "omega_" << m+1 << "_" << it.col() << it.row() << " = " << omega_m_kj << std::endl;
#ifdef ABSOLUTE_FREQUENCY_DEPENDENT_CORRECTIONS
                    //std::cout << "k_factor_m = " << k_factor_m << std::endl;
                    std::complex<double> xi_nm_jk = common_factor_mn*omega_m_kj/std::sqrt(omega_attached[n]*omega_attached[m])*exp_factor(k_factor_m, k0x_vals[m], k0x_vals[n]);
#else // ABSOLUTE_FREQUENCY_DEPENDENT_CORRECTIONS
                    std::complex<double> xi_nm_jk = common_factor_mn*exp_factor(1, k0x_vals[m], k0x_vals[n]);
#endif // ABSOLUTE_FREQUENCY_DEPENDENT_CORRECTIONS
                    //std::cout << "xi_" << n+1 << m+1 << "_" << it.col() << it.row() << " = " << xi_nm_jk << std::endl;
                    //std::cout << "C_" << m+1 << "_" << it.row() << it.col() << " = " << it.value() << std::endl;
                    const Sigma s_m_kj(it.col(), it.row(), m);
                    std::vector<Eigen::Triplet<std::complex<double>>> triplets
                        = s_m_kj.triplets(basis);
                    const int num_triplets = triplets.size();
                    std::vector<Eigen::Triplet<std::complex<double>>> scaled_triplets;
                    scaled_triplets.reserve(num_triplets);
                    for (int l = 0; l < num_triplets; ++l) {
                        scaled_triplets.emplace_back(triplets[l].row(),
                                                     triplets[l].col(),
                                                     std::conj(triplets[l].value()*it.value()));
                    }
                    std::vector<Eigen::Triplet<std::complex<double>>> scaled_triplets_xi;
                    scaled_triplets_xi.reserve(num_triplets);
                    for (int l = 0; l < num_triplets; ++l) {
                        scaled_triplets_xi.emplace_back(triplets[l].row(),
                                                        triplets[l].col(),
                                                        std::conj(triplets[l].value()*it.value()*xi_nm_jk));
                    }
                    O_nm_adjoint_triplets.insert(O_nm_adjoint_triplets.end(), scaled_triplets_xi.cbegin(), scaled_triplets_xi.cend());
                    O_m_adjoint_triplets.insert(O_m_adjoint_triplets.end(), scaled_triplets.cbegin(), scaled_triplets.cend());
                }
            }
            SpMat O_nm_adjoint(basis_size, basis_size);
            O_nm_adjoint.setFromTriplets(O_nm_adjoint_triplets.begin(), O_nm_adjoint_triplets.end());
            SpMat O_m_adjoint(basis_size, basis_size);
            O_m_adjoint.setFromTriplets(O_m_adjoint_triplets.begin(), O_m_adjoint_triplets.end());
#ifdef EXPENSIVE_CALCULATION_OF_O_MN_PRODUCTS
            std::vector<Eigen::Triplet<std::complex<double>>> O_m_adjoint_O_mn_triplets;
            std::vector<Eigen::Triplet<std::complex<double>>> O_nm_adjoint_O_n_triplets;
            for (int j1 = 0; j1 < lowering_operators_eigenbasis[n].outerSize(); ++j1) {
                for (SpMat::InnerIterator it1(lowering_operators_eigenbasis[n],j1); it1; ++it1) {
                    for (int j2 = 0; j2 < lowering_operators_eigenbasis[m].outerSize(); ++j2) {
                        for (SpMat::InnerIterator it2(lowering_operators_eigenbasis[m],j2); it2; ++it2) {
                            double omega_m_kj;
                            double omega_n_kj;
                            double k_factor_n;
                            double k_factor_m;
                            if (drivenEvolution) {
#ifdef ALTERNATIVE_DRIVEN_EVOLUTION_APPROXIMATION
                                omega_m_kj
                                    = subsystem_eigenvalues[m](it2.col())
                                      -subsystem_eigenvalues[m](it2.row());
                                omega_n_kj
                                    = subsystem_eigenvalues[n](it1.col())
                                      -subsystem_eigenvalues[n](it1.row());
                                k_factor_n = omega_d/omega_ref;
                                k_factor_m = omega_d/omega_ref;
                                if (omega_m_kj < 0 || omega_n_kj < 0) {
                                    replaced_negative_frequencies_with_positive = true;
                                }
#else // ALTERNATIVE_DRIVEN_EVOLUTION_APPROXIMATION
                                omega_m_kj = omega_d;
                                omega_n_kj = omega_d;
                                k_factor_n = omega_d/omega_ref;
                                k_factor_m = omega_d/omega_ref;
#endif // ALTERNATIVE_DRIVEN_EVOLUTION_APPROXIMATION
                            } else {
                                omega_m_kj
                                    = subsystem_eigenvalues[m](it2.col())
                                      -subsystem_eigenvalues[m](it2.row());
                                omega_n_kj
                                    = subsystem_eigenvalues[n](it1.col())
                                      -subsystem_eigenvalues[n](it1.row());
                                k_factor_n = omega_n_kj/omega_ref;
                                k_factor_m = omega_m_kj/omega_ref;
                            }
                            const double common_factor_mn = 0.5*std::sqrt(decay_factors[m]*decay_factors[n]);
#ifdef ABSOLUTE_FREQUENCY_DEPENDENT_CORRECTIONS
                            std::complex<double> xi_mn_jk = common_factor_mn*omega_n_kj/std::sqrt(omega_attached[n]*omega_attached[m])*exp_factor(k_factor_n, k0x_vals[m], k0x_vals[n]);
                            std::complex<double> xi_nm_jk = common_factor_mn*omega_m_kj/std::sqrt(omega_attached[n]*omega_attached[m])*exp_factor(k_factor_m, k0x_vals[n], k0x_vals[m]);
#else // ABSOLUTE_FREQUENCY_DEPENDENT_CORRECTIONS
                            std::complex<double> xi_mn_jk = common_factor_mn*exp_factor(1, k0x_vals[m], k0x_vals[n]);
                            std::complex<double> xi_nm_jk = common_factor_mn*exp_factor(1, k0x_vals[n], k0x_vals[m]);
#endif // ABSOLUTE_FREQUENCY_DEPENDENT_CORRECTIONS
                            const Sigma s_n_jk(it1.row(), it1.col(), n);
                            const Sigma s_m_kj(it2.col(), it2.row(), m);
                            const SigmaProduct prod_mn({s_m_kj, s_n_jk});
                            std::vector<Eigen::Triplet<std::complex<double>>> triplets
                                = prod_mn.triplets(basis);
                            const int num_triplets = triplets.size();
                            std::vector<Eigen::Triplet<std::complex<double>>> scaled_triplets_xi_mn;
                            scaled_triplets_xi_mn.reserve(num_triplets);
                            std::vector<Eigen::Triplet<std::complex<double>>> scaled_triplets_xi_nm;
                            scaled_triplets_xi_nm.reserve(num_triplets);
                            for (int l = 0; l < num_triplets; ++l) {
                                scaled_triplets_xi_mn.emplace_back(triplets[l].row(),
                                                                   triplets[l].col(),
                                                                   triplets[l].value()*it1.value()*std::conj(it2.value())*xi_mn_jk);
                                scaled_triplets_xi_nm.emplace_back(triplets[l].row(),
                                                                   triplets[l].col(),
                                                                   std::conj(triplets[l].value()*it1.value()*std::conj(it2.value())*xi_nm_jk));
                            }
                            O_m_adjoint_O_mn_triplets.insert(O_m_adjoint_O_mn_triplets.end(), scaled_triplets_xi_mn.cbegin(), scaled_triplets_xi_mn.cend());
                            O_nm_adjoint_O_n_triplets.insert(O_nm_adjoint_O_n_triplets.end(), scaled_triplets_xi_nm.cbegin(), scaled_triplets_xi_nm.cend());
                        }
                    }
                }
            }
            SpMat O_m_adjoint_O_mn(basis_size, basis_size);
            SpMat O_nm_adjoint_O_n(basis_size, basis_size);
            O_m_adjoint_O_mn.setFromTriplets(O_m_adjoint_O_mn_triplets.begin(), O_m_adjoint_O_mn_triplets.end());
            O_nm_adjoint_O_n.setFromTriplets(O_nm_adjoint_O_n_triplets.begin(), O_nm_adjoint_O_n_triplets.end());
#else // EXPENSIVE_CALCULATION_OF_O_MN_PRODUCTS
            const SpMat O_m_adjoint_O_mn = O_m_adjoint*O_mn;
            const SpMat O_nm_adjoint_O_n = O_nm_adjoint*O_n;
#endif // EXPENSIVE_CALCULATION_OF_O_MN_PRODUCTS
#ifdef LESS_RAM_FOR_L
            // Allocate a temporary and then destroy it (when exiting the
            // scope) to reduce the RAM usage.
            {
                const SpMat temp1 = Eigen::kroneckerProduct(O_m_adjoint_O_mn, Identity);
                for (int j = 0; j < temp1.outerSize(); ++j) {
                    for (SpMat::InnerIterator it(temp1,j); it; ++it) {
                        L_triplets.emplace_back(it.row(), it.col(), -0.5*it.value());
                    }
                }
                //ret.L -= 0.5*temp1;
            }
            {
                const SpMat temp2 = Eigen::kroneckerProduct(O_mn, O_m_adjoint.transpose());
                for (int j = 0; j < temp2.outerSize(); ++j) {
                    for (SpMat::InnerIterator it(temp2,j); it; ++it) {
                        L_triplets.emplace_back(it.row(), it.col(), 0.5*it.value());
                    }
                }
                //ret.L += 0.5*temp2;
            }
            {
                const SpMat temp3 = Eigen::kroneckerProduct(Identity, O_nm_adjoint_O_n.transpose());
                for (int j = 0; j < temp3.outerSize(); ++j) {
                    for (SpMat::InnerIterator it(temp3,j); it; ++it) {
                        L_triplets.emplace_back(it.row(), it.col(), -0.5*it.value());
                    }
                }
                //ret.L -= 0.5*temp3;
            }
            {
                const SpMat temp4 = Eigen::kroneckerProduct(O_n, O_nm_adjoint.transpose());
                for (int j = 0; j < temp4.outerSize(); ++j) {
                    for (SpMat::InnerIterator it(temp4,j); it; ++it) {
                        L_triplets.emplace_back(it.row(), it.col(), 0.5*it.value());
                    }
                }
                //ret.L += 0.5*temp4;
            }
#else // LESS_RAM_FOR_L
            const SpMat temp1 = Eigen::kroneckerProduct(O_m_adjoint_O_mn, Identity);
            const SpMat temp2 = Eigen::kroneckerProduct(O_mn, O_m_adjoint.transpose());
            const SpMat temp3 = Eigen::kroneckerProduct(Identity, O_nm_adjoint_O_n.transpose());
            const SpMat temp4 = Eigen::kroneckerProduct(O_n, O_nm_adjoint.transpose());
            ret.L -= 0.5*(temp1-temp2+temp3-temp4);
#endif // LESS_RAM_FOR_L
#else // COMPUTE_AND_USE_O_MN
            for (int j1 = 0; j1 < lowering_operators_eigenbasis[n].outerSize(); ++j1) {
                for (SpMat::InnerIterator it1(lowering_operators_eigenbasis[n],j1); it1; ++it1) {
                    for (int j2 = 0; j2 < lowering_operators_eigenbasis[m].outerSize(); ++j2) {
                        for (SpMat::InnerIterator it2(lowering_operators_eigenbasis[m],j2); it2; ++it2) {
                            double omega_m_kj;
                            double omega_n_kj;
                            double k_factor_n;
                            double k_factor_m;
                            if (drivenEvolution) {
#ifdef ALTERNATIVE_DRIVEN_EVOLUTION_APPROXIMATION
                                omega_m_kj
                                    = subsystem_eigenvalues[m](it2.col())
                                      -subsystem_eigenvalues[m](it2.row());
                                omega_n_kj
                                    = subsystem_eigenvalues[n](it1.col())
                                      -subsystem_eigenvalues[n](it1.row());
                                k_factor_n = omega_d/omega_ref;
                                k_factor_m = omega_d/omega_ref;
                                if (omega_m_kj < 0 || omega_n_kj < 0) {
                                    replaced_negative_frequencies_with_positive = true;
                                }
#else // ALTERNATIVE_DRIVEN_EVOLUTION_APPROXIMATION
                                omega_m_kj = omega_d;
                                omega_n_kj = omega_d;
                                k_factor_n = omega_d/omega_ref;
                                k_factor_m = omega_d/omega_ref;
#endif // ALTERNATIVE_DRIVEN_EVOLUTION_APPROXIMATION
                            } else {
                                omega_m_kj
                                    = subsystem_eigenvalues[m](it2.col())
                                      -subsystem_eigenvalues[m](it2.row());
                                omega_n_kj
                                    = subsystem_eigenvalues[n](it1.col())
                                      -subsystem_eigenvalues[n](it1.row());
                                k_factor_n = omega_n_kj/omega_ref;
                                k_factor_m = omega_m_kj/omega_ref;
                            }
                            const double common_factor_mn = 0.5*std::sqrt(decay_factors[m]*decay_factors[n]);
#ifdef ABSOLUTE_FREQUENCY_DEPENDENT_CORRECTIONS
                            std::complex<double> xi_mn_jk = common_factor_mn*omega_n_kj/std::sqrt(omega_attached[n]*omega_attached[m])*exp_factor(k_factor_n, k0x_vals[m], k0x_vals[n]);
                            std::complex<double> xi_nm_jk = common_factor_mn*omega_m_kj/std::sqrt(omega_attached[n]*omega_attached[m])*exp_factor(k_factor_m, k0x_vals[n], k0x_vals[m]);
#else // ABSOLUTE_FREQUENCY_DEPENDENT_CORRECTIONS
                            std::complex<double> xi_mn_jk = common_factor_mn*exp_factor(1, k0x_vals[m], k0x_vals[n]);
                            std::complex<double> xi_nm_jk = common_factor_mn*exp_factor(1, k0x_vals[n], k0x_vals[m]);
#endif // ABSOLUTE_FREQUENCY_DEPENDENT_CORRECTIONS
                            const Sigma s_jk_n(it1.row(), it1.col(), n);
                            const SpMat M_jk_n = s_jk_n.matrix(basis);
                            const Sigma s_kj_m(it2.col(), it2.row(), m);
                            const SpMat M_kj_m = s_kj_m.matrix(basis);
                            const SigmaProduct prod_mn({s_kj_m, s_jk_n});
                            const SpMat M_prod_mn = prod_mn.matrix(basis);
                            addLindbladTermsSeparateComplex(ret.L, xi_mn_jk*it1.value()*std::conj(it2.value()), xi_nm_jk*it1.value()*std::conj(it2.value()), M_jk_n, M_kj_m, M_prod_mn, Identity);
                            //std::cout << "Added for m = " << m << ", n = " << n << ", j1 = " << j1 << ", k1 = " << it1.col() << ", j2 = " << j2 << ", k2 = " << it2.col() << std::endl;
                        }
                    }
                }
            }
#endif // COMPUTE_AND_USE_O_MN
        }
    }
#ifdef LESS_RAM_FOR_L
    //TODO: Add kappaInternal etc.
    sortTripletsCSR(L_triplets, basis_size_squared);
    const bool swapManually = false;
    if (swapManually) {
        // Manual implementation of swap to disk. The expectation
        // is that L_triplets is so large that it and CSRMatrix
        // initialized from it cannot simultaneously fit into RAM.
        // Thus, write L_triplets to disk, clear the RAM storage,
        // and then use the memory-mapped disk copy to initialize
        // CSRMatrix.
        const int64_t numTriplets = L_triplets.size();
        std::string path = "triplets.dump";
        std::ofstream file(path);
        file.write((char *)(L_triplets.data()),
                   sizeof(Eigen::Triplet<std::complex<double>>)*numTriplets);
        file.close();
        L_triplets.clear();
        int fd = open(path.c_str(), O_RDONLY);
        if (fd == -1) {
            assert(0 && "Could not open triplets dump file!");
        }
        const int64_t length = lseek(fd, 0, SEEK_END);
        lseek(fd, 0, SEEK_SET);
        Eigen::Triplet<std::complex<double>> *triplets_ptr
            = (Eigen::Triplet<std::complex<double>> *)mmap(
                    NULL, length, PROT_READ, MAP_PRIVATE, fd, 0);
        CSRMatrixFromSortedTriplets(ret.L_csr, triplets_ptr,
                numTriplets, basis_size_squared, basis_size_squared);
        munmap(triplets_ptr, length);
        close(fd);
        unlink(path.c_str());
    } else {
        // This is a simpler option of initializing CSRMatrix from
        // the triplets directly if either both fit into RAM or if a
        // swap partition is available to deal with whatever spills
        // over from RAM during the initialization.
        CSRMatrixFromSortedTriplets(ret.L_csr, L_triplets.data(),
                L_triplets.size(), basis_size_squared, basis_size_squared);
        // The two lines below are for testing to compare that
        // CSRMatrix has the same elements as SpMat.
        //ret.L = SpMat(basis_size_squared, basis_size_squared);
        //ret.L.setFromTriplets(L_triplets.begin(), L_triplets.end());
        L_triplets.clear();
    }
#else // LESS_RAM_FOR_L
    for (int i = 0; i < numKappaInternal; ++i) {
        if (kappaInternal[i] == 0) {
            continue;
        }
        addLindbladTerms(ret.L, kappaInternal[i], ret.M_a[i], ret.M_a_adjoint[i], ret.M_n_res[i], Identity);
    }
    for (int i = 0; i < numGammaInternal; ++i) {
        if (gammaInternal[i] == 0) {
            continue;
        }
        addLindbladTerms(ret.L, gammaInternal[i], ret.M_b[i], ret.M_b_adjoint[i], ret.M_n_atom[i], Identity);
    }
    for (int i = 0; i < numGammaDephasing; ++i) {
        if (gammaDephasing[i] == 0) {
            continue;
        }
        addLindbladTerms(ret.L, gammaDephasing[i], M_sqrt_n_atom[i], M_sqrt_n_atom[i], ret.M_n_atom[i], Identity);
    }
    ret.L.makeCompressed();
#endif // LESS_RAM_FOR_L

    for (int m = 0; m < num_subsystems; ++m) {
        std::vector<SpMat> M_sigma_m(subsystem_eigenvalues[m].size());
        for (int j = 0; j < subsystem_eigenvalues[m].size(); ++j) {
            const Sigma s_m_jj(j, j, m);
            const SpMat M_s_m_jj = s_m_jj.matrix(basis);
            M_sigma_m[j] = M_s_m_jj;
        }
        ret.M_sigma.push_back(M_sigma_m);
    }
    for (int m = 0; m < NResonators; ++m) {
        const int psi_up_index = psi_up_eigenstate_indices[m];
        const int psi_down_index = psi_down_eigenstate_indices[m];
        const Sigma s_ee_m(psi_up_index, psi_up_index, m);
        const SpMat M_psi_up_m = s_ee_m.matrix(basis);
        const Sigma s_gg_m(psi_down_index, psi_down_index, m);
        const SpMat M_psi_down_m = s_gg_m.matrix(basis);
        std::vector<sigma_state_t> spec_up(basis[0].size(), 0);
        std::vector<sigma_state_t> spec_down(basis[0].size(), 0);
        spec_up[m] = psi_up_index;
        spec_down[m] = psi_down_index;
        BasisVector vec_up(spec_up);
        BasisVector vec_down(spec_down);
        Eigen::VectorXcd up_eigenstate = Eigen::VectorXcd::Zero(basis_size);
        for (int i = 0; i < basis_size; ++i) {
            if (basis[i] == vec_up) {
                up_eigenstate(i) = 1;
                break;
            }
        }
        Eigen::VectorXcd down_eigenstate = Eigen::VectorXcd::Zero(basis_size);
        for (int i = 0; i < basis_size; ++i) {
            if (basis[i] == vec_down) {
                down_eigenstate(i) = 1;
                break;
            }
        }
        ret.psi_up.push_back(up_eigenstate);
        ret.psi_down.push_back(down_eigenstate);
        ret.M_psi_up.push_back(M_psi_up_m);
        ret.M_psi_down.push_back(M_psi_down_m);
    }
    ret.basis = std::move(basis);
    ret.subsystem_eigenvalues = std::move(subsystem_eigenvalues);
    ret.subsystem_eigenvalue_shifts = std::move(subsystem_eigenvalue_shifts);
    ret.psi_up_eigenstate_indices = std::move(psi_up_eigenstate_indices);
    ret.psi_down_eigenstate_indices = std::move(psi_down_eigenstate_indices);
    ret.replaced_negative_frequencies_with_positive
        = replaced_negative_frequencies_with_positive;

    return ret;
}
