// Copyright (c) 2019-2022 Ivan Iakoupov
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom
// the Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
// OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
// USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "master_equation_roc.h"
#include "roc_util.h"

// Debug the misbehaving ROCSparseMatrixOnDevice::mul_mat_row_major()
// Leave the code here in case it reappears.
//#define DEBUG_SPMM_MUL

#ifdef DEBUG_SPMM_MUL
#include <iostream>
#include "Eigen/Dense"
#include "types.h"
#endif // DEBUG_SPMM_MUL

#define OPTIMIZED_CSRMMTRACE

MasterEquationData evolve_master_equation_roc(
        int basis_size, const SparseMatrixData &L,
        const std::complex<double> *rho0,
        const std::vector<SparseMatrixData> &M_operators,
        const std::vector<const double*> &M_diag_operators,
        const std::vector<const std::complex<double>*> &fidelity_rhos,
        double dt, int64_t N_t, int64_t iterationsBetweenDeviceSynchronize)
{
    MasterEquationData ret;
    const int basis_size_squared = basis_size*basis_size;

    ROCSparseHandleRAII handle;

#ifdef OPTIMIZED_CSRMMTRACE
    std::vector<rocsparse_double_complex> hones(basis_size, rocsparse_double_complex(1, 0));
#else // OPTIMIZED_CSRMMTRACE
    std::vector<rocsparse_double_complex> htrace_mask(basis_size_squared, 0);
    for (int i = 0; i < basis_size; ++i) {
        htrace_mask[i*basis_size+i] = rocsparse_double_complex(1, 0);
    }
#endif // OPTIMIZED_CSRMMTRACE


    ROCSparseMatrixOnDevice rocL(handle.handle(), L);
    const int num_M_operators = M_operators.size();
    std::vector<ROCSparseMatrixOnDevice> dM_operators;
    for (int j = 0; j < num_M_operators; ++j) {
        dM_operators.emplace_back(handle.handle(), M_operators[j]);
    }
    const int num_M_diag_operators = M_diag_operators.size();
    std::vector<ROCArrayXcdOnDevice> dM_diag_operators;
    for (int j = 0; j < num_M_diag_operators; ++j) {
        std::vector<std::complex<double>> cM_diag_operator(basis_size);
        for (int k = 0; k < basis_size; ++k) {
            cM_diag_operator[k] = M_diag_operators[j][k];
        }
        dM_diag_operators.emplace_back(cM_diag_operator.data(), basis_size);
    }
    const int num_fidelity_rhos = fidelity_rhos.size();
    std::vector<ROCArrayXcdOnDevice> dfidelity_rhos;
    for (int j = 0; j < num_fidelity_rhos; ++j) {
        dfidelity_rhos.emplace_back(fidelity_rhos[j], basis_size_squared);
    }

    const rocsparse_double_complex halpha_dt(dt, 0);
    const rocsparse_double_complex halpha_one(1, 0);
    const rocsparse_double_complex halpha_half(0.5, 0);
    const rocsparse_double_complex halpha_zero(0, 0);
    const rocsparse_double_complex hbeta_zero(0, 0);
    const rocsparse_double_complex hbeta_one(1, 0);

    ROCArrayXcdOnDevice drho0(rho0, basis_size_squared);
    ROCArrayXcdOnDevice dk1(basis_size_squared);
    ROCArrayXcdOnDevice dk2(basis_size_squared);
    ROCArrayXcdOnDevice dk3(basis_size_squared);
    ROCArrayXcdOnDevice dk4(basis_size_squared);
    ROCArrayXcdOnDevice dtemp(basis_size_squared);
#ifdef OPTIMIZED_CSRMMTRACE
    ROCArrayXcdOnDevice dones(basis_size);
#else // OPTIMIZED_CSRMMTRACE
    ROCArrayXcdOnDevice dtrace_mask(basis_size_squared);
#endif // OPTIMIZED_CSRMMTRACE
    std::vector<ROCArrayXcdOnDevice> dM_values;
    for (int j = 0; j < num_M_operators; ++j) {
        dM_values.emplace_back(N_t);
    }
    std::vector<ROCArrayXcdOnDevice> dM_diag_values;
    for (int j = 0; j < num_M_diag_operators; ++j) {
        dM_diag_values.emplace_back(N_t);
    }
    std::vector<ROCArrayXcdOnDevice> dfidelities;
    for (int j = 0; j < num_fidelity_rhos; ++j) {
        dfidelities.emplace_back(N_t);
    }

#ifdef OPTIMIZED_CSRMMTRACE
    hipMemcpy(dones.data(), hones.data(), sizeof(rocsparse_double_complex) * basis_size, hipMemcpyHostToDevice);
#else // OPTIMIZED_CSRMMTRACE
    hipMemcpy(dtrace_mask.data(), htrace_mask.data(), sizeof(rocsparse_double_complex) * basis_size_squared, hipMemcpyHostToDevice);
#endif // OPTIMIZED_CSRMMTRACE

    size_t dev_bytes = rocblas_reduction_kernel_workspace_size<ROCM_DOT_NB, rocsparse_double_complex>(basis_size_squared);
    ROCArrayXcdOnDevice mem(dev_bytes);

    ret.time = std::vector<double>(N_t, 0);
    for (int64_t i = 0; i < N_t; ++i) {
        ret.time[i] = dt*(i+1);
        for (int j = 0; j < num_M_operators; ++j) {
#ifdef OPTIMIZED_CSRMMTRACE
            dM_operators[j].mul_mat_for_trace(&halpha_one, drho0.data(), &hbeta_zero, dtemp.data());
            // TODO: We would like to just sum the elements of dtemp here
            //       (not the entire basis_size_squared, but only basis_size part).
            //       Finding it by taking the dot product with
            //       an array where each element is equal to 1 is
            //       inefficient.
            rocblas_zdotc(handle.handle(), basis_size, dones.data(), 1, dtemp.data(), 1, dM_values[j].data() + i, mem.data());
#else // OPTIMIZED_CSRMMTRACE
            // Calculate the entire matrix product even though we only need the diagonal for the trace
            dM_operators[j].mul_mat(&halpha_one, drho0.data(), &hbeta_zero, dtemp.data(), basis_size);

            //We calculate the trace by finding the dot product with a mask.
            rocblas_zdotc(handle.handle(), basis_size_squared, dtemp.data(), 1, dtrace_mask.data(), 1, dM_values[j].data() + i, mem.data());
#endif // OPTIMIZED_CSRMMTRACE
        }
        for (int j = 0; j < num_M_diag_operators; ++j) {
            // Because tr(M,rho) = sum_{i,j}M_ij*rho_ji = sum_{i}M_ii*rho_ii
            // (the last equality is because M is a diagonal matrix), we see
            // that we need only to find the dot product between the array that
            // stores M (dM_diag_operators[j]) and the diagonal of rho_ii.
            // This can be done by choosing the increment basis_size + 1 like
            // we do below. This will exactly only take the diagonal elements
            // of the density matrix.
            //TODO: dM_diag_operators are actually real arrays. We
            //      need a rocblas_zdotc that takes a real array
            //      for the first argument.
            rocblas_zdotc(handle.handle(), basis_size, dM_diag_operators[j].data(), 1, drho0.data(), basis_size + 1, dM_diag_values[j].data() + i, mem.data());
        }
        for (int j = 0; j < num_fidelity_rhos; ++j) {
            rocblas_zdotc(handle.handle(), basis_size_squared, dfidelity_rhos[j].data(), 1, drho0.data(), 1, dfidelities[j].data() + i, mem.data());
        }

        rocL.mul_vec(&halpha_dt, drho0.data(), &hbeta_zero, dk1.data());
        rocblas_zaxpy_store_new(handle.handle(), basis_size_squared, &halpha_half, dk1.data(), 1, drho0.data(), 1, dtemp.data(), 1);
        rocL.mul_vec(&halpha_dt, dtemp.data(), &hbeta_zero, dk2.data());
        rocblas_zaxpy_store_new(handle.handle(), basis_size_squared, &halpha_half, dk2.data(), 1, drho0.data(), 1, dtemp.data(), 1);
        rocL.mul_vec(&halpha_dt, dtemp.data(), &hbeta_zero, dk3.data());
        rocblas_zaxpy_store_new(handle.handle(), basis_size_squared, &halpha_one, dk3.data(), 1, drho0.data(), 1, dtemp.data(), 1);
        rocL.mul_vec(&halpha_dt, dtemp.data(), &hbeta_zero, dk4.data());
        //rk4_finalize_z implements
        //rho0 += (k1/6) + (k2/3) + (k3/3) + (k4/6);
        rk4_finalize_z(handle.handle(), basis_size_squared, drho0.data(), dk1.data(), dk2.data(), dk3.data(), dk4.data());

        // This is a workaround to prevent ROCm runtime from eating
        // all of RAM in certain situations. It seems that ROCm runtime
        // (at least as of version 4.0) does not have any limit to how
        // much memory it will use for queueing kernels. Hence, if this
        // loop has many iterations (i.e., N_t is large) many kernels
        // could be queued and cause the machine to run out of RAM. This
        // workaround calls "hipDeviceSynchronize" from time to time to
        // flush the kernel queue. If we do not call
        // "hipDeviceSynchronize" too often, then it should have
        // negligible impact.
        if (iterationsBetweenDeviceSynchronize > 0
                && i % iterationsBetweenDeviceSynchronize == 0) {
            hipDeviceSynchronize();
        }
    }
    hipDeviceSynchronize();

    std::vector<rocsparse_double_complex> temp_values(N_t, 0);
    ret.M_values.resize(num_M_operators);
    for (int j = 0; j < num_M_operators; ++j) {
        ret.M_values[j] = std::vector<std::complex<double>>(N_t, 0);
        hipMemcpy(temp_values.data(), dM_values[j].data(), sizeof(rocsparse_double_complex) * N_t, hipMemcpyDeviceToHost);
        for (int64_t i = 0; i < N_t; ++i) {
            const rocsparse_double_complex val = temp_values[i];
            ret.M_values[j][i] = std::complex<double>(std::real(val),std::imag(val));
        }
    }
    ret.M_diag_values.resize(num_M_diag_operators);
    std::vector<std::vector<rocsparse_double_complex>> M_diag_values(num_M_diag_operators);
    for (int j = 0; j < num_M_diag_operators; ++j) {
        ret.M_diag_values[j] = std::vector<std::complex<double>>(N_t, 0);
        hipMemcpy(temp_values.data(), dM_diag_values[j].data(), sizeof(rocsparse_double_complex) * N_t, hipMemcpyDeviceToHost);
        for (int64_t i = 0; i < N_t; ++i) {
            const rocsparse_double_complex val = temp_values[i];
            ret.M_diag_values[j][i] = std::complex<double>(std::real(val),std::imag(val));
        }
    }
    ret.fidelities.resize(num_fidelity_rhos);
    std::vector<std::vector<rocsparse_double_complex>> fidelities(num_fidelity_rhos);
    for (int j = 0; j < num_fidelity_rhos; ++j) {
        ret.fidelities[j] = std::vector<std::complex<double>>(N_t, 0);
        hipMemcpy(temp_values.data(), dfidelities[j].data(), sizeof(rocsparse_double_complex) * N_t, hipMemcpyDeviceToHost);
        for (int64_t i = 0; i < N_t; ++i) {
            const rocsparse_double_complex val = temp_values[i];
            ret.fidelities[j][i] = std::complex<double>(std::real(val),std::imag(val));
        }
    }
    return ret;
}

void mulMatAddAdjoint(rocsparse_handle handle, ROCArrayXcdOnDevice &v_dst, const ROCSparseMatrixOnDevice &dH_t, const ROCArrayXcdOnDevice &v_src, rocsparse_double_complex factor)
{
    const int basis_size = dH_t.rows();
    const rocsparse_double_complex hbeta_zero(0, 0);
    const rocsparse_double_complex hbeta_one(1, 0);
    dH_t.mul_mat_row_major(&factor, v_src.data(), &hbeta_zero, v_dst.data(), basis_size);
    addAdjointInPlace(handle, v_dst.data(), basis_size);
}

void mulMatAddAdjoint(rocsparse_handle handle, ROCArrayXcdOnDevice &v_dst, const ROCSparseMatrixOnDevice &dH_t_Re, const ROCSparseMatrixOnDevice &dH_t_Im, const ROCArrayXcdOnDevice &v_src, rocsparse_double_complex factor_Re, rocsparse_double_complex factor_Im)
{
    const int basis_size = dH_t_Re.rows();
    const rocsparse_double_complex hbeta_zero(0, 0);
    const rocsparse_double_complex hbeta_one(1, 0);
    dH_t_Re.mul_mat_row_major(&factor_Re, v_src.data(), &hbeta_zero, v_dst.data(), basis_size);
    dH_t_Im.mul_mat_row_major(&factor_Im, v_src.data(), &hbeta_one, v_dst.data(), basis_size);
    addAdjointInPlace(handle, v_dst.data(), basis_size);
}

void applyL_t(rocsparse_handle handle, ROCArrayXcdOnDevice &v_dst, const ROCSparseMatrixOnDevice &L, const ROCSparseMatrixOnDevice &H_t_Re, const ROCSparseMatrixOnDevice &H_t_Im, const ROCArrayXcdOnDevice &v_src, rocsparse_double_complex factor_Re, rocsparse_double_complex factor_Im, double dt)
{
    const rocsparse_double_complex halpha_dt(dt, 0);
    const rocsparse_double_complex hbeta_zero(0, 0);
    const rocsparse_double_complex hbeta_one(1, 0);
    mulMatAddAdjoint(handle, v_dst, H_t_Re, H_t_Im, v_src, factor_Re, factor_Im);
    L.mul_vec(&halpha_dt, v_src.data(), &hbeta_one, v_dst.data());
}

#ifdef DEBUG_SPMM_MUL
template <typename Matrix>
inline void addAdjointInPlace(Matrix &M)
{
    // Add the adjoint of the matrix M to itself
    const int rows = M.rows();
    const int cols = M.cols();
    for (int j = 0; j < rows; ++j) {
        M(j,j) = 2*M(j,j).real();
        for (int k = j+1; k < cols; ++k) {
            M(j,k) += std::conj(M(k,j));
            M(k,j) = std::conj(M(j,k));
        }
    }
}

inline void mulMatAddAdjoint(Eigen::VectorXcd &v_dst, const SpMat &Mfactor_Re, const SpMat &Mfactor_Im, const Eigen::VectorXcd &v_src, std::complex<double> factor_Re, std::complex<double> factor_Im)
{
    int size = Mfactor_Re.rows();
    Eigen::Map<const MatrixXcdRowMajor> v_src_map(v_src.data(), size, size);
    Eigen::Map<MatrixXcdRowMajor> v_dst_map(v_dst.data(), size, size);
    v_dst_map.noalias() = factor_Re*Mfactor_Re*v_src_map;
    v_dst_map.noalias() += factor_Im*Mfactor_Im*v_src_map;
    addAdjointInPlace(v_dst_map);
}

void smd_to_spmat(SpMat &M, const SparseMatrixData &smd)
{
    std::vector<Eigen::Triplet<std::complex<double>>> triplets;
    for(int i = 0; i < smd.rows; ++i) {
        for(int j = smd.ptr[i]; j < smd.ptr[i + 1]; ++j) {
            triplets.emplace_back(i, smd.col[j], smd.val[j]);
        }
    }
    M.setFromTriplets(triplets.begin(), triplets.end());
}
#endif // DEBUG_SPMM_MUL

MasterEquationData evolve_time_dependent_master_equation_roc(
        int basis_size, const SparseMatrixData &L,
        const SparseMatrixData &H_t_Re,
        const SparseMatrixData &H_t_Im,
        std::function<std::complex<double>(double)> H_t_factor,
        const std::complex<double> *rho0,
        const std::vector<SparseMatrixData> &M_operators,
        const std::vector<const double*> &M_diag_operators,
        const std::vector<const std::complex<double>*> &fidelity_rhos,
        double dt, int64_t N_t, int64_t iterationsBetweenDeviceSynchronize)
{
    MasterEquationData ret;
    const int basis_size_squared = basis_size*basis_size;

    ROCSparseHandleRAII handle;

#ifdef OPTIMIZED_CSRMMTRACE
    std::vector<rocsparse_double_complex> hones(basis_size, rocsparse_double_complex(1, 0));
#else // OPTIMIZED_CSRMMTRACE
    std::vector<rocsparse_double_complex> htrace_mask(basis_size_squared, 0);
    for (int i = 0; i < basis_size; ++i) {
        htrace_mask[i*basis_size+i] = rocsparse_double_complex(1, 0);
    }
#endif // OPTIMIZED_CSRMMTRACE


    ROCSparseMatrixOnDevice rocL(handle.handle(), L);
    ROCSparseMatrixOnDevice dH_t_Re(handle.handle(), H_t_Re);
    ROCSparseMatrixOnDevice dH_t_Im(handle.handle(), H_t_Im);
    const int num_M_operators = M_operators.size();
    std::vector<ROCSparseMatrixOnDevice> dM_operators;
    for (int j = 0; j < num_M_operators; ++j) {
        dM_operators.emplace_back(handle.handle(), M_operators[j]);
    }
    const int num_M_diag_operators = M_diag_operators.size();
    std::vector<ROCArrayXcdOnDevice> dM_diag_operators;
    for (int j = 0; j < num_M_diag_operators; ++j) {
        std::vector<std::complex<double>> cM_diag_operator(basis_size);
        for (int k = 0; k < basis_size; ++k) {
            cM_diag_operator[k] = M_diag_operators[j][k];
        }
        dM_diag_operators.emplace_back(cM_diag_operator.data(), basis_size);
    }
    const int num_fidelity_rhos = fidelity_rhos.size();
    std::vector<ROCArrayXcdOnDevice> dfidelity_rhos;
    for (int j = 0; j < num_fidelity_rhos; ++j) {
        dfidelity_rhos.emplace_back(fidelity_rhos[j], basis_size_squared);
    }

    const rocsparse_double_complex halpha_dt(dt, 0);
    const rocsparse_double_complex halpha_one(1, 0);
    const rocsparse_double_complex halpha_half(0.5, 0);
    const rocsparse_double_complex halpha_zero(0, 0);
    const rocsparse_double_complex hbeta_zero(0, 0);
    const rocsparse_double_complex hbeta_one(1, 0);

#ifdef DEBUG_SPMM_MUL
    std::vector<rocsparse_double_complex> htemp(basis_size_squared);
    std::vector<rocsparse_double_complex> hrho(basis_size_squared);
    std::vector<std::complex<double>> htemp_std(basis_size_squared);
    Eigen::VectorXcd hrho_std(basis_size_squared);
    Eigen::VectorXcd etemp(basis_size_squared);

    SpMat hH_t_Re(basis_size, basis_size);
    SpMat hH_t_Im(basis_size, basis_size);
    smd_to_spmat(hH_t_Re, H_t_Re);
    smd_to_spmat(hH_t_Im, H_t_Im);
#endif // DEBUG_SPMM_MUL

    ROCArrayXcdOnDevice drho0(rho0, basis_size_squared);
    ROCArrayXcdOnDevice dk1(basis_size_squared);
    ROCArrayXcdOnDevice dk2(basis_size_squared);
    ROCArrayXcdOnDevice dk3(basis_size_squared);
    ROCArrayXcdOnDevice dk4(basis_size_squared);
    ROCArrayXcdOnDevice dtemp(basis_size_squared);
    ROCArrayXcdOnDevice dtemp2(basis_size_squared);
#ifdef OPTIMIZED_CSRMMTRACE
    ROCArrayXcdOnDevice dones(basis_size);
#else // OPTIMIZED_CSRMMTRACE
    ROCArrayXcdOnDevice dtrace_mask(basis_size_squared);
#endif // OPTIMIZED_CSRMMTRACE
    std::vector<ROCArrayXcdOnDevice> dM_values;
    for (int j = 0; j < num_M_operators; ++j) {
        dM_values.emplace_back(N_t);
    }
    std::vector<ROCArrayXcdOnDevice> dM_diag_values;
    for (int j = 0; j < num_M_diag_operators; ++j) {
        dM_diag_values.emplace_back(N_t);
    }
    std::vector<ROCArrayXcdOnDevice> dfidelities;
    for (int j = 0; j < num_fidelity_rhos; ++j) {
        dfidelities.emplace_back(N_t);
    }

#ifdef OPTIMIZED_CSRMMTRACE
    hipMemcpy(dones.data(), hones.data(), sizeof(rocsparse_double_complex) * basis_size, hipMemcpyHostToDevice);
#else // OPTIMIZED_CSRMMTRACE
    hipMemcpy(dtrace_mask.data(), htrace_mask.data(), sizeof(rocsparse_double_complex) * basis_size_squared, hipMemcpyHostToDevice);
#endif // OPTIMIZED_CSRMMTRACE

    size_t dev_bytes = rocblas_reduction_kernel_workspace_size<ROCM_DOT_NB, rocsparse_double_complex>(basis_size_squared);
    ROCArrayXcdOnDevice mem(dev_bytes);

    ret.time = std::vector<double>(N_t, 0);
    ret.M_values.resize(num_M_operators);
    for (int64_t i = 0; i < N_t; ++i) {
        const std::complex<double> H_t_factor1 = H_t_factor(dt*i);
        const std::complex<double> H_t_factor2 = H_t_factor(dt*(i+0.5));
        const std::complex<double> H_t_factor3 = H_t_factor(dt*(i+1));
        const rocsparse_double_complex I(0, 1);
        const rocsparse_double_complex factor1_Re = -I*H_t_factor1.real()*dt;
        const rocsparse_double_complex factor1_Im = -I*H_t_factor1.imag()*dt;
        const rocsparse_double_complex factor2_Re = -I*H_t_factor2.real()*dt;
        const rocsparse_double_complex factor2_Im = -I*H_t_factor2.imag()*dt;
        const rocsparse_double_complex factor3_Re = -I*H_t_factor3.real()*dt;
        const rocsparse_double_complex factor3_Im = -I*H_t_factor3.imag()*dt;

        applyL_t(handle.handle(), dk1, rocL, dH_t_Re, dH_t_Im, drho0, factor1_Re, factor1_Im, dt);
#ifdef DEBUG_SPMM_MUL
        // When debugging ROCSparseMatrixOnDevice::mul_mat_row_major,
        // we ignore the incoherent part and only use mulMatAddAdjoint()
        // that calls mul_mat_row_major().

        // This is the GPU multiplication
        mulMatAddAdjoint(handle.handle(), dtemp, dH_t_Re, dH_t_Im, drho0, factor1_Re, factor1_Im);
        hipDeviceSynchronize();
        hipMemcpy(htemp.data(), dtemp.data(), sizeof(rocsparse_double_complex) * basis_size_squared, hipMemcpyDeviceToHost);
        // Also copy the rho matrix data to be able to
        // multipy the Hamiltonian onto it on the host.
        hipMemcpy(hrho.data(), drho0.data(), sizeof(rocsparse_double_complex) * basis_size_squared, hipMemcpyDeviceToHost);
        for (int64_t j = 0; j < basis_size_squared; ++j) {
            const rocsparse_double_complex val = hrho[j];
            hrho_std(j) = std::complex<double>(std::real(val),std::imag(val));
        }
        const std::complex<double> factor1_Re_std(std::real(factor1_Re), std::imag(factor1_Re));
        const std::complex<double> factor1_Im_std(std::real(factor1_Im), std::imag(factor1_Im));
        // This is the CPU multiplication
        mulMatAddAdjoint(etemp, hH_t_Re, hH_t_Im, hrho_std, factor1_Re_std, factor1_Im_std);
        for (int64_t j = 0; j < basis_size_squared; ++j) {
            const rocsparse_double_complex val = htemp[j];
            const std::complex<double> val_std(std::real(val),std::imag(val));
            const int64_t row = j / basis_size;
            const int64_t col = j % basis_size;
            std::cout << "row = " << row << ", col = " << col << ", GPU value = " << val << ", CPU value = " << etemp(j) << std::endl;
        }
#endif // DEBUG_SPMM_MUL

        rocblas_zaxpy_store_new(handle.handle(), basis_size_squared, &halpha_half, dk1.data(), 1, drho0.data(), 1, dtemp.data(), 1);

        applyL_t(handle.handle(), dk2, rocL, dH_t_Re, dH_t_Im, dtemp, factor2_Re, factor2_Im, dt);

        rocblas_zaxpy_store_new(handle.handle(), basis_size_squared, &halpha_half, dk2.data(), 1, drho0.data(), 1, dtemp.data(), 1);

        applyL_t(handle.handle(), dk3, rocL, dH_t_Re, dH_t_Im, dtemp, factor2_Re, factor2_Im, dt);

        rocblas_zaxpy_store_new(handle.handle(), basis_size_squared, &halpha_one, dk3.data(), 1, drho0.data(), 1, dtemp.data(), 1);

        applyL_t(handle.handle(), dk4, rocL, dH_t_Re, dH_t_Im, dtemp, factor3_Re, factor3_Im, dt);

        //rk4_finalize_z implements
        //rho0 += (k1/6) + (k2/3) + (k3/3) + (k4/6);
        rk4_finalize_z(handle.handle(), basis_size_squared, drho0.data(), dk1.data(), dk2.data(), dk3.data(), dk4.data());

        ret.time[i] = dt*(i+1);
        for (int j = 0; j < num_M_operators; ++j) {
#ifdef OPTIMIZED_CSRMMTRACE
            dM_operators[j].mul_mat_for_trace(&halpha_one, drho0.data(), &hbeta_zero, dtemp.data());
            // TODO: We would like to just sum the elements of dtemp here
            //       (not the entire basis_size_squared, but only basis_size part).
            //       Finding it by taking the dot product with
            //       an array where each element is equal to 1 is
            //       inefficient.
            rocblas_zdotc(handle.handle(), basis_size, dones.data(), 1, dtemp.data(), 1, dM_values[j].data() + i, mem.data());
#else // OPTIMIZED_CSRMMTRACE
            // Calculate the entire matrix product even though we only need the diagonal for the trace
            dM_operators[j].mul_mat(&halpha_one, drho0.data(), &hbeta_zero, dtemp.data(), basis_size);

            //We calculate the trace by finding the dot product with a mask.
            rocblas_zdotc(handle.handle(), basis_size_squared, dtemp.data(), 1, dtrace_mask.data(), 1, dM_values[j].data() + i, mem.data());
#endif // OPTIMIZED_CSRMMTRACE
        }
        for (int j = 0; j < num_M_diag_operators; ++j) {
            // Because tr(M,rho) = sum_{i,j}M_ij*rho_ji = sum_{i}M_ii*rho_ii
            // (the last equality is because M is a diagonal matrix), we see
            // that we need only to find the dot product between the array that
            // stores M (dM_diag_operators[j]) and the diagonal of rho_ii.
            // This can be done by choosing the increment basis_size + 1 like
            // we do below. This will exactly only take the diagonal elements
            // of the density matrix.
            //TODO: dM_diag_operators are actually real arrays. We
            //      need a rocblas_zdotc that takes a real array
            //      for the first argument.
            rocblas_zdotc(handle.handle(), basis_size, dM_diag_operators[j].data(), 1, drho0.data(), basis_size + 1, dM_diag_values[j].data() + i, mem.data());
        }
        for (int j = 0; j < num_fidelity_rhos; ++j) {
            rocblas_zdotc(handle.handle(), basis_size_squared, dfidelity_rhos[j].data(), 1, drho0.data(), 1, dfidelities[j].data() + i, mem.data());
        }

        // This is a workaround to prevent ROCm runtime from eating
        // all of RAM in certain situations. It seems that ROCm runtime
        // (at least as of version 4.0) does not have any limit to how
        // much memory it will use for queueing kernels. Hence, if this
        // loop has many iterations (i.e., N_t is large) many kernels
        // could be queued and cause the machine to run out of RAM. This
        // workaround calls "hipDeviceSynchronize" from time to time to
        // flush the kernel queue. If we do not call
        // "hipDeviceSynchronize" too often, then it should have
        // negligible impact.
        if (iterationsBetweenDeviceSynchronize > 0
                && i % iterationsBetweenDeviceSynchronize == 0) {
            hipDeviceSynchronize();
        }
    }
    hipDeviceSynchronize();

    std::vector<rocsparse_double_complex> temp_values(N_t, 0);
    ret.M_values.resize(num_M_operators);
    for (int j = 0; j < num_M_operators; ++j) {
        ret.M_values[j] = std::vector<std::complex<double>>(N_t, 0);
        hipMemcpy(temp_values.data(), dM_values[j].data(), sizeof(rocsparse_double_complex) * N_t, hipMemcpyDeviceToHost);
        for (int64_t i = 0; i < N_t; ++i) {
            const rocsparse_double_complex val = temp_values[i];
            ret.M_values[j][i] = std::complex<double>(std::real(val),std::imag(val));
        }
    }
    ret.M_diag_values.resize(num_M_diag_operators);
    std::vector<std::vector<rocsparse_double_complex>> M_diag_values(num_M_diag_operators);
    for (int j = 0; j < num_M_diag_operators; ++j) {
        ret.M_diag_values[j] = std::vector<std::complex<double>>(N_t, 0);
        hipMemcpy(temp_values.data(), dM_diag_values[j].data(), sizeof(rocsparse_double_complex) * N_t, hipMemcpyDeviceToHost);
        for (int64_t i = 0; i < N_t; ++i) {
            const rocsparse_double_complex val = temp_values[i];
            ret.M_diag_values[j][i] = std::complex<double>(std::real(val),std::imag(val));
        }
    }
    ret.fidelities.resize(num_fidelity_rhos);
    std::vector<std::vector<rocsparse_double_complex>> fidelities(num_fidelity_rhos);
    for (int j = 0; j < num_fidelity_rhos; ++j) {
        ret.fidelities[j] = std::vector<std::complex<double>>(N_t, 0);
        hipMemcpy(temp_values.data(), dfidelities[j].data(), sizeof(rocsparse_double_complex) * N_t, hipMemcpyDeviceToHost);
        for (int64_t i = 0; i < N_t; ++i) {
            const rocsparse_double_complex val = temp_values[i];
            ret.fidelities[j][i] = std::complex<double>(std::real(val),std::imag(val));
        }
    }
    return ret;
}

namespace {
inline double G(double x, int64_t N_t, double sigma_t)
{
    const double L = N_t + 1;
    return std::exp(-std::pow((x-0.5*double(N_t))/(2*L*sigma_t),2));
}

inline double confinedGaussian(double n, int64_t N_t, double t_f, double sigma)
{
    // Approximate confined Gaussian window from
    // [Starosielec, S.; Hägele, D. (2014), "Discrete-time windows with
    // minimal RMS bandwidth for given RMS temporal width",
    // Signal Processing. 102: 240–246.]
    const double L = N_t + 1;
    return G(n, N_t, sigma)-(G(-0.5, N_t, sigma)*(G(n+L, N_t, sigma)+G(n-L, N_t, sigma)))/(G(-0.5+L, N_t, sigma)+G(-0.5-L, N_t, sigma));
}
} // anonymous namespace

#define ADRK4_FILTER_BASIS_FUNCTIONS             (1 << 0)
#define ADRK4_CLAMP_OMEGA                        (1 << 1)
#define ADRK4_PRECOMPUTE_BASIS_FUNCTION_VALUES   (1 << 2)

void adrk4_master_equation_roc(
        int N_amplitudes, const double *x, double *grad,
        double Omega_max, double t_final,
        double sigmaFilter, double sigmaWindow,
        const double *f1, const double *f2,
        std::function<void(double, double, double)> addToGradient,
        std::complex<double> *trace_final,
        int basis_size, const SparseMatrixData &L,
        const SparseMatrixData &L_adjoint,
        const SparseMatrixData &H_t_Re,
        const SparseMatrixData &H_t_Im,
        std::function<std::complex<double>(double)> calculateOmega,
        const std::complex<double> *rho_initial_vec,
        const std::complex<double> *rho_target_vec,
        double dt, int64_t N_t, int64_t iterationsBetweenDeviceSynchronize,
        int64_t cacheSizeTimeSteps, int64_t cacheSizeTimeStepsGPU,
        int64_t timeStepsPerStoredState)
{
    const int basis_size_squared = basis_size*basis_size;
    const rocsparse_double_complex I(0, 1);

    int adrk4_flags = 0;
    if (Omega_max > 0) {
        adrk4_flags |= ADRK4_CLAMP_OMEGA;
    }
    if (sigmaFilter > 0 && sigmaWindow > 0) {
        adrk4_flags |= ADRK4_FILTER_BASIS_FUNCTIONS;
    }
    if (f1 != nullptr && f2 != nullptr) {
        adrk4_flags |= ADRK4_PRECOMPUTE_BASIS_FUNCTION_VALUES;
    }
    auto calculateOmega1 = [&](int64_t i) -> std::complex<double>
    {
        double Omega_Re = 0;
        double Omega_Im = 0;
        for (int k = 0; k < N_amplitudes; ++k) {
            // f1 is assumed to be a column-major matrix
            const int64_t index = k+N_amplitudes*i;
            Omega_Re += x[k]*f1[index];
            Omega_Im += x[k+N_amplitudes]*f1[index];
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
            // f2 is assumed to be a column-major matrix
            const int64_t index = k+N_amplitudes*i;
            Omega_Re += x[k]*f2[index];
            Omega_Im += x[k+N_amplitudes]*f2[index];
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
            // f1 is assumed to be a column-major matrix
            const int64_t index = k+N_amplitudes*(i+1);
            Omega_Re += x[k]*f1[index];
            Omega_Im += x[k+N_amplitudes]*f1[index];
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
            // f1 is assumed to be a column-major matrix
            const int64_t index = k+N_amplitudes*i;
            // Factor of -1 is needed because we return -trace_final.real() below
            grad[k] += -factor_Re*f1[index];
            grad[k+N_amplitudes] += -factor_Im*f1[index];
        }
    };
    auto addToGradient2 = [&](int64_t i, double factor_Re, double factor_Im) -> void
    {
        for (int k = 0; k < N_amplitudes; ++k) {
            // f2 is assumed to be a column-major matrix
            const int64_t index = k+N_amplitudes*i;
            // Factor of -1 is needed because we return -trace_final.real() below
            grad[k] += -factor_Re*f2[index];
            grad[k+N_amplitudes] += -factor_Im*f2[index];
        }
    };
    auto addToGradient3 = [&](int64_t i, double factor_Re, double factor_Im) -> void
    {
        for (int k = 0; k < N_amplitudes; ++k) {
            // f1 is assumed to be a column-major matrix
            const int64_t index = k+N_amplitudes*(i+1);
            // Factor of -1 is needed because we return -trace_final.real() below
            grad[k] += -factor_Re*f1[index];
            grad[k+N_amplitudes] += -factor_Im*f1[index];
        }
    };

    ROCSparseHandleRAII handle;

#ifdef OPTIMIZED_CSRMMTRACE
    std::vector<rocsparse_double_complex> hones(basis_size, rocsparse_double_complex(1, 0));
#else // OPTIMIZED_CSRMMTRACE
    std::vector<rocsparse_double_complex> htrace_mask(basis_size_squared, 0);
    for (int i = 0; i < basis_size; ++i) {
        htrace_mask[i*basis_size+i] = rocsparse_double_complex(1, 0);
    }
#endif // OPTIMIZED_CSRMMTRACE


    ROCSparseMatrixOnDevice rocL(handle.handle(), L);
    ROCSparseMatrixOnDevice rocL_adjoint(handle.handle(), L_adjoint);
    ROCSparseMatrixOnDevice dH_t_Re(handle.handle(), H_t_Re);
    ROCSparseMatrixOnDevice dH_t_Im(handle.handle(), H_t_Im);

    const rocsparse_double_complex halpha_dt(dt, 0);
    const rocsparse_double_complex halpha_one(1, 0);
    const rocsparse_double_complex halpha_half(0.5, 0);
    const rocsparse_double_complex halpha_zero(0, 0);
    const rocsparse_double_complex hbeta_zero(0, 0);
    const rocsparse_double_complex hbeta_one(1, 0);

    ROCArrayXcdOnDevice drho0(rho_initial_vec, basis_size_squared);
    std::vector<rocsparse_double_complex> hrho0(basis_size_squared);
    std::vector<rocsparse_double_complex> hchi(basis_size_squared);
    ROCArrayXcdOnDevice drho_target(rho_target_vec, basis_size_squared);
    ROCArrayXcdOnDevice dk0(basis_size_squared);
    std::vector<rocsparse_double_complex> hk0(basis_size_squared);
    ROCArrayXcdOnDevice dk1(basis_size_squared);
    ROCArrayXcdOnDevice dk2(basis_size_squared);
    ROCArrayXcdOnDevice dk3(basis_size_squared);
    ROCArrayXcdOnDevice dk4(basis_size_squared);
    ROCArrayXcdOnDevice dk5(basis_size_squared);
    ROCArrayXcdOnDevice dk6(basis_size_squared);
    ROCArrayXcdOnDevice dk7(basis_size_squared);
    ROCArrayXcdOnDevice dk8(basis_size_squared);
    ROCArrayXcdOnDevice dk9(basis_size_squared);
    ROCArrayXcdOnDevice dtemp(basis_size_squared);
    ROCArrayXcdOnDevice danother_temp(basis_size_squared);
    ROCArrayXcdOnDevice dtrace_temp(6);
    std::vector<rocsparse_double_complex> htrace_temp(6);

    size_t dev_bytes = rocblas_reduction_kernel_workspace_size<ROCM_DOT_NB, rocsparse_double_complex>(basis_size_squared);
    ROCArrayXcdOnDevice mem(dev_bytes);

    int rho_cache_gpu_block = 0;
    int64_t currentCacheSizeTimeStepsGPU = 0;
    std::vector<rocsparse_double_complex> rho_vec_cached_t(basis_size_squared*cacheSizeTimeSteps, 0);
    ROCArrayXcdOnDevice drho_vec_cached_t(basis_size_squared*cacheSizeTimeStepsGPU);
    auto start_forward = std::chrono::steady_clock::now();
    for (int64_t i = 0; i < N_t; ++i) {
        std::complex<double> H_t_factor1;
        std::complex<double> H_t_factor2;
        std::complex<double> H_t_factor3;
        if (adrk4_flags & ADRK4_PRECOMPUTE_BASIS_FUNCTION_VALUES) {
            H_t_factor1 = calculateOmega1(i);
            H_t_factor2 = calculateOmega2(i);
            H_t_factor3 = calculateOmega3(i);
        } else {
            H_t_factor1 = calculateOmega(dt*i);
            H_t_factor2 = calculateOmega(dt*(i+0.5));
            H_t_factor3 = calculateOmega(dt*(i+1));
        }
        std::complex<double> Omega1 = H_t_factor1;
        std::complex<double> Omega2 = H_t_factor2;
        std::complex<double> Omega3 = H_t_factor3;
        if (adrk4_flags & ADRK4_CLAMP_OMEGA) {
            Omega1 *= Omega_max;
            Omega2 *= Omega_max;
            Omega3 *= Omega_max;
        }
        const rocsparse_double_complex factor1_Re = -I*Omega1.real()*dt;
        const rocsparse_double_complex factor1_Im = -I*Omega1.imag()*dt;
        const rocsparse_double_complex factor2_Re = -I*Omega2.real()*dt;
        const rocsparse_double_complex factor2_Im = -I*Omega2.imag()*dt;
        const rocsparse_double_complex factor3_Re = -I*Omega3.real()*dt;
        const rocsparse_double_complex factor3_Im = -I*Omega3.imag()*dt;

        applyL_t(handle.handle(), dk1, rocL, dH_t_Re, dH_t_Im, drho0, factor1_Re, factor1_Im, dt);

        rocblas_zaxpy_store_new(handle.handle(), basis_size_squared, &halpha_half, dk1.data(), 1, drho0.data(), 1, dtemp.data(), 1);

        applyL_t(handle.handle(), dk2, rocL, dH_t_Re, dH_t_Im, dtemp, factor2_Re, factor2_Im, dt);

        rocblas_zaxpy_store_new(handle.handle(), basis_size_squared, &halpha_half, dk2.data(), 1, drho0.data(), 1, dtemp.data(), 1);

        applyL_t(handle.handle(), dk3, rocL, dH_t_Re, dH_t_Im, dtemp, factor2_Re, factor2_Im, dt);

        rocblas_zaxpy_store_new(handle.handle(), basis_size_squared, &halpha_one, dk3.data(), 1, drho0.data(), 1, dtemp.data(), 1);

        applyL_t(handle.handle(), dk4, rocL, dH_t_Re, dH_t_Im, dtemp, factor3_Re, factor3_Im, dt);

        //rk4_finalize_z implements
        //rho0 += (k1/6) + (k2/3) + (k3/3) + (k4/6);
        if (grad != nullptr && i % timeStepsPerStoredState == 0) {
            const int64_t offset = i/timeStepsPerStoredState - rho_cache_gpu_block*cacheSizeTimeStepsGPU;
            rk4_finalize_cache_z(handle.handle(), basis_size_squared, drho0.data(), drho_vec_cached_t.data() + offset*basis_size_squared, dk1.data(), dk2.data(), dk3.data(), dk4.data());
            ++currentCacheSizeTimeStepsGPU;
            if (currentCacheSizeTimeStepsGPU == cacheSizeTimeStepsGPU) {
                hipMemcpy(rho_vec_cached_t.data() + rho_cache_gpu_block*cacheSizeTimeStepsGPU*basis_size_squared, drho_vec_cached_t.data(), sizeof(rocsparse_double_complex) * cacheSizeTimeStepsGPU * basis_size_squared, hipMemcpyDeviceToHost);
                ++rho_cache_gpu_block;
                currentCacheSizeTimeStepsGPU = 0;
            }
        } else {
            rk4_finalize_z(handle.handle(), basis_size_squared, drho0.data(), dk1.data(), dk2.data(), dk3.data(), dk4.data());
        }

        // This is a workaround to prevent ROCm runtime from eating
        // all of RAM in certain situations. It seems that ROCm runtime
        // (at least as of version 4.0) does not have any limit to how
        // much memory it will use for queueing kernels. Hence, if this
        // loop has many iterations (i.e., N_t is large) many kernels
        // could be queued and cause the machine to run out of RAM. This
        // workaround calls "hipDeviceSynchronize" from time to time to
        // flush the kernel queue. If we do not call
        // "hipDeviceSynchronize" too often, then it should have
        // negligible impact.
        if (iterationsBetweenDeviceSynchronize > 0
                && i % iterationsBetweenDeviceSynchronize == 0) {
            hipDeviceSynchronize();
        }
    }
    rocblas_zdotc(handle.handle(), basis_size_squared, drho_target.data(), 1, drho0.data(), 1, dtrace_temp.data(), mem.data());
    hipDeviceSynchronize();
    rocsparse_double_complex traceFinalComplex;
    hipMemcpy(&traceFinalComplex, dtrace_temp.data(), sizeof(rocsparse_double_complex), hipMemcpyDeviceToHost);
    *trace_final = std::complex<double>(std::real(traceFinalComplex), std::imag(traceFinalComplex));
    // Copy the last incomplete block of the cached rho arrays
    hipMemcpy(rho_vec_cached_t.data() + rho_cache_gpu_block*cacheSizeTimeStepsGPU*basis_size_squared, drho_vec_cached_t.data(), sizeof(rocsparse_double_complex) * currentCacheSizeTimeStepsGPU * basis_size_squared, hipMemcpyDeviceToHost);
    auto end_forward = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff_forward = end_forward-start_forward;
    //std::cout << "Forward propagation: " << diff_forward.count() << " s" << std::endl;

    if (grad != nullptr) {
        for (int k = 0; k < N_amplitudes; ++k) {
            grad[k] = 0;
            grad[k+N_amplitudes] = 0;
        }
        // Backward propagation
        auto start_back = std::chrono::steady_clock::now();
        ROCArrayXcdOnDevice dchi(rho_target_vec, basis_size_squared);
        for (int64_t i = N_t; i > 0; --i) {
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
            if ((i-1) % timeStepsPerStoredState == 0) {
                hipMemcpy(drho0.data(), rho_vec_cached_t.data() + (i-1)/timeStepsPerStoredState*basis_size_squared, sizeof(rocsparse_double_complex) * basis_size_squared, hipMemcpyHostToDevice);
            } else {
                // Omega1, Omega2, Omega3 are used in the reverse order here,
                // because this is propagation backwards in time
                const rocsparse_double_complex I(0, 1);
                const rocsparse_double_complex factor1_Re = -I*Omega3.real()*(-dt);
                const rocsparse_double_complex factor1_Im = -I*Omega3.imag()*(-dt);
                const rocsparse_double_complex factor2_Re = -I*Omega2.real()*(-dt);
                const rocsparse_double_complex factor2_Im = -I*Omega2.imag()*(-dt);
                const rocsparse_double_complex factor3_Re = -I*Omega1.real()*(-dt);
                const rocsparse_double_complex factor3_Im = -I*Omega1.imag()*(-dt);

                applyL_t(handle.handle(), dk1, rocL, dH_t_Re, dH_t_Im, drho0, factor1_Re, factor1_Im, -dt);

                rocblas_zaxpy_store_new(handle.handle(), basis_size_squared, &halpha_half, dk1.data(), 1, drho0.data(), 1, dtemp.data(), 1);

                applyL_t(handle.handle(), dk2, rocL, dH_t_Re, dH_t_Im, dtemp, factor2_Re, factor2_Im, -dt);

                rocblas_zaxpy_store_new(handle.handle(), basis_size_squared, &halpha_half, dk2.data(), 1, drho0.data(), 1, dtemp.data(), 1);

                applyL_t(handle.handle(), dk3, rocL, dH_t_Re, dH_t_Im, dtemp, factor2_Re, factor2_Im, -dt);

                rocblas_zaxpy_store_new(handle.handle(), basis_size_squared, &halpha_one, dk3.data(), 1, drho0.data(), 1, dtemp.data(), 1);

                applyL_t(handle.handle(), dk4, rocL, dH_t_Re, dH_t_Im, dtemp, factor3_Re, factor3_Im, -dt);

                //rk4_finalize_z implements
                //rho0 += (k1/6) + (k2/3) + (k3/3) + (k4/6);
                rk4_finalize_z(handle.handle(), basis_size_squared, drho0.data(), dk1.data(), dk2.data(), dk3.data(), dk4.data());
            }

            const rocsparse_double_complex H_factor = -I*dt;
            const rocsparse_double_complex H_factor1_Re = -I*Omega1.real()*dt;
            const rocsparse_double_complex H_factor1_Im = -I*Omega1.imag()*dt;
            const rocsparse_double_complex H_factor2_Re = -I*Omega2.real()*dt;
            const rocsparse_double_complex H_factor2_Im = -I*Omega2.imag()*dt;
            const rocsparse_double_complex H_factor3_Re = -I*Omega3.real()*dt;
            const rocsparse_double_complex H_factor3_Im = -I*Omega3.imag()*dt;

            const rocsparse_double_complex hone_over_6(1.0/6, 0);
            const rocsparse_double_complex hone_over_12(1.0/12, 0);
            const rocsparse_double_complex hone_over_24(1.0/24, 0);
            const rocsparse_double_complex hone(1.0, 0);
            const rocsparse_double_complex hhalf(0.5, 0);
            const rocsparse_double_complex hquarter(0.25, 0);
            const rocsparse_double_complex htwo_over_3(2.0/3, 0);

            mulMatAddAdjoint(handle.handle(), dk0, dH_t_Re, drho0, H_factor);
            applyL_t(handle.handle(), dk1, rocL, dH_t_Re, dH_t_Im, drho0, H_factor1_Re, H_factor1_Im, dt);
            applyL_t(handle.handle(), dk2, rocL, dH_t_Re, dH_t_Im, drho0, H_factor2_Re, H_factor2_Im, dt);
            mulMatAddAdjoint(handle.handle(), dk3, dH_t_Re, dk1, H_factor);
            mulMatAddAdjoint(handle.handle(), dk4, dH_t_Re, dk2, H_factor);
            applyL_t(handle.handle(), dk5, rocL, dH_t_Re, dH_t_Im, dk0, H_factor2_Re, H_factor2_Im, dt);
            applyL_t(handle.handle(), dk6, rocL, dH_t_Re, dH_t_Im, dk1, H_factor2_Re, H_factor2_Im, dt);
            applyL_t(handle.handle(), dk7, rocL, dH_t_Re, dH_t_Im, dk3, H_factor2_Re, H_factor2_Im, dt);
            applyL_t(handle.handle(), dk8, rocL, dH_t_Re, dH_t_Im, dk5, H_factor2_Re, H_factor2_Im, dt);
            mulMatAddAdjoint(handle.handle(), dk9, dH_t_Re, dk6, H_factor);

            applyL_t(handle.handle(), danother_temp, rocL, dH_t_Re, dH_t_Im, dk8, H_factor3_Re, H_factor3_Im, dt);
            rocblas_zsum4_store_new(handle.handle(), basis_size_squared, dtemp.data(), &hone_over_6, dk0.data(), &hone_over_6, dk5.data(), &hone_over_12, dk8.data(), &hone_over_24, danother_temp.data());
            rocblas_zdotc(handle.handle(), basis_size_squared, dchi.data(), 1, dtemp.data(), 1, dtrace_temp.data(), mem.data());

            rocblas_zsum5_store_new(handle.handle(), basis_size_squared, dtemp.data(), &hone, dk0.data(), &hhalf, dk4.data(), &hhalf, dk5.data(), &hquarter, dk9.data(), &hquarter, dk7.data());
            applyL_t(handle.handle(), danother_temp, rocL, dH_t_Re, dH_t_Im, dtemp, H_factor3_Re, H_factor3_Im, dt);
            rocblas_zsum7_store_new(handle.handle(), basis_size_squared, dtemp.data(), &htwo_over_3, dk0.data(), &hone_over_6, dk3.data(), &hone_over_6, dk4.data(), &hone_over_6, dk5.data(), &hone_over_12, dk9.data(), &hone_over_12, dk7.data(), &hone_over_6, danother_temp.data());
            rocblas_zdotc(handle.handle(), basis_size_squared, dchi.data(), 1, dtemp.data(), 1, dtrace_temp.data() + 1, mem.data());

            rocblas_zsum2_store_new(handle.handle(), basis_size_squared, danother_temp.data(), &hone, dk2.data(), &hhalf, dk6.data());
            applyL_t(handle.handle(), dtemp, rocL, dH_t_Re, dH_t_Im, danother_temp, H_factor2_Re, H_factor2_Im, dt);
            mulMatAddAdjoint(handle.handle(), danother_temp, dH_t_Re, dtemp, H_factor);
            rocblas_zsum3_store_new(handle.handle(), basis_size_squared, dtemp.data(), &hone_over_6, dk0.data(), &hone_over_6, dk4.data(), &hone_over_12, danother_temp.data());
            rocblas_zdotc(handle.handle(), basis_size_squared, dchi.data(), 1, dtemp.data(), 1, dtrace_temp.data() + 2, mem.data());

            mulMatAddAdjoint(handle.handle(), dk0, dH_t_Im, drho0, H_factor);
            applyL_t(handle.handle(), dk1, rocL, dH_t_Re, dH_t_Im, drho0, H_factor1_Re, H_factor1_Im, dt);
            applyL_t(handle.handle(), dk2, rocL, dH_t_Re, dH_t_Im, drho0, H_factor2_Re, H_factor2_Im, dt);
            mulMatAddAdjoint(handle.handle(), dk3, dH_t_Im, dk1, H_factor);
            mulMatAddAdjoint(handle.handle(), dk4, dH_t_Im, dk2, H_factor);
            applyL_t(handle.handle(), dk5, rocL, dH_t_Re, dH_t_Im, dk0, H_factor2_Re, H_factor2_Im, dt);
            applyL_t(handle.handle(), dk6, rocL, dH_t_Re, dH_t_Im, dk1, H_factor2_Re, H_factor2_Im, dt);
            applyL_t(handle.handle(), dk7, rocL, dH_t_Re, dH_t_Im, dk3, H_factor2_Re, H_factor2_Im, dt);
            applyL_t(handle.handle(), dk8, rocL, dH_t_Re, dH_t_Im, dk5, H_factor2_Re, H_factor2_Im, dt);
            mulMatAddAdjoint(handle.handle(), dk9, dH_t_Im, dk6, H_factor);

            applyL_t(handle.handle(), danother_temp, rocL, dH_t_Re, dH_t_Im, dk8, H_factor3_Re, H_factor3_Im, dt);
            rocblas_zsum4_store_new(handle.handle(), basis_size_squared, dtemp.data(), &hone_over_6, dk0.data(), &hone_over_6, dk5.data(), &hone_over_12, dk8.data(), &hone_over_24, danother_temp.data());
            rocblas_zdotc(handle.handle(), basis_size_squared, dchi.data(), 1, dtemp.data(), 1, dtrace_temp.data() + 3, mem.data());

            rocblas_zsum5_store_new(handle.handle(), basis_size_squared, dtemp.data(), &hone, dk0.data(), &hhalf, dk4.data(), &hhalf, dk5.data(), &hquarter, dk9.data(), &hquarter, dk7.data());
            applyL_t(handle.handle(), danother_temp, rocL, dH_t_Re, dH_t_Im, dtemp, H_factor3_Re, H_factor3_Im, dt);
            rocblas_zsum7_store_new(handle.handle(), basis_size_squared, dtemp.data(), &htwo_over_3, dk0.data(), &hone_over_6, dk3.data(), &hone_over_6, dk4.data(), &hone_over_6, dk5.data(), &hone_over_12, dk9.data(), &hone_over_12, dk7.data(), &hone_over_6, danother_temp.data());
            rocblas_zdotc(handle.handle(), basis_size_squared, dchi.data(), 1, dtemp.data(), 1, dtrace_temp.data() + 4, mem.data());

            rocblas_zsum2_store_new(handle.handle(), basis_size_squared, danother_temp.data(), &hone, dk2.data(), &hhalf, dk6.data());
            applyL_t(handle.handle(), dtemp, rocL, dH_t_Re, dH_t_Im, danother_temp, H_factor2_Re, H_factor2_Im, dt);
            mulMatAddAdjoint(handle.handle(), danother_temp, dH_t_Im, dtemp, H_factor);
            rocblas_zsum3_store_new(handle.handle(), basis_size_squared, dtemp.data(), &hone_over_6, dk0.data(), &hone_over_6, dk4.data(), &hone_over_12, danother_temp.data());
            rocblas_zdotc(handle.handle(), basis_size_squared, dchi.data(), 1, dtemp.data(), 1, dtrace_temp.data() + 5, mem.data());


            hipDeviceSynchronize();
            hipMemcpy(htrace_temp.data(), dtrace_temp.data(), sizeof(rocsparse_double_complex)*6, hipMemcpyDeviceToHost);

            double factor1_Re = std::real(htrace_temp[0]);
            double factor2_Re = std::real(htrace_temp[1]);
            double factor3_Re = std::real(htrace_temp[2]);
            double factor1_Im = std::real(htrace_temp[3]);
            double factor2_Im = std::real(htrace_temp[4]);
            double factor3_Im = std::real(htrace_temp[5]);
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
            //Backward step for chi
            applyL_t(handle.handle(), dk1, rocL_adjoint, dH_t_Re, dH_t_Im, dchi, -H_factor2_Re, -H_factor2_Im, dt);
            applyL_t(handle.handle(), dk2, rocL_adjoint, dH_t_Re, dH_t_Im, dchi, -H_factor3_Re, -H_factor3_Im, dt);
            applyL_t(handle.handle(), dk3, rocL_adjoint, dH_t_Re, dH_t_Im, dk1, -H_factor2_Re, -H_factor2_Im, dt);
            applyL_t(handle.handle(), dk4, rocL_adjoint, dH_t_Re, dH_t_Im, dk2, -H_factor2_Re, -H_factor2_Im, dt);
            applyL_t(handle.handle(), dk5, rocL_adjoint, dH_t_Re, dH_t_Im, dk4, -H_factor2_Re, -H_factor2_Im, dt);

            rocblas_zsum4_store_new(handle.handle(), basis_size_squared, danother_temp.data(), &hone_over_6, dchi.data(), &hone_over_6, dk1.data(), &hone_over_12, dk3.data(), &hone_over_24, dk5.data());
            applyL_t(handle.handle(), dtemp, rocL_adjoint, dH_t_Re, dH_t_Im, danother_temp, -H_factor1_Re, -H_factor1_Im, dt);

            rocblas_zsum7_store_new(handle.handle(), basis_size_squared, dchi.data(), &hone, dchi.data(), &hone, dtemp.data(), &htwo_over_3, dk1.data(), &hone_over_6, dk3.data(), &hone_over_6, dk2.data(), &hone_over_6, dk4.data(), &hone_over_12, dk5.data());
        }
        auto end_back = std::chrono::steady_clock::now();
        std::chrono::duration<double> diff_back = end_back-start_back;
        //std::cout << "Backward propagation: " << diff_back.count() << " s" << std::endl;
    }
}
