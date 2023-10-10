// Copyright (c) 2020-2021 Ivan Iakoupov
//
// Based in part upon code from rocBLAS which is:
//
// Copyright 2018-2019 Advanced Micro Devices, Inc.
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

#include "master_equation_evolution_roc.h"

SDEData sme_roc(int basis_size,
                int batch_size,
                const SparseMatrixData &L,
                const SparseMatrixData &RR,
                const std::complex<double> *rho_vec_0,
                const std::complex<double> *rho_vec_1,
                const double *filter,
                const double *DeltaW_matrix,
                const double *integration_times,
                int num_integration_times,
                int N_t_max)
{
    SDEData ret;
    const int basis_size_squared = basis_size*basis_size;
    std::vector<std::vector<rocsparse_double_complex>> hrho(2);
    hrho[0] = std::vector<rocsparse_double_complex>(basis_size_squared);
    hrho[1] = std::vector<rocsparse_double_complex>(basis_size_squared);
    for (int i = 0; i < basis_size_squared; ++i) {
        hrho[0][i] = rocsparse_double_complex(rho0[i].real(), rho0[i].imag());
        hrho[1][i] = rocsparse_double_complex(rho1[i].real(), rho1[i].imag());
    }
    std::vector<rocsparse_double_complex> htrace_mask(basis_size_squared, 0);
    for (int i = 0; i < basis_size; ++i) {
        htrace_mask[i*basis_size+i] = rocsparse_double_complex(1, 0);
    }

    rocsparse_handle handle;
    rocsparse_create_handle(&handle);

    ROCSparseMatrixOnDevice rocL(handle, L, basis_size_squared, basis_size_squared);
    ROCSparseMatrixOnDevice rocRR(handle, RR, basis_size_squared, basis_size_squared);

    hipDeviceProp_t devProp;
    int device_id = 0;

    hipGetDevice(&device_id);
    hipGetDeviceProperties(&devProp, device_id);

    rocsparse_mat_descr descrA;
    rocsparse_create_mat_descr(&descrA);
    const rocsparse_double_complex halpha_dt(dt, 0);
    const rocsparse_double_complex halpha_one(1, 0);
    const rocsparse_double_complex halpha_half(0.5, 0);
    const rocsparse_double_complex halpha_zero(0, 0);
    const rocsparse_double_complex hbeta_zero(0, 0);
    const rocsparse_double_complex hbeta_one(1, 0);

    // Offload data to device
    rocsparse_double_complex* drho = NULL;
    rocsparse_double_complex* drr_applied_rho = NULL;
    rocsparse_double_complex* dtemp = NULL;
    rocsparse_double_complex* dexcitation = NULL;
    rocsparse_double_complex* dtrace_mask = NULL;
    rocsparse_double_complex* dS = NULL;

    hipMalloc((void**)&drho, sizeof(rocsparse_double_complex) * basis_size_squared * batch_size);
    hipMalloc((void**)&drr_applied_rho, sizeof(rocsparse_double_complex) * basis_size_squared * batch_size);
    hipMalloc((void**)&dtemp, sizeof(rocsparse_double_complex) * basis_size_squared * batch_size);
    hipMalloc((void**)&dexcitation, sizeof(rocsparse_double_complex) * N_t);
    hipMalloc((void**)&dtrace_mask, sizeof(rocsparse_double_complex) * basis_size_squared);
    hipMalloc((void**)&dS, sizeof(rocsparse_double_complex) * 2 * num_integration_times * batch_size);

    hipMemcpy(dtrace_mask, htrace_mask.data(), sizeof(rocsparse_double_complex) * basis_size_squared, hipMemcpyHostToDevice);

    for (int m = 0; m < 2; ++m) {
        for (int l = 0; l < batch_size; ++l) {
            hipMemcpy(drho + l * basis_size_squared, hpsi[m].data(), sizeof(rocsparse_double_complex) * basis_size_squared, hipMemcpyHostToDevice);
        }

        size_t dev_bytes = rocblas_reduction_kernel_workspace_size<ROCM_DOT_NB, rocsparse_double_complex>(basis_size_squared);
        rocsparse_double_complex* mem;
        hipMalloc((void**)&mem, sizeof(rocsparse_double_complex) * dev_bytes);

        ret.time.resize(N_t, 0);
        ret.excitation.resize(N_t, 0);
        std::vector<rocsparse_double_complex> excitation(N_t, 0);
        for (int i = 0; i < N_t; ++i) {
            const double t_i = i*dt;
            ret.time[i] = t_i;
#if 0
            //TODO: Below, we first store the coefficientwise product of dpsi0 and dM_11_sum_diag
            //      into dtemp and then take the inner product of dtemp and dpsi0.
            //      This is equivalent to calculating the sum over j std::norm(dpsi0[j]) * dM_11_sum_diag[j]
            //      We can just calculate this sum without using a temporary buffer. Probably the
            //      easiest approach is to modify the zdotc kernel such that it computes
            //      std::norm(dpsi0[j]) * dM_11_sum_diag[j] instead of std::conj(dpsi0[j]) * temp[j]
            rocblas_zd_coeffwise_mul_store_new(handle, basis_size_squared, &halpha_one, dM_11_sum_diag, 1, dpsi0, 1, dtemp, 1);
            rocblas_zdotc(handle, basis_size_squared, dtemp, 1, dpsi0, 1, dexcitation + i, mem);
            for (int j = 0; j < M_op_array_size; ++j) {
                rocblas_zd_coeffwise_mul_store_new(handle, basis_size_squared, &halpha_one, dM_op_array[j], 1, dpsi0, 1, dtemp, 1);
                rocblas_zdotc(handle, basis_size_squared, dtemp, 1, dpsi0, 1, dper_atom_excitation[j] + i, mem);
            }
#endif // 0

            rocL.mul_mat(&halpha_dt, drho, &hbeta_zero, dtemp, batch_size);
            rocRR.mul_mat(&halpha_one, drho, &hbeta_zero, drr_applied_rho, batch_size);
            //TODO: Optimize. It is actually a matrix-vector product rho*mask
            //      Probably need to use or copy gemv (hemv) from rocBLAS
            //      On the other hand, we probably want to remove dtrace_mask
            //      completely and just sum the correct indices of drho for the trace.
            for (int l = 0; l < batch_size; ++l) {
                rocblas_zdotc(handle, basis_size_squared, drr_applied_rho + l * basis_size_squared, 1, dtrace_mask, 1, dS + m*num_integration_times*batch_size + current_S_index*num_integration_times + l, mem);
            }
            // TODO: Add DeltaW to dtemp
            rocblas_zaxpy(handle, basis_size, &halpha_one, dtemp, 1, drho, 1);
        }
        hipDeviceSynchronize();
        hipMemcpy(excitation.data(), dexcitation, sizeof(rocsparse_double_complex) * N_t, hipMemcpyDeviceToHost);
        for (int i = 0; i < N_t; ++i) {
            ret.excitation[i] = std::real(excitation[i]);
        }
    }
    hipFree(drho);
    hipFree(drr_applied_rho);
    hipFree(dtemp);
    hipFree(dexcitation);
    hipFree(dtrace_mask);
    hipFree(mem);

    rocsparse_destroy_handle(handle);
    return ret;
}
