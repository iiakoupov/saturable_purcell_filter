// Copyright (c) 2021-2023 Ivan Iakoupov
//
// Based in part upon code from rocSPARSE which is:
//
// Copyright (c) 2018-2020 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef CSRMMTRACE_ROC_H
#define CSRMMTRACE_ROC_H

#include <hip/hip_runtime_api.h>
#include <rocsparse/rocsparse.h>
#include "roc_spmv_util.h"

template <unsigned int BLOCKSIZE, unsigned int WF_SIZE, typename I, typename J, typename T>
static __device__ void csrmmtracen_general_device(J                    m,
                                             T                    alpha,
                                             const I*             row_offset,
                                             const J*             csr_col_ind,
                                             const T*             csr_val,
                                             const T*             x,
                                             T                    beta,
                                             T*                   y)
{
    int lid = hipThreadIdx_x & (WF_SIZE - 1);

    J gid = hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x;
    J nwf = hipGridDim_x * BLOCKSIZE / WF_SIZE;

    // Loop over rows
    for(J row = gid / WF_SIZE; row < m; row += nwf)
    {
        // Each wavefront processes one row
        I row_start = row_offset[row];
        I row_end   = row_offset[row + 1];

        T sum = static_cast<T>(0);

        // Loop over non-zero elements
        for(I j = row_start + lid; j < row_end; j += WF_SIZE)
        {
            sum = rocsparse_fma(
                alpha * csr_val[j], rocsparse_ldg(x + csr_col_ind[j]*m + row), sum);
        }

        // Obtain row sum using parallel reduction
        sum = rocsparse_wfreduce_sum<WF_SIZE>(sum);

        // First thread of each wavefront writes result into global memory
        if(lid == WF_SIZE - 1)
        {
            if(beta == static_cast<T>(0))
            {
                y[row] = sum;
            }
            else
            {
                y[row] = rocsparse_fma(beta, y[row], sum);
            }
        }
    }
}

template <unsigned int BLOCKSIZE,
          unsigned int WF_SIZE,
          typename I,
          typename J,
          typename T>
__launch_bounds__(BLOCKSIZE) __global__
    void csrmmtracen_general_kernel(J m,
                               T alpha,
                               const I* __restrict__ csr_row_ptr,
                               const J* __restrict__ csr_col_ind,
                               const T* __restrict__ csr_val,
                               const T* __restrict__ x,
                               T beta,
                               T* __restrict__ y)
{
    csrmmtracen_general_device<BLOCKSIZE, WF_SIZE>(
        m, alpha, csr_row_ptr, csr_col_ind, csr_val, x, beta, y);
}

template <typename I, typename J, typename T>
void csrmmtrace_template_dispatch(rocsparse_handle          handle,
                             rocsparse_operation       trans,
                             J                         m,
                             J                         n,
                             I                         nnz,
                             T                        alpha,
                             const rocsparse_mat_descr descr,
                             const T*                  csr_val,
                             const I*                  csr_row_ptr,
                             const J*                  csr_col_ind,
                             const T*                  x,
                             T                        beta,
                             T*                        y)
{
    // Stream
    hipStream_t stream;
    rocsparse_get_stream(handle, &stream);

#define CSRMMTRACEN_DIM 512
    J nnz_per_row = nnz / m;

    dim3 csrmmtracen_blocks((m - 1) / CSRMMTRACEN_DIM + 1);
    dim3 csrmmtracen_threads(CSRMMTRACEN_DIM);

    hipDeviceProp_t devProp;
    int device_id = 0;
    hipGetDevice(&device_id);
    hipGetDeviceProperties(&devProp, device_id);
    int wavefront_size = devProp.warpSize;

    if(wavefront_size == 32)
    {
        // LCOV_EXCL_START
        if(nnz_per_row < 4)
        {
            hipLaunchKernelGGL((csrmmtracen_general_kernel<CSRMMTRACEN_DIM, 2>),
                               csrmmtracen_blocks,
                               csrmmtracen_threads,
                               0,
                               stream,
                               m,
                               alpha,
                               csr_row_ptr,
                               csr_col_ind,
                               csr_val,
                               x,
                               beta,
                               y);
        }
        else if(nnz_per_row < 8)
        {
            hipLaunchKernelGGL((csrmmtracen_general_kernel<CSRMMTRACEN_DIM, 4>),
                               csrmmtracen_blocks,
                               csrmmtracen_threads,
                               0,
                               stream,
                               m,
                               alpha,
                               csr_row_ptr,
                               csr_col_ind,
                               csr_val,
                               x,
                               beta,
                               y);
        }
        else if(nnz_per_row < 16)
        {
            hipLaunchKernelGGL((csrmmtracen_general_kernel<CSRMMTRACEN_DIM, 8>),
                               csrmmtracen_blocks,
                               csrmmtracen_threads,
                               0,
                               stream,
                               m,
                               alpha,
                               csr_row_ptr,
                               csr_col_ind,
                               csr_val,
                               x,
                               beta,
                               y);
        }
        else if(nnz_per_row < 32)
        {
            hipLaunchKernelGGL((csrmmtracen_general_kernel<CSRMMTRACEN_DIM, 16>),
                               csrmmtracen_blocks,
                               csrmmtracen_threads,
                               0,
                               stream,
                               m,
                               alpha,
                               csr_row_ptr,
                               csr_col_ind,
                               csr_val,
                               x,
                               beta,
                               y);
        }
        else
        {
            hipLaunchKernelGGL((csrmmtracen_general_kernel<CSRMMTRACEN_DIM, 32>),
                               csrmmtracen_blocks,
                               csrmmtracen_threads,
                               0,
                               stream,
                               m,
                               alpha,
                               csr_row_ptr,
                               csr_col_ind,
                               csr_val,
                               x,
                               beta,
                               y);
        }
        // LCOV_EXCL_STOP
    }
    else
    {
        assert(wavefront_size == 64);
        if(nnz_per_row < 4)
        {
            hipLaunchKernelGGL((csrmmtracen_general_kernel<CSRMMTRACEN_DIM, 2>),
                               csrmmtracen_blocks,
                               csrmmtracen_threads,
                               0,
                               stream,
                               m,
                               alpha,
                               csr_row_ptr,
                               csr_col_ind,
                               csr_val,
                               x,
                               beta,
                               y);
        }
        else if(nnz_per_row < 8)
        {
            hipLaunchKernelGGL((csrmmtracen_general_kernel<CSRMMTRACEN_DIM, 4>),
                               csrmmtracen_blocks,
                               csrmmtracen_threads,
                               0,
                               stream,
                               m,
                               alpha,
                               csr_row_ptr,
                               csr_col_ind,
                               csr_val,
                               x,
                               beta,
                               y);
        }
        else if(nnz_per_row < 16)
        {
            hipLaunchKernelGGL((csrmmtracen_general_kernel<CSRMMTRACEN_DIM, 8>),
                               csrmmtracen_blocks,
                               csrmmtracen_threads,
                               0,
                               stream,
                               m,
                               alpha,
                               csr_row_ptr,
                               csr_col_ind,
                               csr_val,
                               x,
                               beta,
                               y);
        }
        else if(nnz_per_row < 32)
        {
            hipLaunchKernelGGL((csrmmtracen_general_kernel<CSRMMTRACEN_DIM, 16>),
                               csrmmtracen_blocks,
                               csrmmtracen_threads,
                               0,
                               stream,
                               m,
                               alpha,
                               csr_row_ptr,
                               csr_col_ind,
                               csr_val,
                               x,
                               beta,
                               y);
        }
        else if(nnz_per_row < 64)
        {
            hipLaunchKernelGGL((csrmmtracen_general_kernel<CSRMMTRACEN_DIM, 32>),
                               csrmmtracen_blocks,
                               csrmmtracen_threads,
                               0,
                               stream,
                               m,
                               alpha,
                               csr_row_ptr,
                               csr_col_ind,
                               csr_val,
                               x,
                               beta,
                               y);
        }
        else
        {
            hipLaunchKernelGGL((csrmmtracen_general_kernel<CSRMMTRACEN_DIM, 64>),
                               csrmmtracen_blocks,
                               csrmmtracen_threads,
                               0,
                               stream,
                               m,
                               alpha,
                               csr_row_ptr,
                               csr_col_ind,
                               csr_val,
                               x,
                               beta,
                               y);
        }
    }
#undef CSRMMTRACEN_DIM
}

template <typename I, typename J, typename T>
void csrmmtrace_template(rocsparse_handle          handle,
                    rocsparse_operation       trans,
                    J                         m,
                    J                         n,
                    I                         nnz,
                    const T*                  alpha,
                    const rocsparse_mat_descr descr,
                    const T*                  csr_val,
                    const I*                  csr_row_ptr,
                    const J*                  csr_col_ind,
                    rocsparse_mat_info        info,
                    const T*                  x,
                    const T*                  beta,
                    T*                        y)
{
    csrmmtrace_template_dispatch(handle,
                            trans,
                            m,
                            n,
                            nnz,
                            *alpha,
                            descr,
                            csr_val,
                            csr_row_ptr,
                            csr_col_ind,
                            x,
                            *beta,
                            y);
}

void zcsrmmtrace(
        rocsparse_handle handle,
        rocsparse_operation trans,
        rocsparse_int m,
        rocsparse_int n,
        rocsparse_int nnz,
        const rocsparse_double_complex* alpha,
        const rocsparse_mat_descr descr,
        const rocsparse_double_complex* csr_val,
        const rocsparse_int* csr_row_ptr,
        const rocsparse_int* csr_col_ind,
        rocsparse_mat_info info,
        const rocsparse_double_complex* x,
        const rocsparse_double_complex* beta,
        rocsparse_double_complex* y)
{
    csrmmtrace_template(handle,
                   trans,
                   m,
                   n,
                   nnz,
                   alpha,
                   descr,
                   csr_val,
                   csr_row_ptr,
                   csr_col_ind,
                   info,
                   x,
                   beta,
                   y);
}

#endif // CSRMMTRACE_ROC_H
