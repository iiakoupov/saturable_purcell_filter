// Copyright (c) 2022-2023 Ivan Iakoupov
//
// Based in part upon code from rocSPARSE which is:
//
// Copyright (c) 2020-2022 Advanced Micro Devices, Inc.
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

#ifndef CSRMM_ROW_MAJOR_H
#define CSRMM_ROW_MAJOR_H

#include <rocsparse/rocsparse.h>
#include "roc_spmv_util.h"
//#include <algorithm>

//#include "rocsparse_csrmm.hpp"

//#include "load_scalar.h"
//#include "wrapper_functions.h"

__device__ __forceinline__ float rocsparse_conj(const float& x) { return x; }
__device__ __forceinline__ double rocsparse_conj(const double& x) { return x; }
__device__ __forceinline__ rocsparse_float_complex rocsparse_conj(const rocsparse_float_complex& x) { return std::conj(x); }
__device__ __forceinline__ rocsparse_double_complex rocsparse_conj(const rocsparse_double_complex& x) { return std::conj(x); }

// For host scalars
template <typename T>
__forceinline__ __device__ __host__ T load_scalar_device_host(T x)
{
    return x;
}

// For device scalars
template <typename T>
__forceinline__ __device__ __host__ T load_scalar_device_host(const T* xp)
{
    return *xp;
}

//
// Provide some utility methods for enums.
//
struct rocsparse_enum_utils
{
    template <typename U>
    static inline bool is_invalid(U value_);
};

template <>
inline bool rocsparse_enum_utils::is_invalid(rocsparse_operation value_)
{
    switch(value_)
    {
    case rocsparse_operation_none:
    case rocsparse_operation_transpose:
    case rocsparse_operation_conjugate_transpose:
    {
        return false;
    }
    }
    return true;
};

template <>
inline bool rocsparse_enum_utils::is_invalid(rocsparse_order value_)
{
    switch(value_)
    {
    case rocsparse_order_row:
    case rocsparse_order_column:
    {
        return false;
    }
    }
    return true;
};

template <typename T>
static ROCSPARSE_DEVICE_ILF T conj_val(T val, bool conj)
{
    return conj ? rocsparse_conj(val) : val;
}

template <unsigned int BLOCKSIZE, unsigned int WF_SIZE, typename I, typename J, typename T>
static ROCSPARSE_DEVICE_ILF void csrmmnn_general_device(bool conj_A,
                                                        bool conj_B,
                                                        J    M,
                                                        J    N,
                                                        J    K,
                                                        I    nnz,
                                                        I    offsets_batch_stride_A,
                                                        I    columns_values_batch_stride_A,
                                                        T    alpha,
                                                        const I* __restrict__ csr_row_ptr,
                                                        const J* __restrict__ csr_col_ind,
                                                        const T* __restrict__ csr_val,
                                                        const T* __restrict__ B,
                                                        J ldb,
                                                        I batch_stride_B,
                                                        T beta,
                                                        T* __restrict__ C,
                                                        J                    ldc,
                                                        I                    batch_stride_C,
                                                        rocsparse_order      order,
                                                        rocsparse_index_base idx_base)
{
    int tid = hipThreadIdx_x;
    J   gid = hipBlockIdx_x * BLOCKSIZE + tid;
    int lid = gid & (WF_SIZE - 1);
    int wid = tid / WF_SIZE;
    J   row = gid / WF_SIZE;
    J   col = lid + hipBlockIdx_y * WF_SIZE;

    J batch = hipBlockIdx_z;

    if(row >= M)
    {
        return;
    }

    J colB = col * ldb;

    __shared__ J shared_col[BLOCKSIZE / WF_SIZE][WF_SIZE];
    __shared__ T shared_val[BLOCKSIZE / WF_SIZE][WF_SIZE];

    I row_start = csr_row_ptr[row + offsets_batch_stride_A * batch] - idx_base;
    I row_end   = csr_row_ptr[row + 1 + offsets_batch_stride_A * batch] - idx_base;

    T sum = static_cast<T>(0);

    for(I j = row_start; j < row_end; j += WF_SIZE)
    {
        I k = j + lid;

        __syncthreads();

        if(k < row_end)
        {
            shared_col[wid][lid]
                = csr_col_ind[k + columns_values_batch_stride_A * batch] - idx_base;
            shared_val[wid][lid]
                = conj_val(csr_val[k + columns_values_batch_stride_A * batch], conj_A);
        }
        else
        {
            shared_col[wid][lid] = 0;
            shared_val[wid][lid] = static_cast<T>(0);
        }

        __syncthreads();

        if(col < N)
        {
            for(J i = 0; i < WF_SIZE; ++i)
            {
                sum = rocsparse_fma(
                    shared_val[wid][i],
                    conj_val(B[shared_col[wid][i]*ldb + col + batch_stride_B * batch], conj_B),
                    sum);
            }
        }
    }

    if(col < N)
    {
        if(beta == static_cast<T>(0))
        {
            C[row*ldc + col + batch_stride_C * batch] = alpha * sum;
        }
        else
        {
            C[row*ldc + col + batch_stride_C * batch]
                    = rocsparse_fma(beta, C[row*ldc + col + batch_stride_C * batch], alpha * sum);
        }
    }
}

template <unsigned int BLOCKSIZE,
          unsigned int WF_SIZE,
          typename I,
          typename J,
          typename T,
          typename U>
__launch_bounds__(BLOCKSIZE) ROCSPARSE_KERNEL
    void csrmmnn_general_kernel(bool conj_A,
                                bool conj_B,
                                J    m,
                                J    n,
                                J    k,
                                I    nnz,
                                I    offsets_batch_stride_A,
                                I    columns_values_batch_stride_A,
                                U    alpha_device_host,
                                const I* __restrict__ csr_row_ptr,
                                const J* __restrict__ csr_col_ind,
                                const T* __restrict__ csr_val,
                                const T* __restrict__ B,
                                J ldb,
                                I batch_stride_B,
                                U beta_device_host,
                                T* __restrict__ C,
                                J                    ldc,
                                I                    batch_stride_C,
                                rocsparse_order      order,
                                rocsparse_index_base idx_base)
{
    auto alpha = load_scalar_device_host(alpha_device_host);
    auto beta  = load_scalar_device_host(beta_device_host);

    if(alpha == static_cast<T>(0) && beta == static_cast<T>(1))
    {
        return;
    }

    csrmmnn_general_device<BLOCKSIZE, WF_SIZE>(conj_A,
                                               conj_B,
                                               m,
                                               n,
                                               k,
                                               nnz,
                                               offsets_batch_stride_A,
                                               columns_values_batch_stride_A,
                                               alpha,
                                               csr_row_ptr,
                                               csr_col_ind,
                                               csr_val,
                                               B,
                                               ldb,
                                               batch_stride_B,
                                               beta,
                                               C,
                                               ldc,
                                               batch_stride_C,
                                               order,
                                               idx_base);
}

template <typename I, typename J, typename T, typename U>
rocsparse_status rocsparse_csrmmnn_template_general(rocsparse_handle handle,
                                                    rocsparse_order  order,
                                                    bool             conj_A,
                                                    bool             conj_B,
                                                    J                m,
                                                    J                n,
                                                    J                k,
                                                    I                nnz,
                                                    J                batch_count_A,
                                                    I                offsets_batch_stride_A,
                                                    I                columns_values_batch_stride_A,
                                                    U                alpha_device_host,
                                                    const rocsparse_mat_descr descr,
                                                    const T*                  csr_val,
                                                    const I*                  csr_row_ptr,
                                                    const J*                  csr_col_ind,
                                                    const T*                  B,
                                                    J                         ldb,
                                                    J                         batch_count_B,
                                                    I                         batch_stride_B,
                                                    U                         beta_device_host,
                                                    T*                        C,
                                                    J                         ldc,
                                                    J                         batch_count_C,
                                                    I                         batch_stride_C)
{
    hipStream_t stream;
    rocsparse_get_stream(handle, &stream);
#define CSRMMNN_DIM 256
// In ROCm 5.6.0, "WF_SIZE 8" causes the kernel to return wrong
// results (wrong sign of some of the values). It worked fine in the
// earlier versions (around 5.4.0). One can check whether it is
// still a problem by using "DEBUG_SPMM_MUL" flag in the file
// "master_equation_roc.cpp". We set WF_SIZE 16 as a work around here.
//
// A side note:
// This variable sets the size of the "logical" wavefront (the
// physical wavefront/warp size is typically 32 or 64, depending
// on the GPU), to presumably utilize the hardware more
// efficiently in the case of a narrow matrix B (the one that
// we multiply the sparse matrix on). For wider matrices B, it
// is probably more efficient to increase WF_SIZE anyway. See,
// e.g., the logic in the file "csrmv_roc.h", function
// csrmv_template_dispatch() (also copied from rocSPARSE).
// That function splits the rows of the sparse matrix, while
// here we split the rows of the target matrix, but the idea
// is the same: if there are not enough elements to iterate
// over, it is probably better to split the large physical
// wavefront into multiple logical ones.
#define WF_SIZE 16
    hipLaunchKernelGGL(
        (csrmmnn_general_kernel<CSRMMNN_DIM, WF_SIZE>),
        dim3((WF_SIZE * m - 1) / CSRMMNN_DIM + 1, (n - 1) / WF_SIZE + 1, batch_count_C),
        dim3(CSRMMNN_DIM),
        0,
        stream,
        conj_A,
        conj_B,
        m,
        n,
        k,
        nnz,
        offsets_batch_stride_A,
        columns_values_batch_stride_A,
        alpha_device_host,
        csr_row_ptr,
        csr_col_ind,
        csr_val,
        B,
        ldb,
        batch_stride_B,
        beta_device_host,
        C,
        ldc,
        batch_stride_C,
        order,
        rocsparse_get_mat_index_base(descr));
#undef CSRMMNN_DIM
#undef WF_SIZE

    return rocsparse_status_success;
}

template <typename I, typename J, typename T, typename U>
rocsparse_status rocsparse_csrmm_template_general(rocsparse_handle    handle,
                                                  rocsparse_operation trans_A,
                                                  rocsparse_operation trans_B,
                                                  rocsparse_order     order,
                                                  J                   m,
                                                  J                   n,
                                                  J                   k,
                                                  I                   nnz,
                                                  J                   batch_count_A,
                                                  I                   offsets_batch_stride_A,
                                                  I                   columns_values_batch_stride_A,
                                                  U                   alpha_device_host,
                                                  const rocsparse_mat_descr descr,
                                                  const T*                  csr_val,
                                                  const I*                  csr_row_ptr,
                                                  const J*                  csr_col_ind,
                                                  const T*                  B,
                                                  J                         ldb,
                                                  J                         batch_count_B,
                                                  I                         batch_stride_B,
                                                  U                         beta_device_host,
                                                  T*                        C,
                                                  J                         ldc,
                                                  J                         batch_count_C,
                                                  I                         batch_stride_C,
                                                  bool                      force_conj_A)
{
    bool conj_A = (trans_A == rocsparse_operation_conjugate_transpose || force_conj_A);
    bool conj_B = (trans_B == rocsparse_operation_conjugate_transpose);

    return rocsparse_csrmmnn_template_general(handle,
                                              order,
                                              conj_A,
                                              conj_B,
                                              m,
                                              n,
                                              k,
                                              nnz,
                                              batch_count_A,
                                              offsets_batch_stride_A,
                                              columns_values_batch_stride_A,
                                              alpha_device_host,
                                              descr,
                                              csr_val,
                                              csr_row_ptr,
                                              csr_col_ind,
                                              B,
                                              ldb,
                                              batch_count_B,
                                              batch_stride_B,
                                              beta_device_host,
                                              C,
                                              ldc,
                                              batch_count_C,
                                              batch_stride_C);
}

template <typename I, typename J, typename T, typename U>
rocsparse_status rocsparse_csrmm_template_dispatch(rocsparse_handle    handle,
                                                   rocsparse_operation trans_A,
                                                   rocsparse_operation trans_B,
                                                   rocsparse_order     order,
                                                   J                   m,
                                                   J                   n,
                                                   J                   k,
                                                   I                   nnz,
                                                   J                   batch_count_A,
                                                   I                   offsets_batch_stride_A,
                                                   I columns_values_batch_stride_A,
                                                   U alpha_device_host,
                                                   const rocsparse_mat_descr descr,
                                                   const T*                  csr_val,
                                                   const I*                  csr_row_ptr,
                                                   const J*                  csr_col_ind,
                                                   const T*                  B,
                                                   J                         ldb,
                                                   J                         batch_count_B,
                                                   I                         batch_stride_B,
                                                   U                         beta_device_host,
                                                   T*                        C,
                                                   J                         ldc,
                                                   J                         batch_count_C,
                                                   I                         batch_stride_C,
                                                   void*                     temp_buffer,
                                                   bool                      force_conj_A)
{
    return rocsparse_csrmm_template_general(handle,
                                            trans_A,
                                            trans_B,
                                            order,
                                            m,
                                            n,
                                            k,
                                            nnz,
                                            batch_count_A,
                                            offsets_batch_stride_A,
                                            columns_values_batch_stride_A,
                                            alpha_device_host,
                                            descr,
                                            csr_val,
                                            csr_row_ptr,
                                            csr_col_ind,
                                            B,
                                            ldb,
                                            batch_count_B,
                                            batch_stride_B,
                                            beta_device_host,
                                            C,
                                            ldc,
                                            batch_count_C,
                                            batch_stride_C,
                                            force_conj_A);
}

template <typename I, typename J, typename T>
rocsparse_status rocsparse_csrmm_template(rocsparse_handle          handle,
                                          rocsparse_operation       trans_A,
                                          rocsparse_operation       trans_B,
                                          rocsparse_order           order_B,
                                          rocsparse_order           order_C,
                                          J                         m,
                                          J                         n,
                                          J                         k,
                                          I                         nnz,
                                          J                         batch_count_A,
                                          I                         offsets_batch_stride_A,
                                          I                         columns_values_batch_stride_A,
                                          const T*                  alpha_device_host,
                                          const rocsparse_mat_descr descr,
                                          const T*                  csr_val,
                                          const I*                  csr_row_ptr,
                                          const J*                  csr_col_ind,
                                          const T*                  B,
                                          J                         ldb,
                                          J                         batch_count_B,
                                          I                         batch_stride_B,
                                          const T*                  beta_device_host,
                                          T*                        C,
                                          J                         ldc,
                                          J                         batch_count_C,
                                          I                         batch_stride_C,
                                          void*                     temp_buffer,
                                          bool                      force_conj_A)
{
    // Check for valid handle and matrix descriptor
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }
    else if(descr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Logging TODO bench logging
    //log_trace(handle,
    //          replaceX<T>("rocsparse_Xcsrmm"),
    //          trans_A,
    //          trans_B,
    //          m,
    //          n,
    //          k,
    //          nnz,
    //          batch_count_A,
    //          offsets_batch_stride_A,
    //          columns_values_batch_stride_A,
    //          LOG_TRACE_SCALAR_VALUE(handle, alpha_device_host),
    //          (const void*&)descr,
    //          (const void*&)csr_val,
    //          (const void*&)csr_row_ptr,
    //          (const void*&)csr_col_ind,
    //          (const void*&)B,
    //          ldb,
    //          batch_count_B,
    //          batch_stride_B,
    //          LOG_TRACE_SCALAR_VALUE(handle, beta_device_host),
    //          (const void*&)C,
    //          ldc,
    //          batch_count_C,
    //          batch_stride_C);

    if(rocsparse_enum_utils::is_invalid(trans_A))
    {
        return rocsparse_status_invalid_value;
    }

    if(rocsparse_enum_utils::is_invalid(trans_B))
    {
        return rocsparse_status_invalid_value;
    }

    if(rocsparse_enum_utils::is_invalid(order_B))
    {
        return rocsparse_status_invalid_value;
    }

    if(rocsparse_enum_utils::is_invalid(order_C))
    {
        return rocsparse_status_invalid_value;
    }

    // Check matrix type
    if(rocsparse_get_mat_type(descr) != rocsparse_matrix_type_general)
    {
        return rocsparse_status_not_implemented;
    }

    // Check matrix sorting mode
    if(rocsparse_get_mat_storage_mode(descr) != rocsparse_storage_mode_sorted)
    {
        return rocsparse_status_not_implemented;
    }

    if(order_B != order_C)
    {
        return rocsparse_status_invalid_value;
    }

    // Check sizes
    if(m < 0 || n < 0 || k < 0 || nnz < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Quick return if possible
    if(m == 0 || n == 0 || k == 0)
    {
        return rocsparse_status_success;
    }

    // Check the rest of pointer arguments
    if(alpha_device_host == nullptr || beta_device_host == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    rocsparse_pointer_mode pointer_mode;
    rocsparse_get_pointer_mode(handle, &pointer_mode);
    if(pointer_mode == rocsparse_pointer_mode_host
       && *alpha_device_host == static_cast<T>(0) && *beta_device_host == static_cast<T>(1))
    {
        return rocsparse_status_success;
    }

    // Check the rest of pointer arguments
    if(csr_row_ptr == nullptr || B == nullptr || C == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // value arrays and column indices arrays must both be null (zero matrix) or both not null
    if((csr_val == nullptr && csr_col_ind != nullptr)
       || (csr_val != nullptr && csr_col_ind == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    if(nnz != 0 && (csr_col_ind == nullptr && csr_val == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check leading dimension of matrices
    static constexpr J s_one = static_cast<J>(1);
    switch(trans_A)
    {
    case rocsparse_operation_none:
    {
        // Check leading dimension of C
        if(ldc < std::max(s_one, ((order_C == rocsparse_order_column) ? m : n)))
        {
            return rocsparse_status_invalid_size;
        }

        // Check leading dimension of B
        switch(trans_B)
        {
        case rocsparse_operation_none:
        {
            if(ldb < std::max(s_one, ((order_B == rocsparse_order_column) ? k : n)))
            {
                return rocsparse_status_invalid_size;
            }
            break;
        }
        case rocsparse_operation_transpose:
        case rocsparse_operation_conjugate_transpose:
        {
            if(ldb < std::max(s_one, ((order_B == rocsparse_order_column) ? n : k)))
            {
                return rocsparse_status_invalid_size;
            }
            break;
        }
        }
        break;
    }
    case rocsparse_operation_transpose:
    case rocsparse_operation_conjugate_transpose:
    {
        // Check leading dimension of C
        if(ldc < std::max(s_one, ((order_C == rocsparse_order_column) ? k : n)))
        {
            return rocsparse_status_invalid_size;
        }

        switch(trans_B)
        {
        case rocsparse_operation_none:
        {
            if(ldb < std::max(s_one, ((order_B == rocsparse_order_column) ? m : n)))
            {
                return rocsparse_status_invalid_size;
            }
            break;
        }
        case rocsparse_operation_transpose:
        case rocsparse_operation_conjugate_transpose:
        {
            if(ldb < std::max(s_one, ((order_B == rocsparse_order_column) ? n : m)))
            {
                return rocsparse_status_invalid_size;
            }
            break;
        }
        }
        break;
    }
    }

    // Check batch parameters of matrices
    bool Ci_A_Bi  = (batch_count_A == 1 && batch_count_B == batch_count_C);
    bool Ci_Ai_B  = (batch_count_B == 1 && batch_count_A == batch_count_C);
    bool Ci_Ai_Bi = (batch_count_A == batch_count_C && batch_count_A == batch_count_B);

    if(!Ci_A_Bi && !Ci_Ai_B && !Ci_Ai_Bi)
    {
        return rocsparse_status_invalid_value;
    }

    rocsparse_get_pointer_mode(handle, &pointer_mode);
    if(pointer_mode == rocsparse_pointer_mode_device)
    {
        return rocsparse_csrmm_template_dispatch(handle,
                                                 trans_A,
                                                 trans_B,
                                                 order_B,
                                                 m,
                                                 n,
                                                 k,
                                                 nnz,
                                                 batch_count_A,
                                                 offsets_batch_stride_A,
                                                 columns_values_batch_stride_A,
                                                 alpha_device_host,
                                                 descr,
                                                 csr_val,
                                                 csr_row_ptr,
                                                 csr_col_ind,
                                                 B,
                                                 ldb,
                                                 batch_count_B,
                                                 batch_stride_B,
                                                 beta_device_host,
                                                 C,
                                                 ldc,
                                                 batch_count_C,
                                                 batch_stride_C,
                                                 temp_buffer,
                                                 force_conj_A);
    }
    else
    {
        return rocsparse_csrmm_template_dispatch(handle,
                                                 trans_A,
                                                 trans_B,
                                                 order_B,
                                                 m,
                                                 n,
                                                 k,
                                                 nnz,
                                                 batch_count_A,
                                                 offsets_batch_stride_A,
                                                 columns_values_batch_stride_A,
                                                 *alpha_device_host,
                                                 descr,
                                                 csr_val,
                                                 csr_row_ptr,
                                                 csr_col_ind,
                                                 B,
                                                 ldb,
                                                 batch_count_B,
                                                 batch_stride_B,
                                                 *beta_device_host,
                                                 C,
                                                 ldc,
                                                 batch_count_C,
                                                 batch_stride_C,
                                                 temp_buffer,
                                                 force_conj_A);
    }
}

#define IMPL(NAME, TYPE)                                                  \
    rocsparse_status NAME(rocsparse_handle          handle,      \
                                     rocsparse_operation       trans_A,     \
                                     rocsparse_operation       trans_B,     \
                                     rocsparse_int             m,           \
                                     rocsparse_int             n,           \
                                     rocsparse_int             k,           \
                                     rocsparse_int             nnz,         \
                                     const TYPE*               alpha,       \
                                     const rocsparse_mat_descr descr,       \
                                     const TYPE*               csr_val,     \
                                     const rocsparse_int*      csr_row_ptr, \
                                     const rocsparse_int*      csr_col_ind, \
                                     const TYPE*               B,           \
                                     rocsparse_int             ldb,         \
                                     const TYPE*               beta,        \
                                     TYPE*                     C,           \
                                     rocsparse_int             ldc)         \
    {                                                                       \
        return rocsparse_csrmm_template(handle,                             \
                                        trans_A,                            \
                                        trans_B,                            \
                                        rocsparse_order_row,                \
                                        rocsparse_order_row,                \
                                        m,                                  \
                                        n,                                  \
                                        k,                                  \
                                        nnz,                                \
                                        1,                                  \
                                        0,                                  \
                                        0,                                  \
                                        alpha,                              \
                                        descr,                              \
                                        csr_val,                            \
                                        csr_row_ptr,                        \
                                        csr_col_ind,                        \
                                        B,                                  \
                                        ldb,                                \
                                        1,                                  \
                                        0,                                  \
                                        beta,                               \
                                        C,                                  \
                                        ldc,                                \
                                        1,                                  \
                                        0,                                  \
                                        nullptr,                            \
                                        false);                             \
    }

IMPL(rocsparse_scsrmm_row_major, float);
IMPL(rocsparse_dcsrmm_row_major, double);
IMPL(rocsparse_ccsrmm_row_major, rocsparse_float_complex);
IMPL(rocsparse_zcsrmm_row_major, rocsparse_double_complex);

#undef IMPL

#endif // CSRMM_ROW_MAJOR_H
