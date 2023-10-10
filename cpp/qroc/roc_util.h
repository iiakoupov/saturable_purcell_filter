// Copyright (c) 2019-2023 Ivan Iakoupov
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

#ifndef ROC_UTIL_H
#define ROC_UTIL_H

#include <hip/hip_runtime_api.h>
#include <rocsparse/rocsparse.h>
#include "csrmmtrace_roc.h"
#include "csrmm_row_major.h"

//#define TEST_CSRMV_CODE_COPIED_FROM_ROCSPARSE

#ifdef TEST_CSRMV_CODE_COPIED_FROM_ROCSPARSE
#include "csrmv_roc.h"
#endif // TEST_CSRMV_CODE_COPIED_FROM_ROCSPARSE

typedef int64_t rocblas_stride;
// Load a scalar. If the argument is a pointer, dereference it; otherwise copy
// it. Allows the same kernels to be used for host and device scalars.

// For host scalars
template <typename T>
__forceinline__ __device__ __host__ T load_scalar(T x)
{
    return x;
}

// For device scalars
template <typename T>
__forceinline__ __device__ __host__ T load_scalar(const T* xp)
{
    return *xp;
}

// Load a pointer from a batch. If the argument is a T**, use block to index it and
// add the offset, if the argument is a T*, add block * stride to pointer and add offset.

// For device array of device pointers

// For device pointers
template <typename T>
__forceinline__ __device__ __host__ T*
                                    load_ptr_batch(T* p, rocsparse_int block, ptrdiff_t offset, rocblas_stride stride)
{
    return p + block * stride + offset;
}

// For device array of device pointers
template <typename T>
__forceinline__ __device__ __host__ T*
                                    load_ptr_batch(T* const* p, rocsparse_int block, ptrdiff_t offset, rocblas_stride stride)
{
    return p[block] + offset;
}

template <typename T>
__forceinline__ __device__ __host__ T*
                                    load_ptr_batch(T** p, rocsparse_int block, ptrdiff_t offset, rocblas_stride stride)
{
    return p[block] + offset;
}

template <typename A, typename X, typename Y>
__global__ void axpy_kernel(rocsparse_int    n,
                            A              alpha_device_host,
                            X              x,
                            rocsparse_int    incx,
                            ptrdiff_t      offsetx,
                            rocblas_stride stridex,
                            Y              y,
                            rocsparse_int    incy,
                            ptrdiff_t      offsety,
                            rocblas_stride stridey)
{
    auto alpha = load_scalar(alpha_device_host);
    ptrdiff_t tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if(tid < n)
    {
        auto tx = load_ptr_batch(x, hipBlockIdx_y, offsetx + tid * incx, stridex);
        auto ty = load_ptr_batch(y, hipBlockIdx_y, offsety + tid * incy, stridey);
        *ty += alpha * (*tx);
    }
}

template <int NB, typename A, typename X, typename Y>
static void axpy_template(rocsparse_handle handle,
                          rocsparse_int    n,
                          const A*       alpha,
                          X              x,
                          rocsparse_int    incx,
                          rocblas_stride stridex,
                          Y              y,
                          rocsparse_int    incy,
                          rocblas_stride stridey,
                          rocsparse_int    batch_count)
{
        ptrdiff_t offsetx = (incx < 0) ? ptrdiff_t(incx) * (1 - n) : 0;
        ptrdiff_t offsety = (incy < 0) ? ptrdiff_t(incy) * (1 - n) : 0;

        dim3 blocks((n - 1) / NB + 1, batch_count);
        dim3 threads(NB);
        hipStream_t stream;
        rocsparse_get_stream(handle, &stream);
        hipLaunchKernelGGL(axpy_kernel,
                           blocks,
                           threads,
                           0,
                           stream,
                           n,
                           *alpha,
                           x,
                           incx,
                           offsetx,
                           stridex,
                           y,
                           incy,
                           offsety,
                           stridey);
}

template <int NB, typename T>
static void rocblas_axpy_template(rocsparse_handle handle,
                                  rocsparse_int    n,
                                  const T*       alpha,
                                  const T*       x,
                                  rocsparse_int    incx,
                                  T*             y,
                                  rocsparse_int    incy)
{
    static constexpr rocblas_stride stride_0 = 0;
    static constexpr rocsparse_int batch_count_1 = 1;
    axpy_template<NB>(handle, n, alpha, x, incx, stride_0, y, incy, stride_0, batch_count_1);
}

inline void rocblas_zaxpy(rocsparse_handle handle,
                   rocsparse_int n,
                   const rocsparse_double_complex* alpha,
                   const rocsparse_double_complex* x,
                   rocsparse_int incx,
                   rocsparse_double_complex* y,
                   rocsparse_int incy)
{
    rocblas_axpy_template<256>(handle, n, alpha, x, incx, y, incy);
}

template <typename A, typename X, typename Y, typename Z>
__global__ void axpy_kernel_store_new(rocsparse_int    n,
                            A              alpha_device_host,
                            X              x,
                            rocsparse_int    incx,
                            ptrdiff_t      offsetx,
                            rocblas_stride stridex,
                            Y              y,
                            rocsparse_int    incy,
                            ptrdiff_t      offsety,
                            rocblas_stride stridey,
                            Z              z,
                            rocsparse_int    incz,
                            ptrdiff_t      offsetz,
                            rocblas_stride stridez)
{
    auto alpha = load_scalar(alpha_device_host);
    ptrdiff_t tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if(tid < n)
    {
        auto tx = load_ptr_batch(x, hipBlockIdx_y, offsetx + tid * incx, stridex);
        auto ty = load_ptr_batch(y, hipBlockIdx_y, offsety + tid * incy, stridey);
        auto tz = load_ptr_batch(z, hipBlockIdx_y, offsetz + tid * incz, stridez);
        *tz = alpha * (*tx) + (*ty);
    }
}

template <int NB, typename A, typename X, typename Y, typename Z>
static void axpy_template_store_new(rocsparse_handle handle,
                          rocsparse_int    n,
                          const A*       alpha,
                          X              x,
                          rocsparse_int    incx,
                          rocblas_stride stridex,
                          Y              y,
                          rocsparse_int    incy,
                          rocblas_stride stridey,
                          Z              z,
                          rocsparse_int    incz,
                          rocblas_stride stridez,
                          rocsparse_int    batch_count)
{
        ptrdiff_t offsetx = (incx < 0) ? ptrdiff_t(incx) * (1 - n) : 0;
        ptrdiff_t offsety = (incy < 0) ? ptrdiff_t(incy) * (1 - n) : 0;
        ptrdiff_t offsetz = (incz < 0) ? ptrdiff_t(incz) * (1 - n) : 0;

        dim3 blocks((n - 1) / NB + 1, batch_count);
        dim3 threads(NB);
        hipStream_t stream;
        rocsparse_get_stream(handle, &stream);
        hipLaunchKernelGGL(axpy_kernel_store_new,
                           blocks,
                           threads,
                           0,
                           stream,
                           n,
                           *alpha,
                           x,
                           incx,
                           offsetx,
                           stridex,
                           y,
                           incy,
                           offsety,
                           stridey,
                           z,
                           incz,
                           offsetz,
                           stridez);
}

template <int NB, typename T>
static void rocblas_axpy_template_store_new(rocsparse_handle handle,
                                  rocsparse_int    n,
                                  const T*       alpha,
                                  const T*       x,
                                  rocsparse_int    incx,
                                  const T*             y,
                                  rocsparse_int    incy,
                                  T*             z,
                                  rocsparse_int    incz)
{
    static constexpr rocblas_stride stride_0 = 0;
    static constexpr rocsparse_int batch_count_1 = 1;
    axpy_template_store_new<NB>(handle, n, alpha, x, incx, stride_0, y, incy, stride_0, z, incz, stride_0, batch_count_1);
}

inline void rocblas_zaxpy_store_new(rocsparse_handle handle,
                   rocsparse_int n,
                   const rocsparse_double_complex* alpha,
                   const rocsparse_double_complex* x,
                   rocsparse_int incx,
                   const rocsparse_double_complex* y,
                   rocsparse_int incy,
                   rocsparse_double_complex* z,
                   rocsparse_int incz)
{
    rocblas_axpy_template_store_new<256>(handle, n, alpha, x, incx, y, incy, z, incz);
}

template <typename A, typename D, typename V>
__global__ void sum2_kernel_store_new(rocsparse_int    n,
                            D              dst,
                            A              factor1_device_host,
                            V              term1,
                            A              factor2_device_host,
                            V              term2)
{
    auto factor1 = load_scalar(factor1_device_host);
    auto factor2 = load_scalar(factor2_device_host);
    ptrdiff_t tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if(tid < n)
    {
        auto tdst = load_ptr_batch(dst, hipBlockIdx_y, tid, 0);
        auto tterm1 = load_ptr_batch(term1, hipBlockIdx_y, tid, 0);
        auto tterm2 = load_ptr_batch(term2, hipBlockIdx_y, tid, 0);
        *tdst = factor1 * (*tterm1) + factor2 * (*tterm2);
    }
}

template <int NB, typename A, typename D, typename V>
static void sum2_template_store_new(rocsparse_handle handle,
                          rocsparse_int    n,
                          D              dst,
                          const A*       factor1,
                          V              term1,
                          const A*       factor2,
                          V              term2)
{
        dim3 blocks((n - 1) / NB + 1);
        dim3 threads(NB);
        hipStream_t stream;
        rocsparse_get_stream(handle, &stream);
        hipLaunchKernelGGL(sum2_kernel_store_new,
                           blocks,
                           threads,
                           0,
                           stream,
                           n,
                           dst,
                           *factor1,
                           term1,
                           *factor2,
                           term2);
}

template <int NB, typename T>
static void rocblas_sum2_template_store_new(rocsparse_handle handle,
                                  rocsparse_int    n,
                                  T*       dst,
                                  const T*       factor1,
                                  const T*       term1,
                                  const T*       factor2,
                                  const T*       term2)
{
    sum2_template_store_new<NB>(handle, n, dst, factor1, term1, factor2, term2);
}

inline void rocblas_zsum2_store_new(rocsparse_handle handle,
                   rocsparse_int n,
                   rocsparse_double_complex* dst,
                   const rocsparse_double_complex* factor1,
                   const rocsparse_double_complex* term1,
                   const rocsparse_double_complex* factor2,
                   const rocsparse_double_complex* term2)
{
    rocblas_sum2_template_store_new<256>(handle, n, dst, factor1, term1, factor2, term2);
}

template <typename A, typename D, typename V>
__global__ void sum3_kernel_store_new(rocsparse_int    n,
                            D              dst,
                            A              factor1_device_host,
                            V              term1,
                            A              factor2_device_host,
                            V              term2,
                            A              factor3_device_host,
                            V              term3)
{
    auto factor1 = load_scalar(factor1_device_host);
    auto factor2 = load_scalar(factor2_device_host);
    auto factor3 = load_scalar(factor3_device_host);
    ptrdiff_t tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if(tid < n)
    {
        auto tdst = load_ptr_batch(dst, hipBlockIdx_y, tid, 0);
        auto tterm1 = load_ptr_batch(term1, hipBlockIdx_y, tid, 0);
        auto tterm2 = load_ptr_batch(term2, hipBlockIdx_y, tid, 0);
        auto tterm3 = load_ptr_batch(term3, hipBlockIdx_y, tid, 0);
        *tdst = factor1 * (*tterm1) + factor2 * (*tterm2) + factor3 * (*tterm3);
    }
}

template <int NB, typename A, typename D, typename V>
static void sum3_template_store_new(rocsparse_handle handle,
                          rocsparse_int    n,
                          D              dst,
                          const A*       factor1,
                          V              term1,
                          const A*       factor2,
                          V              term2,
                          const A*       factor3,
                          V              term3)
{
        dim3 blocks((n - 1) / NB + 1);
        dim3 threads(NB);
        hipStream_t stream;
        rocsparse_get_stream(handle, &stream);
        hipLaunchKernelGGL(sum3_kernel_store_new,
                           blocks,
                           threads,
                           0,
                           stream,
                           n,
                           dst,
                           *factor1,
                           term1,
                           *factor2,
                           term2,
                           *factor3,
                           term3);
}

template <int NB, typename T>
static void rocblas_sum3_template_store_new(rocsparse_handle handle,
                                  rocsparse_int    n,
                                  T*       dst,
                                  const T*       factor1,
                                  const T*       term1,
                                  const T*       factor2,
                                  const T*       term2,
                                  const T*       factor3,
                                  const T*       term3)
{
    sum3_template_store_new<NB>(handle, n, dst, factor1, term1, factor2, term2, factor3, term3);
}

inline void rocblas_zsum3_store_new(rocsparse_handle handle,
                   rocsparse_int n,
                   rocsparse_double_complex* dst,
                   const rocsparse_double_complex* factor1,
                   const rocsparse_double_complex* term1,
                   const rocsparse_double_complex* factor2,
                   const rocsparse_double_complex* term2,
                   const rocsparse_double_complex* factor3,
                   const rocsparse_double_complex* term3)
{
    rocblas_sum3_template_store_new<256>(handle, n, dst, factor1, term1, factor2, term2, factor3, term3);
}

template <typename A, typename D, typename V>
__global__ void sum4_kernel_store_new(rocsparse_int    n,
                            D              dst,
                            A              factor1_device_host,
                            V              term1,
                            A              factor2_device_host,
                            V              term2,
                            A              factor3_device_host,
                            V              term3,
                            A              factor4_device_host,
                            V              term4)
{
    auto factor1 = load_scalar(factor1_device_host);
    auto factor2 = load_scalar(factor2_device_host);
    auto factor3 = load_scalar(factor3_device_host);
    auto factor4 = load_scalar(factor4_device_host);
    ptrdiff_t tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if(tid < n)
    {
        auto tdst = load_ptr_batch(dst, hipBlockIdx_y, tid, 0);
        auto tterm1 = load_ptr_batch(term1, hipBlockIdx_y, tid, 0);
        auto tterm2 = load_ptr_batch(term2, hipBlockIdx_y, tid, 0);
        auto tterm3 = load_ptr_batch(term3, hipBlockIdx_y, tid, 0);
        auto tterm4 = load_ptr_batch(term4, hipBlockIdx_y, tid, 0);
        *tdst = factor1 * (*tterm1) + factor2 * (*tterm2) + factor3 * (*tterm3) + factor4 * (*tterm4);
    }
}

template <int NB, typename A, typename D, typename V>
static void sum4_template_store_new(rocsparse_handle handle,
                          rocsparse_int    n,
                          D              dst,
                          const A*       factor1,
                          V              term1,
                          const A*       factor2,
                          V              term2,
                          const A*       factor3,
                          V              term3,
                          const A*       factor4,
                          V              term4)
{
        dim3 blocks((n - 1) / NB + 1);
        dim3 threads(NB);
        hipStream_t stream;
        rocsparse_get_stream(handle, &stream);
        hipLaunchKernelGGL(sum4_kernel_store_new,
                           blocks,
                           threads,
                           0,
                           stream,
                           n,
                           dst,
                           *factor1,
                           term1,
                           *factor2,
                           term2,
                           *factor3,
                           term3,
                           *factor4,
                           term4);
}

template <int NB, typename T>
static void rocblas_sum4_template_store_new(rocsparse_handle handle,
                                  rocsparse_int    n,
                                  T*       dst,
                                  const T*       factor1,
                                  const T*       term1,
                                  const T*       factor2,
                                  const T*       term2,
                                  const T*       factor3,
                                  const T*       term3,
                                  const T*       factor4,
                                  const T*       term4)
{
    sum4_template_store_new<NB>(handle, n, dst, factor1, term1, factor2, term2, factor3, term3, factor4, term4);
}

inline void rocblas_zsum4_store_new(rocsparse_handle handle,
                   rocsparse_int n,
                   rocsparse_double_complex* dst,
                   const rocsparse_double_complex* factor1,
                   const rocsparse_double_complex* term1,
                   const rocsparse_double_complex* factor2,
                   const rocsparse_double_complex* term2,
                   const rocsparse_double_complex* factor3,
                   const rocsparse_double_complex* term3,
                   const rocsparse_double_complex* factor4,
                   const rocsparse_double_complex* term4)
{
    rocblas_sum4_template_store_new<256>(handle, n, dst, factor1, term1, factor2, term2, factor3, term3, factor4, term4);
}

template <typename A, typename D, typename V>
__global__ void sum5_kernel_store_new(rocsparse_int    n,
                            D              dst,
                            A              factor1_device_host,
                            V              term1,
                            A              factor2_device_host,
                            V              term2,
                            A              factor3_device_host,
                            V              term3,
                            A              factor4_device_host,
                            V              term4,
                            A              factor5_device_host,
                            V              term5)
{
    auto factor1 = load_scalar(factor1_device_host);
    auto factor2 = load_scalar(factor2_device_host);
    auto factor3 = load_scalar(factor3_device_host);
    auto factor4 = load_scalar(factor4_device_host);
    auto factor5 = load_scalar(factor5_device_host);
    ptrdiff_t tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if(tid < n)
    {
        auto tdst = load_ptr_batch(dst, hipBlockIdx_y, tid, 0);
        auto tterm1 = load_ptr_batch(term1, hipBlockIdx_y, tid, 0);
        auto tterm2 = load_ptr_batch(term2, hipBlockIdx_y, tid, 0);
        auto tterm3 = load_ptr_batch(term3, hipBlockIdx_y, tid, 0);
        auto tterm4 = load_ptr_batch(term4, hipBlockIdx_y, tid, 0);
        auto tterm5 = load_ptr_batch(term5, hipBlockIdx_y, tid, 0);
        *tdst = factor1 * (*tterm1) + factor2 * (*tterm2) + factor3 * (*tterm3) + factor4 * (*tterm4) + factor5 * (*tterm5);
    }
}

template <int NB, typename A, typename D, typename V>
static void sum5_template_store_new(rocsparse_handle handle,
                          rocsparse_int    n,
                          D              dst,
                          const A*       factor1,
                          V              term1,
                          const A*       factor2,
                          V              term2,
                          const A*       factor3,
                          V              term3,
                          const A*       factor4,
                          V              term4,
                          const A*       factor5,
                          V              term5)
{
        dim3 blocks((n - 1) / NB + 1);
        dim3 threads(NB);
        hipStream_t stream;
        rocsparse_get_stream(handle, &stream);
        hipLaunchKernelGGL(sum5_kernel_store_new,
                           blocks,
                           threads,
                           0,
                           stream,
                           n,
                           dst,
                           *factor1,
                           term1,
                           *factor2,
                           term2,
                           *factor3,
                           term3,
                           *factor4,
                           term4,
                           *factor5,
                           term5);
}

template <int NB, typename T>
static void rocblas_sum5_template_store_new(rocsparse_handle handle,
                                  rocsparse_int    n,
                                  T*       dst,
                                  const T*       factor1,
                                  const T*       term1,
                                  const T*       factor2,
                                  const T*       term2,
                                  const T*       factor3,
                                  const T*       term3,
                                  const T*       factor4,
                                  const T*       term4,
                                  const T*       factor5,
                                  const T*       term5)
{
    sum5_template_store_new<NB>(handle, n, dst, factor1, term1, factor2, term2, factor3, term3, factor4, term4, factor5, term5);
}

inline void rocblas_zsum5_store_new(rocsparse_handle handle,
                   rocsparse_int n,
                   rocsparse_double_complex* dst,
                   const rocsparse_double_complex* factor1,
                   const rocsparse_double_complex* term1,
                   const rocsparse_double_complex* factor2,
                   const rocsparse_double_complex* term2,
                   const rocsparse_double_complex* factor3,
                   const rocsparse_double_complex* term3,
                   const rocsparse_double_complex* factor4,
                   const rocsparse_double_complex* term4,
                   const rocsparse_double_complex* factor5,
                   const rocsparse_double_complex* term5)
{
    rocblas_sum5_template_store_new<256>(handle, n, dst, factor1, term1, factor2, term2, factor3, term3, factor4, term4, factor5, term5);
}

template <typename A, typename D, typename V>
__global__ void sum7_kernel_store_new(rocsparse_int    n,
                            D              dst,
                            A              factor1_device_host,
                            V              term1,
                            A              factor2_device_host,
                            V              term2,
                            A              factor3_device_host,
                            V              term3,
                            A              factor4_device_host,
                            V              term4,
                            A              factor5_device_host,
                            V              term5,
                            A              factor6_device_host,
                            V              term6,
                            A              factor7_device_host,
                            V              term7)
{
    auto factor1 = load_scalar(factor1_device_host);
    auto factor2 = load_scalar(factor2_device_host);
    auto factor3 = load_scalar(factor3_device_host);
    auto factor4 = load_scalar(factor4_device_host);
    auto factor5 = load_scalar(factor5_device_host);
    auto factor6 = load_scalar(factor6_device_host);
    auto factor7 = load_scalar(factor7_device_host);
    ptrdiff_t tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if(tid < n)
    {
        auto tdst = load_ptr_batch(dst, hipBlockIdx_y, tid, 0);
        auto tterm1 = load_ptr_batch(term1, hipBlockIdx_y, tid, 0);
        auto tterm2 = load_ptr_batch(term2, hipBlockIdx_y, tid, 0);
        auto tterm3 = load_ptr_batch(term3, hipBlockIdx_y, tid, 0);
        auto tterm4 = load_ptr_batch(term4, hipBlockIdx_y, tid, 0);
        auto tterm5 = load_ptr_batch(term5, hipBlockIdx_y, tid, 0);
        auto tterm6 = load_ptr_batch(term6, hipBlockIdx_y, tid, 0);
        auto tterm7 = load_ptr_batch(term7, hipBlockIdx_y, tid, 0);
        *tdst = factor1 * (*tterm1) + factor2 * (*tterm2) + factor3 * (*tterm3) + factor4 * (*tterm4) + factor5 * (*tterm5) + factor6 * (*tterm6) + factor7 * (*tterm7);
    }
}

template <int NB, typename A, typename D, typename V>
static void sum7_template_store_new(rocsparse_handle handle,
                          rocsparse_int    n,
                          D              dst,
                          const A*       factor1,
                          V              term1,
                          const A*       factor2,
                          V              term2,
                          const A*       factor3,
                          V              term3,
                          const A*       factor4,
                          V              term4,
                          const A*       factor5,
                          V              term5,
                          const A*       factor6,
                          V              term6,
                          const A*       factor7,
                          V              term7)
{
        dim3 blocks((n - 1) / NB + 1);
        dim3 threads(NB);
        hipStream_t stream;
        rocsparse_get_stream(handle, &stream);
        hipLaunchKernelGGL(sum7_kernel_store_new,
                           blocks,
                           threads,
                           0,
                           stream,
                           n,
                           dst,
                           *factor1,
                           term1,
                           *factor2,
                           term2,
                           *factor3,
                           term3,
                           *factor4,
                           term4,
                           *factor5,
                           term5,
                           *factor6,
                           term6,
                           *factor7,
                           term7);
}

template <int NB, typename T>
static void rocblas_sum7_template_store_new(rocsparse_handle handle,
                                  rocsparse_int    n,
                                  T*       dst,
                                  const T*       factor1,
                                  const T*       term1,
                                  const T*       factor2,
                                  const T*       term2,
                                  const T*       factor3,
                                  const T*       term3,
                                  const T*       factor4,
                                  const T*       term4,
                                  const T*       factor5,
                                  const T*       term5,
                                  const T*       factor6,
                                  const T*       term6,
                                  const T*       factor7,
                                  const T*       term7)
{
    sum7_template_store_new<NB>(handle, n, dst, factor1, term1, factor2, term2, factor3, term3, factor4, term4, factor5, term5, factor6, term6, factor7, term7);
}

inline void rocblas_zsum7_store_new(rocsparse_handle handle,
                   rocsparse_int n,
                   rocsparse_double_complex* dst,
                   const rocsparse_double_complex* factor1,
                   const rocsparse_double_complex* term1,
                   const rocsparse_double_complex* factor2,
                   const rocsparse_double_complex* term2,
                   const rocsparse_double_complex* factor3,
                   const rocsparse_double_complex* term3,
                   const rocsparse_double_complex* factor4,
                   const rocsparse_double_complex* term4,
                   const rocsparse_double_complex* factor5,
                   const rocsparse_double_complex* term5,
                   const rocsparse_double_complex* factor6,
                   const rocsparse_double_complex* term6,
                   const rocsparse_double_complex* factor7,
                   const rocsparse_double_complex* term7)
{
    rocblas_sum7_template_store_new<256>(handle, n, dst, factor1, term1, factor2, term2, factor3, term3, factor4, term4, factor5, term5, factor6, term6, factor7, term7);
}

template <typename PSI, typename K>
__global__ void rk4_finalize_kernel(rocsparse_int n,
                                    PSI psi,
                                    K k1,
                                    K k2,
                                    K k3,
                                    K k4)
{
    ptrdiff_t tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if(tid < n)
    {
        auto tpsi = load_ptr_batch(psi, hipBlockIdx_y, tid, 0);
        auto tk1 = load_ptr_batch(k1, hipBlockIdx_y, tid, 0);
        auto tk2 = load_ptr_batch(k2, hipBlockIdx_y, tid, 0);
        auto tk3 = load_ptr_batch(k3, hipBlockIdx_y, tid, 0);
        auto tk4 = load_ptr_batch(k4, hipBlockIdx_y, tid, 0);
        //psi0 += (k1/6) + (k2/3) + (k3/3) + (k4/6);
        *tpsi += (*tk1)/6 + (*tk2)/3 + (*tk3)/3 + (*tk4)/6;
    }
}

template <int NB, typename PSI, typename K>
static void rk4_finalize_template(rocsparse_handle handle,
                                  rocsparse_int n,
                                  PSI psi,
                                  K k1,
                                  K k2,
                                  K k3,
                                  K k4)
{
        dim3 blocks((n - 1) / NB + 1);
        dim3 threads(NB);
        hipStream_t stream;
        rocsparse_get_stream(handle, &stream);
        hipLaunchKernelGGL(rk4_finalize_kernel,
                           blocks,
                           threads,
                           0,
                           stream,
                           n,
                           psi,
                           k1,
                           k2,
                           k3,
                           k4);
}

template <int NB, typename T>
static void rk4_finalize(rocsparse_handle handle,
                         rocsparse_int n,
                         T* psi,
                         const T* k1,
                         const T* k2,
                         const T* k3,
                         const T* k4)
{
    rk4_finalize_template<NB>(handle, n, psi, k1, k2, k3, k4);
}

inline void rk4_finalize_z(rocsparse_handle handle,
                    rocsparse_int n,
                    rocsparse_double_complex* psi,
                    const rocsparse_double_complex* k1,
                    const rocsparse_double_complex* k2,
                    const rocsparse_double_complex* k3,
                    const rocsparse_double_complex* k4)
{
    rk4_finalize<256>(handle, n, psi, k1, k2, k3, k4);
}

template <typename PSI, typename K>
__global__ void rk4_finalize_cache_kernel(rocsparse_int n,
                                    PSI psi,
                                    PSI psi_cache,
                                    K k1,
                                    K k2,
                                    K k3,
                                    K k4)
{
    ptrdiff_t tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if(tid < n)
    {
        auto tpsi = load_ptr_batch(psi, hipBlockIdx_y, tid, 0);
        auto tpsi_cache = load_ptr_batch(psi_cache, hipBlockIdx_y, tid, 0);
        auto tk1 = load_ptr_batch(k1, hipBlockIdx_y, tid, 0);
        auto tk2 = load_ptr_batch(k2, hipBlockIdx_y, tid, 0);
        auto tk3 = load_ptr_batch(k3, hipBlockIdx_y, tid, 0);
        auto tk4 = load_ptr_batch(k4, hipBlockIdx_y, tid, 0);
        //psi0 += (k1/6) + (k2/3) + (k3/3) + (k4/6);
        *tpsi_cache = *tpsi;
        *tpsi += (*tk1)/6 + (*tk2)/3 + (*tk3)/3 + (*tk4)/6;
    }
}

template <int NB, typename PSI, typename K>
static void rk4_finalize_cache_template(rocsparse_handle handle,
                                  rocsparse_int n,
                                  PSI psi,
                                  PSI psi_cache,
                                  K k1,
                                  K k2,
                                  K k3,
                                  K k4)
{
        dim3 blocks((n - 1) / NB + 1);
        dim3 threads(NB);
        hipStream_t stream;
        rocsparse_get_stream(handle, &stream);
        hipLaunchKernelGGL(rk4_finalize_cache_kernel,
                           blocks,
                           threads,
                           0,
                           stream,
                           n,
                           psi,
                           psi_cache,
                           k1,
                           k2,
                           k3,
                           k4);
}

template <int NB, typename T>
static void rk4_finalize_cache(rocsparse_handle handle,
                         rocsparse_int n,
                         T* psi,
                         T* psi_cache,
                         const T* k1,
                         const T* k2,
                         const T* k3,
                         const T* k4)
{
    rk4_finalize_cache_template<NB>(handle, n, psi, psi_cache, k1, k2, k3, k4);
}

inline void rk4_finalize_cache_z(rocsparse_handle handle,
                    rocsparse_int n,
                    rocsparse_double_complex* psi,
                    rocsparse_double_complex* psi_cache,
                    const rocsparse_double_complex* k1,
                    const rocsparse_double_complex* k2,
                    const rocsparse_double_complex* k3,
                    const rocsparse_double_complex* k4)
{
    rk4_finalize_cache<256>(handle, n, psi, psi_cache, k1, k2, k3, k4);
}

// This function should take precedence over the operator=*
// method in the rocsparse_complex_num<double>
// (aka rocsparse_double_complex) class which is only defined
// with an argument of type rocsparse_complex_num<double>. Thus,
// that method would first convert double to
// rocsparse_complex_num<double> implicitly and then multiply
// two numbers of the type rocsparse_complex_num<double>.
// Defining this function should avoid half of the arithmetic
// operations.
__forceinline__ __device__ __host__ rocsparse_double_complex operator*(const rocsparse_double_complex &a, double b)
{
    return rocsparse_double_complex(std::real(a)*b, std::imag(a)*b);
}
// The same operator*() specialization, but the different order
__forceinline__ __device__ __host__ rocsparse_double_complex operator*(double a, const rocsparse_double_complex &b)
{
    return rocsparse_double_complex(a*std::real(b), a*std::imag(b));
}


template <typename A, typename X, typename Y>
__global__ void coeffwise_mul_kernel(rocsparse_int    n,
                            A              alpha_device_host,
                            X              x,
                            rocsparse_int    incx,
                            ptrdiff_t      offsetx,
                            rocblas_stride stridex,
                            Y              y,
                            rocsparse_int    incy,
                            ptrdiff_t      offsety,
                            rocblas_stride stridey)
{
    auto alpha = load_scalar(alpha_device_host);
    ptrdiff_t tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if(tid < n)
    {
        auto tx = load_ptr_batch(x, hipBlockIdx_y, offsetx + tid * incx, stridex);
        auto ty = load_ptr_batch(y, hipBlockIdx_y, offsety + tid * incy, stridey);
        // This is where the above operator* function
        // should be useful, when alpha is
        // rocsparse_double_complex and *tx is double.
        *ty *= alpha * (*tx);
    }
}

template <int NB, typename A, typename X, typename Y>
static void coeffwise_mul_template(rocsparse_handle handle,
                          rocsparse_int    n,
                          const A*       alpha,
                          X              x,
                          rocsparse_int    incx,
                          rocblas_stride stridex,
                          Y              y,
                          rocsparse_int    incy,
                          rocblas_stride stridey,
                          rocsparse_int    batch_count)
{
        ptrdiff_t offsetx = (incx < 0) ? ptrdiff_t(incx) * (1 - n) : 0;
        ptrdiff_t offsety = (incy < 0) ? ptrdiff_t(incy) * (1 - n) : 0;

        dim3 blocks((n - 1) / NB + 1, batch_count);
        dim3 threads(NB);
        hipStream_t stream;
        rocsparse_get_stream(handle, &stream);
        hipLaunchKernelGGL(coeffwise_mul_kernel,
                           blocks,
                           threads,
                           0,
                           stream,
                           n,
                           *alpha,
                           x,
                           incx,
                           offsetx,
                           stridex,
                           y,
                           incy,
                           offsety,
                           stridey);
}

template <int NB, typename T, typename TX>
static void rocblas_coeffwise_mul_template(rocsparse_handle handle,
                                  rocsparse_int    n,
                                  const T*       alpha,
                                  const TX*       x,
                                  rocsparse_int    incx,
                                  T*             y,
                                  rocsparse_int    incy)
{
    static constexpr rocblas_stride stride_0 = 0;
    static constexpr rocsparse_int batch_count_1 = 1;
    coeffwise_mul_template<NB>(handle, n, alpha, x, incx, stride_0, y, incy, stride_0, batch_count_1);
}

inline void rocblas_z_coeffwise_mul(rocsparse_handle handle,
                   rocsparse_int n,
                   const rocsparse_double_complex* alpha,
                   const rocsparse_double_complex* x,
                   rocsparse_int incx,
                   rocsparse_double_complex* y,
                   rocsparse_int incy)
{
    rocblas_coeffwise_mul_template<256>(handle, n, alpha, x, incx, y, incy);
}

inline void rocblas_zd_coeffwise_mul(rocsparse_handle handle,
                   rocsparse_int n,
                   const rocsparse_double_complex* alpha,
                   const double* x,
                   rocsparse_int incx,
                   rocsparse_double_complex* y,
                   rocsparse_int incy)
{
    rocblas_coeffwise_mul_template<256>(handle, n, alpha, x, incx, y, incy);
}

template <typename A, typename X, typename Y, typename Z>
__global__ void coeffwise_mul_kernel_store_new(rocsparse_int    n,
                            A              alpha_device_host,
                            X              x,
                            rocsparse_int    incx,
                            ptrdiff_t      offsetx,
                            rocblas_stride stridex,
                            Y              y,
                            rocsparse_int    incy,
                            ptrdiff_t      offsety,
                            rocblas_stride stridey,
                            Z              z,
                            rocsparse_int    incz,
                            ptrdiff_t      offsetz,
                            rocblas_stride stridez)
{
    auto alpha = load_scalar(alpha_device_host);
    ptrdiff_t tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if(tid < n)
    {
        auto tx = load_ptr_batch(x, hipBlockIdx_y, offsetx + tid * incx, stridex);
        auto ty = load_ptr_batch(y, hipBlockIdx_y, offsety + tid * incy, stridey);
        auto tz = load_ptr_batch(z, hipBlockIdx_y, offsetz + tid * incz, stridez);
        // This is where the above operator* function
        // should be useful, when alpha is
        // rocsparse_double_complex and *tx is double.
        *tz = alpha * (*tx) * (*ty);
    }
}

template <int NB, typename A, typename X, typename Y, typename Z>
static void coeffwise_mul_template_store_new(rocsparse_handle handle,
                          rocsparse_int    n,
                          const A*       alpha,
                          X              x,
                          rocsparse_int    incx,
                          rocblas_stride stridex,
                          Y              y,
                          rocsparse_int    incy,
                          rocblas_stride stridey,
                          Z              z,
                          rocsparse_int    incz,
                          rocblas_stride stridez,
                          rocsparse_int    batch_count)
{
        ptrdiff_t offsetx = (incx < 0) ? ptrdiff_t(incx) * (1 - n) : 0;
        ptrdiff_t offsety = (incy < 0) ? ptrdiff_t(incy) * (1 - n) : 0;
        ptrdiff_t offsetz = (incz < 0) ? ptrdiff_t(incz) * (1 - n) : 0;

        dim3 blocks((n - 1) / NB + 1, batch_count);
        dim3 threads(NB);
        hipStream_t stream;
        rocsparse_get_stream(handle, &stream);
        hipLaunchKernelGGL(coeffwise_mul_kernel_store_new,
                           blocks,
                           threads,
                           0,
                           stream,
                           n,
                           *alpha,
                           x,
                           incx,
                           offsetx,
                           stridex,
                           y,
                           incy,
                           offsety,
                           stridey,
                           z,
                           incz,
                           offsetz,
                           stridez);
}

template <int NB, typename T, typename TX, typename TZ>
static void rocblas_coeffwise_mul_template_store_new(rocsparse_handle handle,
                                  rocsparse_int    n,
                                  const T*       alpha,
                                  const TX*       x,
                                  rocsparse_int    incx,
                                  const T*             y,
                                  rocsparse_int    incy,
                                  TZ*             z,
                                  rocsparse_int    incz)
{
    static constexpr rocblas_stride stride_0 = 0;
    static constexpr rocsparse_int batch_count_1 = 1;
    coeffwise_mul_template_store_new<NB>(handle, n, alpha, x, incx, stride_0, y, incy, stride_0, z, incz, stride_0, batch_count_1);
}

inline void rocblas_z_coeffwise_mul_store_new(rocsparse_handle handle,
                   rocsparse_int n,
                   const rocsparse_double_complex* alpha,
                   const rocsparse_double_complex* x,
                   rocsparse_int incx,
                   const rocsparse_double_complex* y,
                   rocsparse_int incy,
                   rocsparse_double_complex* z,
                   rocsparse_int incz)
{
    rocblas_coeffwise_mul_template_store_new<256>(handle, n, alpha, x, incx, y, incy, z, incz);
}

inline void rocblas_zd_coeffwise_mul_store_new(rocsparse_handle handle,
                   rocsparse_int n,
                   const rocsparse_double_complex* alpha,
                   const double* x,
                   rocsparse_int incx,
                   const rocsparse_double_complex* y,
                   rocsparse_int incy,
                   rocsparse_double_complex* z,
                   rocsparse_int incz)
{
    rocblas_coeffwise_mul_template_store_new<256>(handle, n, alpha, x, incx, y, incy, z, incz);
}

template <typename A, typename X, typename Y>
__global__ void scale_kernel_store_new(rocsparse_int    n,
                            A              alpha_device_host,
                            X              x,
                            rocsparse_int    incx,
                            ptrdiff_t      offsetx,
                            rocblas_stride stridex,
                            Y              y,
                            rocsparse_int    incy,
                            ptrdiff_t      offsety,
                            rocblas_stride stridey)
{
    auto alpha = load_scalar(alpha_device_host);
    ptrdiff_t tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if(tid < n)
    {
        auto tx = load_ptr_batch(x, hipBlockIdx_y, offsetx + tid * incx, stridex);
        auto ty = load_ptr_batch(y, hipBlockIdx_y, offsety + tid * incy, stridey);
        *ty = alpha * (*tx);
    }
}

template <int NB, typename A, typename X, typename Y>
static void scale_template_store_new(rocsparse_handle handle,
                          rocsparse_int    n,
                          const A*       alpha,
                          X              x,
                          rocsparse_int    incx,
                          rocblas_stride stridex,
                          Y              y,
                          rocsparse_int    incy,
                          rocblas_stride stridey,
                          rocsparse_int    batch_count)
{
        ptrdiff_t offsetx = (incx < 0) ? ptrdiff_t(incx) * (1 - n) : 0;
        ptrdiff_t offsety = (incy < 0) ? ptrdiff_t(incy) * (1 - n) : 0;

        dim3 blocks((n - 1) / NB + 1, batch_count);
        dim3 threads(NB);
        hipStream_t stream;
        rocsparse_get_stream(handle, &stream);
        hipLaunchKernelGGL(scale_kernel_store_new,
                           blocks,
                           threads,
                           0,
                           stream,
                           n,
                           *alpha,
                           x,
                           incx,
                           offsetx,
                           stridex,
                           y,
                           incy,
                           offsety,
                           stridey);
}

template <int NB, typename TY, typename TX, typename TA>
static void rocblas_scale_template_store_new(rocsparse_handle handle,
                                  rocsparse_int    n,
                                  const TA*       alpha,
                                  const TX*       x,
                                  rocsparse_int    incx,
                                  TY*             y,
                                  rocsparse_int    incy)
{
    static constexpr rocblas_stride stride_0 = 0;
    static constexpr rocsparse_int batch_count_1 = 1;
    scale_template_store_new<NB>(handle, n, alpha, x, incx, stride_0, y, incy, stride_0, batch_count_1);
}

inline void rocblas_z_scale_store_new(rocsparse_handle handle,
                   rocsparse_int n,
                   const double* alpha,
                   const rocsparse_double_complex* x,
                   rocsparse_int incx,
                   rocsparse_double_complex* y,
                   rocsparse_int incy)
{
    rocblas_scale_template_store_new<256>(handle, n, alpha, x, incx, y, incy);
}

// BLAS Level 1 includes routines and functions performing vector-vector
// operations. Most BLAS 1 routines are about reduction: compute the norm,
// calculate the dot production of two vectors, find the maximum/minimum index
// of the element of the vector. As you may observed, although the computation
// type is different, the core algorithm is the same: scan all element of the
// vector(s) and reduce to one single result.
//
// The reduction algorithm on GPU is called [parallel
// reduction](https://raw.githubusercontent.com/mateuszbuda/GPUExample/master/reduce3.png)
// which is adopted in rocBLAS. At the beginning, all the threads in the thread
// block participate. After each step of reduction (like a tree), the number of
// participating threads decrease by half. At the end of the parallel reduction,
// only one thread (usually thread 0) owns the result in its thread block.
//
// Classically, the BLAS 1 reduction needs more than one GPU kernel to finish,
// because the lack of global synchronization of thread blocks without exiting
// the kernel. The first kernels gather partial results, write into a temporary
// working buffer. The second kernel finishes the final reduction.
//
// For example, BLAS 1 routine i*amax is to find index of the maximum absolute
// value element of a vector. In this routine:
//
// Kernel 1: launch many thread block as needed. Each thread block works on a
// subset of the vector. Each thread block use the parallel reduction to find a
// local index with the maximum absolute value of the subset. There are
// number-of-the-thread-blocks local results.The results are written into a
// temporary working buffer. The working buffer has number-of-the-thread-blocks
// elements.
//
// Kernel 2: launch only one thread block which reads the temporary work buffer and
// reduces to final result still with the parallel reduction.
//
// As you may see, if there is a mechanism to synchronize all the thread blocks
// after local index is obtained in kernel 1 (without ending the kernel), then
// Kernel 2's computation can be merged into Kernel 1. One such mechanism is called
// atomic operation. However, atomic operation is new and is not used in rocBLAS
// yet. rocBLAS still use the classic standard parallel reduction right now.

// Recursively compute reduction
template <rocsparse_int k, typename REDUCE, typename T>
struct rocblas_reduction_s
{
    __forceinline__ __device__ void operator()(rocsparse_int tx, T* x)
    {
        // Reduce the lower half with the upper half
        if(tx < k)
            REDUCE{}(x[tx], x[tx + k]);
        __syncthreads();

        // Recurse down with k / 2
        rocblas_reduction_s<k / 2, REDUCE, T>{}(tx, x);
    }
};

// leaf node for terminating recursion
template <typename REDUCE, typename T>
struct rocblas_reduction_s<0, REDUCE, T>
{
    __forceinline__ __device__ void operator()(rocsparse_int tx, T* x) {}
};

/*! \brief general parallel reduction

    \details

    @param[in]
    n         rocsparse_int. assume a power of 2
    @param[in]
    T         element type of vector x
    @param[in]
    REDUCE    reduction functor
    @param[in]
    tx        rocsparse_int. thread id
    @param[inout]
    x         pointer storing vector x on the GPU.
              usually x is stored in shared memory;
              x[0] store the final result.
    ********************************************************************/
template <rocsparse_int NB, typename REDUCE, typename T>
__attribute__((flatten)) __device__ void rocblas_reduction(rocsparse_int tx, T* x)
{
    static_assert(NB > 1 && !(NB & (NB - 1)), "NB must be a power of 2");
    __syncthreads();
    rocblas_reduction_s<NB / 2, REDUCE, T>{}(tx, x);
}

/*! \brief parallel reduction: sum

    \details

    @param[in]
    n         rocsparse_int. assume a power of 2
    @param[in]
    tx        rocsparse_int. thread id
    @param[inout]
    x         pointer storing vector x on the GPU.
              usually x is stored in shared memory;
              x[0] store the final result.
    ********************************************************************/
struct rocblas_reduce_sum
{
    template <typename T>
    __forceinline__ __device__ void operator()(T& __restrict__ a, const T& __restrict__ b)
    {
        a += b;
    }
};

template <rocsparse_int NB, typename T>
__attribute__((flatten)) __device__ void rocblas_sum_reduce(rocsparse_int tx, T* x)
{
    rocblas_reduction<NB, rocblas_reduce_sum>(tx, x);
}
// end sum_reduce

template <rocsparse_int NB, bool CONJ, typename T, typename U, typename V = T>
__global__ void dot_kernel_part1(rocsparse_int    n,
                                 const U        xa,
                                 ptrdiff_t      shiftx,
                                 rocsparse_int    incx,
                                 rocblas_stride stridex,
                                 const U        ya,
                                 ptrdiff_t      shifty,
                                 rocsparse_int    incy,
                                 rocsparse_int    stridey,
                                 V*             workspace)
{
    ptrdiff_t tx  = hipThreadIdx_x;
    ptrdiff_t tid = hipBlockIdx_x * hipBlockDim_x + tx;

    __shared__ V tmp[NB];
    const T*     x;
    const T*     y;

    // bound
    if(tid < n)
    {
        x       = load_ptr_batch(xa, hipBlockIdx_y, shiftx, stridex);
        y       = load_ptr_batch(ya, hipBlockIdx_y, shifty, stridey);
        tmp[tx] = V(y[tid * incy]) * V(CONJ ? std::conj(x[tid * incx]) : x[tid * incx]);
    }
    else
        tmp[tx] = V(0); // pad with zero

    rocblas_sum_reduce<NB>(tx, tmp);

    if(tx == 0)
        workspace[hipBlockIdx_x + hipBlockIdx_y * hipGridDim_x] = tmp[0];
}

inline size_t rocblas_reduction_kernel_block_count(rocsparse_int n, rocsparse_int NB)
{
    if(n <= 0)
        n = 1; // avoid sign loss issues
    return size_t(n - 1) / NB + 1;
}

/*! \brief rocblas_reduction_batched_kernel_workspace_size
    Work area for reduction must be at lease sizeof(To) * (blocks + 1) * batch_count

    @param[in]
    outputType To*
        Type of output values
    @param[in]
    batch_count rocsparse_int
        Number of batches
    ********************************************************************/
template <rocsparse_int NB, typename To>
size_t rocblas_reduction_kernel_workspace_size(rocsparse_int n, rocsparse_int batch_count = 1)
{
    if(n <= 0)
        n = 1; // allow for return value of empty set
    if(batch_count <= 0)
        batch_count = 1;
    auto blocks = rocblas_reduction_kernel_block_count(n, NB);
    return sizeof(To) * (blocks + 1) * batch_count;
}

// Emulates value initialization T{}. Allows specialization for certain types.
template <typename T>
struct rocblas_default_value
{
    __forceinline__ __host__ __device__ constexpr T operator()() const
    {
        return {};
    }
};

// Identity finalizer
struct rocblas_finalize_identity
{
    template <typename T>
    __forceinline__ __host__ __device__ T&& operator()(T&& x)
    {
        return std::forward<T>(x); // Perfect identity, preserving valueness
    }
};

// kernel 2 is used from non-strided reduction_batched see include file
// kernel 2 gathers all the partial results in workspace and finishes the final reduction;
// number of threads (NB) loop blocks
template <rocsparse_int NB,
          typename REDUCE   = rocblas_reduce_sum,
          typename FINALIZE = rocblas_finalize_identity,
          typename To,
          typename Tr>
__attribute__((amdgpu_flat_work_group_size((NB < 128) ? NB : 128, (NB > 256) ? NB : 256)))
__global__ void
    rocblas_reduction_strided_batched_kernel_part2(rocsparse_int nblocks, To* workspace, Tr* result)
{
    rocsparse_int   tx = hipThreadIdx_x;
    __shared__ To tmp[NB];

    if(tx < nblocks)
    {
        To* work = workspace + hipBlockIdx_y * nblocks;
        tmp[tx]  = work[tx];

        // bound, loop
        for(rocsparse_int i = tx + NB; i < nblocks; i += NB)
            REDUCE{}(tmp[tx], work[i]);
    }
    else
    { // pad with default value
        tmp[tx] = rocblas_default_value<To>{}();
    }

    if(nblocks < 32)
    {
        // no need parallel reduction
        __syncthreads();

        if(tx == 0)
            for(rocsparse_int i = 1; i < nblocks; i++)
                REDUCE{}(tmp[0], tmp[i]);
    }
    else
    {
        // parallel reduction
        rocblas_reduction<NB, REDUCE>(tx, tmp);
    }

    // Store result on device or in workspace
    if(tx == 0)
        result[hipBlockIdx_y] = Tr(FINALIZE{}(tmp[0]));
}

// assume workspace has already been allocated, recommened for repeated calling of dot_strided_batched product
// routine
template <rocsparse_int NB, bool CONJ, typename T, typename U, typename V = T>
void rocblas_dot_template(rocsparse_handle __restrict__ handle,
                          rocsparse_int    n,
                          const U        x,
                          rocsparse_int    offsetx,
                          rocsparse_int    incx,
                          rocblas_stride stridex,
                          const U        y,
                          rocsparse_int    offsety,
                          rocsparse_int    incy,
                          rocblas_stride stridey,
                          rocsparse_int    batch_count,
                          T*             results,
                          V*             workspace)
{
    // At least two kernels are needed to finish the reduction
    // kennel 1 write partial results per thread block in workspace, number of partial results is
    // blocks
    // kernel 2 gather all the partial results in workspace and finish the final reduction. number of
    // threads (NB) loop blocks

    // in case of negative inc shift pointer to end of data for negative indexing tid*inc
    auto shiftx = incx < 0 ? offsetx - ptrdiff_t(incx) * (n - 1) : offsetx;
    auto shifty = incy < 0 ? offsety - ptrdiff_t(incy) * (n - 1) : offsety;

    rocsparse_int blocks = rocblas_reduction_kernel_block_count(n, NB);
    dim3        grid(blocks, batch_count);
    dim3        threads(NB);

    hipStream_t stream;
    rocsparse_get_stream(handle, &stream);
    hipLaunchKernelGGL((dot_kernel_part1<NB, CONJ, T>),
                       grid,
                       threads,
                       0,
                       stream,
                       n,
                       x,
                       shiftx,
                       incx,
                       stridex,
                       y,
                       shifty,
                       incy,
                       stridey,
                       workspace);
    //Here we assume that "results" is in the device memory
    hipLaunchKernelGGL(rocblas_reduction_strided_batched_kernel_part2<NB>,
                       dim3(1, batch_count),
                       threads,
                       0,
                       stream,
                       blocks,
                       workspace,
                       results);
    //The commented out version is for the case when "results"
    //is in the host memory.
    //This is why the values are first put in the "workspace"
    //memory (which is on the device) and then copied to the
    //host memory.
    //hipLaunchKernelGGL(rocblas_reduction_strided_batched_kernel_part2<NB>,
    //                   dim3(1, batch_count),
    //                   threads,
    //                   0,
    //                   stream,
    //                   blocks,
    //                   workspace,
    //                   (V*)(workspace + size_t(batch_count) * blocks));

    //// result is in the beginning of workspace[0]+offset
    //size_t offset = size_t(batch_count) * blocks;
    //V      res_V[batch_count];
    //hipMemcpy(res_V, workspace + offset, sizeof(V) * batch_count, hipMemcpyDeviceToHost);
    //for(rocsparse_int i = 0; i < batch_count; i++)
    //    results[i] = T(res_V[i]);
}

// HIP support up to 1024 threads/work itemes per thread block/work group
// setting to 512 for gfx803.
#define ROCM_DOT_NB 512

inline void rocblas_zdotc(rocsparse_handle handle,
                   rocsparse_int n,
                   const rocsparse_double_complex* x,
                   rocsparse_int incx,
                   const rocsparse_double_complex* y,
                   rocsparse_int incy,
                   rocsparse_double_complex* result,
                   rocsparse_double_complex* mem)
{
    rocblas_dot_template<ROCM_DOT_NB, true, rocsparse_double_complex>(
        handle, n, x, 0, incx, 0, y, 0, incy, 0, 1, result, mem);
}

__global__ void
add_adjoint_in_place_kernel(rocsparse_double_complex* __restrict__ M, int size)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    const int size_squared = size*size;
    int index1 = x * size + y;
    int index2 = y * size + x;
    if (x == y && index1 < size_squared) {
        M[index1] = 2*std::real(M[index1]);
    } else if (y > x && index1 < size_squared && index2 < size_squared) {
        const rocsparse_double_complex M_sum = M[index1] + std::conj(M[index2]);
        M[index1] = M_sum;
        M[index2] = std::conj(M_sum);
    }
}

void addAdjointInPlace(rocsparse_handle handle, rocsparse_double_complex* __restrict__ rho, int basis_size)
{
    const int threads_per_block = 16;
    hipStream_t stream;
    rocsparse_get_stream(handle, &stream);
    const int num_blocks = (basis_size - 1)/threads_per_block + 1;
    hipLaunchKernelGGL(add_adjoint_in_place_kernel,
                       dim3(num_blocks, num_blocks),
                       dim3(threads_per_block, threads_per_block),
                       0, stream,
                       rho, basis_size);
}

__global__ void
add_adjoint_in_place_with_transpose_kernel(rocsparse_double_complex* __restrict__ M, int size)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    const int size_squared = size*size;
    int index1 = x * size + y;
    int index2 = y * size + x;
    if (x == y && index1 < size_squared) {
        M[index1] = 2*std::real(M[index1]);
    } else if (y > x && index1 < size_squared && index2 < size_squared) {
        const rocsparse_double_complex M_sum = std::conj(M[index1]) + M[index2];
        M[index1] = M_sum;
        M[index2] = std::conj(M_sum);
    }
}

void addAdjointInPlaceWithTranspose(rocsparse_handle handle, rocsparse_double_complex* __restrict__ rho, int basis_size)
{
    const int threads_per_block = 16;
    hipStream_t stream;
    rocsparse_get_stream(handle, &stream);
    const int num_blocks = (basis_size - 1)/threads_per_block + 1;
    hipLaunchKernelGGL(add_adjoint_in_place_with_transpose_kernel,
                       dim3(num_blocks, num_blocks),
                       dim3(threads_per_block, threads_per_block),
                       0, stream,
                       rho, basis_size);
}

class ROCArrayXdOnDevice
{
    double* m_data;
    int m_size;
public:
    ROCArrayXdOnDevice() :
        m_data(nullptr),
        m_size(0)
    {}
    explicit ROCArrayXdOnDevice(int size) :
        m_size(size)
    {
        hipMalloc((void**)&m_data, sizeof(double) * size);
    }
    ROCArrayXdOnDevice(const double *host_data, int size) :
        m_size(size)
    {
        hipMalloc((void**)&m_data, sizeof(double) * size);
        hipMemcpy(m_data, host_data, sizeof(double) * size, hipMemcpyHostToDevice);
    }
    // noexcept is necessary for the STL containers (e.g. std::vector) to use
    // this move-constructor instead of the copy-constructor
    ROCArrayXdOnDevice(ROCArrayXdOnDevice &&other) noexcept :
        m_data(other.m_data),
        m_size(other.m_size)
    {
        other.m_data = nullptr;
    }
    ROCArrayXdOnDevice(const ROCArrayXdOnDevice &other) :
        m_size(other.m_size)
    {
        hipMalloc((void**)&m_data, sizeof(double) * m_size);
        hipMemcpy(m_data, other.m_data, sizeof(double) * m_size, hipMemcpyDeviceToDevice);
    }
    ROCArrayXdOnDevice &operator=(const ROCArrayXdOnDevice &other)
    {
        if (this == &other) {
            return *this;
        }
        if (m_size != other.m_size) {
            hipFree(m_data);
            m_data = nullptr;
            m_size = other.m_size;
            hipMalloc((void**)&m_data, sizeof(double) * m_size);
            hipMemcpy(m_data, other.m_data, sizeof(double) * m_size, hipMemcpyDeviceToDevice);
        }
        return *this;
    }
    ROCArrayXdOnDevice &operator=(ROCArrayXdOnDevice &&other)
    {
        if (this == &other) {
            return *this;
        }
        hipFree(m_data);
        m_data = other.m_data;
        other.m_data = nullptr;
        m_size = other.m_size;
        return *this;
    }
    double* data() const
    {
        return m_data;
    }
    ~ROCArrayXdOnDevice()
    {
        hipFree(m_data);
    }
};

class ROCArrayXcdOnDevice
{
    rocsparse_double_complex* m_data;
    int m_size;
public:
    ROCArrayXcdOnDevice() :
        m_data(nullptr),
        m_size(0)
    {}
    explicit ROCArrayXcdOnDevice(int size) :
        m_size(size)
    {
        hipMalloc((void**)&m_data, sizeof(rocsparse_double_complex) * size);
    }
    ROCArrayXcdOnDevice(const std::complex<double> *host_data, int size) :
        m_size(size)
    {
        std::vector<rocsparse_double_complex> hdata(size);
        for (int i = 0; i < size; ++i) {
            hdata[i] = rocsparse_double_complex(host_data[i].real(),
                                                host_data[i].imag());
        }
        hipMalloc((void**)&m_data, sizeof(rocsparse_double_complex) * size);
        hipMemcpy(m_data, hdata.data(), sizeof(rocsparse_double_complex) * size, hipMemcpyHostToDevice);
    }
    // noexcept is necessary for the STL containers (e.g. std::vector) to use
    // this move-constructor instead of the copy-constructor
    ROCArrayXcdOnDevice(ROCArrayXcdOnDevice &&other) noexcept :
        m_data(other.m_data),
        m_size(other.m_size)
    {
        other.m_data = nullptr;
    }
    ROCArrayXcdOnDevice(const ROCArrayXcdOnDevice &other) :
        m_size(other.m_size)
    {
        hipMalloc((void**)&m_data, sizeof(rocsparse_double_complex) * m_size);
        hipMemcpy(m_data, other.m_data, sizeof(rocsparse_double_complex) * m_size, hipMemcpyDeviceToDevice);
    }
    ROCArrayXcdOnDevice &operator=(const ROCArrayXcdOnDevice &other)
    {
        if (this == &other) {
            return *this;
        }
        if (m_size != other.m_size) {
            hipFree(m_data);
            m_data = nullptr;
            m_size = other.m_size;
            hipMalloc((void**)&m_data, sizeof(rocsparse_double_complex) * m_size);
            hipMemcpy(m_data, other.m_data, sizeof(rocsparse_double_complex) * m_size, hipMemcpyDeviceToDevice);
        }
        return *this;
    }
    ROCArrayXcdOnDevice &operator=(ROCArrayXcdOnDevice &&other)
    {
        if (this == &other) {
            return *this;
        }
        hipFree(m_data);
        m_data = other.m_data;
        other.m_data = nullptr;
        m_size = other.m_size;
        return *this;
    }
    rocsparse_double_complex* data() const
    {
        return m_data;
    }
    void copyToDevice(const std::complex<double> *host_data, int size) const
    {
        assert(size <= m_size
               && "Host data to be copied does not fit into device array!");
        std::vector<rocsparse_double_complex> hdata(size);
        for (int i = 0; i < size; ++i) {
            hdata[i] = rocsparse_double_complex(host_data[i].real(),
                                                host_data[i].imag());
        }
        hipMemcpy(m_data, hdata.data(), sizeof(rocsparse_double_complex) * size, hipMemcpyHostToDevice);
    }
    ~ROCArrayXcdOnDevice()
    {
        hipFree(m_data);
    }
};

class ROCSparseHandleRAII
{
    rocsparse_handle m_handle;
public:
    ROCSparseHandleRAII()
    {
        rocsparse_create_handle(&m_handle);
    }
    ~ROCSparseHandleRAII()
    {
        rocsparse_destroy_handle(m_handle);
    }
    const rocsparse_handle handle() const
    {
        return m_handle;
    }
};

class ROCSparseMatrixOnDevice
{
    rocsparse_handle m_handle;
    int m_rows;
    int m_cols;
    int m_nnz;
    rocsparse_int* m_dptr;
    rocsparse_int* m_dcol;
    rocsparse_double_complex* m_dval;
    rocsparse_mat_descr m_descr;
    rocsparse_mat_info m_info;
public:
    ROCSparseMatrixOnDevice() :
        m_handle(nullptr),
        m_rows(0),
        m_cols(0),
        m_nnz(0),
        m_dptr(nullptr),
        m_dcol(nullptr),
        m_dval(nullptr),
        m_descr(nullptr),
        m_info(nullptr)
    {}
    ROCSparseMatrixOnDevice(rocsparse_handle handle,
                            const SparseMatrixData &smd) :
        m_handle(handle),
        m_rows(smd.rows),
        m_cols(smd.cols),
        m_nnz(smd.nnz),
        m_dptr(nullptr),
        m_dcol(nullptr),
        m_dval(nullptr),
        m_descr(nullptr),
        m_info(nullptr)
    {
        std::vector<rocsparse_double_complex> hval(m_nnz);
        std::vector<rocsparse_int> hptr(m_rows + 1);
        std::vector<rocsparse_int> hcol(m_nnz);
        for (int i = 0; i < m_nnz; ++i) {
            hval[i] = rocsparse_double_complex(smd.val[i].real(), smd.val[i].imag());
            hcol[i] = smd.col[i];
        }
        for (int i = 0; i < hptr.size(); ++i) {
            hptr[i] = smd.ptr[i];
        }
        hipMalloc((void**)&m_dptr, sizeof(rocsparse_int) * (m_rows + 1));
        hipMalloc((void**)&m_dcol, sizeof(rocsparse_int) * m_nnz);
        hipMalloc((void**)&m_dval, sizeof(rocsparse_double_complex) * m_nnz);
        hipMemcpy(m_dptr, hptr.data(), sizeof(rocsparse_int) * (m_rows + 1), hipMemcpyHostToDevice);
        hipMemcpy(m_dcol, hcol.data(), sizeof(rocsparse_int) * m_nnz, hipMemcpyHostToDevice);
        hipMemcpy(m_dval, hval.data(), sizeof(rocsparse_double_complex) * m_nnz, hipMemcpyHostToDevice);

        rocsparse_create_mat_descr(&m_descr);

        rocsparse_create_mat_info(&m_info);
        rocsparse_zcsrmv_analysis(
            m_handle, rocsparse_operation_none, m_rows, m_cols, m_nnz, m_descr, m_dval, m_dptr, m_dcol, m_info);
    }
    // noexcept is necessary for the STL containers (e.g. std::vector) to use
    // this move-constructor instead of the copy-constructor
    ROCSparseMatrixOnDevice(ROCSparseMatrixOnDevice &&other) noexcept :
        m_handle(other.m_handle),
        m_rows(other.m_rows),
        m_cols(other.m_cols),
        m_nnz(other.m_nnz),
        m_dptr(other.m_dptr),
        m_dcol(other.m_dcol),
        m_dval(other.m_dval),
        m_descr(other.m_descr),
        m_info(other.m_info)
    {
        other.m_dptr = nullptr;
        other.m_dcol = nullptr;
        other.m_dval = nullptr;
        other.m_descr = nullptr;
        other.m_info = nullptr;
    }
    ROCSparseMatrixOnDevice(const ROCSparseMatrixOnDevice &other) :
        m_handle(other.m_handle),
        m_rows(other.m_rows),
        m_cols(other.m_cols),
        m_nnz(other.m_nnz)
    {
        hipMalloc((void**)&m_dptr, sizeof(rocsparse_int) * (m_rows + 1));
        hipMalloc((void**)&m_dcol, sizeof(rocsparse_int) * m_nnz);
        hipMalloc((void**)&m_dval, sizeof(rocsparse_double_complex) * m_nnz);
        hipMemcpy(m_dptr, other.m_dptr, sizeof(rocsparse_int) * (m_rows + 1), hipMemcpyDeviceToDevice);
        hipMemcpy(m_dcol, other.m_dcol, sizeof(rocsparse_int) * m_nnz, hipMemcpyDeviceToDevice);
        hipMemcpy(m_dval, other.m_dval, sizeof(rocsparse_double_complex) * m_nnz, hipMemcpyDeviceToDevice);

        rocsparse_create_mat_descr(&m_descr);

        rocsparse_create_mat_info(&m_info);
        rocsparse_zcsrmv_analysis(
            m_handle, rocsparse_operation_none, m_rows, m_cols, m_nnz, m_descr, m_dval, m_dptr, m_dcol, m_info);
    }
    ROCSparseMatrixOnDevice &operator=(const ROCSparseMatrixOnDevice &other)
    {
        if (this == &other) {
            return *this;
        }
        m_handle = other.m_handle;
        if (m_rows != other.m_rows) {
            hipFree(m_dptr);
            m_dptr = nullptr;
            m_rows = other.m_rows;
            hipMalloc((void**)&m_dptr, sizeof(rocsparse_int) * (m_rows + 1));
            hipMemcpy(m_dptr, other.m_dptr, sizeof(rocsparse_int) * (m_rows + 1), hipMemcpyDeviceToDevice);
        }
        if (m_nnz != other.m_nnz) {
            hipFree(m_dcol);
            m_dcol = nullptr;
            m_nnz = other.m_nnz;
            hipMalloc((void**)&m_dcol, sizeof(rocsparse_int) * m_nnz);
            hipMemcpy(m_dcol, other.m_dcol, sizeof(rocsparse_int) * m_nnz, hipMemcpyDeviceToDevice);

            hipFree(m_dval);
            m_dval = nullptr;
            hipMalloc((void**)&m_dval, sizeof(rocsparse_double_complex) * m_nnz);
            hipMemcpy(m_dval, other.m_dval, sizeof(rocsparse_double_complex) * m_nnz, hipMemcpyDeviceToDevice);
        }

        if (m_info != nullptr) {
            rocsparse_destroy_mat_info(m_info);
        }
        if (m_descr != nullptr) {
            rocsparse_destroy_mat_descr(m_descr);
        }
        rocsparse_create_mat_descr(&m_descr);

        rocsparse_create_mat_info(&m_info);
        rocsparse_zcsrmv_analysis(
            m_handle, rocsparse_operation_none, m_rows, m_cols, m_nnz, m_descr, m_dval, m_dptr, m_dcol, m_info);
        return *this;
    }
    ROCSparseMatrixOnDevice &operator=(ROCSparseMatrixOnDevice &&other)
    {
        if (this == &other) {
            return *this;
        }
        m_handle = other.m_handle;
        hipFree(m_dptr);
        m_dptr = other.m_dptr;
        other.m_dptr = nullptr;
        m_rows = other.m_rows;

        hipFree(m_dcol);
        m_dcol = other.m_dcol;
        other.m_dcol = nullptr;
        m_nnz = other.m_nnz;

        hipFree(m_dval);
        m_dval = other.m_dval;
        m_dval = nullptr;

        if (m_info != nullptr) {
            rocsparse_destroy_mat_info(m_info);
        }
        if (m_descr != nullptr) {
            rocsparse_destroy_mat_descr(m_descr);
        }
        m_info = other.m_info;
        other.m_info = nullptr;
        m_descr = other.m_descr;
        other.m_descr = nullptr;
        return *this;
    }
    ~ROCSparseMatrixOnDevice()
    {
        hipFree(m_dptr);
        hipFree(m_dcol);
        hipFree(m_dval);
        if (m_info != nullptr) {
            rocsparse_destroy_mat_info(m_info);
        }
        if (m_descr != nullptr) {
            rocsparse_destroy_mat_descr(m_descr);
        }
    }
    void mul_vec(const rocsparse_double_complex *alpha, const rocsparse_double_complex* x, const rocsparse_double_complex *beta, rocsparse_double_complex* y) const
    {
#ifdef TEST_CSRMV_CODE_COPIED_FROM_ROCSPARSE
        zcsrmv(m_handle,
               rocsparse_operation_none,
               m_rows,
               m_cols,
               m_nnz,
               alpha,
               m_descr,
               m_dval,
               m_dptr,
               m_dcol,
               m_info,
               x,
               beta,
               y);
#else // TEST_CSRMV_CODE_COPIED_FROM_ROCSPARSE
        rocsparse_zcsrmv(m_handle,
                         rocsparse_operation_none,
                         m_rows,
                         m_cols,
                         m_nnz,
                         alpha,
                         m_descr,
                         m_dval,
                         m_dptr,
                         m_dcol,
                         m_info,
                         x,
                         beta,
                         y);
#endif // TEST_CSRMV_CODE_COPIED_FROM_ROCSPARSE
    }
    void mul_mat_for_trace(const rocsparse_double_complex *alpha, const rocsparse_double_complex* x, const rocsparse_double_complex *beta, rocsparse_double_complex* y) const
    {
        zcsrmmtrace(m_handle,
                    rocsparse_operation_none,
                    m_rows,
                    m_cols,
                    m_nnz,
                    alpha,
                    m_descr,
                    m_dval,
                    m_dptr,
                    m_dcol,
                    m_info,
                    x,
                    beta,
                    y);
    }
    void mul_mat(const rocsparse_double_complex *alpha, const rocsparse_double_complex* B, const rocsparse_double_complex *beta, rocsparse_double_complex* C, rocsparse_int y_cols) const
    {
        rocsparse_zcsrmm(m_handle,
                         rocsparse_operation_none,
                         rocsparse_operation_none,
                         m_rows,
                         y_cols,
                         m_cols,
                         m_nnz,
                         alpha,
                         m_descr,
                         m_dval,
                         m_dptr,
                         m_dcol,
                         B,
                         m_cols,
                         beta,
                         C,
                         m_rows);
    }
    void mul_mat_row_major(const rocsparse_double_complex *alpha, const rocsparse_double_complex* B, const rocsparse_double_complex *beta, rocsparse_double_complex* C, rocsparse_int y_cols) const
    {
        rocsparse_zcsrmm_row_major(m_handle,
                         rocsparse_operation_none,
                         rocsparse_operation_none,
                         m_rows,
                         y_cols,
                         m_cols,
                         m_nnz,
                         alpha,
                         m_descr,
                         m_dval,
                         m_dptr,
                         m_dcol,
                         B,
                         m_cols,
                         beta,
                         C,
                         m_rows);
    }
    void mul_mat_with_transpose(const rocsparse_double_complex *alpha, const rocsparse_double_complex* B, const rocsparse_double_complex *beta, rocsparse_double_complex* C, rocsparse_int y_cols) const
    {
        rocsparse_zcsrmm(m_handle,
                         rocsparse_operation_none,
                         rocsparse_operation_transpose,
                         m_rows,
                         y_cols,
                         m_cols,
                         m_nnz,
                         alpha,
                         m_descr,
                         m_dval,
                         m_dptr,
                         m_dcol,
                         B,
                         m_cols,
                         beta,
                         C,
                         m_rows);
    }
    int rows() const { return m_rows; };
    int cols() const { return m_cols; };
};

#endif // ROC_UTIL_H
