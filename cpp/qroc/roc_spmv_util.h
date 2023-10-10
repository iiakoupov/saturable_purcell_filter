// Copyright (c) 2020-2023 Ivan Iakoupov
//
// Based in part upon code from rocSPARSE which is:
//
// Copyright (c) 2018 Advanced Micro Devices, Inc.
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


#ifndef ROC_SPMV_UTIL_H
#define ROC_SPMV_UTIL_H

#include <hip/hip_runtime_api.h>
#include <rocsparse/rocsparse.h>

__device__ __forceinline__ double rocsparse_ldg(const double* ptr) { return __ldg(ptr); }
__device__ __forceinline__ rocsparse_double_complex rocsparse_ldg(const rocsparse_double_complex* ptr) { return rocsparse_double_complex(__ldg((const double*)ptr), __ldg((const double*)ptr + 1)); }

__device__ __forceinline__ float rocsparse_fma(float p, float q, float r) { return fma(p, q, r); }
__device__ __forceinline__ double rocsparse_fma(double p, double q, double r) { return fma(p, q, r); }
__device__ __forceinline__ rocsparse_float_complex rocsparse_fma(rocsparse_float_complex p, rocsparse_float_complex q, rocsparse_float_complex r) { return std::fma(p, q, r); }
__device__ __forceinline__ rocsparse_double_complex rocsparse_fma(rocsparse_double_complex p, rocsparse_double_complex q, rocsparse_double_complex r) { return std::fma(p, q, r); }

#if (!defined(__gfx1030__)) && (!defined(__gfx1011__))
// DPP-based wavefront reduction maximum
template <unsigned int WFSIZE>
__device__ __forceinline__ void rocsparse_wfreduce_max(int* maximum)
{
    if(WFSIZE >  1) *maximum = max(*maximum, __hip_move_dpp(*maximum, 0x111, 0xf, 0xf, 0));
    if(WFSIZE >  2) *maximum = max(*maximum, __hip_move_dpp(*maximum, 0x112, 0xf, 0xf, 0));
    if(WFSIZE >  4) *maximum = max(*maximum, __hip_move_dpp(*maximum, 0x114, 0xf, 0xe, 0));
    if(WFSIZE >  8) *maximum = max(*maximum, __hip_move_dpp(*maximum, 0x118, 0xf, 0xc, 0));
    if(WFSIZE > 16) *maximum = max(*maximum, __hip_move_dpp(*maximum, 0x142, 0xa, 0xf, 0));
    if(WFSIZE > 32) *maximum = max(*maximum, __hip_move_dpp(*maximum, 0x143, 0xc, 0xf, 0));
}

// DPP-based wavefront reduction minimum
template <unsigned int WFSIZE>
__device__ __forceinline__ void rocsparse_wfreduce_min(int* minimum)
{
    if(WFSIZE >  1) *minimum = min(*minimum, __hip_move_dpp(*minimum, 0x111, 0xf, 0xf, 0));
    if(WFSIZE >  2) *minimum = min(*minimum, __hip_move_dpp(*minimum, 0x112, 0xf, 0xf, 0));
    if(WFSIZE >  4) *minimum = min(*minimum, __hip_move_dpp(*minimum, 0x114, 0xf, 0xe, 0));
    if(WFSIZE >  8) *minimum = min(*minimum, __hip_move_dpp(*minimum, 0x118, 0xf, 0xc, 0));
    if(WFSIZE > 16) *minimum = min(*minimum, __hip_move_dpp(*minimum, 0x142, 0xa, 0xf, 0));
    if(WFSIZE > 32) *minimum = min(*minimum, __hip_move_dpp(*minimum, 0x143, 0xc, 0xf, 0));
}

template <unsigned int WFSIZE>
__device__ __forceinline__ void rocsparse_wfreduce_min(int64_t* minimum)
{
    typedef union i64_b32
    {
        int64_t i64;
        uint32_t b32[2];
    } i64_b32_t;

    i64_b32_t upper_min;
    i64_b32_t temp_min;
    temp_min.i64 = *minimum;

    if(WFSIZE > 1)
    {
        upper_min.b32[0] = __hip_move_dpp(temp_min.b32[0], 0x111, 0xf, 0xf, false);
        upper_min.b32[1] = __hip_move_dpp(temp_min.b32[1], 0x111, 0xf, 0xf, false);
        temp_min.i64 = min(temp_min.i64, upper_min.i64);
    }

    if(WFSIZE > 2)
    {
        upper_min.b32[0] = __hip_move_dpp(temp_min.b32[0], 0x112, 0xf, 0xf, false);
        upper_min.b32[1] = __hip_move_dpp(temp_min.b32[1], 0x112, 0xf, 0xf, false);
        temp_min.i64 = min(temp_min.i64, upper_min.i64);
    }

    if(WFSIZE > 4)
    {
        upper_min.b32[0] = __hip_move_dpp(temp_min.b32[0], 0x114, 0xf, 0xe, false);
        upper_min.b32[1] = __hip_move_dpp(temp_min.b32[1], 0x114, 0xf, 0xe, false);
        temp_min.i64 = min(temp_min.i64, upper_min.i64);
    }

    if(WFSIZE > 8)
    {
        upper_min.b32[0] = __hip_move_dpp(temp_min.b32[0], 0x118, 0xf, 0xc, false);
        upper_min.b32[1] = __hip_move_dpp(temp_min.b32[1], 0x118, 0xf, 0xc, false);
        temp_min.i64 = min(temp_min.i64, upper_min.i64);
    }

    if(WFSIZE > 16)
    {
        upper_min.b32[0] = __hip_move_dpp(temp_min.b32[0], 0x142, 0xa, 0xf, false);
        upper_min.b32[1] = __hip_move_dpp(temp_min.b32[1], 0x142, 0xa, 0xf, false);
        temp_min.i64 = min(temp_min.i64, upper_min.i64);
    }

    if(WFSIZE > 32)
    {
        upper_min.b32[0] = __hip_move_dpp(temp_min.b32[0], 0x143, 0xc, 0xf, false);
        upper_min.b32[1] = __hip_move_dpp(temp_min.b32[1], 0x143, 0xc, 0xf, false);
        temp_min.i64 = min(temp_min.i64, upper_min.i64);
    }

    *minimum = temp_min.i64;
}

// DPP-based wavefront reduction sum
template <unsigned int WFSIZE>
__device__ __forceinline__ void rocsparse_wfreduce_sum(int* sum)
{
    if(WFSIZE >  1) *sum += __hip_move_dpp(*sum, 0x111, 0xf, 0xf, 0);
    if(WFSIZE >  2) *sum += __hip_move_dpp(*sum, 0x112, 0xf, 0xf, 0);
    if(WFSIZE >  4) *sum += __hip_move_dpp(*sum, 0x114, 0xf, 0xe, 0);
    if(WFSIZE >  8) *sum += __hip_move_dpp(*sum, 0x118, 0xf, 0xc, 0);
    if(WFSIZE > 16) *sum += __hip_move_dpp(*sum, 0x142, 0xa, 0xf, 0);
    if(WFSIZE > 32) *sum += __hip_move_dpp(*sum, 0x143, 0xc, 0xf, 0);
}

template <unsigned int WFSIZE>
__device__ __forceinline__ void rocsparse_wfreduce_sum(int64_t* sum)
{
    typedef union i64_b32
    {
        int64_t i64;
        uint32_t b32[2];
    } i64_b32_t;

    i64_b32_t upper_sum;
    i64_b32_t temp_sum;
    temp_sum.i64 = *sum;

    if(WFSIZE > 1)
    {
        upper_sum.b32[0] = __hip_move_dpp(temp_sum.b32[0], 0x111, 0xf, 0xf, false);
        upper_sum.b32[1] = __hip_move_dpp(temp_sum.b32[1], 0x111, 0xf, 0xf, false);
        temp_sum.i64 += upper_sum.i64;
    }

    if(WFSIZE > 2)
    {
        upper_sum.b32[0] = __hip_move_dpp(temp_sum.b32[0], 0x112, 0xf, 0xf, false);
        upper_sum.b32[1] = __hip_move_dpp(temp_sum.b32[1], 0x112, 0xf, 0xf, false);
        temp_sum.i64 += upper_sum.i64;
    }

    if(WFSIZE > 4)
    {
        upper_sum.b32[0] = __hip_move_dpp(temp_sum.b32[0], 0x114, 0xf, 0xe, false);
        upper_sum.b32[1] = __hip_move_dpp(temp_sum.b32[1], 0x114, 0xf, 0xe, false);
        temp_sum.i64 += upper_sum.i64;
    }

    if(WFSIZE > 8)
    {
        upper_sum.b32[0] = __hip_move_dpp(temp_sum.b32[0], 0x118, 0xf, 0xc, false);
        upper_sum.b32[1] = __hip_move_dpp(temp_sum.b32[1], 0x118, 0xf, 0xc, false);
        temp_sum.i64 += upper_sum.i64;
    }

    if(WFSIZE > 16)
    {
        upper_sum.b32[0] = __hip_move_dpp(temp_sum.b32[0], 0x142, 0xa, 0xf, false);
        upper_sum.b32[1] = __hip_move_dpp(temp_sum.b32[1], 0x142, 0xa, 0xf, false);
        temp_sum.i64 += upper_sum.i64;
    }

    if(WFSIZE > 32)
    {
        upper_sum.b32[0] = __hip_move_dpp(temp_sum.b32[0], 0x143, 0xc, 0xf, false);
        upper_sum.b32[1] = __hip_move_dpp(temp_sum.b32[1], 0x143, 0xc, 0xf, false);
        temp_sum.i64 += upper_sum.i64;
    }

    *sum = temp_sum.i64;
}

// DPP-based float wavefront reduction sum
template <unsigned int WFSIZE>
__device__ __forceinline__ float rocsparse_wfreduce_sum(float sum)
{
    typedef union flt_b32
    {
        float val;
        uint32_t b32;
    } flt_b32_t;

    flt_b32_t upper_sum;
    flt_b32_t temp_sum;
    temp_sum.val = sum;

    if(WFSIZE > 1)
    {
        upper_sum.b32 = __hip_move_dpp(temp_sum.b32, 0x111, 0xf, 0xf, false);
        temp_sum.val += upper_sum.val;
    }

    if(WFSIZE > 2)
    {
        upper_sum.b32 = __hip_move_dpp(temp_sum.b32, 0x112, 0xf, 0xf, false);
        temp_sum.val += upper_sum.val;
    }

    if(WFSIZE > 4)
    {
        upper_sum.b32 = __hip_move_dpp(temp_sum.b32, 0x114, 0xf, 0xe, false);
        temp_sum.val += upper_sum.val;
    }

    if(WFSIZE > 8)
    {
        upper_sum.b32 = __hip_move_dpp(temp_sum.b32, 0x118, 0xf, 0xc, false);
        temp_sum.val += upper_sum.val;
    }

    if(WFSIZE > 16)
    {
        upper_sum.b32 = __hip_move_dpp(temp_sum.b32, 0x142, 0xa, 0xf, false);
        temp_sum.val += upper_sum.val;
    }

    if(WFSIZE > 32)
    {
        upper_sum.b32 = __hip_move_dpp(temp_sum.b32, 0x143, 0xc, 0xf, false);
        temp_sum.val += upper_sum.val;
    }

    sum = temp_sum.val;
    return sum;
}

// DPP-based double wavefront reduction
template <unsigned int WFSIZE>
__device__ __forceinline__ double rocsparse_wfreduce_sum(double sum)
{
    typedef union dbl_b32
    {
        double val;
        uint32_t b32[2];
    } dbl_b32_t;

    dbl_b32_t upper_sum;
    dbl_b32_t temp_sum;
    temp_sum.val = sum;

    if(WFSIZE > 1)
    {
        upper_sum.b32[0] = __hip_move_dpp(temp_sum.b32[0], 0x111, 0xf, 0xf, false);
        upper_sum.b32[1] = __hip_move_dpp(temp_sum.b32[1], 0x111, 0xf, 0xf, false);
        temp_sum.val += upper_sum.val;
    }

    if(WFSIZE > 2)
    {
        upper_sum.b32[0] = __hip_move_dpp(temp_sum.b32[0], 0x112, 0xf, 0xf, false);
        upper_sum.b32[1] = __hip_move_dpp(temp_sum.b32[1], 0x112, 0xf, 0xf, false);
        temp_sum.val += upper_sum.val;
    }

    if(WFSIZE > 4)
    {
        upper_sum.b32[0] = __hip_move_dpp(temp_sum.b32[0], 0x114, 0xf, 0xe, false);
        upper_sum.b32[1] = __hip_move_dpp(temp_sum.b32[1], 0x114, 0xf, 0xe, false);
        temp_sum.val += upper_sum.val;
    }

    if(WFSIZE > 8)
    {
        upper_sum.b32[0] = __hip_move_dpp(temp_sum.b32[0], 0x118, 0xf, 0xc, false);
        upper_sum.b32[1] = __hip_move_dpp(temp_sum.b32[1], 0x118, 0xf, 0xc, false);
        temp_sum.val += upper_sum.val;
    }

    if(WFSIZE > 16)
    {
        upper_sum.b32[0] = __hip_move_dpp(temp_sum.b32[0], 0x142, 0xa, 0xf, false);
        upper_sum.b32[1] = __hip_move_dpp(temp_sum.b32[1], 0x142, 0xa, 0xf, false);
        temp_sum.val += upper_sum.val;
    }

    if(WFSIZE > 32)
    {
        upper_sum.b32[0] = __hip_move_dpp(temp_sum.b32[0], 0x143, 0xc, 0xf, false);
        upper_sum.b32[1] = __hip_move_dpp(temp_sum.b32[1], 0x143, 0xc, 0xf, false);
        temp_sum.val += upper_sum.val;
    }

    sum = temp_sum.val;
    return sum;
}
#else
template <unsigned int WFSIZE>
__device__ __forceinline__ void rocsparse_wfreduce_max(int* maximum)
{
    for(int i = WFSIZE >> 1; i > 0; i >>= 1)
    {
        *maximum = max(*maximum, __shfl_xor(*maximum, i));
    }
}

template <unsigned int WFSIZE>
__device__ __forceinline__ void rocsparse_wfreduce_min(int* minimum)
{
    for(int i = WFSIZE >> 1; i > 0; i >>= 1)
    {
        *minimum = min(*minimum, __shfl_xor(*minimum, i));
    }
}

template <unsigned int WFSIZE>
__device__ __forceinline__ void rocsparse_wfreduce_min(int64_t* minimum)
{
    for(int i = WFSIZE >> 1; i > 0; i >>= 1)
    {
        *minimum = min(*minimum, __shfl_xor(*minimum, i));
    }
}

template <unsigned int WFSIZE>
__device__ __forceinline__ void rocsparse_wfreduce_sum(int* sum)
{
    for(int i = WFSIZE >> 1; i > 0; i >>= 1)
    {
        *sum += __shfl_xor(*sum, i);
    }
}

template <unsigned int WFSIZE>
__device__ __forceinline__ void rocsparse_wfreduce_sum(int64_t* sum)
{
    for(int i = WFSIZE >> 1; i > 0; i >>= 1)
    {
        *sum += __shfl_xor(*sum, i);
    }
}

template <unsigned int WFSIZE>
__device__ __forceinline__ float rocsparse_wfreduce_sum(float sum)
{
    for(int i = WFSIZE >> 1; i > 0; i >>= 1)
    {
        sum += __shfl_xor(sum, i);
    }

    return sum;
}

template <unsigned int WFSIZE>
__device__ __forceinline__ double rocsparse_wfreduce_sum(double sum)
{
    for(int i = WFSIZE >> 1; i > 0; i >>= 1)
    {
        sum += __shfl_xor(sum, i);
    }

    return sum;
}
#endif

// DPP-based complex float wavefront reduction sum
template <unsigned int WFSIZE>
__device__ __forceinline__ rocsparse_float_complex rocsparse_wfreduce_sum(rocsparse_float_complex sum)
{
    return rocsparse_float_complex(rocsparse_wfreduce_sum<WFSIZE>(std::real(sum)),
                                   rocsparse_wfreduce_sum<WFSIZE>(std::imag(sum)));
}

// DPP-based complex double wavefront reduction
template <unsigned int WFSIZE>
__device__ __forceinline__ rocsparse_double_complex rocsparse_wfreduce_sum(rocsparse_double_complex sum)
{
    return rocsparse_double_complex(rocsparse_wfreduce_sum<WFSIZE>(std::real(sum)),
                                    rocsparse_wfreduce_sum<WFSIZE>(std::imag(sum)));
}

#endif // ROC_SPMV_UTIL_H
