// Copyright (c) 2020-2021 Ivan Iakoupov
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

#ifndef MASTER_EQUATION_H
#define MASTER_EQUATION_H

#include "types.h"
#include "unsupported/Eigen/KroneckerProduct"

// The expression is
// exp(I*k_n*|x_m-x_n|)+exp(I*k_n*|x_m+x_n|)
// but we rewrite it as
// exp(I*(k_n/k_0)*|k_0*x_m-k_0*x_n|)+exp(I*(k_n/k_0)*|k_0*x_m+k_0*x_n|)
// where k_0 is some chosen wave vector, e.g., the average one for
// the ensemble of the atoms. This way we do not have to input
// absolute values of the wavevectors k_n, but only the rescaled
// value k_n/k_0, which could even be set to approximately unity.
inline std::complex<double> exp_factor(double k_factor_n, double k0x_m, double k0x_n)
{
    const std::complex<double> I(0,1);
    return std::exp(I*M_PI*k_factor_n*std::abs(k0x_m-k0x_n))
           +std::exp(I*M_PI*k_factor_n*std::abs(k0x_m+k0x_n));
}

// The expression is
// cos(k_n*(x_m-x_n))+cos(k_n*(x_m+x_n))
// but we rewrite it as
// cos((k_n/k_0)*(k_0*x_m-k_0*x_n))+cos((k_n/k_0)*(k_0*x_m+k_0*x_n))
// where k_0 is some chosen wave vector, e.g., the average one for
// the ensemble of the atoms. This way we do not have to input
// absolute values of the wavevectors k_n, but only the rescaled
// value k_n/k_0, which could even be set to approximately unity.
inline double cos_factor(double k_factor_n, double k0x_m, double k0x_n)
{
    return std::cos(M_PI*k_factor_n*(k0x_m-k0x_n))
           +std::cos(M_PI*k_factor_n*(k0x_m+k0x_n));
}

// The expression is
// sin(k_n*(x_m-x_n))+sin(k_n*(x_m+x_n))
// but we rewrite it as
// sin((k_n/k_0)*(k_0*x_m-k_0*x_n))+sin((k_n/k_0)*(k_0*x_m+k_0*x_n))
// (See above for the reasoning)
inline double sin_factor(double k_factor_n, double k0x_m, double k0x_n)
{
    return std::sin(M_PI*k_factor_n*std::abs(k0x_m-k0x_n))
           +std::sin(M_PI*k_factor_n*std::abs(k0x_m+k0x_n));
}

inline void addLindbladTerms(SpMat &L, std::complex<double> gamma, const SpMat &L_left, const SpMat &L_right, const SpMat &L2, const SpMat &Identity)
{
    if (std::abs(gamma.real()) < 1e-15 && std::abs(gamma.imag()) < 1e-15) {
        return;
    }
    const SpMat temp1 = Eigen::kroneckerProduct(L2, Identity);
    const SpMat temp2 = Eigen::kroneckerProduct(Identity, L2.transpose());
    const SpMat temp3 = Eigen::kroneckerProduct(L_left, L_right.transpose());
    const SpMat temp_sum2 = temp1+temp2-2*temp3;
    const SpMat temp_sum = 0.5*gamma*(temp_sum2);
    L -= temp_sum;
}

inline void addLindbladTermsSeparate(SpMat &L, double gamma, double delta, double gamma_sum, double delta_diff, const SpMat &L_left, const SpMat &L_right, const SpMat &L2, const SpMat &Identity)
{
    if (std::abs(gamma) < 1e-15 && std::abs(delta) < 1e-15) {
        return;
    }
    const std::complex<double> I(0,1);
    const SpMat temp1 = Eigen::kroneckerProduct(L2, Identity);
    const SpMat temp2 = Eigen::kroneckerProduct(Identity, L2.transpose());
    const SpMat temp3 = Eigen::kroneckerProduct(L_left, L_right.transpose());
    const SpMat temp_sum2 = (gamma+I*delta)*temp1+(gamma-I*delta)*temp2-(gamma_sum+I*delta_diff)*temp3;
    const SpMat temp_sum = 0.5*temp_sum2;
    L -= temp_sum;
}

inline void addLindbladTermsSeparateComplex(SpMat &L, std::complex<double> xi_mn, std::complex<double> xi_nm, const SpMat &L_left, const SpMat &L_right, const SpMat &L2, const SpMat &Identity)
{
    if (std::abs(xi_mn) < 1e-15 && std::abs(xi_nm) < 1e-15) {
        return;
    }
    const SpMat temp1 = Eigen::kroneckerProduct(L2, Identity);
    const SpMat temp2 = Eigen::kroneckerProduct(Identity, L2.transpose());
    const SpMat temp3 = Eigen::kroneckerProduct(L_left, L_right.transpose());
    //const SpMat temp_sum = 0.5*xi_mn*(temp1-temp3)+std::conj(0.5*xi_nm)*(temp2-temp3);
    const SpMat temp_sum = 0.5*xi_mn*temp1+std::conj(0.5*xi_nm)*temp2-(0.5*xi_mn+std::conj(0.5*xi_nm))*temp3;
    L -= temp_sum;
}

template<typename Matrix>
std::complex<double> trace_of_product(const SpMat &A, const Matrix &B)
{
    // Trace of a matrix product tr(A*B) can be calculated
    // by \sum_{i,j} A_{i,j}*B_{j,i}
    std::complex<double> ret(0,0);
    for (int k = 0; k < A.outerSize(); ++k) {
        for (SpMat::InnerIterator it(A, k); it; ++it) {
            ret += it.value()*B(it.col(), it.row());
        }
    }
    return ret;
}

#endif // MASTER_EQUATION_H
