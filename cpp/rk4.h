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

#ifndef RK4_H
#define RK4_H

#include "types.h"
#include "csrmatrix.h"

#ifdef EIGEN_USE_MKL_ALL
#include "mkl_support.h"
#endif // EIGEN_USE_MKL_ALL

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

inline void mulMatAddAdjoint(Eigen::VectorXcd &v_dst, const SpMat &Mfactor, const Eigen::VectorXcd &v_src, std::complex<double> factor)
{
    int size = Mfactor.rows();
    Eigen::Map<const MatrixXcdRowMajor> v_src_map(v_src.data(), size, size);
    Eigen::Map<MatrixXcdRowMajor> v_dst_map(v_dst.data(), size, size);
    v_dst_map.noalias() = factor*Mfactor*v_src_map;
    addAdjointInPlace(v_dst_map);
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

inline void applyL_t(Eigen::VectorXcd &v_dst, const SpMat &L, const SpMat &H_t_Re, const SpMat &H_t_Im, const Eigen::VectorXcd &v_src, std::complex<double> factor_Re, std::complex<double> factor_Im, double dt)
{
    mulMatAddAdjoint(v_dst, H_t_Re, H_t_Im, v_src, factor_Re, factor_Im);
    v_dst.noalias() += dt*L*v_src;
}

inline void applyL_t(Eigen::VectorXcd &v_dst, const SpMat &L, const SpMat &H_t, const Eigen::VectorXcd &v_src, std::complex<double> factor, double dt)
{
    mulMatAddAdjoint(v_dst, H_t, v_src, factor);
    v_dst.noalias() += dt*L*v_src;
}

#ifdef EIGEN_USE_MKL_ALL
inline void applyL_t_mkl(Eigen::VectorXcd &v_dst, const MKLSparseMatrix &mklL, const SpMat &H_t, const Eigen::VectorXcd &v_src, std::complex<double> factor, double dt)
{
    mulMatAddAdjoint(v_dst, H_t, v_src, factor);
    mklL.mul_vector(v_dst, v_src, dt, 1);
}

inline void mulMatAddAdjoint_mkl(Eigen::VectorXcd &v_dst, const MKLSparseMatrix &mklMfactor_Re, const MKLSparseMatrix &mklMfactor_Im, const Eigen::VectorXcd &v_src, std::complex<double> factor_Re, std::complex<double> factor_Im)
{
    int size = mklMfactor_Re.rows();
    Eigen::Map<MatrixXcdRowMajor> v_dst_map(v_dst.data(), size, size);
    mklMfactor_Re.mul_matrix_row_major(v_dst.data(), size, size, v_src.data(), size, factor_Re, 0);
    mklMfactor_Im.mul_matrix_row_major(v_dst.data(), size, size, v_src.data(), size, factor_Im, 1);
    addAdjointInPlace(v_dst_map);
}

inline void applyL_t_mkl(Eigen::VectorXcd &v_dst, const MKLSparseMatrix &mklL, const MKLSparseMatrix &mklH_t_Re, const MKLSparseMatrix &mklH_t_Im, const Eigen::VectorXcd &v_src, std::complex<double> factor_Re, std::complex<double> factor_Im, double dt)
{
    mulMatAddAdjoint_mkl(v_dst, mklH_t_Re, mklH_t_Im, v_src, factor_Re, factor_Im);
    mklL.mul_vector(v_dst, v_src, dt, 1);
}
#endif // EIGEN_USE_MKL_ALL

inline void rk4_step(Eigen::VectorXcd &rho_vec, const SpMat &L, Eigen::VectorXcd &k1, Eigen::VectorXcd &k2, Eigen::VectorXcd &k3, Eigen::VectorXcd &k4, Eigen::VectorXcd &temp, double dt)
{
    k1.noalias() = dt*L*rho_vec;
    temp = rho_vec+k1*0.5;
    k2.noalias() = dt*L*temp;
    temp = rho_vec+k2*0.5;
    k3.noalias() = dt*L*temp;
    temp = rho_vec+k3;
    k4.noalias() = dt*L*temp;
    rho_vec = rho_vec + (k1/6) + (k2/3) + (k3/3) + (k4/6);
}

inline void rk4_step(Eigen::VectorXcd &rho_vec, const CSRMatrix &L, Eigen::VectorXcd &k1, Eigen::VectorXcd &k2, Eigen::VectorXcd &k3, Eigen::VectorXcd &k4, Eigen::VectorXcd &temp, double dt)
{
    L.mul_vec(k1, rho_vec, dt);
    temp = rho_vec+k1*0.5;
    L.mul_vec(k2, temp, dt);
    temp = rho_vec+k2*0.5;
    L.mul_vec(k3, temp, dt);
    temp = rho_vec+k3;
    L.mul_vec(k4, temp, dt);
    rho_vec = rho_vec + (k1/6) + (k2/3) + (k3/3) + (k4/6);
}

#ifdef EIGEN_USE_MKL_ALL
inline void rk4_step_mkl(Eigen::VectorXcd &rho_vec, const MKLSparseMatrix &mklL, Eigen::VectorXcd &k1, Eigen::VectorXcd &k2, Eigen::VectorXcd &k3, Eigen::VectorXcd &k4, Eigen::VectorXcd &temp, double dt)
{
    mklL.mul_vector(k1, rho_vec, dt);
    temp = rho_vec+k1*0.5;
    mklL.mul_vector(k2, temp, dt);
    temp = rho_vec+k2*0.5;
    mklL.mul_vector(k3, temp, dt);
    temp = rho_vec+k3;
    mklL.mul_vector(k4, temp, dt);
    rho_vec = rho_vec + (k1/6) + (k2/3) + (k3/3) + (k4/6);
}
#endif // EIGEN_USE_MKL_ALL

inline void rk4_step_t(Eigen::VectorXcd &rho_vec, const SpMat &L, const SpMat &H_t, double H_t_factor, Eigen::VectorXcd &k1, Eigen::VectorXcd &k2, Eigen::VectorXcd &k3, Eigen::VectorXcd &k4, Eigen::VectorXcd &temp, double dt)
{
    const std::complex<double> I(0,1);
    const std::complex<double> factor = -I*H_t_factor*dt;

    applyL_t(k1, L, H_t, rho_vec, factor, dt);

    temp = rho_vec+k1*0.5;
    applyL_t(k2, L, H_t, temp, factor, dt);

    temp = rho_vec+k2*0.5;
    applyL_t(k3, L, H_t, temp, factor, dt);

    temp = rho_vec+k3;
    applyL_t(k4, L, H_t, temp, factor, dt);

    rho_vec = rho_vec + (k1/6) + (k2/3) + (k3/3) + (k4/6);
}

inline void rk4_step_t(Eigen::VectorXcd &rho_vec, const SpMat &L, const SpMat &H_t1, const SpMat &H_t2, const SpMat &H_t3, double H_t_factor, Eigen::VectorXcd &k1, Eigen::VectorXcd &k2, Eigen::VectorXcd &k3, Eigen::VectorXcd &k4, Eigen::VectorXcd &temp, double dt)
{
    const std::complex<double> I(0,1);
    const std::complex<double> factor = -I*H_t_factor*dt;

    applyL_t(k1, L, H_t1, rho_vec, factor, dt);

    temp = rho_vec+k1*0.5;
    applyL_t(k2, L, H_t2, temp, factor, dt);

    temp = rho_vec+k2*0.5;
    applyL_t(k3, L, H_t2, temp, factor, dt);

    temp = rho_vec+k3;
    applyL_t(k4, L, H_t3, temp, factor, dt);

    rho_vec = rho_vec + (k1/6) + (k2/3) + (k3/3) + (k4/6);
}

inline void rk4_step_t(Eigen::VectorXcd &rho_vec, const SpMat &L, const SpMat &H_t_Re, const SpMat &H_t_Im, std::complex<double> H_t_factor1, std::complex<double> H_t_factor2, std::complex<double> H_t_factor3, Eigen::VectorXcd &k1, Eigen::VectorXcd &k2, Eigen::VectorXcd &k3, Eigen::VectorXcd &k4, Eigen::VectorXcd &temp, double dt)
{
    const std::complex<double> I(0,1);
    const std::complex<double> factor1_Re = -I*H_t_factor1.real()*dt;
    const std::complex<double> factor1_Im = -I*H_t_factor1.imag()*dt;
    const std::complex<double> factor2_Re = -I*H_t_factor2.real()*dt;
    const std::complex<double> factor2_Im = -I*H_t_factor2.imag()*dt;
    const std::complex<double> factor3_Re = -I*H_t_factor3.real()*dt;
    const std::complex<double> factor3_Im = -I*H_t_factor3.imag()*dt;

    applyL_t(k1, L, H_t_Re, H_t_Im, rho_vec, factor1_Re, factor1_Im, dt);

    temp = rho_vec+k1*0.5;
    applyL_t(k2, L, H_t_Re, H_t_Im, temp, factor2_Re, factor2_Im, dt);

    temp = rho_vec+k2*0.5;
    applyL_t(k3, L, H_t_Re, H_t_Im, temp, factor2_Re, factor2_Im, dt);

    temp = rho_vec+k3;
    applyL_t(k4, L, H_t_Re, H_t_Im, temp, factor3_Re, factor3_Im, dt);

    rho_vec = rho_vec + (k1/6) + (k2/3) + (k3/3) + (k4/6);
}

inline void adrk4_backward_step_t(Eigen::VectorXcd &chi_vec,
                                  const SpMat &L,
                                  const SpMat &H_t1,
                                  const SpMat &H_t2,
                                  const SpMat &H_t3,
                                  double H_t_factor,
                                  Eigen::VectorXcd &mu1,
                                  Eigen::VectorXcd &mu2,
                                  Eigen::VectorXcd &mu3,
                                  Eigen::VectorXcd &mu4,
                                  Eigen::VectorXcd &mu5,
                                  Eigen::VectorXcd &temp,
                                  Eigen::VectorXcd &temp2,
                                  double dt)
{
    const std::complex<double> I(0,1);
    const std::complex<double> factor = I*H_t_factor*dt;

    applyL_t(mu1, L, H_t2, chi_vec, factor, dt);
    applyL_t(mu2, L, H_t3, chi_vec, factor, dt);
    applyL_t(mu3, L, H_t2, mu1, factor, dt);
    applyL_t(mu4, L, H_t2, mu2, factor, dt);
    applyL_t(mu5, L, H_t2, mu4, factor, dt);

    temp2=(1.0/6)*chi_vec+(1.0/6)*mu1+(1.0/12)*mu3+(1.0/24)*mu5;
    applyL_t(temp, L, H_t1, temp2, factor, dt);

    chi_vec += temp+(2.0/3)*mu1+(1.0/6)*mu2+(1.0/6)*mu3+(1.0/6)*mu4+(1.0/12)*mu5;
}

#ifdef EIGEN_USE_MKL_ALL
inline void rk4_step_t_mkl(Eigen::VectorXcd &rho_vec, const MKLSparseMatrix &mklL, const SpMat &H_t, double H_t_factor, Eigen::VectorXcd &k1, Eigen::VectorXcd &k2, Eigen::VectorXcd &k3, Eigen::VectorXcd &k4, Eigen::VectorXcd &temp, double dt)
{
    const std::complex<double> I(0,1);
    const std::complex<double> factor = -I*H_t_factor*dt;

    applyL_t_mkl(k1, mklL, H_t, rho_vec, factor, dt);

    temp = rho_vec+k1*0.5;
    applyL_t_mkl(k2, mklL, H_t, temp, factor, dt);

    temp = rho_vec+k2*0.5;
    applyL_t_mkl(k3, mklL, H_t, temp, factor, dt);

    temp = rho_vec+k3;
    applyL_t_mkl(k4, mklL, H_t, temp, factor, dt);

    rho_vec = rho_vec + (k1/6) + (k2/3) + (k3/3) + (k4/6);
}

inline void rk4_step_t_mkl(Eigen::VectorXcd &rho_vec, const MKLSparseMatrix &mklL, const SpMat &H_t1, const SpMat &H_t2, const SpMat &H_t3, double H_t_factor, Eigen::VectorXcd &k1, Eigen::VectorXcd &k2, Eigen::VectorXcd &k3, Eigen::VectorXcd &k4, Eigen::VectorXcd &temp, double dt)
{
    const int basis_size = H_t1.rows();
    const std::complex<double> I(0,1);
    const std::complex<double> factor = -I*H_t_factor*dt;

    applyL_t_mkl(k1, mklL, H_t1, rho_vec, factor, dt);

    temp = rho_vec+k1*0.5;
    applyL_t_mkl(k2, mklL, H_t2, temp, factor, dt);

    temp = rho_vec+k2*0.5;
    applyL_t_mkl(k3, mklL, H_t2, temp, factor, dt);

    temp = rho_vec+k3;
    applyL_t_mkl(k4, mklL, H_t3, temp, factor, dt);

    rho_vec = rho_vec + (k1/6) + (k2/3) + (k3/3) + (k4/6);
}

inline void rk4_step_t_mkl(Eigen::VectorXcd &rho_vec, const MKLSparseMatrix &mklL, const MKLSparseMatrix &mklH_t_Re, const MKLSparseMatrix &mklH_t_Im, std::complex<double> H_t_factor1, std::complex<double> H_t_factor2, std::complex<double> H_t_factor3, Eigen::VectorXcd &k1, Eigen::VectorXcd &k2, Eigen::VectorXcd &k3, Eigen::VectorXcd &k4, Eigen::VectorXcd &temp, double dt)
{
    const std::complex<double> I(0,1);
    const std::complex<double> factor1_Re = -I*H_t_factor1.real()*dt;
    const std::complex<double> factor1_Im = -I*H_t_factor1.imag()*dt;
    const std::complex<double> factor2_Re = -I*H_t_factor2.real()*dt;
    const std::complex<double> factor2_Im = -I*H_t_factor2.imag()*dt;
    const std::complex<double> factor3_Re = -I*H_t_factor3.real()*dt;
    const std::complex<double> factor3_Im = -I*H_t_factor3.imag()*dt;

    applyL_t_mkl(k1, mklL, mklH_t_Re, mklH_t_Im, rho_vec, factor1_Re, factor1_Im, dt);

    temp = rho_vec+k1*0.5;
    applyL_t_mkl(k2, mklL, mklH_t_Re, mklH_t_Im, temp, factor2_Re, factor2_Im, dt);

    temp = rho_vec+k2*0.5;
    applyL_t_mkl(k3, mklL, mklH_t_Re, mklH_t_Im, temp, factor2_Re, factor2_Im, dt);

    temp = rho_vec+k3;
    applyL_t_mkl(k4, mklL, mklH_t_Re, mklH_t_Im, temp, factor3_Re, factor3_Im, dt);

    rho_vec = rho_vec + (k1/6) + (k2/3) + (k3/3) + (k4/6);
}

inline void adrk4_backward_step_t_mkl(Eigen::VectorXcd &chi_vec,
                                      const MKLSparseMatrix &mklL,
                                      const SpMat &H_t1,
                                      const SpMat &H_t2,
                                      const SpMat &H_t3,
                                      double H_t_factor,
                                      Eigen::VectorXcd &mu1,
                                      Eigen::VectorXcd &mu2,
                                      Eigen::VectorXcd &mu3,
                                      Eigen::VectorXcd &mu4,
                                      Eigen::VectorXcd &mu5,
                                      Eigen::VectorXcd &temp,
                                      Eigen::VectorXcd &temp2,
                                      double dt)
{
    const int basis_size = H_t1.rows();
    const std::complex<double> I(0,1);
    const std::complex<double> factor = I*H_t_factor*dt;

    applyL_t_mkl(mu1, mklL, H_t2, chi_vec, factor, dt);
    applyL_t_mkl(mu2, mklL, H_t3, chi_vec, factor, dt);
    applyL_t_mkl(mu3, mklL, H_t2, mu1, factor, dt);
    applyL_t_mkl(mu4, mklL, H_t2, mu2, factor, dt);
    applyL_t_mkl(mu5, mklL, H_t2, mu4, factor, dt);

    temp2=(1.0/6)*chi_vec+(1.0/6)*mu1+(1.0/12)*mu3+(1.0/24)*mu5;
    applyL_t_mkl(temp, mklL, H_t1, temp2, factor, dt);

    chi_vec += temp+(2.0/3)*mu1+(1.0/6)*mu2+(1.0/6)*mu3+(1.0/6)*mu4+(1.0/12)*mu5;
}
#endif // EIGEN_USE_MKL_ALL

#endif // RK4_H
