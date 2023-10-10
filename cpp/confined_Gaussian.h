// Copyright (c) 2022 Ivan Iakoupov
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

#ifndef CONFINED_GAUSSIAN_H
#define CONFINED_GAUSSIAN_H

#include <complex>

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

inline double confinedGaussianUpDown(double n, int64_t N_t, double t_f, double sigma)
{
    // Uses confinedGaussian() to make a piecewise function
    // that starts at 0, rises to (approximately) 1 using
    // half of a confined Gaussian, stays at 1 for some time,
    // and finally goes down to 0 again using the other half
    // of a confined Gaussian.
    const int64_t N_t_up = int64_t(N_t*sigma);
    const int64_t N_t_down = int64_t(N_t*sigma);
    const double t_f_up = t_f*sigma;
    const double t_f_down = t_f*sigma;
    const double sigmaUp = 0.15;
    const double sigmaDown = 0.15;
    if (n < N_t_up) {
        return confinedGaussian(n, 2*N_t_up, t_f_up, sigmaUp);
    } else if (n > N_t-N_t_down) {
        double n_down = n-(N_t-N_t_down)+N_t_down;
        return confinedGaussian(n_down, 2*N_t_down, t_f_down, sigmaDown);
    } else {
        // It is slightly less than 1, because "confinedGaussian"
        // is using an approximate analytical formula.
        return confinedGaussian(N_t_up, 2*N_t_up, t_f_up, sigmaUp);
    }
}

#endif // CONFINED_GAUSSIAN_H
