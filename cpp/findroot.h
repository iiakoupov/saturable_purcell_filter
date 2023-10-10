// Copyright (c) 2022 Ivan Iakoupov
//
// Based in part upon code from mpmath which is:
//
// Copyright (c) 2005-2010 Fredrik Johansson and mpmath contributors
//
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//  a. Redistributions of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//  b. Redistributions in binary form must reproduce the above copyright
//     notice, this list of conditions and the following disclaimer in the
//     documentation and/or other materials provided with the distribution.
//  c. Neither the name of mpmath nor the names of its contributors
//     may be used to endorse or promote products derived from this software
//     without specific prior written permission.
//
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
// OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
// DAMAGE.

#ifndef FINDROOT_H
#define FINDROOT_H

#include <cassert>
#include <complex>
#include <functional>

template <typename Scalar>
Scalar find_root_secant(std::function<Scalar(Scalar)> f, Scalar x1, Scalar x2,
                        double tol)
{
    Scalar tempX1 = x1;
    Scalar tempX2 = x2;
    //printf(" x1 = %.5f\n", x1);
    //printf(" x2 = %.5f\n", x2);
    assert(x1 != x2 && "The two starting points should be different");
    for (int i = 0; i < 100000; ++i) {
        Scalar fOnTempX1 = f(tempX1);
        Scalar fOnTempX2 = f(tempX2);
        //printf(" fOnTempX2 = %.5f\n", fOnTempX2);
        const Scalar tempXNew = tempX2 - fOnTempX2*(tempX2-tempX1)
                                         /(fOnTempX2-fOnTempX1);
        //printf(" tempXNew = %.5f\n", tempXNew);
        if (std::isnan(std::norm(tempXNew)) || std::isinf(std::norm(tempXNew))) {
            return tempXNew;
        }
        tempX1 = tempX2;
        tempX2 = tempXNew;
        if (std::abs(tempX1-tempX2) < tol) {
            break;
        }
    }
    return tempX2;
}

#endif // FINDROOT_H

