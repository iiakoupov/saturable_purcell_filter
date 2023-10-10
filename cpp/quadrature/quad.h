/* Copyright (c) 2012-2021 Ivan Iakoupov
 *
 * Based in part upon code from mpmath which is:
 *
 * Copyright (c) 2005-2010 Fredrik Johansson and mpmath contributors
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  a. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *  b. Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *  c. Neither the name of mpmath nor the names of its contributors
 *     may be used to endorse or promote products derived from this software
 *     without specific prior written permission.
 *
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
 * DAMAGE.
 */

#ifndef QUAD_H
#define QUAD_H

#include <functional>
#include <vector>
#include <complex>
#include <cassert>
//#include <iostream>

#define QUAD_BOTH_BOUNDS_FINITE 0
#define QUAD_LOWER_BOUND_INFINITE_UPPER_FINITE 1
#define QUAD_LOWER_BOUND_FINITE_UPPER_INFINITE 2
#define QUAD_BOTH_BOUNDS_INFINITE 3

struct Node
{
    double node;
    double weight;
};

typedef std::function<double(double)> funcRtoR;
typedef std::function<std::complex<double>(double)> funcRtoC;

double quad_midpoint(funcRtoR f, double a, double b, unsigned N);
std::vector<Node> calc_nodes_tanh_sinh(int degree, double userTolerance = -1);

std::vector<Node> calc_nodes_tanh_sinh_final(double a, double b, int max_degree,
                                             int finiteBounds = QUAD_BOTH_BOUNDS_FINITE);
std::vector<Node> all_nodes_tanh_sinh_semi_inf_quad(double a, int max_degree);

template <typename Scalar>
double quad_estimate_error(Scalar thisResult,
                           const std::vector<Scalar> &previousResults)
{
    assert(!previousResults.empty());
    const double errToOneBack = std::abs(thisResult - previousResults.back());
    double ret = errToOneBack;
    if (previousResults.size() > 1) {
        /* If there're more than one previous result then take
         * the one before the last one */
        const double errToTwoBack = std::abs(thisResult - (*(previousResults.end()-2)));
        ret = std::max(errToOneBack, errToTwoBack);
    }
    return ret;
}

template <typename Scalar>
Scalar quad_tanh_sinh(std::function<Scalar(double)> f, double a, double b,
                      double absTol = -1, double relTol = -1,
                      double *absErr = 0, double *relErr = 0,
                      int finiteBounds = QUAD_BOTH_BOUNDS_FINITE,
                      int max_degree = 15)
{
    // Absolute tolerance
    double toleranceAbs = absTol;
    // Relative tolerance
    double toleranceRel = relTol;

    const int doublePrecisionBits = 53;
    const int tolExponent = -doublePrecisionBits;
    double machine_tol = std::ldexp(1, tolExponent);
    //std::cout << "machine_tol = " << machine_tol << std::endl;
    if (toleranceAbs <= 0) {
        toleranceAbs = machine_tol;
    }

    std::vector<Scalar> retVals;
    retVals.reserve(max_degree);
    std::vector<Node> nodes;
    for (int i = 1; i <= max_degree; ++i) {
        //std::cout << "Degree " << i << std::endl;
        nodes = calc_nodes_tanh_sinh(i);
        //std::cout << " Total nodes: " << nodes.size() << std::endl;
        auto iter = nodes.cbegin();
        auto end = nodes.cend();
        const double h = ldexp(1.0, -i);
        Scalar sum = 0.0;
        switch (finiteBounds) {
        case QUAD_BOTH_BOUNDS_FINITE:
        {
            //Simple linear change of variables
            const double C = (b-a)/2;
            const double D = (b+a)/2;
            for (; iter != end; ++iter) {
                const double w = C*(iter->weight);
                const double x = D+C*(iter->node);
                sum += f(x)*w;
            }
            break;
        }
        case QUAD_LOWER_BOUND_FINITE_UPPER_INFINITE:
        {
            // We integrate over the interval
            // (a, \infty). We use change of
            // variables t = \frac{2}{x+1} + (a-1)
            const double a1 = a-1;
                //a1 = a-1
                //for x, w in nodes:
                //    u = 2/(x+one)
                //    x = a1+u
                //    w *= half*u**2
            for (; iter != end; ++iter) {
                const double u = 2.0/(iter->node+1);
                const double x = a1+u;
                const double w = (iter->weight)*0.5*u*u;
                sum += f(x)*w;
            }
        }
        case QUAD_BOTH_BOUNDS_INFINITE:
            // We integrate over the interval
            // (-\infty, \infty). We use change of
            // variables t = \frac{x}{\sqrt{1-x^2}}
            for (; iter != end; ++iter) {
                const double px1 = 1-(iter->node)*(iter->node);
                const double spx1 = pow(px1,-0.5);
                const double x = (iter->node)*spx1;
                const double w = (iter->weight)*spx1/px1;
                sum += f(x)*w;
            }
            break;
        default:
            assert(0 && "bounds type not implemented!");
        }
        sum *= h;
        if (!retVals.empty()) {
            sum += retVals.back()/2.0;

            // Enforce going at least to degree 3 before
            // returning;
            if (retVals.size() > 1) {
                const double err = quad_estimate_error(sum, retVals);
                double errRel;
                const double sumAbs = std::abs(sum);
                if (sumAbs == 0) {
                    errRel = 0;
                } else {
                    errRel = err/std::abs(sum);
                }
                if (absErr != 0) {
                    *absErr = err;
                }
                if (relErr != 0) {
                    *relErr = errRel;
                }
                bool relativeToleranceSatisfied = false;
                if (toleranceRel <= 0) {
                    // The relative tolerance was not given. Hence,
                    // we cannot enforce it.
                    relativeToleranceSatisfied = true;
                } else if (errRel < toleranceRel) {
                    relativeToleranceSatisfied = true;
                }
                if (err < toleranceAbs && relativeToleranceSatisfied) {
                    return sum;
                }
            }
        }
        retVals.push_back(sum);
    }
#ifdef DEBUG
    std::cout << " Didn't converge!" << std::endl;
#endif
    // FIXME: Need to tell the caller that we didn't converge
    return retVals.back();
}

#endif // QUAD_H
