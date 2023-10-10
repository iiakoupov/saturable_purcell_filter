/* Copyright (c) 2012-2017 Ivan Iakoupov
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

#include "quad.h"

#include <cmath>
#include <iostream>


double quad_midpoint(funcRtoR f, double a, double b, unsigned N)
{
    const double step_x = std::abs(b-a)/N;
    double ret = 0.0;
    for (unsigned i = 0; i < N; ++i) {
        double x_i = a + b*step_x*(i+0.5);
        ret += f(x_i)*step_x;
    }
    if (b >= a) {
        return ret;
    } else {
        return -ret;
    }
}

double estimate_error(std::complex<double> thisResult,
                      const std::vector<std::complex<double> > &previousResults)
{
    assert(!previousResults.empty());
    if (previousResults.size() > 1) {
        /* If there're more than one previous result then take
         * the one before the last one */
        return std::abs(thisResult - (*(previousResults.end()-2)));
    } else {
        return std::abs(thisResult - previousResults.back());
    }
}

#define DOUBLE_PRECISION_BITS 53

std::vector<Node> calc_nodes_tanh_sinh(int degree, double userTolerance)
{
    std::vector<Node> ret;
    double tol = userTolerance;
    if (tol <= 0) {
        tol = std::ldexp(1, -DOUBLE_PRECISION_BITS-10);
    }
    const double t0 = std::ldexp(1, -degree);
    double h = t0*2;
    /* For simplicity, we work in steps h = 1/2^n, with the first point
     * offset so that we can reuse the sum from the previous degree */

    /* We define degree 1 to include the "degree 0" steps, including
     * the point x = 0. (It doesn't work well otherwise; not sure why.) */
    if (degree == 1) {
        Node zero_node = {0, M_PI_2};
        ret.push_back(zero_node);
        h = t0;
    }
    const double expt0 = std::exp(t0);
    double a = M_PI_4 * expt0;
    double b = M_PI_4 / expt0;
    const double udelta = std::exp(h);
    const double urdelta = 1/udelta;
    const int max_k = static_cast<int>(std::ldexp(20, degree+1));

    for (int k = 0; k < max_k; ++k) {
        /* Reference implementation:
         * t = t0 + k*h
         * x = tanh(pi/2 * sinh(t))
         * w = pi/2 * cosh(t) / cosh(pi/2 * sinh(t))**2
         */

        /* Fast implementation. Note that c = exp(pi/2 * sinh(t)) */
        const double c = std::exp(a-b);
        const double d = 1/c;
        const double co = (c+d)/2;
        const double si = (c-d)/2;
        const double x = si/co;
        const double w = (a+b) / std::pow(co, 2);
        const double diff = std::abs(x-1);
        if (diff <= tol) {
            break;
        }

        Node positive_node = {x, w};
        Node negative_node = {-x, w};
        ret.push_back(positive_node);
        ret.push_back(negative_node);

        a *= udelta;
        b *= urdelta;
    }
    return ret;
}

std::vector<Node> calc_nodes_tanh_sinh_final(double a, double b, int max_degree,
                                             int finiteBounds)
{
    std::vector<Node> all_nodes;
    int degree = 0;
    // Ensure that the maximal degree is at least 3
    for (degree = 1; degree <= max_degree; ++degree) {
        std::vector<Node> nodes = calc_nodes_tanh_sinh(degree);
        all_nodes.reserve(all_nodes.size() + nodes.size());
        all_nodes.insert(all_nodes.end(), nodes.begin(), nodes.end());
    }
    int N_t = all_nodes.size();

    const double h = ldexp(1.0, -max_degree);

    for (int i = 0; i < N_t; ++i) {
        all_nodes[i].weight *= h;
    }
    const double C = (b-a)/2;
    const double D = (b+a)/2;
    for (int i = 0; i < N_t; ++i) {
        all_nodes[i].node = D+C*all_nodes[i].node;
        all_nodes[i].weight = C*all_nodes[i].weight;
    }
    return all_nodes;
}

std::vector<Node> all_nodes_tanh_sinh_semi_inf_quad(double a, int max_degree)
{
    std::vector<Node> all_nodes;
    int degree = 0;
    // Ensure that the maximal degree is at least 3
    for (degree = 1; degree <= max_degree; ++degree) {
        std::vector<Node> nodes = calc_nodes_tanh_sinh(degree);
        all_nodes.reserve(all_nodes.size() + nodes.size());
        all_nodes.insert(all_nodes.end(), nodes.begin(), nodes.end());
    }
    int N_t = all_nodes.size();

    const double h = ldexp(1.0, -max_degree);

    for (int i = 0; i < N_t; ++i) {
        all_nodes[i].weight *= h;
    }
    // We integrate over the interval
    // (a, \infty). We use change of
    // variables t = \frac{2}{x+1} + (a-1)
    const double a1 = a-1;
    //a1 = a-1
    //for x, w in nodes:
    //    u = 2/(x+one)
    //    x = a1+u
    //    w *= half*u**2
    for (int i = 0; i < N_t; ++i) {
        const double u = 2.0/(all_nodes[i].node+1);
        const double x = a1+u;
        const double w = (all_nodes[i].weight)*0.5*u*u;
        all_nodes[i].node = x;
        all_nodes[i].weight = w;
    }
    return all_nodes;
}
