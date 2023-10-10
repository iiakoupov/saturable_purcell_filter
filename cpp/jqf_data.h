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

#ifndef JQF_DATA_H
#define JQF_DATA_H

#include <vector>

struct JQFData
{
    std::vector<double> time;
    std::vector<double> Omega_Re;
    std::vector<double> Omega_Im;
    std::vector<std::vector<double>> res_populations;
    std::vector<std::vector<double>> aa_populations;
    std::vector<std::vector<std::vector<double>>> aa_level_populations;
    std::vector<double> purity_1;
    std::vector<double> purity_1r;
    std::vector<std::vector<double>> F;
    std::vector<std::vector<double>> F_down;
    std::vector<double> tilde_F;
    std::vector<double> fidelity_under_optimization;
    std::vector<double> window_array;
    std::vector<std::vector<double>> x_array;
};

#endif // JQF_DATA_H
