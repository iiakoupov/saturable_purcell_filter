// Copyright (c) 2021 Ivan Iakoupov
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

#ifndef GET_PSI_FOR_EXCITED_ATOMS_H
#define GET_PSI_FOR_EXCITED_ATOMS_H

#include <vector>
#include "operator.h"

inline Eigen::VectorXcd get_psi_for_excited_atoms(const std::vector<int> &excited_atom_indices, const std::vector<BasisVector> &basis)
{
    const int basis_size = basis.size();
    Eigen::VectorXcd psi = Eigen::VectorXcd::Zero(basis_size);
    std::vector<sigma_state_t> spec(basis[0].size(), 0);
    const int num_excited_atoms = excited_atom_indices.size();
    for (int i = 0; i < num_excited_atoms; ++i) {
        spec[excited_atom_indices[i]] = 1;
    }
    BasisVector vec(spec);
    for (int i = 0; i < basis_size; ++i) {
        if (basis[i] == vec) {
            psi[i] = 1;
            break;
        }
    }
    return psi;
}

#endif // GET_PSI_FOR_EXCITED_ATOMS_H
