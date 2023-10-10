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

#ifndef SMD_FROM_SPMAT_H
#define SMD_FROM_SPMAT_H

#include "types.h"
#include "qroc/master_equation_roc.h"

inline SparseMatrixData smd_from_spmat(const SpMat &M)
{
    SparseMatrixData smd;
    smd.rows = M.rows();
    smd.cols = M.cols();
    smd.nnz = M.nonZeros();
    smd.val = M.valuePtr();
    smd.col = M.innerIndexPtr();
    smd.ptr = M.outerIndexPtr();
    return smd;
}

#endif // SMD_FROM_SPMAT_H
