#ifndef MKL_SUPPORT_H
#define MKL_SUPPORT_H

#include "types.h"
#include "csrmatrix.h"

class MKLSparseMatrix
{
private:
    bool m_initialized;
    sparse_matrix_t csrM;
    int m_rows;
    int m_cols;
public:
    MKLSparseMatrix() :
        m_initialized(false),
        m_rows(0),
        m_cols(0)
    {}
    explicit MKLSparseMatrix(const SpMat &M) :
        m_initialized(true),
        m_rows(M.rows()),
        m_cols(M.cols())
    {
        assert(M.isCompressed() && "MKL needs compressed sparse matrices!");
        mkl_sparse_z_create_csr(&csrM, SPARSE_INDEX_BASE_ZERO,
                                M.outerSize(),  // number of rows
                                M.innerSize(),  // number of cols
                                (MKL_INT *)M.outerIndexPtr(),
                                (MKL_INT *)(M.outerIndexPtr()+1),
                                (MKL_INT *)M.innerIndexPtr(),
                                (MKL_Complex16 *)M.valuePtr());
        mkl_sparse_optimize(csrM);
    }
    explicit MKLSparseMatrix(const CSRMatrix &M) :
        m_initialized(true),
        m_rows(M.rows),
        m_cols(M.cols)
    {
        mkl_sparse_z_create_csr(&csrM, SPARSE_INDEX_BASE_ZERO,
                                M.rows,  // number of rows
                                M.cols,  // number of cols
                                (MKL_INT *)M.ptr.data(),
                                (MKL_INT *)(M.ptr.data()+1),
                                (MKL_INT *)M.col.data(),
                                (MKL_Complex16 *)M.val.data());
        mkl_sparse_optimize(csrM);
    }
    ~MKLSparseMatrix()
    {
        if (m_initialized) {
            mkl_sparse_destroy(csrM);
        }
    }
    void setFromSpMat(const SpMat &M)
    {
        assert(M.isCompressed() && "MKL needs compressed sparse matrices!");
        m_initialized = true;
        m_rows = M.rows();
        m_cols = M.cols();
        mkl_sparse_z_create_csr(&csrM, SPARSE_INDEX_BASE_ZERO,
                                M.outerSize(),  // number of rows
                                M.innerSize(),  // number of cols
                                (MKL_INT *)M.outerIndexPtr(),
                                (MKL_INT *)(M.outerIndexPtr()+1),
                                (MKL_INT *)M.innerIndexPtr(),
                                (MKL_Complex16 *)M.valuePtr());
        mkl_sparse_optimize(csrM);
    }
    void setFromCSRMatrix(const CSRMatrix &M)
    {
        m_initialized = true;
        m_rows = M.rows;
        m_cols = M.cols;
        mkl_sparse_z_create_csr(&csrM, SPARSE_INDEX_BASE_ZERO,
                                M.rows,  // number of rows
                                M.cols,  // number of cols
                                (MKL_INT *)M.ptr.data(),
                                (MKL_INT *)(M.ptr.data()+1),
                                (MKL_INT *)M.col.data(),
                                (MKL_Complex16 *)M.val.data());
        mkl_sparse_optimize(csrM);
    }
    void mul_vector(Eigen::VectorXcd &ret, const Eigen::VectorXcd &v, double factor, double factor_ret = 0) const
    {
        MKL_Complex16 alpha = {factor,0};
        MKL_Complex16 beta = {factor_ret,0};
        struct matrix_descr descrM;
        descrM.type = SPARSE_MATRIX_TYPE_GENERAL;

        mkl_sparse_z_mv(SPARSE_OPERATION_NON_TRANSPOSE,
                        alpha,
                        csrM,
                        descrM,
                        (MKL_Complex16 *)v.data(),
                        beta,
                        (MKL_Complex16 *)ret.data());
    }
    void mul_matrix(Eigen::MatrixXcd &ret, const Eigen::MatrixXcd &m, double factor, double factor_ret = 0) const
    {
        MKL_Complex16 alpha = {factor,0};
        MKL_Complex16 beta = {factor_ret,0};
        struct matrix_descr descrM;
        descrM.type = SPARSE_MATRIX_TYPE_GENERAL;
        mkl_sparse_z_mm(SPARSE_OPERATION_NON_TRANSPOSE,
                        alpha,
                        csrM,
                        descrM,
                        SPARSE_LAYOUT_COLUMN_MAJOR,
                        (MKL_Complex16 *)m.data(),
                        (MKL_INT)ret.cols(),
                        (MKL_INT)m.rows(),
                        beta,
                        (MKL_Complex16 *)ret.data(),
                        (MKL_INT)ret.rows());
    }
    void mul_matrix_row_major(std::complex<double> * __restrict__ ret, int ret_rows, int ret_cols, const std::complex<double> * __restrict__ m, int m_cols, std::complex<double> factor, double factor_ret = 0) const
    {
        MKL_Complex16 alpha = {factor.real(),factor.imag()};
        MKL_Complex16 beta = {factor_ret,0};
        struct matrix_descr descrM;
        descrM.type = SPARSE_MATRIX_TYPE_GENERAL;
        mkl_sparse_z_mm(SPARSE_OPERATION_NON_TRANSPOSE,
                        alpha,
                        csrM,
                        descrM,
                        SPARSE_LAYOUT_ROW_MAJOR,
                        (MKL_Complex16 *)m,
                        (MKL_INT)ret_cols,
                        (MKL_INT)m_cols,
                        beta,
                        (MKL_Complex16 *)ret,
                        (MKL_INT)ret_cols);
    }
    int rows() const { return m_rows; };
    int cols() const { return m_cols; };
};

class MKLSparseMatrixReal
{
private:
    sparse_matrix_t csrM;
    int m_rows;
    int m_cols;
    bool m_symmetric;
public:
    explicit MKLSparseMatrixReal(const SpMatReal &M) :
        m_rows(M.rows()),
        m_cols(M.cols()),
        m_symmetric(false)
    {
        assert(M.isCompressed() && "MKL needs compressed sparse matrices!");
        mkl_sparse_d_create_csr(&csrM, SPARSE_INDEX_BASE_ZERO,
                                M.outerSize(),  // number of rows
                                M.innerSize(),  // number of cols
                                (MKL_INT *)M.outerIndexPtr(),
                                (MKL_INT *)(M.outerIndexPtr()+1),
                                (MKL_INT *)M.innerIndexPtr(),
                                (double *)M.valuePtr());
        mkl_sparse_optimize(csrM);
    }
    void setSymmetric(bool value)
    {
        m_symmetric = value;
    }
    bool symmetric() const
    {
        return m_symmetric;
    }
    ~MKLSparseMatrixReal()
    {
        mkl_sparse_destroy(csrM);
    }
    void mul_vector(Eigen::VectorXd &ret, const Eigen::VectorXd &v, double factor, double factor_ret = 0) const
    {
        const double alpha = factor;
        const double beta = factor_ret;
        struct matrix_descr descrM;
        if (m_symmetric) {
            descrM.type = SPARSE_MATRIX_TYPE_SYMMETRIC;
            descrM.mode = SPARSE_FILL_MODE_UPPER;
            descrM.diag = SPARSE_DIAG_NON_UNIT;
        } else {
            descrM.type = SPARSE_MATRIX_TYPE_GENERAL;
        }

        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,
                        alpha,
                        csrM,
                        descrM,
                        (double *)v.data(),
                        beta,
                        (double *)ret.data());
    }
    int rows() const { return m_rows; };
    int cols() const { return m_cols; };
};
#endif // MKL_SUPPORT_H
