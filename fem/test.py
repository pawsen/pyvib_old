#!/usr/bin/env python3


def concatenate_csr_matrices_by_rows(matrix1, matrix2):
    """Using hstack, vstack, or concatenate, is dramatically slower than
    concatenating the inner data objects themselves. The reason is that
    hstack/vstack converts the sparse matrix to coo format which can be
    very slow when the matrix is very large not and not in coo format.

    """
    new_data = np.concatenate((matrix1.data, matrix2.data))
    new_indices = np.concatenate((matrix1.indices, matrix2.indices))
    new_ind_ptr = matrix2.indptr + len(matrix1.data)
    new_ind_ptr = new_ind_ptr[1:]
    new_ind_ptr = np.concatenate((matrix1.indptr, new_ind_ptr))

    return csr_matrix((new_data, new_indices, new_ind_ptr))


def transpose_csr_matrix(matrix1):
    """Transpose sparse csr-matrix. Because I cannot get
    concatenate_csr_matrices_by_columns(k1, k2.T).T
    to work
    """
    from scipy.sparse._sparsetools import csr_tocsc

    indptr = np.empty(matrix1.shape[1] + 1, dtype=int)
    indices = np.empty(matrix1.nnz, dtype=int)
    data = np.empty(matrix1.nnz)
    csr_tocsc(matrix1.shape[0], matrix1.shape[1],
              matrix1.indptr, matrix1.indices, matrix1.data,
              indptr, indices, data)
    return csr_matrix((data, indices, indptr))


                    A1 = concatenate_csr_matrices_by_rows(
                        transpose_csr_matrix(
                            concatenate_csr_matrices_by_rows(
                                self.C,
                                transpose_csr_matrix(self.K))),
                        transpose_csr_matrix(
                            concatenate_csr_matrices_by_rows(
                                -identity(dim[0],format='csr'),
                                csr_matrix(dim)))  # sparse zero matrix
                    )
                    B1 = concatenate_csr_matrices_by_rows(
                        transpose_csr_matrix(
                            concatenate_csr_matrices_by_rows(
                                self.M,
                                csr_matrix(dim, dim))),
                        transpose_csr_matrix(
                            concatenate_csr_matrices_by_rows(
                                csr_matrix(dim),
                                identity(dim[0],format='csr')))
                    )

       # Check that the first eigenvalue and vector satifies the QEP eq
        # l = vals[0]
        # x = vecs[:,0]
        # (M.dot(lambda**2) + C.dot(lambda) + K).dot(x)

# C = np.array([[0.4,0, -0.3, 0],[0, 0, 0, 0],
#               [-0.3, 0, 0.5, -0.2],[0, 0, -0.2, 0.2]])
# K = np.array([[-7, 2, 4, 0],[2, -4, 2, 0],
#               [4, 2, -9, 3],[0, 0, 3, -3]])
# from scipy.sparse import csr_matrix
# M = csr_matrix(M)
# C = csr_matrix(C)
# K = csr_matrix(K)
# vals: -2.4498, -2.1536, -1.6248, 2.2279, 2.0364, 1.4752, 0.3353, -0.3466
# https://se.mathworks.com/help/matlab/ref/polyeig.html

import numpy as np

def is_pos_def(x):
    # if all the eigenvalues of matrix are positive, the matrix is positive
    # definite.
    return np.all(np.linalg.eigvals(x) > 0)
