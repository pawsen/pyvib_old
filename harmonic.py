#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy import linalg
from scipy.sparse.linalg import eigs
from scipy.sparse import csr_matrix, vstack, hstack, identity

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


class nonlinearity(object):
    def __init__(self, dof, order, value):
        self.dof = dof
        self.order = order
        self.value = value


class pseudo_random(object):
    """
    Periodic random signal with user controlled amplitude spectrum, and a
    random phase spectrum drawn from a uniform distribution.
    Band limited by f_min and f_max

    Input:
        RMS (Root mean square) amplitude of signal
        n: number of periods. [Integer]
        f_min, f_max: lower and upper frequency bound.
    """
    def __init__(self, rms, n, band):
        self.rms = rms
        self.n = n
        self.band = band

    def eval_force(self, u, t):
        return

class Sweep(object):
    """
    Sine sweep

    Input:
        n: sweep rate
        f_min, f_max: lower and upper bound for frequency sweep.
    """
    def __init__(self, A, n, omega):
        self.A = A
        self.n = n
        self.omega = omega

    def eval_force(self, u, t):
        return self.A * np.sin(self.omega * t)

class Sine(object):
    """
    Constant frequency sine excitation.

    Input:
        omega: angular frequency (rad/s)
    """
    def __init__(self, dof, A, omega):
        self.dof = dof
        self.A = A
        self.omega = omega

    def eval_force(self, u, du, t):
        return self.A * np.sin(np.pi * self.omega * t)


class Solver(object):
    def __init__(self, M, K, C=None):
        self.M = M
        self.C = C
        self.K = K

        self.list_nonlin = []
        self.list_dfnonlin = []
        self.list_force = []

        self.w0 = []
        self.w0d = []
        self.phi = []
        self.vecs = []

    def add_nonlin(self, dof, order, value):
        self.list_nonlin.append(nonlinearity(dof, order, value))
        # derivative of the nonlinearity
        self.list_dfnonlin.append(nonlinearity(dof, order - 1, value * order))

    def add_force(self, ftype):
        self.list_force.append(ftype)

    def integrate(self, x0, dx0, t):
        return Newmark.newmark_beta(self, x0, dx0, t)

    def eigen(self, neigs=6, damped=False):
        """Calculate damped and undamped eigenfrequencies in (rad/s).
        See Jakob S. Jensen & Niels Aage: "Lecture notes: FEMVIB", eq. 3.50 for
        the state space formulation of the EVP.

        Use eigs (Works on sparse matrices and calls Arpack under the hood)

        Returns
        -------
        vals: The eigenvalues, each repeated according to its multiplicity.
            The eigenvalues are not necessarily ordered.

        vesc: the normalized (unit “length”) eigenvectors, such that the
            column vals[:,i] is the eigenvector corresponding to the
            eigenvalue vecs[i].


        Undamped frequencies can be found directly from (matlab syntax). Remove
        diag when using python.
        * [phi, w2] = eigs(K,M,6,'sm');
        * f = sort(sqrt(diag(w2))) / (2*pi)

        """
        from scipy.sparse import issparse

        dim = np.shape(self.K)
        if damped:
            """ The generalized eigenvalue problem can be stated in different
            ways. scipy.sparse.linalg.eigs requires B to positive definite. A
            common formulation[1]:
            A = [C, K; -eye(size(K)), zeros(size(K))]
            B = [M, zeros(size(K)); zeros(size(K)), eye(size(K))]

            with corresponding eigenvector
            z = [lambda x; x]
            We can then take the first neqs components of z as the eigenvector x

            Note that the system coefficient matrices are always positive
            definite for elastic problems. Thus the formulation:
            A = [zeros(size(K)),K; K, C];
            B = [K, zeros(size(K)); zeros(size(K)), -M];
            does not work with scipy.sparse. But does work with
            scipy.linalg.eig. See also [2] and [3]

            [1]
            https://en.wikipedia.org/wiki/Quadratic_eigenvalue_problem
            [2]
            https://mail.python.org/pipermail/scipy-dev/2011-October/016670.html
            [3]
            https://scicomp.stackexchange.com/questions/10940/solving-a-generalised-eigenvalue-problem
            """

            if issparse(self.K):
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

                A = hstack([self.C, self.K], format='csr')
                A = vstack((A, hstack([-identity(dim[0]), csr_matrix(dim)])), format='csr')
                B = hstack([self.M, csr_matrix(dim)], format='csr')
                B = vstack((B, hstack([csr_matrix(dim), identity(dim[0])])), format='csr')

            else:
                A = np.column_stack([self.C, self.K])
                A = np.row_stack((A, np.column_stack([-np.eye(dim[0]), np.zeros(dim)])))
                B = np.column_stack([self.M, np.zeros(dim)])
                B = np.row_stack((B, np.column_stack([np.zeros(dim), np.eye(dim[0])])))

        else:
            A = self.K
            B = self.M

        # For SDOF eigs is not useful.
        if dim[0] < 12 and not issparse(self.K):
            vals, vecs = linalg.eig(A, b=B)

        else:
            vals, vecs = eigs(A, k=neigs, M=B, which='SR')

            if damped:
                # extract eigenvectors as the first half
                vecs = np.split(vecs, 2)[0]


        # remove complex conjugate elements(ie. cut vector/2d array in half)
        # Be sure to remove conjugate and the ones we want to keep!
        # TODO: THIS used to be commented out. Why did I comment it out?
        # Maybe because they are not sorted?
        # vals = np.split(vals,2)[0]
        # vecs = np.hsplit(vecs,2)[0]

        # NB! Remember when sorting the eigenvalues, that the eigenvectors
        # should be sorted accordingly, so they still match
        idx = vals.argsort()[::1]
        w2_complex = vals[idx]
        self.vecs = vecs[:, idx]

        # extract complex eigenvalues.
        # taking the absolute is simply: sqrt(real(w2)**2 + imag(w2)**2)
        self.w0 = np.abs(w2_complex)
        self.w0d = np.abs(np.imag(w2_complex))
        # damping ration
        self.psi = -np.real(w2_complex) / self.w0

        # TODO maybe scale eigenvectors so that \phi^T * [M] * \phi = 1

        return self.w0, self.w0d, self.psi, self.vecs

    def convert_freqs(self):
        """ Returns undamped damped and eigenfrequencies in (Hz)"""
        return \
            self.w0 / (2 * np.pi), \
            self.w0d / (2 * np.pi)

    def scale_eigenvec(self):
        for i in range(self.vecs.shape[1]):
            self.vecs[:,i] = self.vecs[:,i].dot(self.M.dot(self.vecs[:,i]))

class Newmark(Solver):
    def f(self, u):
        ndof = len(u)
        force = np.zeros(ndof)
        for nonlin in self.list_nonlin:
            dof = nonlin.dof
            value = nonlin.value
            order = nonlin.order
            force[dof-1] = force[dof-1] + value * u[dof-1]**order
        return force

    def df(self, u):
        ndof = len(u)
        force = np.zeros(ndof)
        for nonlin in self.list_dfnonlin:
            dof = nonlin.dof
            value = nonlin.value
            order = nonlin.order
            force[dof-1] = force[dof-1] + value * u[dof-1]**order
        return force

    def r_ext(self, u, du, t):
        ndof = len(u)
        fvec = np.zeros(ndof)
        for force in self.list_force:
            dof = force.dof
            fvec[dof-1] = fvec[dof-1] + \
                          force.eval_force(u[dof-1], du[dof-1], t)

        return fvec

    def newmark_beta(self, x0, dx0, t):
        '''
        Newmark-beta nonlinear integration.
        With gamma = 1/2, beta = 1/4, this correspond to the "Average acceleration"
        Method. Unconditional stable. Convergence: O(dt**2).

        No enforcing of boundary conditions, eg. only solves IVP.
        Input:
            xo, dx0
            - Initial conditions. Size [ndof]
            t
            - Time vector. Size[nsteps]
            r_ext(t)
            - External force function.
            - Takes the current time as input.
            - Returns an array. Size [ndof]

        Output:
            x, dx, ddx
            - State arrays. Size [nsteps, ndof]

        Equations are from Krenk: "Non-linear Modeling and analysis of Solids
            and Structures"
        See also: Cook: "Concepts and applications of FEA, chap 11 & 17"
        '''
        gamma = 1./2
        beta = 1./4

        nsteps = len(t)
        ndof = len(self.K)
        # Pre-allocate arrays. Use empty
        ddx = np.zeros([nsteps, ndof], dtype=float)
        dx = np.zeros([nsteps, ndof], dtype=float)
        x = np.zeros([nsteps, ndof], dtype=float)
        delta_x = np.zeros([nsteps, ndof], dtype=float)

        x[0,:] = x0
        dx[0,:] = dx0
        # initial acceleration. eq. 11.12-13
        f = Newmark.f(self, x[0,:])
        r_ext = Newmark.r_ext(self, x[0,:], dx[0,:], t[0])
        r_int = np.dot(self.K,x[0,:]) + np.dot(self.C,dx[0,:]) + f
        ddx[0,:] = linalg.solve(self.M, r_ext - r_int)

        # time stepping
        for j in range(1, nsteps):
            dt = t[j] - t[j-1]
            # Prediction step
            ddx[j,:] = ddx[j-1,:]
            dx[j,:] = dx[j-1,:] + dt * ddx[j-1,:]
            x[j,:] = x[j-1,:] + dt * dx[j-1,:] + 1/2 * dt**2 * ddx[j-1,:]

            # correct prediction step
            for i in range(0,20):
                # system matrices and increment correction
                """ calculate tangent stiffness.
                r(u) : residual
                Kt   : Tangent stiffness. Kt = ∂r/∂u

                r(u,u̇) = Cu̇ + Ku + f(u,u̇) - p
                Kt = ∇{u}r = K + ∇{u}f
                Ct = ∇{u̇}r = C + ∇{u̇}f
                """
                # residual calculation
                f = Newmark.f(self, x[j,:])
                res = Newmark.r_ext(self, x[j,:], dx[j,:], t[j]) - \
                      np.dot(self.M, ddx[j,:]) - \
                      np.dot(self.C, dx[j,:]) - np.dot(self.K, x[j,:]) - f

                df = Newmark.df(self, x[j,:])
                Kt = self.K + np.diag(df)
                Keff = 1 / (beta * dt**2) * self.M + \
                       gamma * dt / (beta * dt**2) * self.C + Kt

                delta_x = linalg.solve(Keff, res)

                x[j,:] = x[j,:] + delta_x
                dx[j,:] = dx[j,:] + gamma * dt / (beta * dt**2) * delta_x
                ddx[j,:] = ddx[j,:] + 1 / (beta * dt**2) * delta_x

                res_norm = (linalg.norm(res))
                delta_x_norm = (linalg.norm(delta_x))
                #print("j: {}, i: {}, delta_x: {}, res: {}, dx_norm: {}".
                #      format(j,i,delta_x,res_norm,delta_x_norm))
                if (res_norm <= 1e-10) or (delta_x_norm <= 1e-10):
                    break

                # energy norm. Could be used instead. Eq: Cook: 17.7-6
                # conv = dt * np.dot(dx1, res)
                # if (conv) <= 1e-6:
                #    break

        return x, dx, ddx


# M, C, K = np.array([[2]]), np.array([[10]]), np.array([[100e3]])
# k3 = 100e6
# sys = Solver(M, C, K)
# sys.add_nonlin(1, 3, k3)
# sys.add_force(Sine(1,1,1))
# x, dx, ddx = sys.integrate(x0, y0, t)
