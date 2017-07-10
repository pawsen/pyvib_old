#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy import linalg
from scipy.sparse.linalg import eigs
from scipy.sparse import csr_matrix, vstack, hstack, identity
from scipy.sparse import issparse


class nonlinearity(object):
    def __init__(self, dof, order, value):
        self.dof = dof
        self.order = order
        self.value = value


class pseudo_random(object):
    """Periodic random signal with user controlled amplitude spectrum, and a
    random phase spectrum drawn from a uniform distribution. Band limited by
    f_min and f_max

    Parameters
    ----------
    RMS : float
        (Root mean square) amplitude of signal
    n : int
        number of periods.
    f_min, f_max : float
        lower and upper frequency bound.

    """
    def __init__(self, rms, n, band):
        self.rms = rms
        self.n = n
        self.band = band

    def eval_force(self, u, t):
        return

class Sweep(object):
    """Sine sweep

    Parameters
    ----------
    n : int
        sweep rate
    f_min, f_max : float
        lower and upper bound for frequency sweep.

    """
    def __init__(self, A, n, omega):
        self.A = A
        self.n = n
        self.omega = omega

    def eval_force(self, u, t):
        return self.A * np.sin(self.omega * t)

class Sine(object):
    """Constant frequency sine excitation.

    Parameters
    ----------
    dof : int
    A : float
        Amplitude
    omega : float
        Angular frequency (rad/s)

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
        vals : float[neigs]
            The eigenvalues, each repeated according to its multiplicity. The
            eigenvalues are not necessarily ordered.
        vesc: float[neqs,neigs]
            the mass normalized (unit “length”) eigenvectors. The column
            vals[:,i] is the eigenvector corresponding to the eigenvalue
            vecs[i].

        Undamped frequencies can be found directly from (matlab syntax). Remove
        diag when using python.
        * [phi, w2] = eigs(K,M,6,'sm');
        * f = sort(sqrt(diag(w2))) / (2*pi)

        """

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

        # Scale eigenvectors so that \phi^T * [M] * \phi = 1
        self.scale_eigenvec()

        # Check that the first eigenvalue and vector satifies the QEP eq
        # l = vals[0]
        # x = vecs[:,0]
        # (M.dot(l**2) + C.dot(l) + K).dot(x)

        return self.w0, self.w0d, self.psi, self.vecs

    def freqs_hz(self, w):
        """Returns eigenfrequencies in (Hz)"""
        return w / (2 * np.pi)

    def scale_eigenvec(self):
        """Mass normalization

        v.T*M*v = 1
        This in turn implies that
        v.T*M*v = omega"""
        for i in range(self.vecs.shape[1]):
            self.vecs[:,i] = self.vecs[:,i] / np.sqrt(self.vecs[:,i].dot(self.M.dot(self.vecs[:,i])))

    def time_harmonic(self, Omega, rhs, damped=False):
        """Time harmonic(steady state) solution of S*u=f

        u is the displacement and f the forcing.
        Omega is the frequency of the harmonic exitation.

        If damping is present, the solution is complex:
        u = u_r + iu_i
        The time-dependent solution is then
        u(t) = Real(u*e^(i*Omega*t)) = u_r*cos(Omega*t) + u_i*sin(Omega*t)

        Parameters
        ----------
        rhs : float[ndof]
            force vector
        """
        from scipy.sparse.linalg import spsolve

        if damped:
            S = csr_matrix(-Omega**2*self.M + 1j*Omega*self.C + self.K,
                           dtype=complex)
        else:
            S = csr_matrix(-Omega**2*self.M + self.K)

        u = spsolve(S, rhs)

        return u


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
        """Newmark-beta nonlinear integration.

        With gamma = 1/2, beta = 1/4, this correspond to the "Average
        acceleration" Method. Unconditional stable. Convergence: O(dt**2).

        No enforcing of boundary conditions, eg. only solves IVP.

        Parameters
        ----------
        xo, dx0 : float[ndof]
            Initial conditions
        t : float[nsteps]
            Time vector
        r_ext(t) : float[ndof]
            External force function. Takes the current time as input.
            Returns an array. Size [ndof]

        Returns
        -------
        x, dx, ddx - float[nsteps, ndof]
            State arrays. Size [nsteps, ndof]

        Note
        ----
        Equations are from Krenk: "Non-linear Modeling and analysis of
        Solids and Structures"
        See also: Cook: "Concepts and applications of FEA, chap 11 & 17"
        """
        gamma = 1./2
        beta = 1./4

        if issparse(self.K):
            self.K = self.K.A
            self.C = self.C.A
            self.M = self.M.A

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

# M = np.eye([3, 1, 3, 1])
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
