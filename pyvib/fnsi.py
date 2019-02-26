#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.linalg import (solve, norm, expm)
from scipy.interpolate import interp1d
from numpy.fft import fft

from .common import meanVar
from .common import (lm, mmul_weight, weightfcn)
from .subspace import subspace, discrete2cont, modal_list, ss2phys
from .modal import modal_ac, stabilization
from .pnlss import transient_indices_periodic, remove_transient_indices_periodic
from .spline import spline
from .helper.modal_plotting import plot_frf, plot_stab

from collections import namedtuple

class FNSI():
    def __init__(self, u, y, fmin=None, fmax=None, xpowers=None, flines=None,
                 yd=None, fs=None):

        self.R = 1
        npp, m, P = u.shape
        npp, p, P = y.shape
        self.u = u
        self.y = y
        self.yd = yd

        if flines is None:
            f1 = int(np.floor(fmin/fs * npp))
            f2 = int(np.ceil(fmax/fs * npp))
            flines = np.arange(f1,f2+1)

        self.flines = flines
        self.fs = fs
        self.fmin = fmin
        self.fmax = fmax
        self.T1 = None
        self.T2 = None
        self.dt = 1/fs
        self.xpowers = np.atleast_2d(xpowers)
        self.xd_powers, self.xd_coeff = poly_deriv(self.xpowers)
        self.n_nx = self.xpowers.shape[0]
        self.n_ny = self.n_nx
        self.npp = npp

    def calc_EY(self, isnoise=False, vel=False):
        """Calculate FFT of the extended input vector e(t) and the measured
        output y.

        The concatenated extended input vector e(t), is e=[u(t), g(t)].T, see
        eq (5). (E is called the Extended input spectral matrix and used for
        forming Ei, eq. (12)). Notice that the stacking order is reversed here.
        u(t) is the input force and g(y(t),ẏ(t)) is the functional nonlinear
        force calculated from the specified polynomial nonlinearity, see eq.(2)

        Returns
        -------
        E : ndarray (complex)
            FFT of the concatenated extended input vector e(t)
        Y : ndarray (complex)
            FFT of y.

        Notes
        -----
        Method by J.P Noel. Described in article
        "Frequency-domain subspace identification for nonlinear mechanical
        systems"
        http://dx.doi.org/10.1016/j.ymssp.2013.06.034
        Equation numbers refers to this article

        """
        print('E and Y comp.')
        npp, self.p, P = self.y.shape
        npp, self.m, P = self.u.shape

        U = fft(self.u,axis=0) / np.sqrt(npp)
        Y = fft(self.y,axis=0) / np.sqrt(npp)

        # average over periods
        Ymean = np.sum(Y,axis=2) / P
        Umean = np.sum(U,axis=2) / P

        # Set weights to none, if the signal is not noisy
        if isnoise is False:
            WY = None

        # calculate derivative if yd is not given
        if self.yd is None and vel is True:
            yd = derivative(Y)
        else:
            yd = self.yd

        if yd is not None:
            Yd = fft(yd,axis=0) / np.sqrt(npp)
            Ydmean = np.sum(Yd,axis=2) / P
            yd = np.sum(yd, axis=2) / P

        Yext = np.hstack((Ymean, Ydmean)) if yd is not None else Ymean

        # In case of no nonlinearities
        if self.xpowers is None:
            scaling = []
            E = Umean
        else:
            # average displacement
            y = np.sum(self.y, axis=2) / P
            # ynl: [npp, 2*p]
            ynl = np.hstack((y, yd)) if yd is not None else y

            nnl = self.xpowers.shape[0]
            repmat_x = np.ones(nnl)

            # einsum does np.outer(repmat_x, ynl[i]) for all i
            fnl = np.prod(np.einsum('i,jk->jik',repmat_x, ynl)**self.xpowers,
                          axis=2)

            scaling = np.zeros(nnl)
            for j in range(nnl):
                scaling[j] = np.std(self.u[:,0]) / np.std(fnl[:,j])
                fnl[:,j] *= scaling[j]

            FNL = np.fft.fft(fnl, axis=0) / np.sqrt(npp)
            # concatenate to form extended input spectra matrix
            E = np.hstack((Umean, -FNL))

        self.U = E
        self.Y = Yext
        self.W = WY
        self.scaling = scaling

    def svd_comp(self, r):
        self.ims = r

    def id(self, n, bd_method='explicit'):

        flines = self.flines
        U = self.U[flines]
        Y = self.Y[flines]

        # normalized frequency [0-0.5]
        freq = self.flines / self.npp
        r = self.ims
        covarG = None
        G = None
        dt = 1/self.fs
        Ad, Bd, Cd, Dd, z, isstable = \
            subspace(G, covarG, freq, n, r, U, Y, bd_method)
        # convert to cont. time. Only A and B changes
        Ac, Bc, Cc, Dc = discrete2cont(Ad, Bd, Cd, Dd, dt)

        m, p = self.m, self.p
        self.n = n

        # extract nonlinear coefficients
        n_nx = self.xpowers.shape[0]
        E = np.zeros((n, n_nx))
        F = np.zeros((p, n_nx))
        for i in range(n_nx):
            E[:,i] = - self.scaling[i]*Bd[:,m+i]
            F[:,i] = - self.scaling[i]*Dd[:,m+i]

        self.Ac = Ac
        self.Bc = Bc[:,:m]
        self.C = Cd
        self.D = Dd[:,:m]
        self.Ad = Ad
        self.Bd = Bd[:,:m]
        self.A = self.Ad
        self.B = self.Bd

        self.E = E
        self.F = F
        self.xactive = np.arange(E.size)
        self.yactive = np.arange(F.size)

    def setss(self, *system):
        if len(system) == 6:
            self.A, self.B, self.C, self.D, self.E, self.F = system

    def weightfcn(self):
        try:
            self.covY
        except AttributeError:
            self.covY = covariance(self.signal.y)
        return weightfcn(self.covY)

    def optimize(self, method=None, weight=True, info=True, nmax=50, lamb=None,
                 ftol=1e-8, xtol=1e-8, gtol=1e-8, copy=False):
        """Optimize the estimated the nonlinear state space matrices"""

        if weight is True:
            try:
                self.weight
            except AttributeError:
                self.weight = self.weightfcn()
        else:
            self.weight = weight

        self.freq_weight = True
        if self.weight is None:
            self.freq_weight = False

        if info:
            print('\nStarting PNLSS optimization')

        x0 = self.flatten_ss()
        if method is None:
            res = lm(costfcn, x0, jacobian, system=self, weight=self.weight,
                     info=info, nmax=nmax, lamb=lamb, ftol=ftol, xtol=xtol,
                     gtol=gtol)
        else:
            res = least_squares(costfcn,x0,jacobian, method='lm',
                                x_scale='jac',
                                kwargs={'system':self,'weight':self.weight})

        if copy:
            # restore state space matrices as they are
            self.A, self.B, self.C, self.D, self.E, self.F = extract_ss(x0, self)

            nmodel = deepcopy(self)
            nmodel.A, nmodel.B, nmodel.C, nmodel.D, nmodel.E, nmodel.F = \
                extract_ss(res['x'], nmodel)
            nmodel.res = res
            return nmodel

        self.A, self.B, self.C, self.D, self.E, self.F = extract_ss(res['x'], self)
        self.res = res

    def cost(self, weight=None):

        if weight is True:
            try:
                weight = self.weight
            except AttributeError:
                weight = self.weightfcn()

        x0 = self.flatten_ss()
        err = costfcn(x0, self, weight=weight)
        # TODO maybe divide by 2 to match scipy's implementation of minpack
        self.cost = np.dot(err, err)
        return self.cost

    def flatten_ss(self):
        """Returns the state space as flattened array"""

        # index of active elements
        xact = np.arange(self.E.size)
        yact = np.arange(self.F.size)
        ne = len(xact)
        nf = len(yact)

        # samples per period
        n, m, p = self.n, self.m, self.p
        self.npar = n**2 + n*m + p*n + p*m + ne + nf

        # initial guess
        x0 = np.empty(self.npar)
        x0[:n**2] = self.A.ravel()
        x0[n**2 + np.r_[:n*m]] = self.B.ravel()
        x0[n**2 + n*m + np.r_[:n*p]] = self.C.ravel()
        x0[n*(p+m+n) + np.r_[:p*m]] = self.D.ravel()
        x0[n*(p+m+n)+p*m + np.r_[:ne]] = self.E.flat[xact]
        x0[n*(p+m+n)+p*m+ne + np.r_[:nf]] = self.F.flat[yact]
        return x0

    def nl_coeff(self, iu, dofs):
        """Form the extended FRF (transfer function matrix) He(ω) and ectract
        nonlinear coefficients

        H(ω) is the linear FRF matrix, eq. (46)
        He(ω) is formed using eq (47)

        Parameters
        ----------
        fs : float
            Sampling frequency.
        N : int
            The number of time-domain samples.
        flines : ndarray(float)
            Vector of frequency lines where the nonlinear coefficients are
            computed.
        'A,B,C,D' : ndarray x ndarray
            The continuous-time state-space matrices.
        inl : ndarray
            A matrix contaning the locations of the nonlinearities in the system.
        iu : int
            The location of the force.
        dofs : ndarray (int)
            DOFs to calculate H(ω) for.

        Returns
        -------
        knl : ndarray(complex)
            The nonlinear coefficients (frequency-dependent and complex-valued)
        H(ω) : ndarray(complex)
            Estimate of the linear FRF
        He(ω) : ndarray(complex)
            The extended FRF (transfer function matrix)
        """

        dofs = np.atleast_1d(dofs)
        fs = self.fs
        nsper = self.nsper
        flines = self.flines
        A = self.A
        B = self.B
        C = self.C
        D = self.D
        scaling = self.scaling

        freq = np.arange(0,nsper)*fs/nsper
        F = len(flines)

        l, m = D.shape

        # just return in case of no nonlinearities
        if len(self.nonlin.nls) == 0:
            nnl = 0
        else:
            nnl = 0
            for nl in self.nonlin.nls:
                nnl += nl.nnl
                nl.knl = np.empty((nl.nnl,F),dtype=complex)

        m = m - nnl

        # Extra rows of zeros in He is for ground connections
        # It is not necessary to set inl's connected to ground equal to l, as
        # -1 already point to the last row.
        H = np.empty((len(dofs), F),dtype=complex)
        He = np.empty((l+1, m+nnl, F),dtype=complex)
        He[-1,:,:] = 0

        for k in range(F):
            # eq. 47
            He[:-1,:,k] = C @ solve(np.eye(*A.shape,dtype=complex)*2j*np.pi*
                                    freq[flines[k]] - A, B) + D

            i = 0
            for nl in self.nonlin.nls:
                # number of nonlin connections for the given nl type
                ninl = int(nl.nnl/nl.inl.shape[0])
                for j in range(nl.nnl):
                    idx = j//ninl
                    nl.knl[j,k] = scaling[i] * He[iu, m+i, k] / \
                        (He[nl.inl[idx,0],0,k] - He[nl.inl[idx,1],0,k])
                    i += 1

            for j, dof in enumerate(dofs):
                H[j,k] = He[dof, 0, k]

        self.H = H
        self.He = He
        return H, He

    def stabilization(self, nvec, tol_freq=1, tol_damping=5, tol_mode=0.98,
                      macchoice='complex'):
        """
        Parameters
        ----------
        # tol for freq and damping is in %

        Returns
        -------
        SD : defaultdict of defaultdict(list)
            Stabilization data. Key is model number(int), v is the properties
            for the given model number.
        """
        print('FNSI stabilisation diagram')
        fs = self.fs
        # for postprocessing
        fmin = self.fmin
        fmax = self.fmax
        self.nvec = nvec

        flines = self.flines
        E = self.E.T[flines]
        Y = self.Y.T[flines]

        freq = self.flines / self.nsper
        r = self.ims
        covG = None
        G = None

        ml = modal_list(G, covG, freq, nvec, r, fs, E, Y)
        # postprocessing
        sd = stabilization(ml, fmin, fmax, tol_freq, tol_damping, tol_mode,
                           macchoice)
        self.sr = ml
        self.sd = sd
        return sd

    def calc_modal(self):
        """Calculate modal properties after identification is done

        Remember to do it on continuous time matrices.
        """
        self.modal = modal_ac(self.A, self.C)

    def ss2phys(self):
        """Calculate state space matrices in physical domain using a similarity
        transform T
        """

        # returns A, B, C, T. T is similarity transform
        return ss2phys(self.A, self.B, self.C)

    def simulate(self, u, t=None, x0=None, T1=None, T2=None):
        """Calculate the output and the states of a nonlinear state-space model with
        transient handling.

        """

        # Number of samples
        ns = u.shape[0]
        if T1 is None:
            T1 = self.T1
            T2 = self.T2
            if T1 is not None:
                idx = self.idx_trans
        else:
            idx = transient_indices_periodic(T1, ns)

        if T1 is not None:
            # Prepend transient samples to the input
            u = u[idx]

        t, y, x = dnlsim(self, u, t, x0)

        if T1 is not None:
            # remove transient samples. p=1 is correct. TODO why?
            idx = remove_transient_indices_periodic(T1, ns, p=1)
            x = x[idx]
            y = y[idx]
            t = t[idx]

        self.x_mod = x
        self.y_mod = y
        return t, y, x

    def transient(self, signal, T1=None, T2=None):
        """Transient handling. t1: periodic, t2: aperiodic
        Get transient index. Only needed to run once
        """

        self.signal = signal
        self.T1 = T1
        self.T2 = T2
        sig = self.signal
        ns = sig.R * sig.npp
        # Extract the transient part of the input
        self.idx_trans = transient_indices_periodic(T1, ns)
        self.idx_remtrans = remove_transient_indices_periodic(T1, ns, self.p)
        self.umt = sig.um[self.idx_trans]
        self.n_trans = self.umt.shape[0]

        # without_T2 = remove_transient_indices_nonperiodic(system.T2,N,system.p)
        self.without_T2 = np.s_[:ns]

    def plot_frf(self, p=0, m=0, sca=1, fig=None, ax=None, **kwargs):
        # Plot identified frequency response function
        m = np.atleast_1d(m)
        p = np.atleast_1d(p)

        H = np.atleast_3d(self.H.T)
        fs = self.fs
        nsper = self.nsper
        flines = self.flines

        freq = np.arange(0,nsper)*fs/nsper
        freq_plot = freq[flines]

        return plot_frf(freq_plot, H, p=p, m=m, sca=sca, fig=fig, ax=ax,
                        **kwargs)

    def plot_stab(self, sca=1, fig=None, ax=None):
        " plot stabilization"
        return plot_stab(self.sd, self.fmin, self.fmax, sca, fig, ax)

class NL_force(object):

    def __init__(self, nls=None):
        self.nls = []
        if nls is not None:
            self.add(nls)

    def add(self, nls):
        if not isinstance(nls, list):
            nls = [nls]
            for nl in nls:
                self.nls.append(nl)

    def force(self, x, xd):

        # return empty array in case of no nonlinearities
        if len(self.nls) == 0:
            return np.array([])

        fnl = []
        for nl in self.nls:
            fnl_t = nl.compute(x, xd)
            fnl.extend(fnl_t)

        fnl = np.asarray(fnl)
        return fnl

class NL_polynomial():
    """Calculate force contribution for polynomial nonlinear stiffness or
    damping, see eq(2)

    Parameters
    ----------
    x : ndarray (ndof, ns)
        displacement or velocity.
    inl : ndarray (nbln, 2)
        Matrix with the locations of the nonlinearities,
        ex: inl = np.array([[7,0],[7,0]])
    enl : ndarray
        List of exponents of nonlinearity
    knl : ndarray (nbln)
        Array with nonlinear coefficients. ex. [1,1]
    idof : ndarray
        Array with node mapping for x.

    Returns
    -------
    f_nl : ndarray (nbln, ns)
        Nonlinear force
    """

    def __init__(self, inl, enl, knl, is_force=True):
        self.inl = inl
        self.enl = enl
        self.knl = knl
        self.is_force = is_force
        self.nnl = inl.shape[0]

    def compute(self, x, xd):
        inl = self.inl
        nbln = inl.shape[0]
        if self.is_force is False:
            # TODO: Overskriver dette x i original funktion? Ie pass by ref?
            x = xd

        ndof, nsper = x.shape
        idof = np.arange(ndof)
        fnl = np.zeros((nbln, nsper))

        for j in range(nbln):
            # connected from
            i1 = inl[j,0]
            # conencted to
            i2 = inl[j,1]

            # Convert to the right index
            idx1 = np.where(i1==idof)
            # if connected to ground
            if i2 == -1:
                x12 = x[idx1]
            else:
                idx2 = np.where(i2==idof)
                x12 = x[idx1] - x[idx2]
            # in case of even functional
            if (self.enl[j] % 2 == 0):
                x12 = np.abs(x12)

            fnl[j,:] = self.knl[j] * x12**self.enl[j]

        return fnl

class NL_spline():
    def __init__(self, inl, nspl, is_force=True):
        self.nspline = nspl
        self.is_force = is_force
        self.inl = inl

        # number of nonlinearities * number of knots
        self.nnl = inl.shape[0]*(nspl+1)

    def compute(self, x, xd):
        inl = self.inl
        nbln = inl.shape[0]
        ndof, nsper = x.shape
        idof = np.arange(ndof)
        if self.is_force is False:
            x = xd

        fnl = []
        for j in range(nbln):
            i1 = inl[j,0]
            i2 = inl[j,1]
            idx1 = np.where(i1==idof)
            # if connected to ground
            if i2 == -1:
                x12 = x[idx1]
            else:
                idx2 = np.where(i2==idof)
                x12 = x[idx1] - x[idx2]
            fnl_t, kn, dx = spline(x12.squeeze(), self.nspline)

            fnl.extend(fnl_t)
        fnl = np.asarray(fnl)

        self.kn = kn
        self.fnl = fnl

        return fnl

def dnlsim(system, u, t=None, x0=None):
    """Simulate output of a discrete-time nonlinear system.

    Calculate the output and the states of a nonlinear state-space model.
        x(t+1) = A x(t) + B u(t) + E zeta(x(t),u(t))
        y(t)   = C x(t) + D u(t) + F eta(x(t),u(t))
    where zeta and eta are polynomials whose exponents are given in xpowers and
    ypowers, respectively. The maximum degree in one variable (a state or an
    input) in zeta or eta is given in max_nx and max_ny, respectively. The
    initial state is given in x0.

    """

    u = np.asarray(u)

    if u.ndim == 1:
        u = np.atleast_2d(u).T

    if t is None:
        out_samples = len(u)
        stoptime = (out_samples - 1) * system.dt
    else:
        stoptime = t[-1]
        out_samples = int(np.floor(stoptime / system.dt)) + 1

    # Pre-build output arrays
    xout = np.empty((out_samples, system.A.shape[0]))
    yout = np.empty((out_samples, system.C.shape[0]))
    tout = np.linspace(0.0, stoptime, num=out_samples)

    # Check initial condition
    if x0 is None:
        xout[0, :] = np.zeros((system.A.shape[1],))
    else:
        xout[0, :] = np.asarray(x0)

    # Pre-interpolate inputs into the desired time steps
    if t is None:
        u_dt = u
    else:
        if len(u.shape) == 1:
            u = u[:, np.newaxis]

        u_dt_interp = interp1d(t, u.transpose(), copy=False, bounds_error=True)
        u_dt = u_dt_interp(tout).transpose()

    # prepare nonlinear part
    m = system.B.shape[1] - system.E.shape[1]
    repmat_x = np.ones(system.xpowers.shape[0])
    # Simulate the system
    for i in range(0, out_samples - 1):
        # Output equation y(t) = C*x(t) + D*u(t)
        yout[i, :] = (np.dot(system.C, xout[i, :]) +
                      np.dot(system.D, u_dt[i, :]))
        # State equation x(t+1) = A*x(t) + B*u(t) + E*zeta(y(t),ẏ(t))
        zeta_t = np.prod(np.outer(repmat_x, yout[i])**system.xpowers,
                         axis=1)
        xout[i+1, :] = (np.dot(system.Ad, xout[i, :]) +
                        np.dot(system.Bd, u_dt[i, :]) +
                        np.dot(system.E, zeta_t))

    # Last point
    # zeta_t = np.hstack((u_dt[-1, :],xout[-1,idx]**system.xpowers))
    yout[-1, :] = (np.dot(system.C, xout[-1, :]) +
                   np.dot(system.D, u_dt[-1, :]))

    return tout, yout, xout

def derivative(Y):
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.fftpack.diff.html
    N = npp
    if N % 2 == 0:
        # NOTE k is sequence of ints
        k = np.r_[np.arange(0, N//2), [0], np.arange(-N//2+1, 0)]
    else:
        k = np.r_[np.arange(0, (N-1)//2), [0], np.arange(-(N-1)//2, 0)]

    freq = self.flines / self.npp
    k *= 2 * np.pi / freq
    yd = np.real(np.fft.ifft(1j*k*Y*np.sqrt(npp),axis=0))

    return yd

def jacobian(x0, system, weight=None):
    """Compute the Jacobians of a steady state gray-box state-space model

    Jacobians of a gray-box state-space model

        x(t+1) = A x(t) + B u(t) + E zeta(y(t),ẏ(t))
        y(t)   = C x(t) + D u(t) + F eta(y(t),ẏ(t))

    i.e. the partial derivatives of the modeled output w.r.t. the active
    elements in the A, B, C, D, E and F matrices, fx: JA = ∂y/∂Aᵢⱼ

    x0 : ndarray
        flattened array of state space matrices

    """

    n, m, p = system.n, system.m, system.p
    R, p, npp = system.signal.R, system.signal.p, system.signal.npp
    nfd = npp//2
    # total number of points
    N = R*npp  # system.signal.um.shape[0]
    without_T2 = system.without_T2

    A, B, C, D, E, F = extract_ss(x0, system)

    # Collect states and outputs with prepended transient sample
    y_trans = system.y_mod[system.idx_trans]
    x_trans = system.x_mod[system.idx_trans]
    u_trans = system.umt
    # (n_var, nt)
    contrib = np.atleast_2d(np.hstack((y_trans)).T)
    n_trans = u_trans.shape[0]

    # E∂ₓζ + A(n,n,NT)
    ny = contrib.shape[0]
    Edwxdx = multEdwdx(contrib,system.xd_powers,system.xd_coeff, E,ny)
    zeta = nl_terms(contrib, system.xpowers).T  # (NT,n_nx)

    # calculate jacobians wrt state space matrices
    JC = np.kron(np.eye(p), system.x_mod)  # (p*N,p*n)
    JD = np.kron(np.eye(p), system.signal.um)  # (p*N, p*m)

    # calculate Jacobian by filtering an alternative state-space model
    JA = element_jacobian(x_trans, A, C, Edwxdx, None, np.arange(n**2))
    JA = JA.transpose((1,0,2)).reshape((p*n_trans, n**2))
    JA = JA[system.idx_remtrans]  # (p*N,n**2)

    JB = element_jacobian(u_trans, A, C, Edwxdx, None, np.arange(n*m))
    JB = JB.transpose((1,0,2)).reshape((p*n_trans, n*m))
    JB = JB[system.idx_remtrans]  # (p*N,n*m)

    if system.xactive.size:
        JE = element_jacobian(zeta, A, C, Edwxdx, None, system.xactive)
        JE = JE.transpose((1,0,2)).reshape((p*n_trans, len(system.xactive)))
        JE = JE[system.idx_remtrans]  # (p*N,nactiveE)

        JF = np.kron(np.eye(p), zeta)  # Jacobian wrt all elements in F
        JF = JF[:,system.yactive]  # all active elements in F. (p*NT,nactiveF)
        JF = JF[system.idx_remtrans]  # (p*N,nactiveF)
    else:
        JE = np.array([]).reshape(p*N,0)
        JF = np.array([]).reshape(p*N,0)

    jac = np.hstack((JA, JB, JC, JD, JE, JF))[without_T2]
    npar = jac.shape[1]

    # add frequency weighting
    if weight is not None and system.freq_weight:
        # (p*ns, npar) -> (Npp,R,p,npar) -> (Npp,p,R,npar) -> (Npp,p,R*npar)
        jac = jac.reshape((npp,R,p,npar),
                          order='F').swapaxes(1,2).reshape((-1,p,R*npar),
                                                           order='F')
        # select only the positive half of the spectrum
        jac = fft(jac, axis=0)[:nfd]
        # TODO should we test if weight is None or just do it in mmul_weight
        if weight is not None:
            jac = mmul_weight(jac, weight)
        # (nfd,p,R*npar) -> (nfd,p,R,npar) -> (nfd,R,p,npar) -> (nfd*R*p,npar)
        jac = jac.reshape((-1,p,R,npar),
                          order='F').swapaxes(1,2).reshape((-1,npar), order='F')

        J = np.empty((2*nfd*R*p,npar))
        J[:nfd*R*p] = jac.real
        J[nfd*R*p:] = jac.imag
    elif weight is not None:
        raise ValueError('Time weighting not possible')
    else:
        return jac

    return J


def element_jacobian(samples, A, C, Edwdy, Fdwdy, active):
    """Compute Jacobian of the output y wrt. A, B, and E

    The Jacobian is calculated by filtering an alternative state-space model

    ∂x∂Aᵢⱼ(t+1) = Iᵢⱼx(t) + A*∂x∂Aᵢⱼ(t) + E∂ζ∂y*∂y∂Aᵢⱼ(t)
    ∂y∂Aᵢⱼ(t) = C*∂x∂Aᵢⱼ(t) + F*∂η∂y*∂y∂Aᵢⱼ(t)

    where JA = ∂y∂Aᵢⱼ

    Parameters
    ----------
    samples : ndarray
       x, u or zeta corresponding to JA, JB, or JE
    A_Edwdx : ndarray (n,n,NT)
       The result of ``A + E*∂ζ∂x``
    C_Fdwdx : ndarray (p,n,NT)
       The result of ``C + F*∂η∂x``
    active : ndarray
       Array with index of active elements. For JA: np.arange(n**2), JB: n*m or
       JE: xactive

    Returns
    -------
    JA, JB or JE depending on the samples given as input

    See fJNL

    """
    # Number of outputs and number of states
    p, n = C.shape
    n_nx, ny, NT = Edwdy.shape
    # Number of samples and number of inputs in alternative state-space model
    N, npar = samples.shape
    nactive = len(active)  # Number of active parameters in A, B, or E

    out = np.zeros((p,N,nactive))
    for k, activ in enumerate(active):
        # Which column in A, B, or E matrix
        j = np.mod(activ, npar)
        # Which row in A, B, or E matrix
        i = (activ-j)//npar
        # partial derivative of x(0) wrt. A(i,j), B(i,j), or E(i,j)
        Jprev = np.zeros(n)
        for t in range(0,N-1):
            # Calculate output alternative state-space model at time t
            out[:,t,k] = C @ Jprev
            # Calculate state update alternative state-space model at time t
            # Terms in alternative states at time t-1
            J = A @ Jprev + Edwdy[:,:,t] @ out[:,t,k]
            # Term in alternative input at time t-1
            J[i] += samples[t,j]
            # Update previous state alternative state-space model
            Jprev = J
        out[:,-1,k] = C @ J

    return out

def nl_terms(contrib,power):
    """Construct polynomial terms.

    Computes polynomial terms, where contrib contains the input signals to the
    polynomial and pow contains the exponents of each term in each of the
    inputs. The maximum degree of an individual input is given in max_degree.

    Parameters
    ----------
    contrib : ndarray(n+m,N)
        matrix with N samples of the input signals to the polynomial.
        Typically, these are the n states and the m inputs of the nonlinear
        state-space model.
    power : ndarray(nterms,n+m)
        matrix with the exponents of each term in each of the inputs to the
        polynomial

    Returns
    -------
    out : ndarray(nterms,N)
        matrix with N samples of each term

    Examples
    --------
    >>> n = 2  # Number of states
    >>> m = 1  # Number of inputs
    >>> N = 1000  # Number of samples
    >>> x = np.random.randn(N,n)  # States
    >>> u = np.random.randn(N,m)  # Input
    >>> contrib = np.hstack((x, u)).T  # States and input combined
    All possible quadratic terms in states and input: x1^2, x1*x2, x1*u, x2^2,
    x2*u, u^2
    >>> pow = np.array([[2,0,0],
                        [1,1,0],
                        [1,0,1],
                        [0,2,0],
                        [0,1,1],
                        [0,0,2]])
    >>> nl_terms(contrib,pow)
    array([x[:,0]**2,
           x[:,0]*x[:,1],
           x[:,0]*u.squeeze(),
           x[:,1]**2,
           x[:,1]*u.squeeze(),
           u.squeeze()**2])
    """

    # Number of samples
    N = contrib.shape[1]
    # Number of terms
    nterms = power.shape[0]
    out = np.empty((nterms,N))
    for i in range(nterms):
        # All samples of term i
        out[i] = np.prod(contrib**power.T[:,None,i], axis=0)

    return out


def multEdwdx(contrib, power, coeff, E, n):
    """Multiply a matrix E with the derivative of a polynomial w(x,u) wrt. x

    Multiplies a matrix E with the derivative of a polynomial w(x,u) wrt the n
    elements in x. The samples of x and u are given by `contrib`. The
    derivative of w(x,u) wrt. x, is given by the exponents in x and u (given in
    power) and the corresponding coefficients (given in coeff).

    Parameters
    ----------
    contrib : ndarray(n+m,N)
        N samples of the signals x and u
    power : ndarray(n_nx,n+m,n+m)
        The exponents of the derivatives of w(x,u) w.r.t. x and u, i.e.
        power(i,j,k) contains the exponent of contrib j in the derivative of
        the ith monomial w.r.t. contrib k.
    coeff : ndarray(n_nx,n+m)
        The corresponding coefficients, i.e. coeff(i,k) contains the
        coefficient of the derivative of the ith monomial in w(x,u) w.r.t.
        contrib k.
    E : ndarray(n_out,n_nx)
    n : int
        number of x signals w.r.t. which derivatives are taken

    Returns
    -------
    out : ndarray(n_out,n,N)
        Product of E and the derivative of the polynomial w(x,u) w.r.t. the
        elements in x at all samples.

    Examples
    --------
    Consider w(x1,x2,u) = [x1^2    and E = [1 3 5
                           x1*x2            2 4 6]
                           x2*u^2]
    then the derivatives of E*w wrt. x1 and x2 are given by
    E*[2*x1 0
       1*x2 1*x1
       0    1*u^2]
    and the derivative of w wrt. u is given by [0,0,2*x2*u]^T

    >>> E = np.array([[1,3,5],[2,4,6]])
    >>> pow = np.zeros((3,3,3))
    Derivative wrt. x1 has terms 2*x1, 1*x2 and 0
    >>> pow[:,:,0] = np.array([[1,0,0],
                               [0,1,0],
                               [0,0,0]])
    Derivative wrt. x2 has terms 0, 1*x1 and 1*u^2
    >>> pow[:,:,1] = np.array([[0,0,0],
                               [1,0,0],
                               [0,0,2]])
    Derivative wrt. u has terms 0, 0 and 2*x2*u
    >>> pow[:,:,2] = np.array([[0,0,0],
                               [0,0,0],
                               [0,1,1]])
    >>> coeff = np.array([[2,0,0],
                          [1,1,0],
                          [0,1,2]])
    >>> n = 2  # Two signals x
    Ten random samples of signals x1, x2, and u
    >>> contrib = np.random.randn(3,10)
    >>> out = multEdwdx(contrib,pow,coeff,E,n)
    >>> t = 0
    out[:,:,t] = E @ np.array([[2*contrib[0,t],0],
                              [1*contrib[1,t],1*contrib[0,t]],
                              [0             ,1*contrib[2,t]**2]])

    """

    # n_all = number of signals x and u; N = number of samples
    n_all, N = contrib.shape
    # n_out = number of rows in E; n_nx = number of monomials in w
    n_out, n_nx = E.shape
    out = np.zeros((n_out,n,N))
    # Loop over all signals x w.r.t. which derivatives are taken
    for k in range(n):
        # Repeat coefficients of derivative of w w.r.t. x_k
        A = np.outer(coeff[:,k], np.ones(N))
        for j in range(n_all):     # Loop over all signals x and u
            for i in range(n_nx):  # Loop over all monomials
                # Derivative of monomial i wrt x_k
                A[i,:] *= contrib[j,:]**power[i,j,k]
        # E times derivative of w wrt x_k
        out[:,k,:] = np.matmul(E,A)

    return out

def extract_ss(x0, system):

    n, m, p = system.n, system.m, system.p
    n_nx, n_ny = system.n_nx, system.n_ny
    # index of active elements
    xact = system.xactive
    yact = system.yactive
    ne = len(xact)
    nf = len(yact)

    E = np.zeros((n, n_nx))
    F = np.zeros((p, n_ny))
    A = x0.flat[:n**2].reshape((n,n))
    B = x0.flat[n**2 + np.r_[:n*m]].reshape((n,m))
    C = x0.flat[n**2+n*m + np.r_[:p*n]].reshape((p,n))
    D = x0.flat[n*(p+m+n) + np.r_[:p*m]].reshape((p,m))

    E.flat[xact] = x0.flat[n*(p+m+n)+p*m + np.r_[:ne]]
    F.flat[yact] = x0.flat[n*(p+m+n)+p*m+ne + np.r_[:nf]]

    return A, B, C, D, E, F

def costfcn(x0, system, weight=None):
    # TODO fix transient
    T2 = system.T2
    R, p, npp = system.signal.R, system.signal.p, system.signal.npp
    nfd = npp//2
    without_T2 = system.without_T2

    # update the state space matrices from x0
    # TODO find a way to avoid explicitly updating the state space model.
    # It is not the expected behavior that calculating the cost should change
    # the model! Right now it is done because simulating is using the systems
    # ss matrices

    # A, B, C, D, E, F = extract_ss(x0, system)
    system.setss(*extract_ss(x0, system))
    # Compute the (transient-free) modeled output and the corresponding states
    t_mod, y_mod, x_mod = system.simulate(system.signal.um)

    # Compute the (weighted) error signal without transient
    err = system.y_mod[without_T2] - system.signal.ym[without_T2]
    if weight is not None and system.freq_weight:
        err = err.reshape((npp,R,p),order='F').swapaxes(1,2)
        # Select only the positive half of the spectrum
        err = fft(err, axis=0)[:nfd]
        err = mmul_weight(err, weight)
        #cost = np.vdot(err, err).real
        err = err.swapaxes(1,2).ravel(order='F')
        err_w = np.hstack((err.real.squeeze(), err.imag.squeeze()))
    elif weight is not None:
        # TODO time domain weighting. Does not work
        err_w = err * weight[without_T2]
        #cost = np.dot(err,err)
    else:
        # no weighting
        # TODO are we sure this is the right order?
        return err.ravel(order='F')

    return err_w

def poly_deriv(powers):
    """Calculate derivative of a multivariate polynomial

    """
    # Polynomial coefficients of the derivative
    d_coeff = powers
    n = powers.shape[1]
    #  Terms of the derivative
    d_powers = np.repeat(powers[...,None],n, axis=2)
    for i in range(n):
        d_powers[:,i,i] = np.abs(powers[:,i]-1)

    return d_powers, d_coeff
