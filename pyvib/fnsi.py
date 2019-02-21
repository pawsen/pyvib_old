#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.linalg import (solve, norm, expm)
from scipy.interpolate import interp1d

from .common import meanVar
from .subspace import subspace, discrete2cont, modal_list, ss2phys
from .modal import modal_ac, stabilization
from .pnlss import transient_indices_periodic, remove_transient_indices_periodic
from .spline import spline
from .helper.modal_plotting import plot_frf, plot_stab

class FNSI():
    def __init__(self, u, y, fmin=None, fmax=None, xpowers=None, flines=None,
                 yd=None, fs=None):

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
        npp, p, P = self.y.shape
        npp, m, P = self.y.shape

        U = np.fft.fft(self.u,axis=0) / np.sqrt(npp)
        Y = np.fft.fft(self.y,axis=0) / np.sqrt(npp)

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
            Yd = np.fft.fft(yd,axis=0) / np.sqrt(npp)
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

        self.E = E
        self.Y = Yext
        self.W = WY
        self.scaling = scaling

    def svd_comp(self, r):
        self.ims = r

    def id(self, n, bd_method='explicit'):

        flines = self.flines
        E = self.E[flines]
        Y = self.Y[flines]

        freq = self.flines / self.npp
        r = self.ims
        covarG = None
        G = None
        dt = 1/self.fs
        Ad, Bd, Cd, Dd, z, isstable = \
            subspace(G, covarG, freq, n, r, E, Y, bd_method)
        Ac, Bc, Cc, Dc = discrete2cont(Ad, Bd, Cd, Dd, dt)

        #import ipdb; ipdb.set_trace()

        self.A = Ac
        self.B = Bc
        self.C = Cd
        self.D = Dd
        self.Ad = Ad
        self.Bd = Bd

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
        """Calculate modal properties after identification is done"""
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
                      np.dot(system.D[:,:m], u_dt[i, :]))
        # State equation x(t+1) = A*x(t) + B*u(t) + E*zeta(y(t),ẏ(t))
        zeta_t = np.prod(np.outer(repmat_x, yout[i])**system.xpowers,
                         axis=1)
        xout[i+1, :] = (np.dot(system.Ad, xout[i, :]) +
                        np.dot(system.Bd[:,:m], u_dt[i, :]) +
                        np.dot(system.E, zeta_t))

    # Last point
    # zeta_t = np.hstack((u_dt[-1, :],xout[-1,idx]**system.xpowers))
    yout[-1, :] = (np.dot(system.C, xout[-1, :]) +
                   np.dot(system.D[:,:m], u_dt[-1, :]))

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
