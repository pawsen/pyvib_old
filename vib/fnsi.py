#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.linalg import (lstsq, solve, qr, svd, logm, pinv)

from .common import meanVar
from .modal import modal_ac, stabilization
from .spline import spline
from .helper.modal_plotting import plot_frf, plot_stab

class FNSI():
    def __init__(self, signal, nonlin, idof, fmin, fmax, iu=[], nldof=[]):
        # self.signal = signal
        self.nonlin = nonlin

        # nper : number of periods
        # nsper : number of samples per period
        # ns : total samples in periodic signal
        fs = signal.fs
        y = signal.y_per
        u = signal.u_per
        if u.ndim != 2:
            u = u.reshape(-1,u.shape[0])
        nper = signal.nper
        nsper = signal.nsper

        f1 = int(np.floor(fmin/fs * nsper))
        f2 = int(np.ceil(fmax/fs*nsper))
        flines = np.arange(f1,f2+1)

        # Make sure dof of force and dof of nonlin is included in idofs
        # if all arrays are int, then resulting array is also int. But if some
        # are empty, the resulting array is float
        idof = np.unique(np.hstack((idof, iu, nldof)))
        idof = idof.astype(int)
        y = y[idof,:]
        fdof, ns = u.shape
        ndof, ns = y.shape

        # some parameters. Dont know what.. Maybe npp: number per period!
        p1 = 0
        p2 = 0
        npp = nper - p1 - p2

        # TODO could be done in Signal class
        # TODO make C-order. Should just be swap nsper and npp
        self.u = np.reshape(u[:,p1*nsper:(nper-p2)*nsper], (fdof, nsper, npp), order='F')
        self.y = np.reshape(y[:,p1*nsper:(nper-p2)*nsper], (ndof, nsper, npp), order='F')

        self.idof = idof
        self.npp = npp
        self.nper = nper
        self.nsper = nsper
        self.flines = flines
        self.fs = fs
        self.fmin = fmin
        self.fmax = fmax

    def calc_EY(self, isnoise=False):
        """Calculate FFT of the extended input vector e(t) and the measured
        output y.

        The concatenated extended input vector e(t), is e=[u(t), g(t)].T, see
        eq (5). (E is called the Extended input spectral matrix and used for
        forming Ei, eq. (12)). Notice that the stacking order is reversed here.
        u(t) is the input force and g(y(t),ẏ(t)) is the functional nonlinear
        force calculated from the specified polynomial nonlinearity, see eq.(2)

        Returns
        ------
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
        u = self.u
        y = self.y
        nsper = self.nsper
        npp = self.npp

        U = np.fft.fft(self.u,axis=1) / np.sqrt(nsper)
        Y = np.fft.fft(self.y,axis=1) / np.sqrt(nsper)

        Umean, WU = meanVar(U, isnoise=False)
        Ymean, WY = meanVar(Y, isnoise=isnoise)

        # Set weights to none, if the signal is not noisy
        if isnoise is False:
            WY = None

        # In case of no nonlinearities
        if len(self.nonlin.nls) == 0:
            scaling = []
            E = Umean
        else:
            ynl = y
            # average displacement
            ynl = np.sum(ynl, axis=2) / npp
            fnl = self.nonlin.force(ynl, 0)
            nnl = fnl.shape[0]

            scaling = np.zeros(nnl)
            for j in range(nnl):
                scaling[j] = np.std(u[0,:]) / np.std(fnl[j,:])
                fnl[j,:] *= scaling[j]

            FNL = np.fft.fft(fnl, axis=1) / np.sqrt(nsper)
            # concatenate to form extended input spectra matrix
            E = np.vstack((Umean, -FNL))

        self.E = E
        self.Y = Ymean
        self.W = WY
        self.scaling = scaling

    def svd_comp(self, ims, svd_plot=False):
        """
        ims: Number of matrix block rows. Determining the size of matrices used.
        W : some weights

        # flines : ndarray
        #     Vector of frequency lines where the nonlinear coefficients are
        #     computed.
        """
        print('FNSI analysis part 1')

        nper = self.nper
        nsper = self.nsper
        ns = nper * nsper
        fs = self.fs
        flines = self.flines
        E = self.E
        Y = self.Y
        W = self.W

        # dimensions
        # l: measured applied displacements
        m = np.shape(E)[0]
        l = np.shape(Y)[0]
        F = len(flines)

        # z-transform variable
        # Remember that e^(a+ib) = e^(a)*e^(ib) = e^(a)*(cos(b) + i*sin(b))
        # i.e. the exponential of a imaginary number is complex.
        zvar = np.empty(flines.shape, dtype=complex)
        zvar.real = np.zeros(flines.shape)
        zvar.imag = 2 * np.pi * flines / nsper
        zvar = np.exp(zvar)

        # dz is an array containing powers of zvar. Eg. the scaling z's in eq.
        # (10) (ξ is not used, but dz relates to ξ)
        dz = np.zeros((ims+1, F), dtype=complex)
        for j in range(ims+1):
            dz[j,:] = zvar**j

        # 2:
        # Concatenate external forces and nonlinearities to form the extended
        # input  spectra Ei
        # Initialize Ei and Yi
        Emat = np.empty((m * (ims + 1), F), dtype=complex)
        Ymat = np.empty((l * (ims + 1), F), dtype=complex)
        for j in range(F):
            # Implemented as the formulation eq. (10), not (11) and (12)
            Emat[:,j] = np.kron(dz[:,j], E[:, flines[j]])
            Ymat[:,j] = np.kron(dz[:,j], Y[:, flines[j]])

        # Emat is implicitly recreated as dtype: float64
        Emat = np.hstack([np.real(Emat), np.imag(Emat)])
        Ymat = np.hstack([np.real(Ymat), np.imag(Ymat)])

        # 3:
        # Compute the orthogonal projection P = Yi/Ei using QR-decomposition,
        # eq. (20)
        print('QR decomposition')
        P = np.vstack([Emat, Ymat])
        R, = qr(P.T, mode='r')
        _slice = np.s_[:(ims+1)*(m+l)]
        Rtr = R[_slice, _slice].T
        _slice = np.s_[(ims+1)*m:ims*(m+l)+m]
        R22 = Rtr[_slice, _slice]
        if np.size(R22) == 0:
            raise ValueError('Not enough frequency content to allow for the'
                             ' number of nonlinear basis functions and'
                             ' matrix block rows(ims).')

        # Calculate weight CY from filter W if present.
        if W is None:
            CY = np.eye(l*ims)
        else:
            Wmat = np.zeros((l*ims,F), dtype=complex)
            for j in range(F):
                Wmat[:,j] = np.sqrt(np.kron(dz[:ims,j], W[:, flines[j]]))
            CY = np.real(Wmat @ Wmat.T)

        # full_matrices=False is equal to matlabs economy-size decomposition.
        # Use gesdd as driver. Matlab uses gesvd. The only reason to use dgesvd
        # instead is for accuracy and workspace memory. The former should only
        # be important if you care about high relative accuracy for tiny
        # singular values. Mem: gesdd: O(min(m,n)^2) vs O(max(m,n)) gesvd
        UCY, scy, _ = svd(CY, full_matrices=False)

        # it is faster to work on the diagonal scy, than the full matrix SCY
        sqCY = UCY @ np.diag(np.sqrt(scy)) @ UCY.T
        isqCY = UCY @ np.diag(1/np.sqrt(scy)) @ UCY.T

        # 4:
        # Compute the SVD of P
        print('SV decomposition')
        Un, sn, _ = svd(isqCY.dot(R22), full_matrices=False)

        self.sn = sn
        self.Un = Un
        self.Rtr = Rtr
        self.sqCY = sqCY
        self.m = m
        self.l = l
        self.F = F
        self.ims = ims

    def id(self, nmodel, bd_method=None):
        """Frequency-domain Nonlinear Subspace Identification (FNSI)
        """
        ims = self.ims
        fs = self.fs
        l = self.l
        m = self.m
        Un  =self.Un
        sn = self.sn
        Rtr = self.Rtr
        sqCY = self.sqCY

        # 5:
        # Truncate Un and Sn based on the model order n. The model order can be
        # determined by inspecting the singular values in Sn or using
        # stabilization diagram.
        U1 = Un[:,:nmodel]
        S1 = sn[:nmodel]

        # 6:
        # Estimation of the extended observability matrix, Γi, eq (21)
        # Here np.diag(np.sqrt(S1)) creates an diagonal matrix from an array
        G = sqCY @ U1 @ np.diag(np.sqrt(S1))

        # 7:
        # Estimate A from eq(24) and C as the first block row of G.
        # Recompute G from A and C, eq(13). G plays a major role in determining
        # B and D, thus Noel suggest that G is recalculated

        A, *_ = lstsq(G[:-l,:], G[l:,:])
        C = G[:l,:].copy()

        # Equal to G[] = C @ np.linalg.matrix_power(A,j)
        for j in range(1,ims):
            G[j*l:(j+1)*l,:] = G[(j-1)*l:(j)*l,:] @ A

        # 8:
        # Estimate B and D
        print('(B,D) estimation using no optimisation')

        # R_U: Ei+1, R_Y: Yi+1
        R_U = Rtr[:m*(ims+1),:(m+l)*(ims+1)]
        R_Y = Rtr[m*(ims+1):(m+l)*(ims+1),:(m+l)*(ims+1)]

        # eq. 30
        G_inv = pinv(G)
        Q = np.vstack([
            G_inv @ np.hstack([np.zeros((l*ims,l)), np.eye(l*ims)]) @ R_Y,
            R_Y[:l,:]]) - \
            np.vstack([
                A,
                C]) @ G_inv @ np.hstack([np.eye(l*ims), np.zeros((l*ims,l))]) @ R_Y

        Rk = R_U

        # eq (34) with zeros matrix appended to the end. eq. L1,2 = [L1,2, zeros]
        L1 = np.hstack([A @ G_inv, np.zeros((nmodel,l))])
        L2 = np.hstack([C @ G_inv, np.zeros((l,l))])

        # The pseudo-inverse of G. eq (33), prepended with zero matrix.
        # eq. MM = [zeros, G_inv]
        MM = np.hstack([np.zeros((nmodel,l)), G_inv])

        # The reason for appending/prepending zeros in L and MM, is to easily
        # form the submatrices of N, given by eq. 40. Thus ML is equal to first
        # row of N1
        ML = MM - L1

        # rhs multiplicator of eq (40)
        Z = np.vstack([
            np.hstack([np.eye(l), np.zeros((l,nmodel))]),
            np.hstack([np.zeros((l*ims,l)),G])
        ])

        # Assemble the kron_prod in eq. 44.
        for kk in range(ims+1):
            # Submatrices of N_k. Given by eq (40).
            # eg. N1 corresspond to first row, N2 to second row of the N_k's
            # submatrices
            N1 = np.zeros((nmodel,l*(ims+1)))
            N2 = np.zeros((l,l*(ims+1)))

            N1[:, :l*(ims-kk+1)] = ML[:, kk*l:l*(ims+1)]
            N2[:, :l*(ims-kk)] = -L2[:, kk*l:l*ims]

            if kk == 0:
                # add the identity Matrix
                N2[:l, :l] = np.eye(l) + N2[:l, :l]

            # Evaluation of eq (40)
            Nk = np.vstack([
                N1,
                N2
            ]).dot(Z)

            if kk == 0:
                kron_prod = np.kron(Rk[kk*m:(kk+1)*m,:].T, Nk)
            else:
                kron_prod = kron_prod + np.kron(Rk[kk*m:(kk+1)*m,:].T, Nk)

        # like flatten, just faster. Flatten row wise.
        Q = Q.ravel(order='F')
        Q_real = np.hstack([
            np.real(Q),
            np.imag(Q)
        ])

        kron_prod_real = np.vstack([
            np.real(kron_prod),
            np.imag(kron_prod)
        ])

        # Solve for DB, eq. (44)
        DB, *_ = lstsq(kron_prod_real, Q_real)
        DB = DB.reshape(nmodel+l,m, order='F')
        D = DB[:l,:]
        B = DB[l:l+nmodel,:]

        ## End of (B,D) estimation using no optimisation ##

        # 9:
        # Convert A, B, C, D into continous-time arrays using eq (8)
        # Ad, Bd is the discrete time(frequency domain) matrices.
        # A, B is continuous time
        Ad = A
        A = fs * logm(Ad)

        Bd = B
        B = A @ (solve(Ad - np.eye(len(Ad)),Bd))

        self.A = A
        self.B = B
        self.C = C
        self.D = D

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
            He[:-1,:,k] = C @ solve(np.eye(*A.shape,dtype=complex)*1j*2*np.pi*
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

    def stabilization(self, nlist, tol_freq=1, tol_damping=5, tol_mode=0.98,
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
        l = self.l
        m = self.m
        F = self.F
        U = self.Un
        s = self.sn
        sqCY = self.sqCY
        # for postprocessing
        fmin = self.fmin
        fmax = self.fmax
        self.nlist = nlist

        # estimate modal properties for increasing model order
        sr = [None]*len(nlist)
        for k, n in enumerate(nlist):
            U1 = U[:,:n]
            S1 = s[:n]
            # Estimation of the extended observability matrix, Γi, eq (21)
            G = sqCY @ U1 @ np.diag(np.sqrt(S1))

            # Estimate A from eq(24) and C as the first block row of G.
            G_up = G[l:,:]
            G_down = G[:-l,:]
            A, *_ = lstsq(G_down, G_up)
            C = G[:l, :]
            # Convert A into continous-time arrays using eq (8)
            A = fs * logm(A)
            sr[k] = modal_ac(A, C)

        # postprocessing
        sdout = stabilization(sr, nlist, fmin, fmax, tol_freq, tol_damping,
                              tol_mode, macchoice)
        self.sr = sr
        self.sd = sdout
        return sdout

    def calc_modal(self):
        """Calculate modal properties after identification is done"""
        self.modal = modal_ac(self.A, self.C)

    def state_space(self):
        """Calculate state space matrices in physical domain using a similarity
        transform T

        See eq 20.10 in
        Etienne Gourc, JP Noel, et.al
        "Obtaining Nonlinear Frequency Responses from Broadband Testing"
        """

        # Similarity transform
        T = np.vstack((self.C, self.C @ self.A))
        C = solve(T.T, self.C.T).T  # (C = C*T^-1)
        A = solve(T.T, (T @ self.A).T).T  # (A = T*A*T^-1)
        B = T @ self.B
        self.state = {
            'T': T, 'C': C, 'A': A, 'B': B,
        }
        return

    def plot_frf(self, dofs=[0], sca=1, fig=None, ax=None, **kwargs):
        # Plot identified frequency response function
        H = self.H
        fs = self.fs
        nsper = self.nsper
        flines = self.flines

        freq = np.arange(0,nsper)*fs/nsper
        freq_plot = freq[flines]

        return plot_frf(freq_plot, H, dofs, sca, fig, ax, **kwargs)

    def plot_stab(self, sca=1, fig=None, ax=None):
        " plot stabilization"
        return plot_stab(self.sd, self.nlist, self.fmin, self.fmax, sca, fig,
                         ax)

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
            x1 = x[idx1]

            # if connected to ground
            if i2 == -1:
                x2 = 0
            else:
                idx2 = np.where(i2==idof)
                x2 = x[idx2]
            x12 = x1 - x2
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
            x1 = x[idx1]
            # if connected to ground
            if i2 == -1:
                x2 = 0
            else:
                idx2 = np.where(i2==idof)
                x2 = x[idx2]
            x12 = x1 - x2

            fnl_t, kn, dx = spline(x12.squeeze(), self.nspline)

            fnl.extend(fnl_t)
        fnl = np.asarray(fnl)

        self.kn = kn
        self.fnl = fnl

        return fnl
