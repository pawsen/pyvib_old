#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy import linalg
from common import meanVar, db, modal_properties, ModalAC, ModalACX
from signal2 import Signal
from collections import defaultdict

# just for debugging
from pprint import pprint
from scipy.linalg import norm

class FNSI():
    def __init__(self, signal, inl, enl, knl, idof, fmin, fmax,
                 iu=[], nldof=[]):
        # self.signal = signal
        self.inl = np.asarray(inl)
        self.enl = enl
        self.knl = knl

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

        # some parameters. Dont know what..
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



    def _force_nl(self, x):
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
        inl = self.inl
        enl = self.enl
        knl = self.knl
        idof = self.idof

        # return empty array in case of no nonlinearities
        if inl.size == 0:
            return np.array([])

        nbln = inl.shape[0]
        nsper = x.shape[1]
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
            if (enl[j] % 2 == 0):
                x12 = np.abs(x12)

            fnl[j,:] = knl[j] * x12**enl[j]

        return fnl


    def calc_EY(self, isnoise=False):
        """Calculate FFT of the extended input vector e(t) and the measured output
        y.

        The concatenated extended input vector e(t), is e=[u(t), g(t)].T, see eq
        (5). (E is called the Extended input spectral matrix and used for forming
        Ei, eq. (12)). Notice that the stacking order is reversed here.
        u(t) is the input force and g(y(t),ẏ(t)) is the functional nonlinear force
        calculated from the specified polynomial nonlinearity, see eq.(2)

        Returns
        ------
        E : ndarray (complex)
            FFT of the concatenated extended input vector e(t)
        Y : ndarray (complex)
            FFT of y.

        Notes
        -----
        Method by J.P Noel. Described in article
        "Frequency-domain subspace identification for nonlinear mechanical systems"
        http://dx.doi.org/10.1016/j.ymssp.2013.06.034
        Equation numbers refers to this article
        """
        print('E and Y comp.')
        u = self.u
        y = self.y
        nsper = self.nsper
        npp = self.npp
        inl = self.inl

        U = np.fft.fft(self.u,axis=1) / np.sqrt(nsper)
        Y = np.fft.fft(self.y,axis=1) / np.sqrt(nsper)

        Umean, WU = meanVar(U)
        Ymean, WY = meanVar(Y)

        # Set weights to none, if the signal is not noisy
        if not isnoise:
            WY = None

        # In case of no nonlinearities
        if inl.size == 0:
            scaling = []
            E = Umean
        else:
            ynl = y
            # average displacement
            ynl = np.sum(ynl, axis=2) / npp
            nl = self._force_nl(ynl)
            nnl = nl.shape[0]

            scaling = np.zeros(nnl)
            for j in range(nnl):
              scaling[j] = np.std(u[0,:]) / np.std(nl[j,:])
              nl[j,:] = nl[j,:] * scaling[j]

            NL = np.fft.fft(nl, axis=1) / np.sqrt(nsper)
            # concatenate to form extended input spectra matrix
            E = np.vstack((Umean, -NL))

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


        # dz is an array containing powers of zvar. Eg. the scaling z's in eq. (10)
        # (ξ is not used, but dz relates to ξ)
        dz = np.zeros((ims+1, F), dtype=complex)
        for j in range(ims+1):
            dz[j,:] = zvar**j

        # 2:
        # Concatenate external forces and nonlinearities to form the extended input
        # spectra Ei
        # Initialize Ei and Yi
        Emat = np.empty((m * (ims + 1), F), dtype=complex)
        Ymat = np.empty((l * (ims + 1), F), dtype=complex)
        for j in range(F):
            # Implemented as the formulation eq. (10), not (11) and (12)
            Emat[:,j] = np.kron(dz[:,j], E[:, flines[j]])
            Ymat[:,j] = np.kron(dz[:,j], Y[:, flines[j]])
        print('dtype: {}'.format(Emat.dtype))

        # Emat is implicitly recreated as dtype: float64
        Emat = np.hstack([np.real(Emat), np.imag(Emat)])
        Ymat = np.hstack([np.real(Ymat), np.imag(Ymat)])
        print('dtype: {}'.format(Emat.dtype))

        # 3:
        # Compute the orthogonal projection P = Yi/Ei using QR-decomposition, eq. (20)
        print('QR decomposition')
        P = np.vstack([Emat, Ymat])
        R, = linalg.qr(P.T, mode='r')
        Rtr = R[:(ims+1)*(m+l),:(ims+1)*(m+l)].T
        R22 = Rtr[(ims+1)*m:ims*(m+l)+m,(ims+1)*m:ims*(m+l)+m]

        # Calculate weight CY from filter W if present.
        if W is None:
            CY = np.eye(l*ims)
        else:
            Wmat = np.zeros((l*ims,F))
            for j in range(F):
                Wmat[:,j] = np.sqrt(np.kron(dz[:ims,j], W[:, flines[j]]))
            CY = np.real(Wmat @ Wmat.T)


        # full_matrices=False is equal to matlabs economy-size decomposition.
        # gesvd is the driver used in matlab,
        UCY, scy, _ = linalg.svd(CY, full_matrices=False, lapack_driver='gesvd')
        SCY = np.diag(scy)

        sqCY = UCY.dot(np.sqrt(SCY).dot(UCY.T))
        isqCY = UCY.dot(np.diag(np.diag(SCY)**(-0.5)).dot(UCY.T))
        print('nan value: ', np.argwhere(np.isnan(isqCY)))

        # 4:
        # Compute the SVD of P
        print('SV decomposition')

        Un, sn, _ = linalg.svd(isqCY.dot(R22), full_matrices=False,
                               lapack_driver='gesvd')

        self.Sn = np.diag(np.diag(sn))
        self.Un = Un
        self.Rtr = Rtr
        self.sqCY = sqCY
        self.m = m
        self.l = l
        self.F = F
        self.ims = ims

        if svd_plot:
            #plt.ion()
            plt.figure(1)
            plt.clf()
            plt.semilogy(Sn/np.sum(Sn),'sk', markersize=6)
            plt.xlabel('Singular value')
            plt.ylabel('Magnitude')
            #plt.show()


    def id(self, nmodel, bd_method=None):
        """Frequency-domain Nonlinear Subspace Identification (FNSI)
        """
        ims = self.ims
        fs = self.fs
        l = self.l
        m = self.m
        Un  =self.Un
        Sn = self.Sn
        Rtr = self.Rtr
        sqCY = self.sqCY

        # 5:
        # Truncate Un and Sn based on the model order n. The model order can be
        # determined by inspecting the singular values in Sn or using stabilization
        # diagram.
        U1 = Un[:,:nmodel]
        S1 = Sn[:nmodel]

        # 6:
        # Estimation of the extended observability matrix, Γi, eq (21)
        # Here np.diag(np.sqrt(S1)) creates an diagonal matrix from an array
        G = sqCY.dot(U1.dot(np.diag(np.sqrt(S1))))

        # 7:
        # Estimate A from eq(24) and C as the first block row of G.
        # Recompute G from A and C, eq(13). G plays a major role in determining B and D,
        # thus Noel suggest that G is recalculated

        A = linalg.pinv(G[:-l,:]).dot(G[l:,:])
        C = G[:l,:]

        G1 = np.empty(G.shape)
        G1[:l,:] = C
        for j in range(1,ims):
            G1[j*l:(j+1)*l,:] = C.dot(np.linalg.matrix_power(A,j))

        G = G1

        # 8:
        # Estimate B and D
        ## Start of (B,D) estimation using no optimisation ##
        print('(B,D) estimation using no optimisation')

        # R_U: Ei+1, R_Y: Yi+1
        R_U = Rtr[:m*(ims+1),:(m+l)*(ims+1)]
        R_Y = Rtr[m*(ims+1):(m+l)*(ims+1),:(m+l)*(ims+1)]

        # eq. 30
        G_inv = linalg.pinv(G)
        Q = np.vstack([
            G_inv.dot(np.hstack([np.zeros((l*ims,l)), np.eye(l*ims)]).dot(R_Y)),
            R_Y[:l,:]]) - \
            np.vstack([
                A,
                C]).dot(G_inv.dot(np.hstack([np.eye(l*ims), np.zeros((l*ims,l))]).dot(R_Y)))

        Rk = R_U

        # eq (34) with zeros matrix appended to the end. eq. L1,2 = [L1,2, zeros]
        L1 = np.hstack([A.dot(G_inv), np.zeros((nmodel,l))])
        L2 = np.hstack([C.dot(G_inv), np.zeros((l,l))])

        # The pseudo-inverse of G. eq (33), prepended with zero matrix.
        # eq. MM = [zeros, G_inv]
        MM = np.hstack([np.zeros((nmodel,l)), G_inv])

        # The reason for appending/prepending zeros in L and MM, is to easily form the
        # submatrices of N, given by eq. 40. Thus ML is equal to first row of N1
        ML = MM - L1

        # rhs multiplicator of eq (40)
        Z = np.vstack([
            np.hstack([np.eye(l), np.zeros((l,nmodel))]),
            np.hstack([np.zeros((l*ims,l)),G])
        ])

        # Assemble the kron_prod in eq. 44.
        for kk in range(ims+1):
            # Submatrices of N_k. Given by eq (40).
            # eg. N1 corresspond to first row, N2 to second row of the N_k's submatrices
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
        DB = linalg.pinv(kron_prod_real).dot(Q_real)
        DB = DB.reshape(nmodel+l,m, order='F')
        D = DB[:l,:]
        B = DB[l:l+nmodel,:]

        ## End of (B,D) estimation using no optimisation ##

        # 9:
        # Convert A, B, C, D into continous-time arrays using eq (8)
        # Ad, Bd is the discrete time(frequency domain) matrices.
        # A, B is continuous time
        Ad = A
        A = fs * linalg.logm(Ad)

        Bd = B
        B = A.dot(linalg.solve(Ad - np.eye(len(Ad)),Bd))

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
            The nonlinear coefficients (frequency-dependent and complex-valued).
        H(ω) : ndarray(complex)
            The extended FRF (transfer function matrix)

        """
        from copy import deepcopy

        dofs = np.asarray(dofs)
        fs = self.fs
        nsper = self.nsper
        flines = self.flines
        A = self.A
        B = self.B
        C = self.C
        D = self.D
        inl = deepcopy(self.inl)
        scaling = self.scaling

        freq = np.arange(0,nsper)*fs/nsper
        F = len(flines)


        l, m = D.shape

        # just return in case of no nonlinearities
        if inl.size == 0:
            #return np.array([]), np.array([])
            knl = np.array([])
            nnl = 0
        else:
            nnl = inl.shape[0]
            # connected from
            inl1 = inl[:,0]
             # connected to
            inl2 = inl[:,1]
            # if connected to ground.
            inl2[np.where(inl2 == -1)] = l

            knl = np.empty((nnl, F), dtype=complex)

        m = m - nnl

        # Extra rows of zeros in He is for ground connections
        H = np.empty((len(dofs), F),dtype=complex)
        He = np.empty((l+1, m+nnl, F),dtype=complex)
        He[-1,:,:] = 0
        for k in range(F):
            He[:-1,:,k] = C.dot(linalg.pinv(np.eye(*A.shape,dtype=complex)*1j*2*np.pi*
                                            freq[flines[k]] - A)).dot(B) + D

            for i in range(nnl):
                knl[i, k] = scaling[i] * He[iu, m+i, k] / (He[inl1[i],0,k] -
                                                          He[inl2[i],0,k] )

            for j, dof in enumerate(dofs):
                H[j,k] = He[dof, 0, k]

        return knl, H, He


    def stabilisation_diagram(self, nlist):
        """

        Returns
        -------
        SD : defaultdict of defaultdict(list)
            Stabilization data. Key is model number(int), v is the properties for the
            given model number.
        """
        print( 'FNSI stabilisation diagram' );
        fs = self.fs
        l = self.l
        m = self.m
        F = self.F
        U  =self.Un
        S = self.Sn
        sqCY = self.sqCY
        # for postprocessing
        fmin = self.fmin
        fmax = self.fmax

        SD = [None]*len(nlist)
        for k, n in enumerate(nlist):
            U1 = U[:,:n]
            S1 = S[:n]
            # Estimation of the extended observability matrix, Γi, eq (21)
            G = sqCY.dot(U1.dot(np.diag(np.sqrt(S1))))

            # Estimate A from eq(24) and C as the first block row of G.
            G_up = G[l:,:]
            G_down = G[:-l,:]
            A = linalg.pinv(G_down).dot(G_up)
            C = G[:l, :]
            # Convert A into continous-time arrays using eq (8)
            A = fs * linalg.logm(A)
            SD[k] = modal_properties(A, C)


        # postprocessing
        # tol for freq and damping is in %
        tol_freq = 1
        tol_damping = 5
        tol_mode = 0.98
        macchoice = 'complex'

        # Initialize SDout as 2 nested defaultdict
        SDout = defaultdict(lambda: defaultdict(list))
        # loop over model orders
        for ior, nval in enumerate(nlist[:-1]):
            # loop over frequencies for current model order
            for ifr, natfreq in enumerate( SD[ior]['natfreq']):
                if natfreq < fmin or natfreq > fmax:
                    continue

                # compare with frequencies from one model order higher.
                nfreq = SD[ior+1]['natfreq']
                tol_low = (1 - tol_freq / 100) * natfreq
                tol_high = (1 + tol_freq / 100) * natfreq
                ifreqS, = np.where( (nfreq >= tol_low) & (nfreq <= tol_high) )
                if ifreqS.size == 0:  # ifreqS is empty
                    # the current natfreq is not stabilized
                    SDout[nval]['stab'].append(False)
                    SDout[nval]['freq'].append(natfreq)
                    SDout[nval]['ep'].append(False)
                    SDout[nval]['mode'].append(False)
                else:
                    # Stabilized in natfreq
                    SDout[nval]['stab'].append(True)
                    SDout[nval]['freq'].append(natfreq)
                    # Only in very rare cases, ie multiple natfreqs are very close,
                    # is len(ifreqS) != 1
                    for ii in ifreqS:
                        nep = SD[ior+1]['ep'][ii]
                        tol_low = (1 - tol_damping / 100) * SD[ior]['ep'][ifr]
                        tol_high = (1 + tol_damping / 100) * SD[ior]['ep'][ifr]
                        # TODO: matlab have find(cond ,1). Ie only return the first match
                        iepS, = np.where( (nep >= tol_low) & (nep <= tol_high) )
                        if iepS.size == 0:
                            SDout[nval]['ep'].append(False)
                        else:
                            SDout[nval]['ep'].append(True)
                    if macchoice == 'complex':
                        m1 = SD[ior]['cpxmode'][ifr]
                        m2 = SD[ior+1]['cpxmode'][ifreqS]
                        MAC = ModalACX(m1, m2)
                    else:
                        m1 = SD[ior]['realmode'][ifr]
                        m2 = SD[ior+1]['realmode'][ifreqS]
                        MAC = ModalAC(m1, m2)
                    if np.max(MAC) >= tol_mode:
                        SDout[nval]['mode'].append(True)
                    else:
                        SDout[nval]['mode'].append(False)

        return SDout

