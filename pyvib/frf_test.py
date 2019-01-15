import numpy as np
from numpy.fft import fft, ifft
from numpy.linalg import pinv, norm
import scipy.io as sio
from vib.common import db, import_npz
import matplotlib.pyplot as plt

def periodic(U, Y):  #(u, y, nper, fs, fmin, fmax):
    """Calculate the frequency response matrix, and the corresponding noise and
    total covariance matrices from the spectra of periodic input/output data.

    Note that the term stochastic nonlinear contribution term is a bit
    misleading. The NL contribution is deterministic given the same forcing
    buy differs between realizations.

    H(f) = FRF(f) = Y(f)/U(f) (Y/F in classical notation)
    Y and U is the output and input of the system in frequency domain.

    Parameters
    ----------
    u : ndarray
        Forcing signal
    y : ndarray
        Response signal (displacements)
    nper : int
        Number of periods in signal
    fs : float
        Sampling frequency
    fmin : float
        Starting frequency in Hz
    fmax : float
        Ending frequency in Hz

    Returns
    -------
    G : ndarray
        Frequency response matrix(FRM)
    covGML : ndarray
        Total covariance (= stochastic nonlinear contributions + noise)
    covGn : ndarray
        Noise covariance
    """

    # Number of inputs, realization, periods and frequencies
    m, R, P, F = U.shape
    p = Y.shape[0]  # number of outputs
    M = np.floor(R/m).astype(int)  # number of block of experiments
    if M*m != R:
        print('Warning: suboptimal number of experiments: B*m != M')
    # Reshape in M blocks of m experiments
    U = U[:,:m*M].reshape((m,m,M,P,F))
    Y = Y[:,:m*M].reshape((p,m,M,P,F))

    if P > 1:
        # average input/output spectra over periods
        U_mean = np.mean(U,axis=3)  # m x m x M x F
        Y_mean = np.mean(Y,axis=3)

        # Estimate noise spectra
        # create new axis. We could have used U_m = np.mean(U,3, keepdims=True)
        NU = U - U_mean[:,:,:,None,:]  # m x m x M x P x F
        NY = Y - Y_mean[:,:,:,None,:]

        # Calculate input/output noise (co)variances on averaged(over periods)
        # spectra
        covU = np.empty((m*m,m*m,M,F), dtype=complex)
        covY = np.empty((p*m,p*m,M,F), dtype=complex)
        covYU = np.empty((p*m,m*m,M,F), dtype=complex)
        for mm in range(M):  # Loop over experiment blocks
            # noise spectrum of experiment block mm (m x m x P x F)
            NU_m = NU[:,:,mm]
            NY_m = NY[:,:,mm]
            for f in range(F):
                # TODO extend this using einsum, so we avoid all loops
                # TODO fx: NU_m[...,f].reshape(-1,*NU_m[...,f].shape[2:])
                # flatten the m x m dimension and use einsum to take the outer
                # product of m*m x m*m and then sum over the p periods.
                tmpUU = NU_m[...,f].reshape(-1, P)  # create view
                tmpYY = NY_m[...,f].reshape(-1, P)
                covU[:,:,mm,f] = np.einsum('ij,kj->ik',tmpUU,tmpUU.conj()) / (P-1)/P
                covY[:,:,mm,f] = np.einsum('ij,kj->ik',tmpYY,tmpYY.conj()) / (P-1)/P
                covYU[:,:,mm,f] = np.einsum('ij,kj->ik',tmpYY,tmpUU.conj()) / (P-1)/P

        # Further calculations with averaged spectra
        U = U_mean  # m x m x M x F
        Y = Y_mean

    # Compute FRM and noise and total covariance on averaged(over experiment
    # blocks and periods) FRM
    G = np.empty((p,m,F), dtype=complex)
    covGML = np.empty((m*p,m*p,F), dtype=complex)
    covGn = np.empty((m*p,m*p,F), dtype=complex)
    Gm = np.empty((p,m,M), dtype=complex)
    U_inv_m = np.empty((m,m,M), dtype=complex)
    covGn_m = np.empty((m*p,m*p,M), dtype=complex)
    for f in range(F):
        # Estimate the frequency response matrix (FRM)
        for mm in range(M):
            # psudo-inverse using svd. A = u*s*v, then A* = vᵀs*u where s*=1/s
            U_inv_m[:,:,mm] = pinv(U[:,:,mm,f])
            # FRM of experiment block m at frequency f
            Gm[:,:,mm] = Y[:,:,mm,f] @ U_inv_m[:,:,mm]

        # Average FRM over experiment blocks
        G[:,:,f] = Gm.mean(2)

        # Estimate the total covariance on averaged FRM
        NG = G[:,:,f,None] - Gm
        tmp = NG.reshape(-1, M)
        covGML[:,:,f] = np.einsum('ij,kj->ik',tmp,tmp.conj()) / M/(M-1)

        # Estimate noise covariance on averaged FRM (only if P > 1)
        if P > 1:
            for mm in range(M):
                U_invT = U_inv_m[:,:,mm].T
                A = np.kron(U_invT, np.eye(p))
                B = -np.kron(U_invT, Gm[:,:,mm])
                AB = A @ covYU[:,:,mm,f] @ B.conj().T
                covGn_m[:,:,mm] = A @ covY[:,:,mm,f] @ A.conj().T + \
                    B @ covU[:,:,mm,f] @ B.conj().T + \
                    (AB + AB.conj().T)

            covGn[:,:,f] = covGn_m.mean(2)/M

    # No total covariance estimate possible if only one experiment block
    if M < 2:
        covGML = 0
    # No noise covariance estimate possible if only one period
    if P == 1:
        covGn = 0

    return G, covGML, covGn


data = sio.loadmat('data.mat')

Y = data['Y'].transpose((1,2,3,0))
U = data['U'].transpose((1,2,3,0))
G_data = data['G']
covGML_data = data['covGML']
covGn_data = data['covGn']

lines = data['lines'].squeeze()
non_exc_even = data['non_exc_even'].squeeze()
non_exc_odd = data['non_exc_odd'].squeeze()

N = 1024
freq = (lines-1)/N  # Excited frequencies (normalized)

G, covGML, covGn = periodic(U, Y)


plt.ion()
f4 = plt.figure(4)
plt.clf()
plt.plot(freq, db(G.squeeze()),'*')
plt.plot(freq, db(covGML.squeeze()),'^', mfc='none')
plt.plot(freq, db(covGn.squeeze()),'v', mfc='none')  # korrekt

plt.plot(freq, db(G_data.squeeze()),'o', mfc='none')
plt.plot(freq, db(covGML_data.squeeze()),'>', mfc='none')
plt.plot(freq, db(covGn_data.squeeze()),'<', mfc='none')

#plt.plot(freq[lines_odd_det]/fAut,db(Y_mean[0,lines_odd_det]),'^', mfc='none')
#plt.plot(freq[lines_even]/fAut,db(Y_mean[0,lines_even]),'o', mfc='none')
#plt.xlim([0,fs/2/fAut])
#plt.xlim([0,3])
plt.xlabel('frequency f/fAut')
plt.ylabel('magnitude (dB)')
plt.title('FFT output')
#plt.legend(('excited lines','odd detection lines','even detection lines'))
