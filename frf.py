#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

def periodic( u, y, nper, fs, fmin, fmax):
    """ Calculate FRF for a periodic signal.

    H(f) = FRF(f) = Y(f)/U(f) (Y/F in classical notation)
    Y and U is the output and input of the system in frequency domain.

    The receptance frequency response function is defined as the ration between
    a harmonic displacement and a harmonic force. This ratio is complex, since
    there is both an amplitude ratio, np.abs(FRF), and a phase angle,
    np.angle(FRF)/np.pi*180 for the phase in degrees.

    Other definitions can be used.

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
    freq : ndarray
        Frequencies for the FRF
    FRF : ndarray
        Receptance vector. Also called H.
    sigN : ndarray
        Standard deviation between each period
    """

    # TODO: Only relevant with multiple forcings
    # M: number of forced dofs
    # ns: number of points per forced dof
    # M, ns = u.shape
    # if ns < M:
    #     u = u.T
    #     M, ns = u.shape

    ns = len(u)

    # points per period
    nsper = ns // nper

    freq = np.arange(0, nsper//2) * fs/nsper
    flines = np.where( (freq >= fmin) & (freq <= fmax))
    freq = freq[flines]
    nfreq = len(freq)

    G = np.empty((nper, nfreq), dtype='complex')
    for p in range (nper):

        # do fft on current period
        U = np.fft.fft(u[p*nsper : (p+1)*nsper]) / np.sqrt(nsper)

        Y = np.fft.fft(y[p*nsper : (p+1)*nsper]) / np.sqrt(nsper)

        U = U[flines]
        Y = Y[flines]

        G[p,:] = Y / U

    """In case of multiple forcings, one could put an extra for-loop around the
    p-loop, ie. G[m,p,:] = Y/U, where m is the current forcing. We would then
    have the current points as
    Y = fft(y[m, p*nsper:...]) and visa versa for u.
    Then take the frf as
    FRF = np.mean(G, (1,0))
    ie. first mean over periods, then forcings.
    """

    # Make the FRF the average of all periods
    FRF = np.mean(G, axis=0)

    # deviation between FFTs for each period
    if nper > 1:
        sigN = np.sqrt(np.std(G, axis=0)**2 / nper)
    else:
        sigN = None


    return freq, FRF, sigN

def nonperiodic(u, y, N, fs, fmin, fmax):
    """Calculate FRF for a nonperiodic signal.

    A nonperiodic signal could be a hammer test. For nonperiodid signals,
    normally H1 or H2 is calculated. H2 is most commonly used with random
    excitation.
    H1 is used when the output is expected to be noisy compared to the input.
    H2 is used when the input is expected to be noisy compared to the output.

    H1 = Suy/Suu
    H2 = Syy/Syu

    All spectral densities are calculated in frequency domain
    Suy is the Cross Spectral Density of the input and output
    Suu/Syy is the Auto Spectral Density of the input/output. Also called power
        spectral density(PSD)
    Suy = Syu* (complex conjugate). See [1]_


    Parameters
    ----------
    u : ndarray
        Forcing signal
    y : ndarray
        Response signal (displacements)
    N : int
        Number of points used for fft. N =  2^nfft, where nfft standard is
        chosen as 8.
    fs : float
        Sampling frequency
    fmin : float
        Starting frequency in Hz
    fmax : float
        Ending frequency in Hz

    Returns
    -------
    freq : ndarray
        Frequencies for the FRF
    FRF : ndarray
        Receptance vector. Also called H.
    sigT : ndarray

    gamma : ndarray
        Coherence. Between 0 and 1 that measures the correlation between u(n)
        and y(n) at the frequency f, ie. can y be predicted from u. The
        coherence of a linear system therefore represents the fractional part
        of the output signal power that is produced by the input at that
        frequency. 1 is perfect correlation, 0 is none.

    Notes -----
    [1]: Ewins, D. J. "Modal testing: theory, practice and application (2003),
    pages 141.
    https://en.wikipedia.org/wiki/Spectral_density
    https://en.wikipedia.org/wiki/Coherence_(signal_processing)

    """

    while N > len(u)/2:
      N = int(N//2)

    M = int(np.floor(len(u)/N))


    freq = (np.arange(1, N/2 - 1) + 0.5) * fs/ N
    flines = np.where( (freq >= fmin) & (freq <= fmax))
    freq = freq[flines]
    nfreq = len(freq)


    u = u[:M * N]
    y = y[:M * N]

    u = np.reshape(u, (N, M))
    y = np.reshape(y, (N, M))

    U = np.fft.fft(u, axis=0) / np.sqrt(N)
    U = U[1:N/2+1,:]
    Y = np.fft.fft(y, axis=0) / np.sqrt(N)
    Y = Y[1:N/2+1,:]


    U = diff(U, axis=0)
    Y = diff(Y, axis=0)

    # Syu: Cross spectral density
    # Suu: power spectral density
    # (mayby not correct: taking the mean, is the same as taking the expectance
    # E)
    Syu = np.mean(Y * U.conj(), axis=1)
    Suu = np.mean(np.abs(U)**2, axis=1)
    Syy = np.mean(np.abs(Y)**2, axis=1)
    FRF = Syu / Suu
    FRF = FRF[flines]

    if M > 1:
        sigT = 1/(M-1) * (Syy - np.abs(Syu)**2/Suu) / Suu
        # coherence
        gamma = np.abs(Syu)**2 / (Suu*Syy)
        sigT = sigT[flines]
        gamma = gamma[flines]
    else:
        sigT = None
        gamma = None


    return freq, FRF, sigT, gamma
