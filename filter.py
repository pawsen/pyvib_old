#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy import signal
from scipy import integrate as sp_integrate


def integrate(ddx, fs, order=3, lowcut=0.1, highcut=0.6):
    """Integrate acceleration to get vel and displacement.

    Numerical integration is prone to low-frequency problems.
    Trapez-rule only have problems with low-freq, but is imprecise from
    0.1*nyquist freq.
    Simpsons and Tick's rule or give better precision, but have problems with
    high frequencies as well as low. They can(and should) be used until 1/4 of
    nyquist freq. See fig. I.9 p. 618 [1]_.

    Instead of using band-pass filter, using low pass filter for measurement
    (acceleration) to remove noise and then high pass filter for integrated
    signals (velocity and position) yields better integrated signals.
    The filter on the integrated signal removes drift.

    Parameters:
    ----------
    cutoff : scalar (0-1)
        The freq where the gain drops to 1/sqrt(2) that of the passband (the
        "-3 dB point"). Normalized from 0 to 1, where 1 is the Nyquist
        frequency

    Notes
    -----
    .
    [1]
    K. Worden: Nonlinearity in structural dynamics. Appendix I

    """

    #TODO: skal det være samme cutoff-freq for begge filtre?

    # Create an order bandpass butterworth filter:
    b, a = signal.butter(order, highcut, btype='lowpass')
    ddy = signal.filtfilt(b, a, ddx)

    b, a = signal.butter(order, lowcut, btype='highpass')

    dy = sp_integrate.cumtrapz(ddy)/fs
    dy = signal.filtfilt(b, a, dy)
    y = sp_integrate.cumtrapz(dy)/fs
    y = signal.filtfilt(b, a, y)

    return y, dy

def differentiate(x, fs, order = 3, cutoff=0.5):
    """ Differentiate x twice to get vel and acc

    5-point stencil offers fairly good results in most cases [1]_

    Differentiate signals should normally be avoided. Only use when distance
    sensors cannot be avoided, ie, rotating machinery.

    In frequency domain, differentiation is given by
    Y(omega) = -omega**2*Y(omega)
    which shows that differentiation amplifies high-frequency noise

    The normalized cutoff can be calculated as
    cutoff = 100 Hz
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq

    Parameters:
    ----------
    cutoff : scalar (0-1)
        The freq where the gain drops to 1/sqrt(2) that of the passband (the
        "-3 dB point"). Normalized from 0 to 1, where 1 is the Nyquist
        frequency

    Notes
    -----
    .
    [1]
    K. Worden: Nonlinearity in structural dynamics. Appendix I
    """
    # Create an order lowpass butterworth filter:
    b, a = signal.butter(order, cutoff)

    # Use filtfilt to apply the filter:
    y = signal.filtfilt(b, a, x)

    #sampling frequency
    # fs = 1./dt
    dt = 1/fs
    nyq_freq= fs/2

    #w, gd = signal.group_delay((b, a))

    # 5-point stencil
    b_diff = np.array([1, -8, 0, 8, -1])/12
    a_diff = 1

    """filtfilt: is zero-phase filtering, which doesn't shift the signal as it
    filters. Since the phase is zero at all frequencies, it is also
    linear-phase. Filtering backwards in time requires you to predict the
    future, so it can't be used in "online" real-life applications, only for
    offline processing of recordings of signals.

    lfilter: is causal forward-in-time filtering only, similar to a real-life
    electronic filter. It can't be zero-phase. It can be linear-phase
    (symmetrical FIR), but usually isn't. Usually it adds different amounts of
    delay at different frequencies.
    https://dsp.stackexchange.com/a/19086
    """

    # manual doing the same
    dy = np.zeros(x.shape)
    ddy = np.zeros(x.shape)
    for i in range(len(dy)-4):
        dy[i+2] = 1/dt * b_diff.dot(y[i:i+5])
    #dy = signal.filtfilt(b, a, dy)
    for i in range(2,len(dy)-8):
        ddy[i+2] = 1/dt * b_diff.dot(dy[i:i+5])
    #dyd = signal.filtfilt(b, a, ddy)

    # # The phase delay of the filtered signal
    # n = len(b)
    # delay = 0.5 * (n +1 ) / fs
    # dy = signal.lfilter(b_diff, a_diff, y)/dt
    # #dy[:3], dy[-3:] = 0, 10
    # #dy = signal.filtfilt(b, a, dy)
    # ddy = signal.lfilter(b_diff, a_diff, dy)/dt
    # #ddy[:20], ddy[-20:] = 0, 0
    # #ddy = signal.filtfilt(b, a, ddy)

    return dy, ddy

"""
Test:
import sympy

t = sympy.Symbol('t')
exp = (sympy.sin(2*sympy.pi*0.75*t + 2.1) + 0.1*sympy.sin(2*sympy.pi*1.25*t + 1)
      + 0.18*sympy.cos(2*sympy.pi*3.85*t))
ddx = sympy.diff(exp,t,2)

nsamples = 10000
sample_rate = 500.0
t = np.arange(nsamples) / sample_rate
dt = np.diff(t[:2])[0]
x = (np.sin(2*np.pi*0.75*t + 2.1) + 0.1*np.sin(2*np.pi*1.25*t + 1)
      + 0.18*np.cos(2*np.pi*3.85*t))
xn = x + np.random.randn(len(t)) * 0.00008
ddx = -np.pi**2*(2.25*np.sin(1.5*np.pi*t + 2.1) + 0.625*np.sin(2.5*np.pi*t + 1) + 10.6722*np.cos(7.7*np.pi*t))

plt.figure(1)
plt.clf()
plt.ion()
plt.plot(t, xn, 'b', alpha=0.75)
#plt.plot(t, y, 'k', t - delay, ddy, 'r--',t, ddy2, 'g--',t, ddy3, 'g', t, z2, 'r')
plt.plot(t, y, 'k',t, ddx, 'y', t - delay, ddy, 'r--',t, ddy2, 'g--', t, ddy3,'--')

plt.legend(('noisy signal', 'filtfilt','ddy_symbolic','ddy', 'ddy2',
            ), loc='best')
plt.grid(True)
plt.show()

"""
