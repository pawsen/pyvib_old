#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pylab as plt

from .common import db, prime_factor
from .filter import integrate, differentiate
from scipy.signal import decimate
# from .morletWT import morletWT
# from collections import namedtuple

class Signal(object):
    """ Holds common properties for a signal
    """
    def __init__(self, u, fs, y=None, yd=None, ydd=None):
        """
        Parameters
        ----------
        y : ndarray(ns, ndof)
            accelerations
        fs : int
            Sampling frequency

        """
        # cast to 2d. Format is now y[ndofs,ns]. For 1d cases ndof=0
        self.y = _set_signal(y)
        self.yd = _set_signal(yd)
        self.ydd = _set_signal(ydd)

        self.isset_y = False
        self.isset_yd = False
        self.isset_ydd = False
        # ns: total sample points
        if y is not None:
            self.ndof, self.ns = self.y.shape
            self.isset_y = True
        elif yd is not None:
            self.ndof, self.ns = self.yd.shape
            self.isset_yd = True
        elif ydd is not None:
            self.ndof, self.ns = self.ydd.shape
            self.isset_ydd = True

        self.u = _set_signal(u)
        self.fs = fs

        self.y_per = None
        self.u_per = None
        self.iscut = False
        self.isnumeric = False

    def cut(self, nsper, per, offset=0):
        """Extract periodic signal from original signal

        Parameters
        ----------
        nsper : int
            Number of samples per period
        per : list
            List of periods to use. 0-based. Ie. [0,1,2] etc
        """
        per = np.atleast_1d(per)
        # number of periods. Only used for this check
        ns = self.ns
        _nper = int(np.floor(ns / nsper))
        if any(p > _nper - 1 for p in per):
            raise ValueError('Period too high. Only {} periods in data.'.
                             format(_nper),per)

        self.iscut = True
        self.nper = len(per)
        self.nsper = int(nsper)
        # number of sample for cut'ed signal
        ns = self.nper * self.nsper

        # extract periodic signal
        if self.isset_y:
            self.y_per = _cut(self.y, per, self.nsper,offset)
        if self.isset_yd:
            self.yd_per = _cut(self.yd, per, self.nsper,offset)
        if self.isset_ydd:
            self.ydd_per = _cut(self.ydd, per, self.nsper,offset)

        self.u_per = _cut(self.u, per, self.nsper, offset)


    def periodicity(self, nsper=None, dof=0, offset=0, fig=None, ax=None,
                    **kwargs):
        """Shows the periodicity for the signal


        Parameters:
        ----------
        nsper : int
            Number of points per period
        dof : int
            DOF where periodicity is plotted for
        offset : int
            Use offset as first index
        """
        if nsper is None:
            nsper = self.nsper

        fs = self.fs

        y = self.y[dof,offset:]
        ns = len(y)
        ndof = self.ndof
        # number of periods
        nper = int(np.floor(ns / nsper))
        # in case the signal contains more than an integer number of periods
        ns = nper*nsper
        t = np.arange(ns)/fs

        # first index of last period
        ilast = ns - nsper

        # reference period. The last measured period
        yref = y[ilast:ilast+nsper]
        yscale = np.amax(y) - np.amin(y)
        # holds the similarity of the signal, compared to reference period
        va = np.empty(nsper*(nper-1))

        nnvec = np.arange(-nper,nper+1, dtype='int')
        va_per = np.empty(nnvec.shape)
        for i, n in enumerate(nnvec):
            # index of the current period. Moving from 1 to nn-1 because of
            # if/break statement
            ist = ilast + n * nsper
            if ist < 0:
                continue
            elif ist > ns - nsper - 1:
                break

            idx = np.arange(ist,ist+nsper)
            # difference between signals in dB
            va[idx] = db(y[idx] - yref)
            va_per[i] = np.amax(va[idx])

        signal = db(y[:ns])
        if fig is None:
            fig, ax = plt.subplots()
            ax.clear()

        ax.set_title('Periodicity of signal for DOF {}'.format(dof))
        ax.plot(t,signal,'--', label='signal', **kwargs)  # , rasterized=True)
        ax.plot(t[:ilast],va, label='periodicity', **kwargs)
        for i in range(1,nper):
            x = t[nsper * i]
            ax.axvline(x, color='k', linestyle='--')

        ax.set_xlabel('Time (s)')
        ax.set_ylabel(r'$\varepsilon$ dB')
        ax.legend()

        return fig, ax

    def get_displ(self, lowcut, highcut, isnumeric=False):
        """Integrate signals to get velocity and displacement"""

        self.isnumeric = isnumeric
        ydd = self.ydd
        ndof = self.ndof
        fs = self.fs

        y = np.empty(ydd.shape)
        yd = np.empty(ydd.shape)
        for i in range(ndof):
            y[i,:], yd[i,:] = integrate(ydd[i,:], fs, lowcut, highcut,
                                        isnumeric=isnumeric)

        self.isset_y = True
        self.isset_yd = True
        self.y = y
        self.yd = yd

    def get_accel(self, isnumeric=False):
        """ Differentiate signals to get velocity and accelerations
        """
        self.isnumeric = isnumeric
        y = self.y
        ndof = self.ndof
        fs = self.fs

        yd = np.empty(y.shape)
        ydd = np.empty(y.shape)
        for i in range(ndof):
            yd[i,:], ydd[i,:] = differentiate(y[i,:], fs, isnumeric=isnumeric)

        self.yd = yd
        self.ydd = ydd

    def set_signal(self, y=None, yd=None, ydd=None):
        self.y = _set_signal(y)
        self.yd = _set_signal(yd)
        self.ydd = _set_signal(ydd)
        if y is not None:
            self.ndof, self.ns = self.y.shape
            self.isset_y = True
        elif yd is not None:
            self.ndof, self.ns = self.yd.shape
            self.isset_yd = True
        elif ydd is not None:
            self.ndof, self.ns = self.ydd.shape
            self.isset_ydd = True
    # def wt(self, f1, f2, nf=50, f00=10, dof=0, pad=0):
    #     fs = self.fs
    #     x = self.y[dof]
    #     finst, wtinst, time, freq, y = morletWT(x, fs, f1, f2, nf, f00, pad)

    #     # poor mans object
    #     WT = namedtuple('WT', 'f1 f2 nf f00  dof pad finst wtinst time freq y')
    #     self.wt = WT(f1, f2, nf, f00, dof, pad,finst, wtinst, time, freq, y)

    def filter(self, lowcut, highcut, order=3):
        from scipy import signal
        fn = 0.5 * self.fs
        if highcut > fn:
            raise ValueError('Highcut frequency is higher than nyquist\
            frequency of the signal', highcut, fn)
        elif lowcut <= 0:
            raise ValueError('Lowcut frequency is 0 or lower', lowcut, fn)

        b, a = signal.butter(order, highcut, btype='lowpass')
        return signal.filtfilt(b, a, self.y)

    def downsample(self, n, nsper=None, remove=False):
        """Filter and downsample signals

        The displacement is decimated(low-pass filtered and downsampled) where
        forcing is only downsampled by the sampling factor n, ie. every n'th
        sample is kept.

        Parameters
        ----------
        n: scalar
            downsample rate, ie. keep every n'th sample
        nsper : int
            Number of samples per period
        remove: bool
            Removal of the last simulated period to eliminate the edge effects
            due to the low-pass filter.

        """
        y = self.y
        u = self.u

        # axis to operate along
        axis = -1

        # filter and downsample
        # prime factor decomposition.
        for k in prime_factor(n):
            y = decimate(y, q=k, ftype='fir', axis=-1)

        # index for downsampling u
        sl = [slice(None)] * u.ndim
        sl[axis] = slice(None, None, n)
        u = u[sl]

        # Removal of the last simulated period to eliminate the edge effects
        # due to the low-pass filter.
        if remove:
            ns = self.ns
            nper = int(np.floor(ns/nsper))
            y = y[:,:(nper-1)*nsper]
            u = u[:,:(nper-1)*nsper]

        return u, y


def _set_signal(y):
    if y is not None:
        y = np.atleast_2d(y)
        return y
    return None


def _cut(x,per,nsper,offset=0):
    """Extract periodic signal from original signal"""

    nper = len(per)
    ndof = x.shape[0]
    x_per = np.empty((ndof, nper*nsper))

    for i, p in enumerate(per):
        x_per[:,i*nsper: (i+1)*nsper] = x[:,offset + p*nsper:
                                          offset+(p+1)*nsper]
    return x_per
