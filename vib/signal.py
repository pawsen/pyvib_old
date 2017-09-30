#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pylab as plt

from .common import db
from .filter import integrate, differentiate


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

        self.u = u
        self.fs = fs

        # ns: total sample points
        if y is not None:
            self.ndof, self.ns = self.y.shape
        elif yd is not None:
            self.ndof, self.ns = self.y.shape
        elif ydd is not None:
            self.ndof, self.ns = self.y.shape

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

        # number of periods. Only used for this check
        ns = self.ns
        _nper = int(np.floor(ns / nsper))
        if any(p > _nper - 1 for p in per):
            raise ValueError('Period too high. Only {} periods in data.'.
                             format(_nper),per)

        self.iscut = True
        ndof = self.ndof
        self.nper = len(per)
        self.nsper = int(nsper)
        # number of sample for cut'ed signal
        ns = self.nper * self.nsper

        # extract periodic signal
        self.y_per = np.empty((ndof, self.nper*self.nsper))
        self.u_per = np.empty(self.nper*self.nsper)
        for i, p in enumerate(per):
            self.y_per[:,i*nsper: (i+1)*nsper] = self.y[:,offset + p*nsper:
                                                        offset+(p+1)*nsper]
            self.u_per[i*nsper: (i+1)*nsper] = self.u[offset + p*nsper:
                                                      offset + (p+1)*nsper]

    def periodicity(self, nsper, dof=0, offset=0, fig=None, ax=None):
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
        ax.plot(t,signal,'--', label='signal')  # , rasterized=True)
        ax.plot(t[:ilast],va, label='periodicity')
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


def _set_signal(y):
    if y is not None:
        if y.ndim != 2:
            y = y.reshape(-1,y.shape[0])
        return y
    return None
