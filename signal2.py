#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import scipy.io
import numpy as np
from scipy import linalg
import matplotlib.pylab as plt
from common import db
from filter import integrate, differentiate


class Signal(object):
    """ Holds common properties for a signal
    """
    def __init__(self, u, y, fs):
        """
        Parameters
        ----------
        y : ndarray(ns, ndof)
            accelerations
        fs : int
            Sampling frequency

        """

        # cast to 2d. Format is now y[ndofs,ns]. For 1d cases ndof=0
        if y.ndim != 2:
            y = y.reshape(-1,y.shape[0])

        self.u = u
        self.y = y
        self.fs = fs

        # number of measurement/sensors
        self.ndof = self.y.shape[0]
        # total sample points
        self.ns = self.y.shape[1]

        self.dy = None
        self.ddy = None
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
        if any(p > _nper- 1 for p in per):
            raise ValueError('Period too high. Only {} periods in data.'.format(nper),per)

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
            self.y_per[:,i*nsper : (i+1)*nsper] = self.y[:,offset + p*nsper :offset+(p+1)*nsper]
            self.u_per[i*nsper : (i+1)*nsper] = self.u[offset + p*nsper:offset + (p+1)*nsper]


    def periodicity(self, nsper, ido=0, offset=0, savefig={'save':False,'fname':''}):
        """Shows the periodicity for the signal

        Parameters:
        ----------
        nsper : int
            Number of points per period
        ido : int
            DOF where periodicity is plotted for
        offset : int
            Use offset as first index
        """

        fs = self.fs

        y = self.y[ido,offset:]
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
        va =  np.empty(nsper*(nper-1))

        nnvec = np.arange(-nper,nper+1, dtype='int')
        va_per = np.empty(nnvec.shape)
        for i, n in enumerate(nnvec):
            # index of the current period. Moving from 1 to nn-1 because of if/break statement
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

        plt.figure(1)
        plt.clf()
        #plt.ion()

        plt.title('Periodicity of signal for DOF {}'.format(ido))
        plt.plot(t,signal,'--', label='signal')  # , rasterized=True)
        plt.plot(t[:ilast],va, label='periodicity')
        for i in range(1,nper):
            x = t[nsper * i]
            plt.axvline(x, color='k', linestyle='--')

        plt.xlabel('Time (s)')
        plt.ylabel(r'$\varepsilon$ dB')
        plt.legend()

        if savefig['save']:
            fname = savefig['fname']
            plt.savefig(fname + '.png')
            plt.savefig(fname + '.pdf')
            print('plot saved as {}'.format(fname))

        plt.show()

    def get_displ(self, lowcut, highcut, isnumeric=False):
        """Integrate signals to get velocity and displacement"""

        self.isnumeric = isnumeric
        ddy = self.ddy
        ndof = self.ndof
        fs = self.fs

        y = np.empty(ddy.shape)
        dy = np.empty(ddy.shape)
        for i in range(ndof):
            y[i,:], dy[i,:] = integrate(ddy[i,:], fs, lowcut, highcut, isnumeric=isnumeric)

        self.y = y
        self.dy = dy

    def get_accel(self, isnumeric=False):
        """ Differentiate signals to get velocity and accelerations
        """
        self.isnumeric = isnumeric
        y = self.y
        ndof = self.ndof
        fs = self.fs

        dy = np.empty(val.shape)
        ddy = np.empty(val.shape)
        for i in range(ndof):
            dy[i,:], ddy[i,:] = differentiate(y[i,:], fs, isnumeric=isnumeric)

        self.dy = dy
        self.ddy = ddy

    def set_values(self, y=None, dy=None, ddy=None):
        if y is not None:
            #print(np.linalg.norm(y), np.linalg.norm(self.y))
            # cast to 2d. Format is now y[ndofs,ns]. For 1d cases ndof=0
            if y.ndim != 2:
                self.y = y.reshape(-1,y.shape[0])
            else:
                self.y = y
        if dy is not None:
            #print(np.linalg.norm(dy), np.linalg.norm(self.dy))
            if dy.ndim != 2:
                self.dy = dy.reshape(-1,dy.shape[0])
            else:
                self.dy = dy
        if ddy is not None:
            if ddy.ndim != 2:
                self.ddy = ddy.reshape(-1,ddy.shape[0])
            else:
                self.ddy = ddy
