#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import scipy.io
import numpy as np
from scipy import linalg
import matplotlib.pylab as plt
import filter as myfilter

def db(x):
    """relative value in dB

    TODO: Maybe x should be rescaled to ]0..1].?
    log10(0) = inf.
    """
    return 20*np.log10( np.abs(x **2 ))

def rescale(x):
    """Rescale data to range

    To 0..1:
    z_i = (x_i− min(x)) / (max(x)−min(x))

    Or custom range:
    a = (maxval-minval) / (max(x)-min(x))
    b = maxval - a * max(x)
    z = a * x + b

    """
    if hasattr(x, "__len__") is False:
        return x
    else:
        return (x - np.amin(x)) / (np.amax(x) - np.amin(x))

def signal_per(u,y,fs, ido=0):
    """Shows the periodicity for the signal

    Parameters:
    ----------
    u : float(ns)
        force signal
    y : float(ns, ndof)
        accelerations
    ido : int
        DOF where periodicity is plotted for
    """

    # number of measurement/sensors
    ndof = ymat.shape[0]
    # total sample points
    ns = ymat.shape[1]
    # number of periods
    nn = int(np.ceil(ns / nper))
    ts = 1/fs
    t = np.arange(ns)*ts

    # dof of shaker
    # id_force = u['iu'].item()

    # dof of measurement to use
    # ie. where is the nonlinearity
    y = ymat[ido]

    # y = np.array( [ np.arange(0,10)+np.random.normal(), np.arange(0,10)+np.random.normal()*0.01, np.arange(0,10)]).flatten()
    # nper = 10
    # ns = len(y)
    # nn = int(np.ceil(ns / nper))

    # first index of last period
    ilast = ns - nper

    # reference period. The last measured period
    yref = y[ilast:ilast+nper]
    yscale = np.amax(y) - np.amin(y)

    # holds the similarity of the signal, compared to reference period
    va =  np.empty(nper*(nn-1))

    nnvec = np.arange(-nn,nn+1, dtype='int')
    va_per = np.empty(nnvec.shape)
    for i, n in enumerate(nnvec):
        # index of the current period. Moving from 1 to nn-1 because of if/break statement
        ist = ilast + n * nper
        if ist < 0:
            continue
        elif ist > ns - nper - 1:
            break

        idx = np.arange(ist,ist+nper)
        # difference between signals in dB
        va[idx] = db(y[idx] - yref)
        va_per[i] = np.amax(va[idx])

    va2 = db(y)

    plt.figure(1)
    plt.clf()
    plt.ion()
    plt.title('Periodicity of signal for DOF {}'.format(ido))
    plt.plot(t,va2, label='signal')  # , rasterized=True)
    plt.plot(t[:ilast],va, label='periodicity')
    for i in range(1,nn):
        x = t[nper * i]
        plt.axvline(x, color='k', linestyle='--')

    plt.xlabel('Time (s)')
    plt.ylabel(r'$\varepsilon$ dB')
    plt.legend()
    plt.show()

class RFS(object):
    def __init__(self,val, fs, dofs=None, show_damped = False, displ = False):
        """
        Parameters:
        -----------
        val : ndarray(ns)
            either accelerations or displacements
        dofs : int(2)
            dofs to compare. If none, then compare to ground, ie. zero signal
        show_damp : bool
            show stifness or damping coeff.
        """

        if dofs is None and val.ndim is 2:
            val = val[0,:]
        else:
            val = val[dofs,:]
        if val.ndim is not 2:
            # cast to 2d array
            val = val[None,:]


        self.displ = displ
        self.show_damped = show_damped
        self.dofs = dofs
        self.fs = fs
        self.ns = val.shape[1]
        self.tol_slice = 1e-2

        # displacement. Differentiate
        if self.displ:
            y = val
            dy = np.empty(val.hape)
            self.ddy = np.empty(val.shape)
            for i in range(val.shape[0]):
                dy[i,:] , self.ddy[i,:] = myfilter.differentiate(y[i,:], self.fs)
        else:
            # accelerations. Integrate
            self.ddy = val
            y = np.empty(val.shape)
            dy = np.empty(val.shape)
            for i in range(val.shape[0]):
                y[i,1:-1] , dy[i,1:] = myfilter.integrate(self.ddy[i,:], self.fs)

        import scipy.io

        # directory = 'data/T03a_Data/'
        # mat =  scipy.io.loadmat(directory + 'f16_x.mat')

        # y = mat['x'][dofs,:]
        # dy = mat['xd'][dofs,:]
        # ddy = mat['xdd'][dofs,:]

        # g( x_i - x_j, dx_i - dx_j) = -ddx_i
        if val.shape[0] is 2:
            # connected to another dof
            self.y = y[0,:] - y[1,:]
            self.dy = dy[0,:] - dy[1,:]
        else:
            # connected to ground
            self.y = y
            self.dy = dy


    def update_sel(self, id0, id1=-1):
        """ Update RFS for the given selection

        Parameters:
        ----------
        id0/id1 : int
            index start/end for the selection
        """
        #t = self.t( id0:di1 );
        y = self.y[id0:id1]
        dy = self.dy[id0:id1]
        ddy = self.ddy[0,id0:id1]

        if self.show_damped:
            tol =  self.tol_slice * max( np.abs(y))
            ind_tol = np.where(np.abs(y) < tol)
        else:
            tol =  self.tol_slice * max( np.abs(dy))
            ind_tol = np.where(np.abs(dy) < tol)
        y_tol = y[ind_tol]
        dy_tol = dy[ind_tol]
        ddy_tol = ddy[ind_tol]

        return y, dy, ddy, y_tol, dy_tol, ddy_tol



# from mpl_toolkits.mplot3d import Axes3D

# plt.figure(2)
# plt.clf()
# ax = plt.axes(projection='3d')
# ax.plot(y1, dy, -ddy, '.k', markersize=10)
# ax.plot(y1_tol, y1_tol, -ddy_tol, '.r', markersize=12)
# ax.set_title("Restoring force surface")
# ax.set_xlabel('Displacement (m)')
# ax.set_ylabel('Velocity (m/s)')
# ax.set_zlabel('-Acceleration (m/s²)')


# plt.figure(3)
# plt.clf()
# plt.title('Stiffness curve')
# plt.xlabel('Displacement (m)')
# plt.ylabel('-Acceleration (m/s²)')
# plt.plot(y1_tol,-ddy_tol,'.k', markersize=12)

# #plt.show()
