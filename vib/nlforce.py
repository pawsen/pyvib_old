#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.sparse import coo_matrix

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

class NL_force(object):

    def __init__(self):
        self.nls = []
        self.dnls_force = []
        self.dnls_damp = []

    def add(self, _NL_compute):
        self.nls.append(_NL_compute)
        if _NL_compute.is_force:
            self.dnls_force.append(_NL_compute)
        else:
            self.dnls_damp.append(_NL_compute)

    def force(self, x, xd):

        if x.ndim == 1:
            ndof, = x.shape
            ns = 1
        else:
            ndof, ns = x.shape
        fnl = np.zeros((ndof+1, ns))

        for nl in self.nls:
            fnl = nl.compute(x, xd, fnl)
        # squeeze in case ns = 1
        fnl = fnl[:ndof,:].squeeze()
        return fnl

    def dforce(self, x, xd, is_force=True):
        """Derivative of nonlinear functional
        """
        if x.ndim == 1:
            ndof, = x.shape
            ns = 1
        else:
            ndof, ns = x.shape

        dfnl = np.zeros((ndof+1, ns*(ndof+1)))

        if is_force:
            for nl in self.dnls_force:
                dfnl = nl.dcompute(x, xd, dfnl)
        else:
            for nl in self.dnls_damp:
                dfnl = nl.dcompute(x, xd, dfnl)

        if ns == 1:
            return dfnl[:ndof,:ndof].squeeze()

        # create sparse structure from dfnl
        # TODO: dont create dfnl in the first place...:)
        ind = np.arange(ns*(ndof+1))
        ind = np.delete(ind, np.s_[ndof::ndof+1])
        dfnl = dfnl[:ndof, ind]
        dfnl = np.reshape(dfnl, (ndof**2,ns), order='F')
        # dont ask...
        ind = np.outer(np.ones(ndof), np.arange(ndof)) * ns*ndof + \
            np.outer(np.arange(ndof), np.ones(ndof))
        ind = np.outer(ind.T, np.ones(ns)) + \
            ns*ndof * np.outer(np.ones(ndof**2), np.arange(0,(ns-1)*ndof+1, ndof)) + \
            ndof * np.outer(np.ones(ndof**2), np.arange(ns))

        ind = ind.ravel(order='F').astype('int')

        #https://stackoverflow.com/questions/28995146/matlab-ind2sub-equivalent-in-python
        arr = ns*ndof*np.array([1,1])
        ii, jj = np.unravel_index(ind, tuple(arr), order='F')
        dfnl_s = coo_matrix((dfnl.ravel(order='F'), (ii, jj)),
                            shape=(ndof*ns, ndof*ns)).tocsr()

        return dfnl_s
        # return dfnl

    def energy(self, x, xd):
        if x.ndim == 1:
            ndof, = x.shape
            ns = 1
        else:
            ndof, ns = x.shape
        energy = 0
        for nl in self.nls:
            energy = nl.energy(x, xd, energy)
        return energy


class _NL_compute(object):
    is_force = True

    def compute(self, x, fnl):
        pass
    def dcompute(self, x, fnl):
        pass
    def energy(self, x, fnl):
        pass

class NL_polynomial(_NL_compute):
    def __init__(self, inl, enl, knl, is_force=True):
        self.inl = inl
        self.enl = enl
        self.knl = knl
        self.is_force = is_force

    def compute(self, x, xd, fnl):
        inl = self.inl
        ndof = fnl.shape[0] - 1
        idof = np.arange(ndof)
        nbln = inl.shape[0]
        if self.is_force is False:
            x = xd

        for j in range(nbln):
            # connected from
            i1 = inl[j,0]
            # conencted to
            i2 = inl[j,1]

            # Convert to the right index
            idx1 = np.where(i1 == idof)
            x1 = x[idx1]

            # if connected to ground
            if i2 == -1:
                idx2 = ([ndof],)
                x2 = 0
            else:
                idx2 = np.where(i2 == idof)
                x2 = x[idx2]
            x12 = x1 - x2
            # in case of even functional
            if (self.enl[j] % 2 == 0):
                x12 = np.abs(x12)

            f12 = self.knl[j] * x12**self.enl[j]
            fnl[idx1] += f12
            fnl[idx2] -= f12
        return fnl

    def dcompute(self, x, xd, dfnl):
        """Derivative of nonlinear functional
        """
        inl = self.inl
        ndof = dfnl.shape[0]-1
        idof = np.arange(ndof)
        nbln = inl.shape[0]
        if self.is_force is False:
            x = xd

        for j in range(nbln):
            i1 = inl[j,0]
            i2 = inl[j,1]
            idx1 = np.where(i1 == idof)
            x1 = x[idx1]
            # extraction needed due to the way the slice is made in dfnl.
            idx1 = idx1[0][0]
            if i2 == -1:
                idx2 = ndof
                x2 = 0
            else:
                idx2 = np.where(i2 == idof)
                x2 = x[idx2]
                idx2 = idx2[0][0]
            x12 = x1 - x2
            df12 = self.knl[j] * self.enl[j] * np.abs(x12)**(self.enl[j]-1)
            # in case of even functional
            if (self.enl[j] % 2 == 0):
                idx = np.where(x12 < 0)
                df12[idx] = -df12[idx]

            # add the nonlinear force to the right dofs
            dfnl[idx1, idx1::ndof+1] += df12
            dfnl[idx2, idx1::ndof+1] -= df12
            dfnl[idx1, idx2::ndof+1] -= df12
            dfnl[idx2, idx2::ndof+1] += df12
        return dfnl

    def energy(self, x, xd, energy):
        inl = self.inl
        if self.is_force is False:
            x = xd
        if x.ndim == 1:
            ndof, = x.shape
            ns = 1
        else:
            ndof, ns = x.shape
        idof = np.arange(ndof)
        nbln = inl.shape[0]
        for j in range(nbln):
            i1 = inl[j,0]
            i2 = inl[j,1]
            idx1 = np.where(i1 == idof)
            x1 = x[idx1]
            if i2 == -1:
                idx2 = ([ndof],)
                x2 = 0
            else:
                idx2 = np.where(i2 == idof)
                x2 = x[idx2]
            x12 = x1 - x2
            if (self.enl[j] % 2 == 0):
                x12 = np.abs(x12)
            e12 = self.knl[j] / (self.enl[j]+1) * abs(x12)**(self.enl[j]+1)
            # TODO there should be more to this if-statement
            if x12 < 0:
                pass
                #e12 = 0
            energy += e12
        return energy

class NL_tanh_damping(_NL_compute):
    def __init__(self, inl, enl, knl):
        self.inl = inl
        self.enl = enl
        self.knl = knl
        self.is_force = False

    def compute(self, x, xd, fnl):
        inl = self.inl
        ndof = fnl.shape[0] - 1
        idof = np.arange(ndof)
        nbln = inl.shape[0]
        for j in range(nbln):
            i1 = inl[j,0]
            i2 = inl[j,1]
            idx1 = np.where(i1 == idof)
            xd1 = xd[idx1]
            if i2 == -1:
                idx2 = ([ndof],)
                xd2 = 0
            else:
                idx2 = np.where(i2 == idof)
                xd2 = xd[idx2]
            xd12 = xd1 - xd2
            f12 = self.knl[j] * np.tanh(xd12 * self.enl[j])
            fnl[idx1] += f12
            fnl[idx2] -= f12
        return fnl

    def dcompute(self, x, xd, dfnl):
        inl = self.inl
        ndof = dfnl.shape[0]-1
        idof = np.arange(ndof)
        nbln = inl.shape[0]

        for j in range(nbln):
            i1 = inl[j,0]
            i2 = inl[j,1]
            idx1 = np.where(i1 == idof)[0][0]
            xd1 = xd[idx1]
            if i2 == -1:
                idx2 = ndof
                xd2 = 0
            else:
                idx2 = np.where(i2 == idof)[0][0]
                xd2 = xd[idx2]
            xd12 = xd1 - xd2
            df12 = self.enl[j] * self.knl[j] * (1 - np.tanh(xd12 * self.enl[j])**2)
            dfnl[idx1, idx1::ndof+1] += df12
            dfnl[idx2, idx1::ndof+1] -= df12
            dfnl[idx1, idx2::ndof+1] -= df12
            dfnl[idx2, idx2::ndof+1] += df12
        return dfnl

class NL_piecewise_linear(_NL_compute):
    def __init__(self, x, y, slope, delta, symmetric, inl, is_force=True):
        """
        Parameters
        ----------
        x: ndarray [nbln, n_knots]
            x-coordinates for knots
        y: ndarray [nbln, n_knots]
            y-coordinates for knots
        slope: ndarray [nbln, n_knots+1]
            Slope for each linear segment
        delta: ndarray [nbln, n_knots]
            ??
        inl: ndarray [nbln, 2]
            DOFs for nonlinear connection
        """
        self.x = x
        self.y = y
        self.slope = slope
        self.delta = delta
        self.symmetric = symmetric
        self.inl = inl
        self.is_force = is_force

    def compute(self, x, xd, fnl):
        inl = self.inl
        nbln = inl.shape[0]
        ndof = fnl.shape[0] - 1
        idof = np.arange(ndof)
        if self.is_force is False:
            x = xd

        for j in range(nbln):
            i1 = inl[j,0]
            i2 = inl[j,1]
            idx1 = np.where(i1 == idof)
            x1 = x[idx1]
            if i2 == -1:
                idx2 = ([ndof],)
                x2 = 0
            else:
                idx2 = np.where(i2 == idof)
                x2 = x[idx2]
            x12 = x1 - x2
            if self.symmetric[j] is True and x12 < 0:
                x12 = -x12
            f12 = piecewise_linear(self.x[j], self.y[j], self.slope[j],
                                   self.delta[j], x12)
            fnl[idx1] += f12
            fnl[idx2] -= f12
        return fnl

    def dcompute(self, x, xd, dfnl):
        inl = self.inl
        ndof = dfnl.shape[0]-1
        idof = np.arange(ndof)
        nbln = inl.shape[0]
        if self.is_force is False:
            x = xd

        for j in range(nbln):
            i1 = inl[j,0]
            i2 = inl[j,1]
            idx1 = np.where(i1 == idof)[0][0]
            x1 = x[idx1]
            if i2 == -1:
                idx2 = ndof
                x2 = 0
            else:
                idx2 = np.where(i2 == idof)[0][0]
                x2 = x[idx2]
            x12 = x1 - x2
            if self.symmetric[j] is True and x12 < 0:
                x12 = -x12
            df12 = piecewise_linear_der(self.x[j], self.y[j], self.slopes[j],
                                        self.delta[j], x12)
            dfnl[idx1, idx1::ndof+1] += df12
            dfnl[idx2, idx1::ndof+1] -= df12
            dfnl[idx1, idx2::ndof+1] -= df12
            dfnl[idx2, idx2::ndof+1] += df12
        return dfnl

class NL_spline(_NL_compute):
    def __init__(self, x, coeff, symmetric, inl, is_force=True):
        self.x = x
        self.coeff = coeff
        self.symmetric = symmetric
        self.inl = inl
        self.is_force = is_force

    def compute(self, x, xd, fnl):
        inl = self.inl
        nbln = inl.shape[0]
        ndof = fnl.shape[0] - 1
        idof = np.arange(ndof)
        if self.is_force is False:
            x = xd

        for j in range(nbln):
            i1 = inl[j,0]
            i2 = inl[j,1]
            idx1 = np.where(i1 == idof)
            x1 = x[idx1]
            if i2 == -1:
                idx2 = ([ndof],)
                x2 = 0
            else:
                idx2 = np.where(i2 == idof)
                x2 = x[idx2]
            x12 = x1 - x2
            if self.symmetric[j] is True and x12 < 0:
                x12 = -x12
            f12, _ = spline_interp(self.x[j], self.coeff[j], x12)
            fnl[idx1] += f12
            fnl[idx2] -= f12
        return fnl

    def dcompute(self, x, xd, dfnl):
        inl = self.inl
        ndof = dfnl.shape[0]-1
        idof = np.arange(ndof)
        nbln = inl.shape[0]
        if self.is_force is False:
            x = xd

        for j in range(nbln):
            i1 = inl[j,0]
            i2 = inl[j,1]
            idx1 = np.where(i1 == idof)[0][0]
            x1 = x[idx1]
            if i2 == -1:
                idx2 = ndof
                x2 = 0
            else:
                idx2 = np.where(i2 == idof)[0][0]
                x2 = x[idx2]
            x12 = x1 - x2
            if self.symmetric[j] is True and x12 < 0:
                x12 = -x12
            _, df12 = spline_interp(self.x[j], self.coeff[j], x12)
            dfnl[idx1, idx1::ndof+1] += df12
            dfnl[idx2, idx1::ndof+1] -= df12
            dfnl[idx1, idx2::ndof+1] -= df12
            dfnl[idx2, idx2::ndof+1] += df12
        return dfnl


def spline_interp(x, a, xv):
    """
    
    Parameters
    ----------
    x: ndarray
        Spline knot (x)-coordinate
    a: ndarray [(len(x),4)]
        Coefficients for each cubic spline
    xv: float
        Current point
    
    Returns
    -------
    yv: float
        Ordinate(y) for point xv
    yvp: float
        Derivative of y for point xv
    """


    # Point is smaller than the first spline knot
    if xv < x[0]:
        x0 = x[0]
        a0 = a[0,1]
        a1 = a[0,2]
        a2 = a[0,3]
        a3 = a[0,4]
        y0 = a0 + a1 * x0 + a2 * x0**2 + a3 * x0**3
        s = a1 + 2 * a2 * x0 + 3 * a3 * x0**2
        p = y0 - s * x0
        yv = s * xv + p
        yvp = s
        return yv, yvp

    # Point is larger than the last spline knot
    if xv > x[-1]:
        x0 = x[-1]
        a0 = a[-1,0]
        a1 = a[-1,1]
        a2 = a[-1,2]
        a3 = a[-1,3]
        y0 = a0 + a1 * x0 + a2 * x0**2 + a3 * x0**3
        s = a1 + 2 * a2 * x0 + 3 * a3 * x0**2
        p = y0 - s * x0
        yv = s * xv + p
        yvp = s
        return yv, yvp

    # Find the segment the point is in
    iseg, = np.where(x <= xv)
    iseg = iseg[-1]
    aa = a[iseq]
    a0 = aa[0]
    a1 = aa[1]
    a2 = aa[2]
    a3 = aa[3]

    yv = a0 + a1 * xv + a2 * xv**2 + a3 * xv**3
    yvp = a1 + 2 * a2 * xv + 3 * a3 * xv**2

    return yv, yvp

def piecewise_linear(x, y, s, delta=[], xv=[]):
    """Interpolate piecewise segments.

    Parameters
    ----------
    x: ndarray
        x-coordinate for knots
    y: ndarray
        y-coordinate for knots
    s: ndarray. [len(x)+1]
        Slopes for line segments. Len: 'number of knots' + 1
    delta: ndarray
        ??
    xv: ndarray
        x-coordinates for points to be interpolated

    Return
    ------
    yv: ndarray
        Interpolated values
    """

    n = len(x)
    nv = len(xv)

    # Find out which segments, the xv points are located in.
    indv = np.outer(x[:,None],np.ones(nv)) - \
           np.outer(np.ones((n,)),xv)
    indv = np.floor((n - sum(np.sign(indv),0)) / 2)

    yv = np.zeros(nv)
    for i in range(1,n+1):
        ind = np.where(indv == i)
        yv[ind] = s[i] * xv[ind] + y[i-1] - s[i] * x[i-1]

    ind = np.where(indv == 0)
    yv[ind] = s[0] * xv[ind] + y[0] - s[0] * x[0]

    if any(delta) is False:
        return yv

    for i in range(n):
        dd = delta[i]
        indv = np.where(abs(xv - x[i]) <= dd)

        xa = x[i] - dd
        sa = s[i]
        sb = s[i+1]
        ya = y[i] - sa * dd
        yb = y[i] + sb * dd

        t = (xv[indv] - xa) / 2/dd
        h00 = 2*t**3 - 3*t**2 + 1
        h10 = t**3 - 2*t**2 + t
        h01 = - 2*t**3 + 3*t**2
        h11 = t**3 - t**2
        yv[indv] = h00*ya + 2*h10*dd*sa + h01*yb + 2*h11*dd*sb
    return yv

def piecewise_linear_der(x, y, s, delta=[], xv=[]):
    n = len(x)
    nv = len(xv)

    indv = np.outer(x[:,None],np.ones(nv)) - \
           np.outer(np.ones((n,)),xv)
    indv = np.floor((n - sum(np.sign(indv),0)) / 2)

    yvd = np.zeros(nv)
    for i in range(0,n+1):
        ind = np.argwhere(indv == i)
        yvd[ind] = s[i]

    if any(delta) is False:
        return yvd

    for i in range(n):
        dd = delta[i]
        indv = np.argwhere(abs(xv - x[i]) <= dd)

        xa = x[i] - dd
        sa = s[i]
        sb = s[i+1]
        ya = y[i] - sa * dd
        yb = y[i] + sb * dd

        t = (xv[indv] - xa) / 2/dd
        dh00 = 6*t**2 - 6*t
        dh10 = 3*t**2 - 4*t + 1
        dh01 = -6*t**2 + 6*t
        dh11 = 3*t**2 - 2*t
        yvd[indv] = (dh00*ya + 2*dh10*dd*sa + dh01*yb + 2*dh11*dd*sb) \
            / 2/dd

    return yvd
    
def piecewise_linear_y(x, s):

    n = len(x)
    y = np.zeros(n)


    if n == 1: 
        if x > 0:
            dx = x
            dy = s[0] * dx
        else:
            dx = abs(x)
            dy = - s[1] * dx
        y = y + dy
        return y

    for i in range(1,n): 
        y[i] = y[i-1] + s[i] * (x[i] - x[i-1])


    if x[0] > 0:
        dy = s[0] * x[0]
    else:
        ind = np.argwhere(x < 0)
        ind = ind[-1]
        dy = s[ind + 1] * x[ind] - y[ind]

    y = y + dy

    return y
    
from scipy.linalg import solve
def piecewise_cubic_spline(x):
    """
    
    Let q be divided into L
    segments of arbitrary length and defined by their abscissas,
    denoted by qk for k=1,...,L+1.

    qk:
        Abcissa(x-value) of knot k
    gk:
        Ordinate(y-value) of knot k
    q:
        Displacement value between knot k and k+1
        
    Returns
    -------
    g:
        Interpolated value for displacement q
    """

    pass

    # calculate first derivatives gm from linear constraints
    # Force the cubic spline and its first two derivatives to be continuous across each of the interior knots (L-1 linear constraints)
    # A = np.zeros((L+1,L+1))
    # b = np.zeros(L+1)
    # for k in range(1,L):
    #     A[k-1,k-1] = 1/(q[k]-q[k-1])
    #     A[k-1,k] = 2*(1/(q[k]-q[k-1]) + 1/(q[k+1]-q[k]))
    #     A[k-1,k+1] = 1/(q[k+1]-q[k])
        
    #     b[k-1] = 3*((g[k]-g[k-1)/(q[k]-q[k-1])**2 +
    #               (g[k+1]-g[k)/(q[k+1]-q[k])**2)
    
    # # Since the essentially non-linear restoring force g is zero and
    # # has zero slope at equilibrium, one should also enforce, in the
    # # segment containing the abscissa of the equilibrium point, that ..
    # k, = np.where(q<0)
    # k = k[-1]
    # t0 = -q[k]/(q[k+1]-q[k])
    # A[L-1,k] = (t0**3-2*t0**2+t0) * (q[k+1]-q[k]) # *gm[k]
    # A[L-1,k+1] = (t0**3-t0**2) * (q[k+1]-q[k])  # *gm[k+1]
    # b[L-1] = -(2*t0**3-3*t0**2+1)*q[k] - (-2*t0**3+3*t0**2)*g[k+1]
    
    # A[L,k] = (3*t0**3-4*t0+1) * (q[k+1]-q[k])  # *gm[k]
    # A[L,k+1] = (3*t0**2-2*t0) * (q[k+1]-q[k])  # *gm[k+1]
    # b[L] = 6*(t0-t0**2) * (g[k]-g[k+1])
    
    # gm = solve(A,b)


    # # Normalized displacement
    # t = (x-q[k]) / (q[k+1] + q[k])
    
    # # interpolation
    # gv = (2*t**3-3*t**2+1) * g[k] + (-2*t**3+3*t**2) * g[k+1] + \
    #      (t**3-2*t**2+t) * (q[k+1]-q[k]) * gm[k] + \
    #      (t**3-t**2) * (q[k+1]-q[k]) * gm[k]

    # return gv
