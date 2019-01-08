#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
dnlsys -- a collection of classes and functions for modeling nonlinear
linear state space systems.
"""

from scipy.interpolate import interp1d

class StateSpace()

class Root:
    def draw(self):
        # the delegation chain stops here
        assert not hasattr(super(), 'draw')

class Shape(Root):
    def __init__(self, shapename, **kwds):
        self.shapename = shapename
        super().__init__(**kwds)
    def draw(self):
        print('Drawing.  Setting shape to:', self.shapename)
        super().draw()

class ColoredShape(Shape):
    def __init__(self, color, **kwds):
        self.color = color
        super().__init__(**kwds)
    def draw(self):
        print('Drawing.  Setting color to:', self.color)
        super().draw()

from collections import Counter, OrderedDict

class OrderedCounter(Counter, OrderedDict):
     'Counter that remembers the order elements are first seen'
     def __repr__(self):
         return '%s(%r)' % (self.__class__.__name__,
                            OrderedDict(self))
     def __reduce__(self):
         return self.__class__, (OrderedDict(self),)

oc = OrderedCounter('abracadabra')

# https://github.com/scipy/scipy/blob/master/scipy/signal/ltisys.py
def dnlsim(system, u, t=None, x0=None):
    """Simulate output of a discrete-time nonlinear system.

	Calculate the output and the states of a nonlinear state-space model.
        x(t+1) = A x(t) + B u(t) + E zeta(x(t),u(t))
        y(t)   = C x(t) + D u(t) + F eta(x(t),u(t))
    where zeta and eta are polynomials whose exponents are given in xpowers and
    ypowers, respectively. The maximum degree in one variable (a state or an
    input) in zeta or eta is given in max_nx and max_ny, respectively. The
    initial state is given in x0.

    """

    u = np.asarray(u)

    if u.ndim == 1:
        u = np.atleast_2d(u).T

    if t is None:
        out_samples = len(u)
        stoptime = (out_samples - 1) * system.dt
    else:
        stoptime = t[-1]
        out_samples = int(np.floor(stoptime / system.dt)) + 1

    # Pre-build output arrays
    xout = np.empty((out_samples, system.A.shape[0]))
    yout = np.empty((out_samples, system.C.shape[0]))
    tout = np.linspace(0.0, stoptime, num=out_samples)

    # Check initial condition
    if x0 is None:
        xout[0, :] = np.zeros((system.A.shape[1],))
    else:
        xout[0, :] = np.asarray(x0)

    # Pre-interpolate inputs into the desired time steps
    if t is None:
        u_dt = u
    else:
        if len(u.shape) == 1:
            u = u[:, np.newaxis]

        u_dt_interp = interp1d(t, u.transpose(), copy=False, bounds_error=True)
        u_dt = u_dt_interp(tout).transpose()

    # prepare nonlinear part
    repmat_x = np.ones(xpowers.shape[0])
    repmat_y = np.ones(ypowers.shape[0])
    # Simulate the system
    for i in range(0, out_samples - 1):
        # State equation x(t+1) = A*x(t) + B*u(t) + E*zeta(x(t),u(t))
        zeta_t = np.prod((repmat_x @ np.vstack(xout[i, :],
                                               u_dt[i, :]))**xpowers, axis=1)
        xout[i+1, :] = (np.dot(system.A, xout[i, :]) +
                        np.dot(system.B, u_dt[i, :]) +
                        np.dot(system.E, zeta_t))
        # Output equation y(t) = C*x(t) + D*u(t) + F*eta(x(t),u(t))
        eta_t = np.prod((repmat_x @ np.vstack(xout[i, :],
                                              u_dt[i, :]))**ypowers, axis=1)
        yout[i, :] = (np.dot(system.C, xout[i, :]) +
                      np.dot(system.D, u_dt[i, :]) +
                      np.dot(system.F, eta_t))

    # Last point
    eta_t = np.prod((repmat_x @ np.vstack(xout[out_samples-1, :],
                                          u_dt[out_samples-1, :]))**ypowers, axis=1)
    yout[out_samples-1, :] = (np.dot(system.C, xout[out_samples-1, :]) +
                              np.dot(system.D, u_dt[out_samples-1, :]) +
                              np.dot(system.F, eta_t))

    return tout, yout, xout
