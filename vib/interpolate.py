import numpy as np

def spline(x, a, xv):
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
    aa = a[iseg]
    a0 = aa[0]
    a1 = aa[1]
    a2 = aa[2]
    a3 = aa[3]

    yv = a0 + a1 * xv + a2 * xv**2 + a3 * xv**3
    yvp = a1 + 2 * a2 * xv + 3 * a3 * xv**2

    return yv, yvp

def piecewise_linear(x, y, s, delta=None, xv=[]):
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
        Regularization interval, ie. enforce continuity of the derivative
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

    if delta is None:
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

def piecewise_linear_der(x, y, s, delta=None, xv=[]):
    n = len(x)
    nv = len(xv)

    indv = np.outer(x[:,None],np.ones(nv)) - \
           np.outer(np.ones((n,)),xv)
    indv = np.floor((n - sum(np.sign(indv),0)) / 2)

    yvd = np.zeros(nv)
    for i in range(0,n+1):
        ind = np.argwhere(indv == i)
        yvd[ind] = s[i]

    if delta is None:
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
    from scipy.linalg import solve

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
