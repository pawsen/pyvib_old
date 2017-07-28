 # -*- coding: utf-8 -*-
"""
femutil.py
----------

Functions to compute kinematics variables for the Finite
Element Analysis.

The elements included are:
    1. 4 node bilinear quadrilateral.
    2. 6 node quadratic triangle.
    3. 3 node linear triangle.

The notation used is similar to the one used by Bathe [1]_.


References
----------
.. [1] Bathe, Klaus-Jürgen. Finite element procedures. Prentice Hall,
   Pearson Education, 2006.

"""
from gauss_int import gauss_int
import numpy as np


class Triang3(object):
    # def __init__(self):
    #     self.name = '3 node linear triangle.'
    #     self.ndof = 6
    #     self.nnodes = 3
    #     self.ngauss = 3

    def N( x, y):
        N = np.zeros((2, 6))
        H = np.array([
            (1 - x - y),
            x,
            y])
        N[0, ::2] = H
        N[1, 1::2] = H
        return N

    def dhdx(r ,s):
        dhdx = np.array([
            [-1, 1, 0],
            [-1, 0, 1]])
        return dhdx

class Quad4():
    def dhdx(r, s):
        dhdx = 0.25*np.array([
            [s - 1, -s + 1, s + 1, -s - 1],
            [r - 1, -r - 1, r + 1, -r + 1]])
        return dhdx

    def N(x, y):
        N = np.zeros((2, 8))
        H = 0.25*np.array(
            [(1 - x)*(1 - y),
             (1 + x)*(1 - y),
             (1 + x)*(1 + y),
             (1 - x)*(1 + y)])
        N[0, ::2] = H
        N[1, 1::2] = H
        return N


class Triag6():
    def dhdx(r,s):
        dhdx = np.array([
            [4*r + 4*s - 3, 4*r - 1, 0, -8*r - 4*s + 4, 4*s,  -4*s],
            [4*r + 4*s - 3, 0, 4*s - 1,  -4*r, 4*r, -4*r - 8*s + 4]])
        return dhdx

    def N(x,y):
        # Shape functions for a 6-noded triangular element
        N = np.zeros((2, 12))
        H = np.array(
            [(1 - x - y) - 2*x*(1 - x - y) - 2*y*(1 - x - y),
             x - 2*x*(1 - x - y) - 2*x*y,
             y - 2*x*y - 2*y*(1-x-y),
             4*x*(1 - x - y),
             4*x*y,
             4*y*(1 - x - y)])
        N[0, ::2] = H
        N[1, 1::2] = H
        return N

elem = {
    2 :{
        'name' : '3 node linear triangle',
        'ndof' : 6,
        'nnodes' : 3,
        'ngauss' : 3,
        'N' : Triang3.N,
        'dhdx' : Triang3.dhdx
    },
    3 :{
        'name' : '4 node bilinear quadrilateral.',
        'ndof' : 8,
        'nnodes' : 4,
        'ngauss' : 4,
        'N' : Quad4.N,
        'dhdx' : Quad4.dhdx
    },
    9 :{
        'name' : '6 node quadratic triangle.',
        'ndof' : 12,
        'nnodes' : 6,
        'ngauss' : 7,
        'N' : Triag6.N,
        'dhdx' : Triag6.dhdx
    }
}


def elem_strain(idx, coord, ul):
    """Compute the strains at each element integration point

    Parameters
    ----------
    coord : ndarray
      Coordinates of the nodes of the element
    ul : ndarray
      Array with displacements for the element.

    Returns
    -------
    epsGT : ndarray
      Strain components for the Gauss points.
    xl : ndarray
      Configuration of the Gauss points after deformation.

    """
    nn = elem[idx]['nnodes']
    ngauss = elem[idx]['ngauss']
    epsl = np.zeros([3])
    epsG = np.zeros([3, ngauss])
    xl = np.zeros([ngauss, 2])
    XW, XP = gauss_int(ngauss)
    for i in range(ngauss):
        ri = XP[i, 0]
        si = XP[i, 1]
        ddet, B = strain_disp(idx, ri, si, coord)
        epsl = np.dot(B, ul)
        epsG[:, i] = epsl[:]
        N = elem[idx]['N'](ri, si)
        xl[i, 0] = sum(N[0, 2*i]*coord[i, 0] for i in range(nn))
        xl[i, 1] = sum(N[0, 2*i]*coord[i, 1] for i in range(nn))
    return epsG.T, xl


def strain_disp(idx, r ,s, coord):
    """Strain-displacement interpolator B for a element and determinant for the
    jacobi matrix

    Parameters
    ----------
    r : float
      r component in the natural space.
    s : float
      s component in the natural space.
    coord : ndarray
      Coordinates of the nodes of the element
    (4, 2) for a 4-noded quad element
    (3, 2) for a 3-noded triang element
    (6, 2) for a 6-noded triang element

    Returns
    -------
    ddet : float
      Determinant evaluated at `(r, s)`.
    B : ndarray
      Strain-displacement interpolator evaluated at `(r, s)`.
    """

    nn = elem[idx]['nnodes']
    B = np.zeros((3, 2*nn))
    dhdx = elem[idx]['dhdx'](r, s)
    det, jaco_inv = jacoper(dhdx, coord)
    dhdx = np.dot(jaco_inv, dhdx)
    B[0, ::2] = dhdx[0, :]
    B[1, 1::2] = dhdx[1, :]
    B[2, ::2] = dhdx[1, :]
    B[2, 1::2] = dhdx[0, :]
    return det, B


def jacoper(dhdx, coord):
    """
    Parameters
    ----------
    dhdx : ndarray
      Derivatives of the interpolation function with respect to the
      natural coordinates.
    coord : ndarray
      Coordinates of the nodes of the element (nn, 2).

    Returns
    -------
    xja : ndarray (2, 2)
      Jacobian of the transformation evaluated at `(r, s)`.

    """
    jaco = dhdx.dot(coord)
    det = np.linalg.det(jaco)
    jaco_inv = np.linalg.inv(jaco)
    return det, jaco_inv


def ke(idx, coord, thk, nu, E):
    """Local element stiffness

    Parameters
    ----------
    coord : ndarray
      Coordinates for the nodes of the element (4, 2).
    enu : float
      Poisson coefficient (-1, 0.5).
    Emod : float
      Young modulus (>0).

    Returns
    -------
    kl : ndarray
      Local stiffness matrix for the element

    """
    ndof = elem[idx]['ndof']
    ngauss = elem[idx]['ngauss']
    kl = np.zeros([ndof, ndof])
    C = mat_prop(nu, E)
    XW, XP = gauss_int(ngauss)
    for i in range(0, ngauss):
        ri = XP[i, 0]
        si = XP[i, 1]
        xw = XW[i]
        ddet, B = strain_disp(idx, ri, si, coord)
        kl = kl + thk * np.dot(np.dot(B.T, C), B) * xw*ddet
    return kl


def me(idx, coord, thk, rho):
    """Local element mass matrix
    """
    ndof = elem[idx]['ndof']
    ngauss = elem[idx]['ngauss']
    me = np.zeros([ndof, ndof])
    XW, XP = gauss_int(ngauss)
    for i in range(0, ngauss):
        ri = XP[i, 0]
        si = XP[i, 1]
        xw = XW[i]
        dhdx = elem[idx]['dhdx'](ri, si)
        ddet, jaco_inv = jacoper(dhdx, coord)
        N = elem[idx]['N'](ri, si)

        me = me + thk * np.dot(N.T, N) * rho * xw*ddet
    return me


def mat_prop(nu, E):
    """2D Elasticity consitutive matrix in plane stress

    For plane strain use effective properties.

    Parameters
    ----------
    nu : float
      Poisson coefficient (-1, 0.5).
    E : float
      Young modulus (>0).

    Returns
    -------
    C : ndarray
      Constitutive tensor in Voigt notation.
    """
    C = np.zeros((3, 3))
    enu = E/(1 - nu**2)
    mnu = (1 - nu)/2
    C[0, 0] = enu
    C[0, 1] = nu*enu
    C[1, 0] = C[0, 1]
    C[1, 1] = enu
    C[2, 2] = enu*mnu

    return C
