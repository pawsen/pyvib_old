#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from scipy import io

import numpy as np
from scipy.linalg import solve, svd, norm, lstsq, eigvals
from scipy import linalg
from scipy import sparse

from ..forcing import sineForce, toMDOF

class Bifurcation(object):
    def __init__(self, hb, fdofs, nldofs_ext, marker, stype, max_it_secant=10,
                 tol_sec=1e-12):

        self.max_it_secant = max_it_secant
        self.tol_sec = tol_sec

        self.hb = hb
        self.fdofs = fdofs
        self.nldofs_ext = nldofs_ext
        self.idx = [0]
        self.nbif = 0
        self.isbif = False
        # add description of this type of bifurcation. For plotting
        self.marker = marker
        self.stype = stype

class Fold(Bifurcation):

    def __init__(self,*args, **kwargs):
        kwargs.update({'marker':'s', 'stype':'fold'})
        super().__init__(*args, **kwargs)

    def detect(self, omega, z, A, J_z, B, force, it_cont):
        """Fold bifurcation traces the lotus of the frequency response peaks

        At a fold bifurcation these four conditions are true:
        * h(z,ω) = 0
        * Rank h_z(z,ω) = nz-1
        * h_ω(z,ω) ∉ range h_z(z,ω). Ie:  Rank [h_z,h_ω] = nz
        * There is a parametrization z(σ) and ω(σ), with z(σ₀) = z₀,
          ω(σ₀) = z₀ and d²ω(σ)/d²σ ≠ 0

        Ie fold bifurcations are detected when h_z is singular and Rank
        [h_z,h_ω] = nz. It is however more efficient to detect fold
        bifurcations, when the component of the tangent vector related to ω
        changes sign.
        """

        nz = self.hb.nz
        max_it_secant = self.max_it_secant
        tol_sec = self.tol_sec
        nu = self.hb.nu
        scale_x = self.hb.scale_x
        n = self.hb.n
        f_amp = self.hb.f_amp
        omega_save = omega.copy()
        z_save = z.copy()

        G = J_z
        omega2 = omega/nu
        A = self.hb.assembleA(omega2)

        Gq0 = null_approx(G, 'LP2')
        Gp0 = null_approx(G.T, 'LP2')

        print('-----> Detecting LP bifurcation...')
        H = self.hb.state_sys(z, A, force)
        F_it_NR, M_ext, w = extended_state_sys(H, G, Gp0, Gq0, 'LP2')
        print('Test function = {}'.format(norm(F_it_NR) / norm(z)))

        it = 0
        while (it < max_it_secant*10 and norm(F_it_NR)/norm(z) > tol_sec):

            J_ext = extended_jacobian5_detect(self.hb, omega, z, A, B, M_ext,
                                              w, self.fdofs, self.nldofs_ext, 'LP2')

            dNR, *_ = lstsq(J_ext[:,:nz+1], -F_it_NR)
            z = z + dNR[:nz]
            omega = omega + dNR[nz]

            omega2 = omega/nu
            t = self.hb.assemblet(omega2)
            u, _ = sineForce(f_amp, omega=omega, t=t)
            force = toMDOF(u, n,  self.fdofs)
            A = self.hb.assembleA(omega2)
            J_z = self.hb.hjac(z, A) / scale_x

            G = J_z
            B = G
            Gq0 = null_approx(G, 'LP2')
            Gp0 = null_approx(G.T, 'LP2')
            H = self.hb.state_sys(z, A, force)

            F_it_NR, M_ext, w = extended_state_sys(H, G, Gp0, Gq0, 'LP2')
            print('Test function = {}'.format(norm(F_it_NR)))
            it += 1

        if it >= max_it_secant:
            print('-----> LP bifurcation detection failed')
            self.isbif = False
            return omega_save, z_save
        else:
            print('-----> LP detected.')
            self.idx.append(it_cont)
            self.nbif += 1
            self.isbif = True
            return omega, z


def extended_state_sys(H, G, p0, q0, bif_type):

    return _something(G, p0, q0, bif_type, H)

def test_func(G, p0, q0, bif_type):
    """
    If f=None, then the test_func is returned.
    Else the extended state sys is returned.

    Parameters
    ----------

    """
    q0 = np.zeros(q0.shape)
    q0[0] = 1
    p0 = np.zeros(p0.shape)
    p0[0] = 1

    return _something(G, p0, q0, bif_type)

def _something(G, p0, q0, bif_type, H=None):
    nG = G.shape[1]
    if bif_type == 'LP' or bif_type == 'NS':

        # data = np.diag(G)
        # data.append(p0)
        # data.append(q0)
        # ii = np.arange(0,nG*nG+nG-1,nG+2)
        #M = sparse.coo_matrix((data, (ii,jj)), shape=(nG+1,nG+1))
        M = sparse.coo_matrix((nG+1, nG+1))
        idx = np.diag_indices(nG+1, ndim=2)
        M[idx] = np.diag(G)
        M[nG,:nG] = p0
        M[:nG,nG] = q0

        Mb = np.zeros(nG+1)
        Mb[nG] = 1

        wg = sparse.linalg.spsolve(M.tocsr(), Mb)

    elif bif_type == 'BP' or bif_type == 'LP2':

        M = np.vstack((
            np.hstack((G, p0[:,None])),
            np.hstack((q0, 0))
        ))

        wg = solve(M, np.append(np.zeros(nG),1))

    w = wg[:nG]
    g = wg[nG]
    if H is None:
        return g, q0, p0

    H = np.append(H,g)
    return H, M, w

def null_approx(A, bif_type):
    """Compute the nullspace of A depending on the type of bifurcation"""
    if bif_type == 'LP2' or bif_type == 'BP':
        # compute nullspace from SVD decomposition
        return nullspace(A, atol=5e-3).squeeze()
    else:
        if not np.all(A == np.diag(np.diagonal(A))):
            raise ValueError('The matrix A should be diagonal')

        eig_A = np.diag(A)
        idx = np.argmin(np.abs(eig_A))

        nullvec = np.zeros(A.shape[0])
        nullvec[idx] = 1

        return nullvec

def nullspace(A, atol=1e-13, rtol=0):
    """Compute an approximate basis for the nullspace of A.

    The algorithm used by this function is based on the singular value
    decomposition of `A`.

    Parameters
    ----------
    A : ndarray
        A should be at most 2-D.  A 1-D array with length k will be treated
        as a 2-D with shape (1, k)
    atol : float
        The absolute tolerance for a zero singular value.  Singular values
        smaller than `atol` are considered to be zero.
    rtol : float
        The relative tolerance.  Singular values less than rtol*smax are
        considered to be zero, where smax is the largest singular value.

    If both `atol` and `rtol` are positive, the combined tolerance is the
    maximum of the two; that is::
        tol = max(atol, rtol * smax)
    Singular values smaller than `tol` are considered to be zero.

    Return value
    ------------
    ns : ndarray
        If `A` is an array with shape (m, k), then `ns` will be an array
        with shape (k, n), where n is the estimated dimension of the
        nullspace of `A`.  The columns of `ns` are a basis for the
        nullspace; each element in numpy.dot(A, ns) will be approximately
        zero.
    """

    A = np.atleast_2d(A)
    # V: Unitary matrix having right singular vectors as rows.
    u, s, vh = svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T

    # In case of multiple occurrences of the minimum values, the indices
    # corresponding to the first occurrence is returned.
    idx = np.argmin(np.abs(s))
    eig_sm = np.abs(s[idx])
    if eig_sm > tol:
        print('The approximation of the bifurcation point has to be improved\n'
              'The smallest eigenvalue of the matrix is {:.2e}'.format(eig_sm))

    return ns

def extended_jacobian5_detect(self, omega, z, A, B, M_ext, w, fdofs, nldofs_ext, bif_type):
    n = self.n
    nG = M_ext.shape[1] -1
    nz = self.nz

    J_z_f = self.hjac(z, A)
    J_w = self.hjac_omega(omega, z)
    # jacobian_param2
    btronc = np.zeros(nz)
    btronc[2*n+fdofs-1] = 1
    J_w2 = -btronc
    J_w_f = np.hstack((J_w[:,None], J_w2[:,None]))

    vh = solve(M_ext.T, np.append(np.zeros(nG),1))
    v = vh[:nG]

    J_part = np.zeros(nz)
    # TODO parallel loop
    for i in range(nz):
        if i in nldofs_ext:
            hess = hessian4_detect(self, omega, z, A, B, i, 'z', bif_type)
            J_g_A =- v @ hess @ w
        else:
            J_g_A = 0
        J_part[i] = np.real(J_g_A)

    # TODO Rewrite so we DONT pass 0.
    hess = hessian4_detect(self, omega, z, A, B, 0,'omega', bif_type)
    J_g_p1 = - v @ hess @ w

    hess = hessian4_detect(self, omega, z, A, B, 0, 'f', bif_type)
    J_g_p2 = - v @ hess @ w

    J_part = np.append(J_part, (np.real(J_g_p1), np.real(J_g_p2)))

    J = np.vstack((
        np.hstack((J_z_f, J_w_f)),
        J_part
    ))

    return J


def hessian4_detect(self, omega, z, A, B, idx, gtype, bif_type):
    scale_x = self.scale_x
    nu = self.nu

    eps = 1e-5
    if gtype == 'z':
        z_pert = z.copy()

        if z_pert[idx] != 0:
            eps = eps * abs(z_pert[idx])

        z_pert[idx] = z_pert[idx] + eps
        J_z_pert = self.hjac(z_pert, A) / scale_x

        if bif_type == 'LP2':
            return (J_z_pert - B) / eps
        elif bif_type == 'BP':
            J_w_pert = self.hjac_omega(omega, z_pert)
            dG_dalpha = (np.vstack((np.hstack((J_z_pert, J_w_pert[:,None])),
                                   tangent_prev)) - B) / eps
            return dG_dalpha
        elif bif_type == 'NS':
            pass

    elif gtype == 'omega':

        if omega != 0:
            eps = eps * abs(omega)
        omega_pert = omega + eps

        omega2_pert = omega_pert / nu
        A_pert = self.assembleA(omega2_pert)
        J_z_pert = self.hjac(z, A_pert) / scale_x

        if bif_type == 'LP2':
            return  (J_z_pert - B ) / eps
        elif bif_type == 'BP':
            J_w_pert = self.hjac_omega(omega_pert, z_pert)
            dG_dalpha = np.vstack((np.hstack((J_z_pert, J_w_pert[:,None])),
                                   tangent_prev))
            return dG_dalpha
        elif bif_type == 'NS':
            pass

    elif gtype == 'f':
         Hess = np.zeros(B.shape)
         if bif_type == 'LP2' or bif_type == 'BP':
             return Hess
         elif bif_type == 'LP':
             pass
         elif bif_type == 'NS':
             pass


# def detect_branchpoint(self, J_z, J_w, tangent):
#     """
#     Branch point bifurcations occur when two branches of periodic solution meet.

#     At a branch bifurcation these four conditions are true:
#     * h(z,ω) = 0
#     * Rank h_z(z,ω) = nz-1
#     * h_ω(z,ω) ∈ range h_z(z,ω). Ie:  Rank [h_z,h_ω] = nz-1
#     * Exactly two branches of periodic solutions intersect with two distinct
#       tangents.

#     """

#     G = np.vstack((
#         np.hstack((J_z, J_w[:,None])),
#         tangent))

#     Gq0 = null_approx(G,'BP')
#     Gp0 = null_approx(G.T,'BP')
#     print('-----> Detecting BP bifurcation...')
#     H = state_sys(self, z, A, force)
#     F_it_NR, M_it_NR, w_it_NR = test_func(G, Gp0, Gq0, 'LP2', H)
#     print('Test function = {}'.format(norm(F_it_NR) / norm(z)))

#     it = 1
#     while (it < max_it_secant and norm(F_it_NR)/norm(z) > tol_sec):
#         J_ext = 1
#         dNR = pinv(J_ext[:,:nz+1]) @ -F_it_NR
#         z = z + dNR[:nz]
#         omega = omega + dNR[nz+1]
#         omega2 = omega / nu
#         t = assemblet(self, omega2)
#         force = sineForce(f_amp, omega, t, n, fdofs)

#         J_w = jac_w(self, omega, z)
#         G = np.vstack((
#             np.hstack((A0, J_w[:,None])),
#             tangent))
#         Gq0 = null_approx(G,'BP')
#         Gp0 = null_approx(G.T,'BP')
#         H = state_sys(self, z, A, force)
#         F_it_NR, M_it_NR, w_it_NR = test_func(G, Gp0, Gq0, 'LP2', H)
#         print('Test function = {}'.format(norm(F_it_NR) / norm(z)))

#         it += 1
#     if it >= max_it_secant:
#         print('-----> BP bifurcation detection failed')
#         z = z_save
#         omega = omega_save
#         A = A_save
#     else:
#         print('-----> BP detected. Computing switching direction...')
#         idx_BP.append(it_cont+1)
#         it_BP += 1
#         is_BP = True

#         J_z = jac_sys(self, z_pert, A)
#         J_w = jac_w(self, omega, z)
#         J_BP = np.vstack((
#             np.hstack((J_z, J_w[:,None])),
#             tangent))
#         phi1 = pinv(J_BP) @ np.append(np.zeros(nz),1)
#         phi1 = phi / norm(phi)
#         psi = null_approx(J_z.T,'BP')
#         phi2 = null_approx(J_BP,'BP')
#         beta2 = 1

#         # perturbation
#         # First pertubate z, then omega
#         eps = 1e-8
#         Hess = np.empty((nz,nz+1,nz+1))
#         # TODO make parallel for loop
#         for i in range(nz):
#             z_pert = z
#             zpert[i] = zpert[i] + eps
#             J_z_pert = jac_sys(self, z_pert, A)
#             J_w_pert = jac_w(self, omega, z_pert)
#             Hess[...,i] = np.hstack(((J_z_pert - J_z)/eps,
#                                      J_w_pert - J_w))
#         omega_pert = omega + eps
#         # TODO should be omega2_pert
#         A = assembleA(self, omega_pert/nu)
#         J_z_pert = jac_sys(self, z, A)
#         J_w_pert = jac_w(self, omega_pert, z)
#         Hess[...,nz] = np.hstack(((J_z_pert - J_z)/eps,
#                                  J_w_pert - J_w))

#         multi_prod12 = np.zeros(nz)
#         multi_prod22 = np.zeros(nz)
#         for i in range(nz):
#             for j in range(nz+1):
#                 for k in range(nz+1):
#                     multi_prod12[i] = multi_prod12[i] + \
#                         Hess[i,j,k] * phi[j] * phi2[k]
#                     multi_prod12[i] = multi_prod12[i] + \
#                         Hess[i,j,k] * phi2[j] * phi2[k]

#         c12 = psi @ multi_prod12
#         c22 = psi @ multi_prod22
#         alpha2 = -1/2 * c22/c12 * beta2
#         Y0_dot = alpha2 * phi1 + beta2 * phi2
#         V0_switch = Y0_dot / norm(Y0_dot)
#         branch_switch = True

# def detect_NS():
#     """Detect Neimark-Sacker(NS) bifurcation.

#     At a NS bifurcation, also called torus bifurcation, quasi periodic
#     oscillation emanates. Quasi periodic oscillations contains the forcing
#     frequency ω and at least one other frequency ω₂(the envelope)
#     These two frequencies are incommensurate, ie ω/ω₂ is irrational.

#     NS bifurcations are detected when the bialternate matrix product of B is
#     singular, B_⊙=B⊙I. The bialternate matrix is singular when two of its
#     eigenvalues μ₁+μ₂=0. For example: μ₁,₂=±jγ

#     """
#     z_save = z
#     omega_save = omega
#     A_save = A

#     mat_B = b2_inv @ b1
#     if rmc_permute:
#         pass

#     Atest, right_vec, B_eig_sort = B_reduc(mat_B, n)
#     right_vec_inv = np.inv(right_vec)
#     G = bialtprod( Atest );
#     Gq0 = null_approx(G, 'NS')
#     Gp0 = null_approx(G.T, 'NS')

#     print('-----> Detecting NS bifurcation...')
#     H = state_sys(self, z, A, force)
#     F_it_NR, M_it_NR, w_it_NR = test_func(G, Gp0, Gq0, 'NS', H)
#     print('Test function = {}'.format(norm(F_it_NR) / norm(z)))

#     it = 1
#     while (it < max_it_secant and norm(F_it_NR)/norm(z) > tol_sec):
#         J_ext = 1
#         dNR = pinv(J_ext[:,:nz+1]) @ -F_it_NR
#         z = z + dNR[:nz]
#         omega = omega + dNR[nz+1]
#         omega2 = omega / nu
#         t = assemblet(self, omega2)
#         force = sineForce(f_amp, omega, t, n, fdofs)
#         A = assembleA(self, omega2)
#         A0 = jac_sys(self, z, A) / scale_x
#         # TODO recalculate Delta1
#         A1 = Delta1
#         #b1 = [ A1, A0; - eye( size( A0 ) ), zeros( size( A0 ) ) ];
#         mat_B = b2_inv @ b1
#         if rmc_permute:
#             pass
#         Atest, right_vec, B_eig_sort = B_reduc(mat_B, n)
#         right_vec_inv = np.inv(right_vec)
#         G = bialtprod(Atest)
#         Gq0 = null_approx(G, 'NS')
#         Gp0 = null_approx(G.T, 'NS')
#         H = state_sys(self, z, A, force)
#         F_it_NR, M_it_NR, w_it_NR = test_func(G, Gp0, Gq0, 'NS', H)
#         print('Test function = {}'.format(norm(F_it_NR) / norm(z)))

#         it += 1
#     if it >= max_it_secant:
#         print( '-----> NS bifurcation detection failed')
#         z = z_save
#         omega = omega_save
#         A = A_save
#     else:
#         print('-----> NS detected.')
#         idx_NS.append(it_cont+1)
#         it_NS += 1
#         is_NS = True
