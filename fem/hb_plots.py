#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
#from hb import hb_signal

def hb_signal(omega, t, c, phi):
    n = c.shape[0]
    NH = c.shape[1]-1
    nt = len(t)
    tt = np.arange(1,NH+1)[:,None] * omega * t

    x = np.zeros((n, nt))
    for i in range(n):

        tmp = tt + np.outer(phi[i,1:],np.ones(nt))
        tmp = c[i,0]*np.ones(nt) + c[i,1:] @ np.sin(tmp)
        x[i] = tmp #np.sum(tmp, axis=0)

    return x

def anim_init(omegas, amps):

    dof = 0
    xx = np.asarray(omegas)/scale_t
    yy = np.asarray(amps).T[dof]

    fig = plt.figure(5)
    fig.clf()
    ax = fig.add_subplot(111)
    #ax.clear()
    #fig, ax = plt.subplots(1, 1)
    #ax.hold(True)
    ax.set_title('Nonlinear FRF')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Amplitude (m)')
    ax.set_xlim((25,omega_cont_max/2/np.pi))
    ax.set_ylim((0,np.max(yy)*1.5))
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
    fig.canvas.draw()

    # cache the background
    background = fig.canvas.copy_from_bbox(ax.bbox)

    points = ax.plot(xx/2/np.pi, yy, '-')[0]
    cur_point = ax.plot(xx[-1]/2/np.pi, yy[-1], 'o')[0]
    return (fig, ax, background, points, cur_point)

def anim_update(x,y,  fig, ax, background, points, cur_point): #*kwargs):

    dof = 0
    xx = np.asarray(x)/scale_t
    yy = np.asarray(y).T[dof]
    # print(xx/2/np.pi)
    # fig, ax, background, points = kwargs
    points.set_data(xx/2/np.pi, yy)
    cur_point.set_data(xx[-1]/2/np.pi, yy[-1])

    # recompute the ax.dataLim
    #ax.relim()
    # update ax.viewLim using the new dataLim
    #ax.autoscale_view(scalex=False)
    #ylim = ax.get_ylim()

    ax.set_ylim((0,max(yy)*1.1))

    # restore background
    fig.canvas.restore_region(background)
    # redraw just the points
    ax.draw_artist(points)
    ax.draw_artist(cur_point)
    # fill in the axes rectangle
    fig.canvas.blit(ax.bbox)


def plots(t, omega, cnorm, c, cd, cdd, dof, B, inl, gtype='displ', savefig=False):

    fig_periodic,_ = periodic(t, omega, c, dof, gtype)
    fig_harmonic,_ = harmonic(cnorm, dof)
    fig_phase,_ = phase(t, omega, c, cd, dof)
    fig_stab,_ = stab(B, dof, gtype='exp')

    abspath='/home/paw/ownCloud/speciale/code/python/vib/'
    relpath = 'plots/hb/'
    path = abspath + relpath
    if inl.size != 0:
        str1 = 'nonlin' + str(len(inl))
    else:
        str1 = 'lin'
    if savefig:
        fig_periodic.savefig(path + 'periodic_' +str1 + '.png')
        fig_harmonic.savefig(path + 'har_' +str1 + '.png')
        fig_phase.savefig(path + 'phase_' +str1 + '.png')
        fig_stab.savefig(path + 'stab_' +str1 + '.png')


def periodic(t, omega, c, dof,gtype='displ'):
    if gtype == 'displ':
        #y = hb_signal(c, phi, omega, t)
        ystr = 'Displacement (m)'
    elif gtype == 'vel':
        #y = hb_signal(cd, phid, omega, t)
        ystr = 'Velocity (m/s)'
    else:
        #y = hb_signal(cdd, phidd, omega, t)
        ystr = 'Acceleration (m/s²)'
    y = hb_signal(omega, t, *c)

    fig = plt.figure(1)
    fig.clf()
    ax = fig.add_subplot(111)
    #fig, ax = plt.subplots()

    ax.plot(t,y[dof],'-')
    ax.axhline(y=0, ls='--', lw='0.5',color='k')
    ax.set_title('Displacement vs time, dof: {}'.format(dof))
    ax.set_xlabel('Time (t)')
    ax.set_ylabel(ystr)
    # use sci format on y axis when figures are out of the [0.01, 99] bounds
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))

    return fig, ax

def harmonic(cnorm, dof):

    nh = cnorm.shape[1] -1

    fig = plt.figure(2)
    fig_har = fig
    fig.clf()
    ax = fig.add_subplot(111)
    ax.clear()
    ax.bar(np.arange(nh+1), cnorm[dof])
    ax.set_title('Displacement harmonic component, dof: {}'.format(dof))
    ax.set_xlabel('Order')
    # use double curly braces to "escape" literal curly braces...
    ax.set_ylabel(r'$C_{{{dof}-h}}$'.format(dof=dof))
    ax.set_xlim([-0.5, nh+0.5])
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))

    return fig, ax


def phase(t, omega, c, cd, dof):
    x = hb_signal(omega, t, *c)
    xd = hb_signal(omega, t, *cd)

    fig  =plt.figure(3)
    fig.clf()
    ax = fig.add_subplot(111)
    ax.plot(x[dof],xd[dof])
    ax.set_title('Phase space, dof: {}'.format(dof))
    ax.set_xlabel('Displacement (m)')
    ax.set_ylabel('Velocity (m/s)')
    ax.ticklabel_format(axis='both', style='sci', scilimits=(-2,2))
    # ax.axis('equal')

    return fig, ax

def conf_space():
    # fig = plt.figure(4)
    # ax = fig.add_subplot(111)
    # ax.clear()
    # ax.plot(x,xd)
    # ax.set_title('Configuration space, dof: {}'.format(dof))
    # ax.set_xlabel('Displacement x₁ (m)')
    # ax.set_ylabel('Displacement x₂ (m)')
    pass


def stab(B, dof, gtype='exp'):
    fig  =plt.figure(5)
    fig_stab = fig
    fig.clf()
    ax = fig.add_subplot(111)

    lamb = B
    if gtype == 'multipliers':
        str1 = 'Floquet multipliers'
        str2 = '$\sigma$'
        T = 1/f0
        sigma = np.exp(lamb*T)
        xx = np.real(sigma)
        yy = np.imag(sigma)
        idx_s = np.where(np.abs(sigma) <= 1)
        idx_u = np.where(np.abs(sigma) > 1)

        circ = plt.Circle((0, 0), radius=1, ec='k', fc='None', ls='-')
        ax.add_patch(circ)

    else:
        str1 = 'Floquet exponent'
        str2 = '$\lambda$'
        xx = np.real(lamb)
        yy = np.imag(lamb)
        idx_s = np.where(xx <= 0)
        idx_u = np.where(xx > 0)

        ax.axhline(y=0, color='k')
        ax.axvline(x=0, color='k')

    if len(idx_u[0]) != 0:
        ax.plot(xx[idx_u],yy[idx_u],'ro', label='unstable')
    ax.plot(xx[idx_s],yy[idx_s],'bx', label='stable')
    ax.set_title('Stability ({}), dof: {}'.format(str1, dof))
    ax.set_xlabel(r'Real({})'.format(str2))
    ax.set_ylabel(r'Imag({})'.format(str2))
    ax.legend()

    xmax = max(np.max(np.abs(xx))*1.1, 1.1)
    ymax = max(np.max(np.abs(yy))*1.1, 1.1)
    ax.set_xlim(xmax * np.array([-1,1]))
    ax.set_ylim(ymax * np.array([-1,1]))
    ax.grid(True, which='both')

    return fig, ax
