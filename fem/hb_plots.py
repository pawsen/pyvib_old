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

def hb_components(z, n, NH):
    z = np.hstack([np.zeros(n), z])
    # reorder so first column is zeros, then one column for each dof
    z = np.reshape(z, (n,2*(NH+1)), order='F')

    # first column in z is zero, thus this will a 'division by zero' warning.
    # Instead we just set the first column in phi to pi/2 (arctan(inf) = pi/2)
    # phi = np.arctan(z[:,1::2] / z[:,::2])
    phi = np.empty((n, NH+1))
    phi[:,1:] =np.arctan(z[:,3::2] / z[:,2::2])
    phi[:,0] = np.pi/2

    c = z[:,::2] / np.cos(phi)
    c[:,0] = z[:,1]

    cnorm = np.abs(c) / (np.max(np.abs(c)))

    return c, phi, cnorm

def assemblet(self, omega2):
    npow2 = self.npow2
    return np.arange(2**npow2) / 2**npow2 * 2*np.pi / omega2

class Anim(object):

    def __init__(self, hb, omegas, amps, omega_cont_min, omega_cont_max,
                 dof=0):


        self.dof = dof
        self.scale_t = hb.scale_t
        xx = np.asarray(omegas)/self.scale_t
        yy = np.asarray(amps).T[dof]

        fig = plt.figure(5)
        #fig.clf()
        ax = fig.add_subplot(111)
        ax.clear()
        #fig, ax = plt.subplots(1, 1)
        #ax.hold(True)
        ax.set_title('Nonlinear FRF')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Amplitude (m)')
        ax.set_xlim((omega_cont_min/2/np.pi, omega_cont_max/2/np.pi))
        ax.set_ylim((0,np.max(yy)*1.5))
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
        fig.canvas.draw()

        # cache the background
        self.background = fig.canvas.copy_from_bbox(ax.bbox)
        self.points = ax.plot(xx/2/np.pi, yy, '-')[0]
        self.cur_point = ax.plot(xx[-1]/2/np.pi, yy[-1], 'o')[0]
        self.ax = ax
        self.fig = fig
        # return (fig, ax, background, points, cur_point)

    def update(self, x,y):

        dof = self.dof
        ax = self.ax
        fig = self.fig

        xx = np.asarray(x)/self.scale_t
        yy = np.asarray(y).T[dof]
        self.points.set_data(xx/2/np.pi, yy)
        self.cur_point.set_data(xx[-1]/2/np.pi, yy[-1])

        # recompute the ax.dataLim
        #ax.relim()
        # update ax.viewLim using the new dataLim
        #ax.autoscale_view(scalex=False)
        #ylim = ax.get_ylim()

        ax.set_ylim((0,max(yy)*1.1))

        # restore background
        fig.canvas.restore_region(self.background)
        # redraw just the points
        ax.draw_artist(self.points)
        ax.draw_artist(self.cur_point)
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


def periodic(t, omega, c, dof,gtype='displ', fig = None, ax = None):
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

    if fig is None:
        fig = plt.figure(1)
        fig.clf()
        ax = fig.add_subplot(111)
    #fig, ax = plt.subplots()

    ax.cla()
    ax.plot(t,y[dof],'-')
    ax.axhline(y=0, ls='--', lw='0.5',color='k')
    ax.set_title('Displacement vs time, dof: {}'.format(dof))
    ax.set_xlabel('Time (t)')
    ax.set_ylabel(ystr)
    # use sci format on y axis when figures are out of the [0.01, 99] bounds
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))

    return fig, ax

def harmonic(cnorm, dof, fig=None, ax=None):

    if fig is None:
        fig = plt.figure(2)
        fig.clf()
        ax = fig.add_subplot(111)

    nh = cnorm.shape[1] -1
    ax.cla()
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


class PointBrowser(object):
    """
    Click on a point to select and highlight it -- the data that
    generated the point will be shown in the lower axes.  Use the 'n'
    and 'p' keys to browse through the next and previous points

    https://matplotlib.org/examples/event_handling/data_browser.html
    """

    def __init__(self, omega_vec, z_vec, xamp_vec, dof, hb, fig, ax, line):
        print('0')
        self.hb = hb
        self.fig = fig
        self.ax = ax
        self.omega_vec = omega_vec
        self.z_vec = z_vec
        self.xamp_vec = xamp_vec
        self.dof = dof
        self.line = line

        self.lastind = 0
        self.selected, = self.ax.plot([self.omega_vec[0]], [self.xamp_vec[0]],
                                      'o', ms=12, alpha=0.4, color='yellow',
                                      visible=False)

    def onpress(self, event):
        if self.lastind is None:
            return
        if event.key not in ('n', 'p'):
            return
        if event.key == 'n':
            inc = 1
        else:
            inc = -1

        self.lastind += inc
        self.lastind = np.clip(self.lastind, 0, len(self.omega_vec) - 1)
        self.update()

    def onpick(self, event):
        if event.artist != self.line:
            return True

        N = len(event.ind)
        if not N:
            return True

        # the click locations
        x = event.mouseevent.xdata
        y = event.mouseevent.ydata

        distances = np.hypot(x - self.omega_vec[event.ind],
                             y - self.xamp_vec[event.ind])
        indmin = distances.argmin()
        dataind = event.ind[indmin]

        self.lastind = dataind
        self.update()

    def update(self):
        if self.lastind is None:
            return

        dataind = self.lastind

        dof = self.dof
        n = self.hb.n
        NH = self.hb.NH
        scale_x = self.hb.scale_x
        z_vec = self.z_vec
        omega_vec = self.omega_vec
        xamp_vec = self.xamp_vec
        z = z_vec[dataind]
        omega = omega_vec[dataind]
        c, phi, cnorm = hb_components(z*scale_x, n, NH)

        try:
            if not plt.fignum_exists(fig_hb.number):
                pass
        except NameError:
            fig_hb = plt.figure(2)
            ax1, ax2 = fig_hb.add_subplot(211), fig_hb.add_subplot(212)

        omega2 = omega / self.hb.nu
        t = assemblet(self.hb, omega2)
        c = (c, phi)
        periodic(t, omega, c, dof,fig=fig_hb, ax=ax1)
        harmonic(cnorm, dof, fig, ax2)

        self.selected.set_visible(True)
        self.selected.set_data(omega_vec[dataind], xamp_vec[dataind])

        self.fig.canvas.draw()
        fig_hb.tight_layout()
        fig_hb.canvas.draw()


def nonlin_frf(hb, omega_vec2, z_vec, xamp_vec2, stab_vec2, dof=0):

    scale_t = hb.scale_t
    omega_vec2 = np.asarray(omega_vec)/scale_t / 2 / np.pi
    xamp_vec2 = np.asarray(xamp_vec).T[dof]


    fig, ax = plt.subplots()
    ax.set_title('Nonlinear FRF')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Amplitude (m)')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))

    # picker: 5 points tolerance
    stab_vec2 = np.array(stab_vec)
    idx1 = ~stab_vec2
    idx2 = stab_vec2
    line, line_uns = ax.plot(np.ma.masked_where(idx1, omega_vec2),
                             np.ma.masked_where(idx1, xamp_vec2), '-ok',
                             np.ma.masked_where(idx2, omega_vec2),
                             np.ma.masked_where(idx2, xamp_vec2), '--ok',
                             ms=1, picker=5)

    browser = PointBrowser(omega_vec2, z_vec, xamp_vec2, dof, hb, fig, ax, line)

    fig.canvas.mpl_connect('pick_event', browser.onpick)
    fig.canvas.mpl_connect('key_press_event', browser.onpress)

    plt.show()

    return fig, ax

#nonlin_frf(self,omega_vec, z_vec, xamp_vec)
