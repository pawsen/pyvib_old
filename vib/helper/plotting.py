#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from scipy.linalg import eigvals
from ..hb.hbcommon import hb_signal
from copy import deepcopy


# from functools import partial
# periodic2 = partial(periodic,ptype='displ')

def phase(y, yd, dof=0, fig=None, ax=None, *args, **kwargs):
    if fig is None:
        fig, ax = plt.subplots()
        ax.clear()
    ax.plot(y[dof],yd[dof])#, **kwargs)
    ax.set_title('Phase space, dof: {}'.format(dof))
    ax.set_xlabel('Displacement (m)')
    ax.set_ylabel('Velocity (m/s)')
    ax.ticklabel_format(axis='both', style='sci', scilimits=(-2,2))
    # ax.axis('equal')
    return fig, ax

def periodic(t, y, dof=0, ptype='displ', fig=None, ax=None, *args, **kwargs):
    if ptype == 'displ':
        ystr = 'Displacement (m)'
    elif ptype == 'vel':
        ystr = 'Velocity (m/s)'
    else:
        ystr = 'Acceleration (m/s²)'

    if fig is None:
        fig, ax = plt.subplots()
        ax.clear()

    ax.plot(t,y[dof])#, **kwargs)
    ax.axhline(y=0, ls='--', lw='0.5',color='k')
    ax.set_title('Displacement vs time, dof: {}'.format(dof))
    ax.set_xlabel('Time (t)')
    ax.set_ylabel(ystr)
    # use sci format on y axis when figures are out of the [0.01, 99] bounds
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
    return fig, ax

def harmonic(cnorm, dof=0, fig=None, ax=None, *args, **kwargs):
    if fig is None:
        fig, ax = plt.subplots()
        ax.clear()

    nh = cnorm.shape[1] - 1
    ax.bar(np.arange(nh+1), cnorm[dof])#, *args, **kwargs)
    ax.set_title('Displacement harmonic component, dof: {}'.format(dof))
    ax.set_xlabel('Harmonic index (-)')
    # use double curly braces to "escape" literal curly braces...
    ax.set_ylabel(r'Harmonic coefficient $C_{{{dof}-h}}$'.format(dof=dof))
    ax.set_xlim([-0.5, nh+0.5])
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
    return fig, ax

def stability(lamb, dof=0, T=None, ptype='exp', fig=None, ax=None,
              *args, **kwargs):
    if fig is None:
        fig, ax = plt.subplots()
        ax.clear()

    if ptype == 'multipliers':
        if T is None:
            raise AttributeError('T not provided')
        str1 = 'Floquet multipliers'
        str2 = '$\sigma$'
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
        lamb = np.log(lamb)/T
        xx = np.real(lamb)
        yy = np.imag(lamb)
        idx_s = np.where(xx <= 0)
        idx_u = np.where(xx > 0)

        ax.axhline(y=0, color='k')
        ax.axvline(x=0, color='k')

    if len(idx_u[0]) != 0:
        ax.plot(xx[idx_u],yy[idx_u],'o', label='unstable')#, **kwargs)
    if len(idx_s[0]) != 0:
        ax.plot(xx[idx_s],yy[idx_s],'x', label='stable')#, **kwargs)
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


def configuration(y, fig=None, ax=None, *args, **kwargs):
    if fig is None:
        fig, ax = plt.subplots()
        ax.clear()

    x1 = y[0]
    x2 = y[1]
    ax.plot(x1,x2)
    ax.set_title('Configuration space')
    ax.set_xlabel('Displacement x₁ (m)')
    ax.set_ylabel('Displacement x₂ (m)')
    ax.ticklabel_format(axis='both', style='sci', scilimits=(-2,2))
    ax.axis('equal')
    return fig, ax


class Anim(object):

    def __init__(self, x, y, dof=0, title='', xstr='', ystr='',
                 xmin=None,xmax=None,ymin=None,ymax=None,
                 xscale=1,yscale=1):

        self.dof = dof

        self.xscale = xscale
        self.yscale = yscale
        x = np.asarray(x) * self.xscale
        y = np.asarray(y) * self.yscale

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.clear()
        ax.set_title(title)
        ax.set_xlabel(xstr)
        ax.set_ylabel(ystr)

        scale = 1.5
        if xmin is None:
            xmin = min(x)
            xmin = xmin*scale if xmin < 0 else xmin*(scale-1)
        if xmax is None:
            xmax = max(y)
            xmax = xmax*(scale-1) if xmax < 0 else xmax*scale
        if ymin is None:
            ymin = 0
        if ymax is None:
            ymax = max(y)*1.5
        ax.set_xlim((xmin, xmax))
        ax.set_ylim((ymin,ymax))
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))

        plt.show(False)
        fig.canvas.draw()

        # cache the clean background
        self.background = fig.canvas.copy_from_bbox(ax.bbox)
        self.points = ax.plot(x, y, '-')[0]
        self.cur_point = ax.plot(x[-1], y[-1], 'o')[0]
        self.ax = ax
        self.fig = fig
        self.ims = []

    def update(self, x,y):
        """The purpose of blit'ing is to avoid redrawing the axes, and all the ticks
        and, more importantly, avoid redrawing the text, which is relatively
        expensive.

        One way to do it, could be to initialize the axes with limits 50%
        larger than data. When data exceed limits, then redraw and copy
        background
        """

        dof = self.dof
        ax = self.ax
        fig = self.fig

        x = np.asarray(x) * self.xscale
        y = np.asarray(y) * self.yscale

        axes_update = False
        scale = 1.5
        ax_xmin, ax_xmax = ax.get_xlim()
        xmin, xmax = min(x), max(x)
        ax_ymin, ax_ymax = ax.get_ylim()
        ymin, ymax = min(y), max(y)
        if xmin <= ax_xmin:
            xmin = xmin*scale if xmin < 0 else xmin*(scale-1)
        else: xmin = ax_xmin
        if xmax >= ax_xmax:
            xmax = xmax*(scale-1) if xmax < 0 else xmax*scale
        else: xmax = ax_xmax
        if (xmin < ax_xmin or xmax > ax_xmax):
            ax.set_xlim((xmin, xmax))
            axes_update = True

        if ymin <= ax_ymin:
            ymin = ymin*scale if ymin < 0 else ymin*(scale-1)
        else: ymin = ax_ymin
        if ymax >= ax_ymax:
            ymax = ymax*(scale-1) if ymax < 0 else ymax*scale
        else: ymax = ax_ymax
        if (ymin < ax_ymin or ymax > ax_ymax):
            ax.set_ylim((ymin, ymax))
            axes_update = True

        if axes_update:
            fig.canvas.draw()
            self.background = fig.canvas.copy_from_bbox(ax.bbox)
        else:
            # restore the clean slate background
            fig.canvas.restore_region(self.background)

        self.points.set_data(x, y)
        self.cur_point.set_data(x[-1], y[-1])
        # redraw just the points
        ax.draw_artist(self.points)
        ax.draw_artist(self.cur_point)
        # fill in the axes rectangle
        fig.canvas.blit(ax.bbox)

        #self.ims.append([deepcopy(self.points), deepcopy(self.cur_point)])

    def save(self):
        # Set up formatting for the movie files
        Writer = animation.writers['ffmpeg'] # ['imagemagick'] #
        writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

        #im_ani = animation.ArtistAnimation(self.fig, self.ims, interval=50, repeat_delay=3000,
        #                                   blit=True)
        #im_ani.save('im.mp4', writer=writer)

class PointBrowser(object):
    """
    Click on a point to select and highlight it -- the data that
    generated the point will be shown in the lower axes.  Use the 'n'
    and 'p' keys to browse through the next and previous points

    https://matplotlib.org/examples/event_handling/data_browser.html
    """

    def __init__(self, x, y, dof, plotlist, fig, ax, lines, hb=None, nnm=None):

        if len(plotlist) == 0:
            raise ValueError('MUST: len(plotlist)>0',plotlist)
        x = np.asarray(x)
        y = np.asarray(y)

        if hb is not None:
            self.ptype = 'hb'
        if nnm is not None:
            self.ptype = 'nnm'
        self.nnm = nnm
        self.hb = hb
        self.x = x
        self.y = y
        self.dof = dof
        self.plotlist = plotlist
        self.fig = fig
        self.ax = ax
        self.lines = lines
        self.lastind = 0
        self.lastchange = 0
        self.selected, = self.ax.plot([self.x[0]], [self.y[0]],
                                      'o', ms=12, alpha=0.4, color='yellow',
                                      visible=False)

        self.fig2, (self.ax1, self.ax2) = plt.subplots(2,1)
        self.fig2.show(False)

    def onpress(self, event):
        if self.lastind is None:
            return
        if event.key not in ('n', 'p', 'a', 'z'):
            return
        inc, change = 0, 0
        if event.key == 'n':
            inc = 1
        elif event.key == 'p':
            inc = -1
        elif event.key == 'a':
            change = 1
        elif event.key == 'z':
            change = -1

        self.lastchange += 2*change
        self.lastchange = np.clip(self.lastchange, 0, len(self.plotlist) - 2)
        self.lastind += inc
        self.lastind = np.clip(self.lastind, 0, len(self.x) - 1)
        self.update()

    def onpick(self, event):
        if event.artist not in self.lines:
            return True
        # if event.artist != self.line:
        #     return True

        N = len(event.ind)
        if not N:
            return True

        # the click locations
        x = event.mouseevent.xdata
        y = event.mouseevent.ydata

        distances = np.hypot(x - self.x[event.ind],
                             y - self.y[event.ind])
        indmin = distances.argmin()
        dataind = event.ind[indmin]

        self.lastind = dataind
        self.update()

    def update(self):
        if self.lastind is None:
            return

        plotind = self.lastchange
        dataind = self.lastind

        try:
            if not plt.fignum_exists(self.fig2.number):
                self.fig2, (self.ax1, self.ax2) = plt.subplots(2,1)
        except AttributeError:  # NameError:
            pass
            #plt.show()
        plotdata = {}
        if self.ptype == 'hb':
            omega = self.hb.omega_vec[dataind]
            omega2 = omega / self.hb.nu
            #t = self.hb.assemblet(omega2)
            z = self.hb.z_vec[dataind]
            t, omegap, zp, cnorm, c, cd, cdd = \
                self.hb.get_components(omega, z)
            lamb = self.hb.lamb_vec[dataind]
            x = hb_signal(omega, t, *c)
            xd = hb_signal(omega, t, *cd)
            xdd = hb_signal(omega, t, *cdd)
            T = t[-1]

            plotdata.update({'cnorm':cnorm})

        elif self.ptype == 'nnm':
            X0 = self.nnm.X0_vec[dataind]
            n = len(X0)
            x0 = X0[:n//2]
            xd0 = X0[n//2:n]
            T = 2*np.pi / self.nnm.omega_vec[dataind]
            dt = T / self.nnm.nppp
            # We want one complete period.
            ns = self.nnm.nppp + 1
            t = np.arange(ns)*dt
            fext = np.zeros(ns)
            x, xd, xdd, Phi = self.nnm.newmark.integrate_nl(x0, xd0, dt, fext,
                                                            sensitivity=True)
            lamb = eigvals(Phi)


        plotdata.update({'t': t, 'y':x, 'yd':xd, 'ydd':xdd, 'dof':self.dof,
                         'T':T, 'lamb':lamb})

        self.ax1.clear()
        self.ax2.clear()
        plot0 = self.plotlist[plotind]
        plot1 = self.plotlist[plotind+1]
        # for plot1, plot2 in zip(plotlist[:-1], plotlist[1:]):
        plot0(**plotdata, fig=self.fig2, ax=self.ax1)
        plot1(**plotdata, fig=self.fig2, ax=self.ax2)
        self.selected.set_visible(True)
        self.selected.set_data(self.x[dataind], self.y[dataind])

        self.fig.canvas.draw()
        self.fig2.tight_layout()
        self.fig2.canvas.draw()


def nfrc(dof=0, plotlist=[], hb=None, nnm=None, energy_plot=False,
         interactive=True, xscale=1/2/np.pi, yscale=1,
         xunit='(Hz)',
         fig=None, ax=None, *args, **kwargs):

    if hb is not None:
        ptype = 'hb'
        stab_vec = hb.stab_vec
        x = np.asarray(hb.omega_vec)/hb.scale_t * xscale
        y = np.asarray(hb.xamp_vec).T[dof]
        titlestr = 'Nonlinear FRF for dof {}'.format(dof)
        ystr = 'Amplitude (m)'
        xstr = 'Frequency ' + xunit
    if nnm is not None:
        ptype = 'nnm'
        stab_vec = nnm.stab_vec
        if energy_plot:
            energy = np.asarray(nnm.energy_vec)#.T[dof]
            x = np.log10(energy)
            y = np.asarray(nnm.omega_vec) * xscale  # / 2/np.pi
            titlestr = 'Frequency Energy plot (FEP)'
            xstr = 'Log10(Energy) (J)'
            ystr = 'Frequency ' + xunit
        else:
            x = np.asarray(nnm.omega_vec) * xscale
            y = np.asarray(nnm.xamp_vec).T[dof]
            titlestr = 'Amplitude of dof {}'.format(dof)
            ystr = 'Amplitude (m)'
            xstr = 'Frequency ' + xunit

    if fig is None:
        fig, ax = plt.subplots()
        ax.clear()

    ax.set_title(titlestr)
    ax.set_xlabel(xstr)
    ax.set_ylabel(ystr)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))

    # picker: 5 points tolerance
    stab_vec = np.array(stab_vec)
    idx1 = ~stab_vec
    idx2 = stab_vec
    lines = ax.plot(np.ma.masked_where(idx1, x),
                    np.ma.masked_where(idx1, y), '-k',
                    np.ma.masked_where(idx2, x),
                    np.ma.masked_where(idx2, y), '--k',
                    ms=1, picker=5, **kwargs)

    if interactive:
        browser = PointBrowser(x, y, dof, plotlist, fig, ax, lines, hb=hb,
                               nnm=nnm)

        fig.canvas.mpl_connect('pick_event', browser.onpick)
        fig.canvas.mpl_connect('key_press_event', browser.onpress)

        plt.show()

    return fig, ax


# a = 1
# plotlist = [periodic, phase, stability]

