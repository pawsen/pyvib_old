#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from ..common import db

class scaler():
    def __init__(self, scale):
        self.scale = scale

    def ratio(self):
        return self.scale
    def label(self):
        if self.scale == 1:
            xstr = '(Hz)'
        else:
            xstr = '(rad/s)'
        return xstr


def fig_ax_getter(fig=None, ax=None):
    if fig is None and ax is None:
        fig, ax = plt.subplots()
    elif fig is None:
        fig = ax.get_figure()
    elif ax is None:
        ax = fig.gca()
    return fig, ax

def plot_knl(fnsi, sca=1):

    fs = fnsi.fs
    nsper = fnsi.nsper
    flines = fnsi.flines

    freq = np.arange(0,nsper)*fs/nsper * sca
    freq_plot = freq[flines]

    if sca == 1:
        xstr = '(Hz)'
    else:
        xstr = '(rad/s)'

    figs = []
    axs = []
    for nl in fnsi.nonlin.nls:
        knl = nl.knl
        try:
            enl = nl.enl
        except:
            enl = np.ones(knl.shape)

        for i in range(knl.shape[0]):
            mu = knl[i]

            mu_mean = np.zeros(2)
            mu_mean[0] = np.mean(np.real(mu))
            mu_mean[1] = np.mean(np.imag(mu))
            # ratio of 1, is a factor of 10. 2 is a factor of 100, etc
            ratio = np.log10(np.abs(mu_mean[0]/mu_mean[1]))
            exponent = enl[i]
            print('exp: {:d}\n ℝ(mu) {:e}\n 𝕀(mu)  {:e}'.format(exponent,
                                                                *mu_mean))
            print(' Ratio log₁₀(ℝ(mu)/𝕀(mu)) {:0.2f}'.format(ratio))

            fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
            ax1.set_title('Exponent: {:d}. Estimated: {:0.3e}'.
                          format(exponent, mu_mean[0]))
            ax1.plot(freq_plot, np.real(mu),label='fnsi')
            ax1.axhline(mu_mean[0], c='k', ls='--', label='mean')
            ax1.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
            ax1.set_xlabel('Frequency ' + xstr)
            ax1.legend()

            str1 = ''
            ymin = np.min(np.real(mu))
            ymax = np.max(np.real(mu))
            if np.abs(ymax - ymin) <= 1e-6:
                ymin = 0.9 * mu_mean[0]
                ymax = 1.1 * mu_mean[0]
                ax1.set_ylim([ymin, ymax])
                str1 = ' 1%'
            ax1.set_ylabel(r'Real($\mu$) $(N/m^{:d})${:s}'.format(exponent, str1))

            ax2.plot(freq_plot, np.imag(mu))
            ax2.axhline(mu_mean[1], c='k', ls='--', label='mean')
            ax2.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
            ax2.set_xlabel('Frequency ' + xstr)
            str1 = ''
            ymin = np.min(np.imag(mu))
            ymax = np.max(np.imag(mu))
            if np.abs(ymax - ymin) <= 1e-6:
                ymin = 0.9 * mu_mean[1]
                ymax = 1.1 * mu_mean[1]
                ax2.set_ylim([ymin, ymax])
                str1 = ' 1%'
            ax2.set_ylabel(r'Imag($\mu$) $(N/m^{:d})$'.format(exponent))
            fig.tight_layout()
            figs.append(fig)
            axs.append([ax1, ax2])

    return figs, axs

def plot_modes(idof, sr, sca=1, fig=None, ax=None, **kwargs):
    fig, ax = fig_ax_getter(fig, ax)
    if sca == 1:
        xstr = '(Hz)'
    else:
        xstr = '(rad/s)'

    ax.set_title('Linear modes')
    ax.set_xlabel('Node id')
    ax.set_ylabel('Displacement (m)')
    # display max 8 modes
    nmodes = min(len(sr['wd']), 8)
    for i in range(nmodes):
        natfreq = sr['wn'][i]
        ax.plot(idof, sr['realmode'][i],'-*', label='{:0.2f} {:s}'.
                format(natfreq*sca, xstr))
    ax.axhline(y=0, ls='--', lw='0.5',color='k')
    ax.autoscale(enable=True, axis='x', tight=True)
    ax.legend()

    return fig, ax

def plot_frf(freq, H, dofs=0, sca=1, fig=None, ax=None, *args, **kwargs):
    fig, ax = fig_ax_getter(fig, ax)

    # If h is only calculated for one dof:
    if H.shape[0] == 1:
        dofs = [0]

    if not isinstance(dofs, list):
        dofs = [dofs]

    for dof in dofs:
        ax.plot(freq*sca, db(np.abs(H[dof])), *args, **kwargs)

    if ax is None:
        ax.set_title('Nonparametric linear FRF')
    if sca == 1:
        xstr = '(Hz)'
    else:
        xstr = '(rad/s)'
    ax.set_xlabel('Frequency ' + xstr)

    # For linear scale: 'Amplitude (m/N)'
    ax.set_ylabel('Amplitude (dB)')
    return fig, ax


def plot_svg(Sn, fig=None, ax=None, **kwargs):
    """Plot singular values of Sn. Alternative to stabilization diagram"""

    fig, ax = fig_ax_getter(fig, ax)
    ax.semilogy(Sn/np.sum(Sn),'sk', markersize=6)
    ax.set_xlabel('Model order')
    ax.set_ylabel('Normalized magnitude')
    return fig, ax


def plot_stab(sd, nlist, fmin=None, fmax=None, sca=1, fig=None, ax=None):
    fig, ax = fig_ax_getter(fig, ax)

    orUNS = []    # Unstabilised model order
    freqUNS = []  # Unstabilised frequency for current model order
    orSfreq = []  # stabilized model order, frequency
    freqS = []    # stabilized frequency for current model order
    orSep = []    # stabilized model order, damping
    freqSep = []  # stabilized damping for current model order
    orSmode = []  # stabilized model order, mode
    freqSmode = []# stabilized damping for current model order
    orSfull = []  # full stabilized
    freqSfull = []
    for k, v in sd.items():
        # Short notation for the explicit for-loop
        # values = zip(v.values())
        # for freq, ep, mode, stab in zip(*values):
        for freq, ep, mode, stab in zip(v['freq'], v['zeta'],
                                        v['mode'], v['stab']):
            freq = freq*sca
            if stab:
                if ep and mode:
                    orSfull.append(k)
                    freqSfull.append(freq)
                elif ep:
                    orSep.append(k)
                    freqSep.append(k)
                elif mode:
                    orSmode.append(k)
                    freqSmode.append(freq)
                else:
                    orSfreq.append(k)
                    freqS.append(freq)
            else:
                orUNS.append(k)
                freqUNS.append(freq)

    # Avoid showing the labels of empty plots
    if len(freqUNS) != 0:
        ax.plot(freqUNS, orUNS, 'xr', ms=7, label='Unstabilized')
    if len(freqS) != 0:
        ax.plot(freqS, orSfreq, '*k', ms=7,
                label='Stabilized in natural frequency')
    if len(freqSep) != 0:
        ax.plot(freqSep, orSep, 'sk', ms=7, mfc='none',
                label='Extra stabilized in damping ratio')
    if len(freqSmode) != 0:
        ax.plot(freqSmode, orSmode, 'ok', ms=7, mfc='none',
                label='Extra stabilized in MAC')
    if len(freqSfull) != 0:
        ax.plot(freqSfull, orSfull, '^k', ms=7, mfc='none',
                label='Full stabilization')

    if fmin is not None:
        ax.set_xlim(left=fmin*sca)
    if fmax is not None:
        ax.set_xlim(right=fmax*sca)

    ax.set_ylim([nlist[0]-2, nlist[-1]])
    step = round(nlist[-2]/5)
    major_ticks = np.arange(0, nlist[-2]+1, step)
    ax.set_yticks(major_ticks)

    if sca == 1:
        xstr = '(Hz)'
    else:
        xstr = '(rad/s)'
    ax.set_xlabel('Frequency ' + xstr)
    ax.set_ylabel('Model order')
    ax.set_title('Stabilization diagram')
    ax.legend(loc='lower right')

    return fig, ax
    #plt.show()

