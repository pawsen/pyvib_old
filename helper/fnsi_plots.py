#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from common import db

def plot_knl(fnsi, knl):

    inl = fnsi.inl
    enl = fnsi.enl
    fs = fnsi.fs
    nsper = fnsi.nsper
    flines = fnsi.flines

    if inl.size == 0:
        return
    freq = np.arange(0,nsper)*fs/nsper
    freq_plot = freq[flines + 1]  # Hz

    # machine precision for float 64. See also
    # https://stackoverflow.com/a/25155518 for an interesting way of doing it.
    eps = np.finfo(float).eps

    figs = []
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
        ax1.set_title('Exponent: {:d}. Estimated: {:0.3e}'.format(exponent,mu_mean[0]))
        ax1.plot(freq_plot, np.real(mu),label='fnsi')
        ax1.axhline(mu_mean[0], c='k', ls='--', label='mean')
        ax1.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
        ax1.set_xlabel('Frequency (Hz)')
        ax1.legend()

        str1 = ''
        ymin = np.min(np.real(mu))
        ymax = np.max(np.real(mu))
        if np.abs(ymax - ymin) <= 1e-6:
            ymin = 0.9 * mu_mean[0]
            ymax = 1.1 * mu_mean[0]
            ax1.set_ylim([ymin-eps, ymax+eps])
            str1 = ' 1%'
        ax1.set_ylabel(r'Real($\mu$) $(N/m^{:d})${:s}'.format(exponent, str1))

        ax2.plot(freq_plot, np.imag(mu))
        ax2.axhline(mu_mean[1], c='k', ls='--', label='mean')
        ax2.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
        ax2.set_xlabel('Frequency (Hz)')
        str1 = ''
        ymin = np.min(np.imag(mu))
        ymax = np.max(np.imag(mu))
        if np.abs(ymax - ymin) <= 1e-6:
            ymin = 0.9 * mu_mean[1]
            ymax = 1.1 * mu_mean[1]
            ax2.set_ylim([ymin-eps, ymax+eps])
            str1 = ' 1%'
        ax2.set_ylabel(r'Imag($\mu$) $(N/m^{:d})$'.format(exponent))
        fig.tight_layout()
        figs.append(fig)

    return figs

def plot_modes(idof, sd):

    fig = plt.figure()
    plt.clf()
    plt.title('Linear modes')
    plt.xlabel('Node id')
    plt.ylabel('Displacement (m)')
    # display max 8 modes
    nmodes = min(len(sd['freq']), 8)
    for i in range(nmodes):
        natfreq = sd['natfreq'][i]
        plt.plot(idof, sd['realmode'][i],'-*', label='{:0.2f} Hz'.format(natfreq))
    plt.axhline(y=0, ls='--', lw='0.5',color='k')
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.legend()

    return fig

def plot_linfrf(fnsi, dofs, H, ax = None, fig = None, **kwargs):

    fs = fnsi.fs
    nsper = fnsi.nsper
    flines = fnsi.flines

    freq = np.arange(0,nsper)*fs/nsper
    freq_plot = freq[flines +1]  # Hz

    # If h is only calculated for one dof:
    if H.shape[0] == 1:
        dofs = [0]

    if ax is None:
        fig, ax = plt.subplots()
        ax.clear()

    for i, dof in enumerate(dofs):
        # label='dof: {:d}'.format(dof))
        ax.plot(freq_plot, db(np.abs(H[i])), **kwargs)

    if ax is None:
        ax.set_title('Nonparametric linear FRF')
    ax.set_xlabel('Frequency (Hz)')
    # For linear scale: 'Amplitude (m/N)'
    ax.set_ylabel('Amplitude (dB)')
    ax.legend()
    return fig, ax


def plot_stab(fnsi, nlist, sd, ax = None, fig = None):
    fmin = fnsi.fmin
    fmax = fnsi.fmax

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
        for freq, ep, mode, stab in zip(v['freq'], v['ep'],
                                        v['mode'], v['stab']):
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
    if ax is None:
        fig, ax = plt.subplots()
        ax.clear()

    # Avoid showing the labels of empty plots
    if len(freqUNS) != 0:
        ax.plot(freqUNS, orUNS, 'xr', ms= 7, label='Unstabilized')
    if len(freqS) != 0:
        ax.plot(freqS, orSfreq, '*k', ms=7, label='Stabilized in natural frequency')
    if len(freqSep) != 0:
        ax.plot(freqSep, orSep, 'sk', ms=7, mfc='none', label='Extra stabilized in damping ratio')
    if len(freqSmode) != 0:
        ax.plot(freqSmode, orSmode, 'ok', ms=7, mfc='none', label='Extra stabilized in MAC')
    if len(freqSfull) != 0:
        ax.plot(freqSfull, orSfull, '^k', ms= 7, mfc='none', label='Full stabilization')

    ax.set_xlim([fmin, fmax])
    ax.set_ylim([nlist[0]-2, nlist[-1]])
    #ax = plt.gca()
    step = round(nlist[-2]/5)
    major_ticks = np.arange(0, nlist[-2]+1, step)
    ax.set_yticks(major_ticks)

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Model order')
    ax.set_title('Stabilization diagram')
    ax.legend(loc='lower right')

    return fig, ax
    #plt.show()

