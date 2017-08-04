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

        plt.figure()
        plt.clf()
        plt.subplot(2, 1, 1)
        plt.title('Exponent: {:d}. Estimated: {:0.3e}'.format(exponent,mu_mean[0]))
        plt.plot(freq_plot, np.real(mu),label='fnsi')
        plt.axhline(mu_mean[0], c='k', ls='--', label='mean')
        plt.xlabel('Frequency (Hz)')
        plt.legend()

        ymin = np.min(np.real(mu))
        ymax = np.max(np.real(mu))
        if np.abs(ymax - ymin) <= 1e-6:
            ymin = 0.9 * mu_mean[0]
            ymax = 1.1 * mu_mean[0]
            plt.ylim([ymin-eps, ymax+eps])
            plt.ylabel(r'Real($\mu$) $(N/m^3)$ 1%')
        else:
            plt.ylabel(r'Real($\mu$) $(N/m^3)$')

        plt.subplot(2, 1, 2)
        plt.plot(freq_plot, np.imag(mu))
        plt.axhline(mu_mean[1], c='k', ls='--', label='mean')
        plt.xlabel('Frequency (Hz)')
        ymin = np.min(np.imag(mu))
        ymax = np.max(np.imag(mu))
        if np.abs(ymax - ymin) <= 1e-6:
            ymin = 0.9 * mu_mean[1]
            ymax = 1.1 * mu_mean[1]
            plt.ylim([ymin-eps, ymax+eps])
            plt.ylabel(r'Imag($\mu$) $(N/m^3)$ 1%')
        else:
            plt.ylabel(r'Imag($\mu$) $(N/m^3)$')


def plot_modes(idof, sd):

    plt.figure()
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

def plot_linfrf(fnsi, dofs, H):

    fs = fnsi.fs
    nsper = fnsi.nsper
    flines = fnsi.flines

    freq = np.arange(0,nsper)*fs/nsper
    freq_plot = freq[flines +1]  # Hz

    # If h is only calculated for one dof:
    if H.shape[0] == 1:
        dofs = [0]

    plt.figure(5)
    plt.clf()
    for i, dof in enumerate(dofs):
        plt.plot(freq_plot, db(np.abs(H[i])),label='dof: {:d}'.format(dof))

    plt.title('Nonparametric linear FRF')
    plt.xlabel('Frequency (Hz)')
    # For linear scale: 'Amplitude (m/N)'
    plt.ylabel('Amplitude (dB)')
    plt.legend()

def plot_stab(fnsi, nlist, sd):
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

    plt.figure()
    plt.clf()
    # Avoid showing the labels of empty plots
    if len(freqUNS) != 0:
        plt.plot(freqUNS, orUNS, 'xr', ms= 7, label='Unstabilized')
    if len(freqS) != 0:
        plt.plot(freqS, orSfreq, 'xk', ms=7, label='Stabilized in natural frequency')
    if len(freqSep) != 0:
        plt.plot(freqSep, orSep, 'sk', ms=7, mfc='none', label='Extra stabilized in damping ratio')
    if len(freqSmode) != 0:
        plt.plot(freqSmode, orSmode, 'ok', ms=7, mfc='none', label='Extra stabilized in MAC')
    if len(freqSfull) != 0:
        plt.plot(freqSfull, orSfull, '^k', ms= 7, mfc='none', label='Full stabilization')

    plt.xlim([fmin, fmax])
    plt.ylim([nlist[0]-2, nlist[-1]])
    ax = plt.gca()
    step = round(nlist[-2]/5)
    major_ticks = np.arange(0, nlist[-2]+1, step)
    ax.set_yticks(major_ticks)

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Model order')
    plt.title('Stabilization diagram')
    plt.legend(loc='lower right')
    #plt.show()

