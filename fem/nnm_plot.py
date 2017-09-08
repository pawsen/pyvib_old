#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

class Anim(object):

    def __init__(self, energy, omega, xamp=None, dof=0, energy_plot=True):

        self.dof = dof

        if energy_plot:
            xx = np.log10(energy)
            titlestr = 'Frequency Energy plot (FEP)'
            xstr = 'Log10(Energy) (J)'
        else:
            xx = xamp
            titlestr = 'Amplitude of dof {}'.format(idof)
            xstr = 'Amplitude (m)'

        xx = np.log10(np.asarray(energy))
        yy = np.asarray(omega)/2/np.pi

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.clear()

        ax.set_title(titlestr)
        ax.set_xlabel(xstr)
        ax.set_ylabel('Frequency (Hz)')

        ax.set_xlim((min(xx)*1.1,max(xx)*1.1))
        ax.set_ylim((0,max(yy)*1.5))
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))

        # cache the background
        self.background = fig.canvas.copy_from_bbox(ax.bbox)
        self.points = ax.plot(xx, yy, '-')[0]
        self.cur_point = ax.plot(xx[-1], yy[-1], 'o')[0]
        self.ax = ax
        self.fig = fig

        plt.show(False)
        fig.canvas.draw()

    def update(self, energy, omega):

        dof = self.dof
        ax = self.ax
        fig = self.fig

        xx = np.log10(np.asarray(energy))
        yy = np.asarray(omega)/2/np.pi
        self.points.set_data(xx, yy)
        self.cur_point.set_data(xx[-1], yy[-1])

        ax.set_xlim((min(xx)*1.1,max(xx)*1.1))
        ax.set_ylim((0,max(yy)*1.1))

        # restore background
        fig.canvas.restore_region(self.background)
        # redraw just the points
        ax.draw_artist(self.points)
        ax.draw_artist(self.cur_point)
        # fill in the axes rectangle
        fig.canvas.blit(ax.bbox)
