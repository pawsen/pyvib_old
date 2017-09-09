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


        xmin, xmax = min(xx), max(xx)
        scale = 1.5
        xmin = xmin*scale if xmin < 0 else xmin*(scale-1)
        xmax = xmax*(scale-1) if xmax < 0 else xmax*scale
        ax.set_xlim((xmin, xmax))
        ax.set_ylim((0,max(yy)*1.5))
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))

        plt.show(False)
        fig.canvas.draw()

        # cache the clean background
        self.background = fig.canvas.copy_from_bbox(ax.bbox)
        self.points = ax.plot(xx, yy, '-')[0]
        self.cur_point = ax.plot(xx[-1], yy[-1], 'o')[0]
        self.ax = ax
        self.fig = fig

    def update(self, energy, omega):
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
        # restore the clean slate background
        fig.canvas.restore_region(self.background)

        xx = np.log10(np.asarray(energy))
        yy = np.asarray(omega)/2/np.pi

        ax_xmin, ax_xmax = ax.get_xlim()
        xmin, xmax = min(xx), max(xx)
        ax_ylim = ax.get_ylim()
        ymin, ymax = min(yy), max(yy)
        if ( xmin <= ax_xmin or xmax >= ax_xmax or
             ymin <= ax_ylim[0] or ymax >= ax_ylim[1] ):
            scale = 1.5
            xmin = xmin*scale if xmin < 0 else xmin*(scale-1)
            xmax = xmax*(scale-1) if xmax < 0 else xmax*scale
            ax.set_xlim((xmin, xmax))
            ymin = ymin*scale if ymin < 0 else ymin*(scale-1)
            ymax = ymax*(scale-1) if ymax < 0 else ymax*scale
            ax.set_ylim((ymin, ymax))

            fig.canvas.draw()
            self.background = fig.canvas.copy_from_bbox(ax.bbox)

        self.points.set_data(xx, yy)
        self.cur_point.set_data(xx[-1], yy[-1])
        # redraw just the points
        ax.draw_artist(self.points)
        ax.draw_artist(self.cur_point)
        # fill in the axes rectangle
        fig.canvas.blit(ax.bbox)
