import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider
import matplotlib.pyplot as plt
import numpy as np

class rfsPlotBuilder(object):
    """
    Could be update to not clear and redraw, but instead just update the
    points.

    First save handle to plot:
    3dplot = ax3d.plot(y,dy,-ddy, '.k', markersize=10)
    ax3d.set_autoscaley_on(True)

    #¤¤ on update:
    3dplot.set_xdata(xdata)
    # Rescale:
    ax3d.relim()
    ax3d.autoscale_view()

    # draw and flush (something like this...)
    fig.canvas.draw()
    fig.canvas.flush_events()

    """
    def __init__(self, rfs):

        self.rfs = rfs

        fig= plt.figure()
        ax = fig.add_subplot(211)
        ax2 = fig.add_subplot(223, projection='3d')
        ax3 = fig.add_subplot(224)
        self.ax3d = ax2
        self.ax2d = ax3
        self.fig = fig
        # fig.subplots_adjust(wspace=0.7)
        # fig.subplots_adjust(hspace=0.5)
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.2)

        # plot acceleration signal
        t = np.arange(self.rfs.ns)/self.rfs.fs
        ax.plot(t, self.rfs.ddy[0,:],'-k')
        #ax.plot(t, self.rfs.y,'-k')

        #¤ back to rectangular settings
        ymin, ymax = ax.get_ylim()
        xmin, xmax = ax.get_xlim()
        xx = [xmin,xmax]

        # Set initial selection and add rectangle
        start_sel = 0.45
        width_sel = 0.1

        x1 = xx[0] + start_sel * np.diff(xx)
        dx = width_sel * np.diff(xx)
        rect = patches.Rectangle((x1, ymin), dx, ymax-ymin, alpha=0.4,fc='y')
        ax.add_patch(rect)

        # set slider
        sliderax = fig.add_axes([0.1, 0.02, 0.6, 0.03], fc='white')
        slider = Slider(sliderax, 'Tol', 0, 0.5, valinit= 0.05)
        slider.on_changed(self.slider_update)
        slider.drawon = True

        canvas = rect.figure.canvas
        self.canvas = canvas
        self.rect = rect
        self.axes = rect.axes
        self.ind = None

        # show rfs for the initial selection
        self.update_rfs()

        canvas.mpl_connect('button_press_event', self.button_press_callback)
        canvas.mpl_connect('button_release_event', self.button_release_callback)
        canvas.mpl_connect('motion_notify_event', self.motion_notify_callback)

        plt.show()

    def slider_update(self, value):

        self.rfs.tol_slice = value
        y, dy, ddy, y_tol, dy_tol, ddy_tol = self.rfs.update_sel(self.idx1, self.idx2)
        self.update_rfs()

        # TODO: works, even if the canvas belong to rect. ie canvas = rect.canvas
        self.canvas.blit(self.ax2d.bbox)
        self.canvas.blit(self.ax3d.bbox)

    def get_ind(self, event):
        # Figure out which line to move, or move the figure if in between lines.

        # get the index of the vertex under point if within epsilon tolerance
        x, y = self.rect.get_x(), self.rect.get_y()
        width = self.rect.get_width()
        # d1: distance to left line, d2: distance to right line
        d1 = np.sqrt((x-event.xdata)**2)
        d2 = np.sqrt(((x+width)-event.xdata)**2)

        # save cursor position in order to move the rectangle a relative
        # distance
        self.press = event.xdata, x, width

        epsilon = 10
        # only move if cursor is sufficiently close to line or over rect
        if d1 < epsilon:
            return 0
        elif d2 < epsilon:
            return 1
        elif event.xdata > x and event.xdata < x+width:
            return 2
        else:
            return None

    def check_rect(self, event):
        # only do stuff if we're on the rectangle
        if event.inaxes != self.rect.axes:
            return False
        elif event.button != 1:
            return False
        else:
            return True

    def button_press_callback(self, event):
        if self.check_rect(event) is False: return

        # get line cursor is close to
        self.ind = self.get_ind(event)

        self.rect.set_animated(True)
        self.canvas.draw()
        # store background for updating the plot
        self.background = self.canvas.copy_from_bbox(self.rect.axes.bbox)
        self.axes.draw_artist(self.rect)
        self.canvas.blit(self.axes.bbox)

    def button_release_callback(self, event):
        if self.check_rect(event) is False: return
        self.ind = None
        # update graphs showing restoring force
        self.update_rfs()

        self.rect.set_animated(False)
        self.background = None
        self.rect.figure.canvas.draw()

    def motion_notify_callback(self, event):
        # Update the rect position while dragging/moving
        if self.check_rect(event) is False: return
        if self.ind is None:
            return

        # get coords from when the mouse was pressed
        x_event, x_rect, w_rect = self.press
        x, y = event.xdata, event.ydata
        dx = x - x_event
        if self.ind is 0:
            self.rect.set_x(x)
            self.rect.set_width(w_rect - dx)
        elif self.ind is 1:
            self.rect.set_width(x-x_rect)
        else:
            self.rect.set_x(x_rect + dx)

        self.canvas.restore_region(self.background)
        self.axes.draw_artist(self.rect)
        self.canvas.blit(self.axes.bbox)

    def set_postion(self, x, width):
        self.rect.set_x(x)
        self.rect.set_width(width)
        self.update_rfs()

    def update_rfs(self):
        # plot RFS

        # get index of selection
        x1, width = self.rect.get_x(), self.rect.get_width()
        x2 = x1 + width
        xmin, xmax = self.axes.get_xlim()
        xx = [xmin, xmax]
        self.idx1 = int(np.floor( self.rfs.ns * (x1 - xx[0]) / np.diff(xx)))
        self.idx2 = int(np.ceil( self.rfs.ns * (x2 - xx[0]) / np.diff(xx)))

        y, dy, ddy, y_tol, dy_tol, ddy_tol = self.rfs.update_sel(self.idx1, self.idx2)
        self.ax3d.clear()
        self.ax3d.plot(y,dy,-ddy, '.k', markersize=10)
        self.ax3d.plot(y_tol,dy_tol,-ddy_tol, '.r', markersize=10)
        self.ax2d.clear()
        self.ax2d.plot(y_tol,-ddy_tol, '.k', markersize=12)

        self.ax3d.set_title("Restoring force surface")
        self.ax3d.set_xlabel('Displacement (m)')
        self.ax3d.set_ylabel('Velocity (m/s)')
        self.ax3d.set_zlabel('-Acceleration (m/s²)')
        self.ax3d.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

        self.ax2d.set_title('Stiffness curve')
        self.ax2d.set_xlabel('Displacement (m)')
        self.ax2d.set_ylabel('-Acceleration (m/s²)')
        self.ax2d.ticklabel_format(style='sci', axis='x', scilimits=(0,0))


import filter as myfilter
class RFS(object):
    def __init__(self,val, fs, dofs=None, show_damped = False, displ = False,
                 numeric = False):
        """
        Parameters:
        -----------
        val : ndarray(ns)
            either accelerations or displacements
        dofs : int(2)
            dofs to compare. If none, then compare to ground, ie. zero signal
        show_damp : bool
            show stifness or damping coeff.
        displ : bool
            Is the signal accelerations or displacements
        """

        if dofs is None and val.ndim is 2:
            val = val[0,:]
        else:
            val = val[dofs,:]
        if val.ndim is not 2:
            # cast to 2d array
            val = val[None,:]

        self.numeric = numeric
        self.displ = displ
        self.dofs = dofs
        self.fs = fs
        self.ns = val.shape[1]
        self.tol_slice = 1e-2

        # displacement. Differentiate
        if self.displ:
            y = val
            dy = np.empty(val.shape)
            self.ddy = np.empty(val.shape)
            for i in range(val.shape[0]):
                dy[i,:] , self.ddy[i,:] = myfilter.differentiate(y[i,:],
                                                                 self.fs, numeric=self.numeric)
        else:
            # accelerations. Integrate
            self.ddy = val
            y = np.empty(val.shape)
            dy = np.empty(val.shape)
            for i in range(val.shape[0]):
                y[i,1:-1] , dy[i,1:] = myfilter.integrate(self.ddy[i,:],
                                                          self.fs, numeric=self.numeric)

        # import scipy.io
        # directory = 'data/T03a_Data/'
        # mat =  scipy.io.loadmat(directory + 'f16_x.mat')

        # y = mat['x'][dofs,:]
        # dy = mat['xd'][dofs,:]
        # ddy = mat['xdd'][dofs,:]

        # g( x_i - x_j, dx_i - dx_j) = -ddx_i
        if val.shape[0] is 2:
            # connected to another dof
            self.y = y[0,:] - y[1,:]
            self.dy = dy[0,:] - dy[1,:]
        else:
            # connected to ground
            self.y = y[0,:]
            self.dy = dy[0,:]


    def update_sel(self, id0, id1=-1, show_damped=False):
        """ Update RFS for the given selection

        Parameters:
        ----------
        id0/id1 : int
            index start/end for the selection
        """
        #t = self.t( id0:di1 );
        y = self.y[id0:id1]
        dy = self.dy[id0:id1]
        ddy = self.ddy[0,id0:id1]

        if show_damped:
            tol =  self.tol_slice * max( np.abs(y))
            ind_tol = np.where(np.abs(y) < tol)
        else:
            tol =  self.tol_slice * max( np.abs(dy))
            ind_tol = np.where(np.abs(dy) < tol)
        y_tol = y[ind_tol]
        dy_tol = dy[ind_tol]
        ddy_tol = ddy[ind_tol]

        return y, dy, ddy, y_tol, dy_tol, ddy_tol



# from mpl_toolkits.mplot3d import Axes3D

# plt.figure(2)
# plt.clf()
# ax = plt.axes(projection='3d')
# ax.plot(y1, dy, -ddy, '.k', markersize=10)
# ax.plot(y1_tol, y1_tol, -ddy_tol, '.r', markersize=12)
# ax.set_title("Restoring force surface")
# ax.set_xlabel('Displacement (m)')
# ax.set_ylabel('Velocity (m/s)')
# ax.set_zlabel('-Acceleration (m/s²)')


# plt.figure(3)
# plt.clf()
# plt.title('Stiffness curve')
# plt.xlabel('Displacement (m)')
# plt.ylabel('-Acceleration (m/s²)')
# plt.plot(y1_tol,-ddy_tol,'.k', markersize=12)

# #plt.show()
