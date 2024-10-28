# plotters.py (Modified Section)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.axes3d import Axes3D
from pyqcams.constants import *
from pyqcams import util
from constants import Boh2m, ttos, K2Har

def traj_plt(traj, ax=None, title=True, legend=True):
    '''
    Plots the internuclear distances over time.
    '''
    if ax is None:
        ax = plt.gca()
    x = traj.wn[:6]
    t = traj.t * ttos  # a.u. to seconds
    r12, r23, r31 = util.jac2cart(x, traj.C1, traj.C2)
    ax.plot(t, r12, label='r_SO', color='blue')
    ax.plot(t, r23, label='r_OO', color='green')
    ax.plot(t, r31, label='r_OS', color='red')
    ax.set_xlabel('$t$ (s)')
    ax.set_ylabel('$r$ ($a_0$)')
    if legend:
        ax.legend()
    if title:
        ax.set_title(f'Collision Energy: {traj.E0 / K2Har} K')
    return ax

def traj_3d(traj, ax=None):
    '''
    Creates a 3D plot of the trajectory.
    '''
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    x = traj.wn[:6]
    r1 = np.array([-traj.C2 * x[i] - traj.m3 / traj.mtot * x[i+3] for i in range(0,3)])  # S
    r2 = np.array([traj.C1 * x[i] - traj.m3 / traj.mtot * x[i+3] for i in range(0,3)])   # O1
    r3 = np.array([(traj.m1 + traj.m2) / traj.mtot * x[i+3] for i in range(0,3)])        # O2
    
    # Plot trajectories
    ax.plot(r1[0], r1[1], r1[2], 'b', label='S')
    ax.plot(r2[0], r2[1], r2[2], 'g', label='O1')
    ax.plot(r3[0], r3[1], r3[2], 'r', label='O2')
    
    # Mark initial positions
    ax.scatter(r1[0][0], r1[1][0], r1[2][0], marker='o', color='b', label='S Start')
    ax.scatter(r2[0][0], r2[1][0], r2[2][0], marker='o', color='g', label='O1 Start')
    ax.scatter(r3[0][0], r3[1][0], r3[2][0], marker='o', color='r', label='O2 Start')
    
    # Mark final positions
    ax.scatter(r1[0][-1], r1[1][-1], r1[2][-1], marker='^', color='b', label='S End')
    ax.scatter(r2[0][-1], r2[1][-1], r2[2][-1], marker='^', color='g', label='O1 End')
    ax.scatter(r3[0][-1], r3[1][-1], r3[2][-1], marker='^', color='r', label='O2 End')
    
    ax.set_xlabel('X ($a_0$)')
    ax.set_ylabel('Y ($a_0$)')
    ax.set_zlabel('Z ($a_0$)')
    ax.legend()
    return ax

def traj_gif(traj, theta=30, phi=30):
    '''
    Create an animation of a 3D trajectory.
    '''
    def update_lines(num, dataLines, lines):
        for line, data in zip(lines, dataLines):
            line.set_data(data[0:2, :num])
            line.set_3d_properties(data[2, :num])
        return lines
    
    r = traj.wn[:6]
    t = traj.t
    m1, m2, m3 = traj.m1, traj.m2, traj.m3
    mtot = m1 + m2 + m3
    c1, c2 = traj.C1, traj.C2
    r1 = np.array([-c2 * r[i] - m3 / mtot * r[i+3] for i in range(0,3)])  # S
    r2 = np.array([c1 * r[i] - m3 / mtot * r[i+3] for i in range(0,3)])   # O1
    r3 = np.array([(m1 + m2) / mtot * r[i+3] for i in range(0,3)])        # O2
    data = [r1, r2, r3]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Initialize lines
    lines = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1])[0] for dat in data]
    
    # Setting the axes properties
    all_r = np.concatenate(data, axis=1)
    ax.set_xlim3d([all_r[0].min() / 2, all_r[0].max()])
    ax.set_ylim3d([all_r[1].min() / 2, all_r[1].max()])
    ax.set_zlim3d([all_r[2].min() / 2, all_r[2].max()])
    ax.set_xlabel('X ($a_0$)')
    ax.set_ylabel('Y ($a_0$)')
    ax.set_zlabel('Z ($a_0$)')
    ax.view_init(theta, phi)
    
    # Creating the Animation object
    line_ani = animation.FuncAnimation(
        fig, update_lines, frames=len(t), fargs=(data, lines),
        interval=30, blit=False
    )
    
    plt.show()
    return ax, line_ani
