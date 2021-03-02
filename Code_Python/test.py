"""
Path Planning Using Particle Swarm Optimization


Implementation of particle swarm optimization (PSO) for path planning when the
environment is known.


Copyright (c) 2021 Gabriele Gilardi


Features
--------
- The code has been written and tested in Python 3.8.5.
- Four types of obstacles: circle, ellipse, convex polygon, generic polygon.
- Start position, goal position, and obstacles can be dynamically changed to
  simulate motion.
- Penalty function of type 1/d with the center in the obstacle centroid.
- To improve the execution speed, the algorithms to determine if a point is
  inside an obstacle have been designed to carry out the determination on all
  points simultaneously.
- Points on the obstacle borders/edges are not considered inside the obstacle.
- Option to run sequential tests with different initial conditions to increase
  the chances to find a global minimum.
- Usage: python test.py <example>.

Main Parameters
---------------
example
    Number of the example to run (1, 2, or 3.)
start
    Start coordinates.
goal
    Goal coordinates.
limits
    Lower and upper boundaries of the map and search space in the PSO.
nRun
    Number of runs.
nPts >= 2
    Number of internal points defining the spline. The number of variables is
    twice this number.
d >= 2
    Number of segments between the spline breakpoints.
nPop >=1, epochs >= 1
    Number of agents (population) and number of iterations.
f_interp = slinear, quadratic, cubic
    Order of the spline (1st, 2nd and 3rd order, respectively.)
Xinit
    Initial value of the variables. Set Xinit=None to pick them randomly. This
    array is organized with first the x-coordinates of all internal points and
    then the y-coordinates of all internal points.
K >= 0
    Average size of each agent's group of informants. If K=0 the entire swarm
    is used as agent's group of informants.
phi >= 2
    Coefficient to calculate the self-confidence and the confidence-in-others
    coefficients.
vel_fact > 0
    Velocity factor to calculate the max. and min. allowed velocities.
conf_type = HY, RB, MX
    Confinement on the velocities: hyperbolic, random-back, mixed.
IntVar
    List of indexes specifying which variable should be treated as integer.
    If all variables are real set IntVar=None, if all variables are integer
    set IntVar='all'.
normalize = True, False
    Specifies if the search space should be normalized.
0 < rad < 1
    Normalized radius of the hypersphere centered on the best particle. The
    higher the number of other particles inside and the better is the solution.

Examples
--------
There are three examples, all of them using the same obstacles:

Example 1
- Multiple runs, cubic spline, optimizer initialized randomly.
- See <Results_Figure_1.png> for the full results.

Example 2
- Multiple runs, quadratic spline, optimizer initialized with the straight line
  between start and goal position.
- See <Results_Example_2.png> for the full results.

Example 3
- Single run, linear spline, optimizer initialized with the previous solution,
  start point chasing a moving goal with one obstacle (the circle) also moving.
- See <Results_Example_3.gif> for the full animation.

References
----------
- PSO: https://github.com/gabrielegilardi/PSO.git.
- Centroid calculation: http://en.wikipedia.org/wiki/Centroid.
- Points inside polygons: http://paulbourke.net/geometry/polygonmesh/.
"""

import sys
from copy import deepcopy
import numpy as np
from matplotlib.animation import FuncAnimation

from PathPlanning import *

# Read example to run (1, 2, or 3)
if len(sys.argv) != 2:
    print("Usage: python test.py <example>")
    sys.exit(1)
example = sys.argv[1]

# Define start, goal, and limits
start = (0, 5)
goal = (8, -2)
limits = [-2, 10, -6, 6]
layout = PathPlanning(start, goal, limits)

# Add obstacles (polygon verteces must be given in counter-clockwise order)
layout.add_ellipse(x=5, y=-1, theta=-np.pi/6, a=1.0, b=0.5, Kv=100)
V = [(2, 1), (5, 1), (5, 5), (4, 5), (2, 4)]
layout.add_convex(V, Kv=100)
V = [(6.5, -1), (9.5, -1), (9.5, 4), (8.5, 3), (8.5, 0), (7.5, 0),
     (7.5, 3), (6.5, 3)]
layout.add_polygon(V, Kv=100)
layout.add_circle(x=2, y=-2, r=1.2, Kv=100)

# Example 1: multiple runs, cubic spline, optimizer initialized randomly.
# Best: run=9, L=11.20, count=0.
# No obstacle violations.
# See <Results_Figure_1.png> for the full results.
if (example == '1'):

    nRun = 15
    nPts = 5
    d = 50                  # ns = 301 (points along the spline)
    nPop = 100
    epochs = 500
    f_interp = 'cubic'
    Xinit = None

# Example 2: multiple runs, quadratic spline, optimizer initialized with
# the straight line between start and goal position.
# Best: run=4, L=11.84, count=0.
# Obstacle violation in run 8, 9, 13, 14, and 15.
# See <Results_Example_2.png> for the full results.
elif (example == '2'):

    # Set new start and goal position
    layout.set_start(0, -3.5)
    layout.set_goal(8, 2)

    # Remove obstacle and add it again with a different scaling factor
    layout.remove_obs(2)
    V = [(6.5, -1), (9.5, -1), (9.5, 4), (8.5, 3), (8.5, 0), (7.5, 0),
         (7.5, 3),  (6.5, 3)]
    layout.add_polygon(V, Kv=200)

    nRun = 15
    nPts = 10
    d = 50                  # ns = 551 (points along the spline)
    nPop = 100
    epochs = 500
    f_interp = 'quadratic'
    Xinit = build_Xinit(layout.start, layout.goal, nPts)

# Example 3: single run, linear spline, optimizer initialized with the
# previous solution, start point chasing a moving goal with one obstacle
# (the circle) also moving.
# Path length goes from 11.53 (run 1) to 4.52 (run 65).
# Start position from [0.00,5.00] to [1.36,-0.81].
# Goal position from [8.00,-2.00] to [-1.27,-4.48].
# Obstacle position from [2.00,-2.00] to [3.04,-2.60].
# No obstacle violations.
# See <Results_Example_3.gif> for the full animation.
elif (example == '3'):

    nRun = 65
    nPts = 5
    d = 50                  # ns = 301 (points along the spline)
    nPop = 100
    epochs = 500
    f_interp = 'slinear'
    Xinit = None            # Re-defined after each run

else:
    print("Example not found")
    sys.exit(1)

# Init other parameters
np.random.seed(1294404794)
np.seterr(all='ignore')
ns = 1 + (nPts + 1) * d         # Number of points along the spline
best_L = np.inf                 # Best length (minimum)
best_run = 0                    # Run corresponding to the best length
best_count = 0                  # count corresponding to the best length
paths = [None] * nRun           # List with the results from all runs

# Run cases
print("\nns = ", ns)
for run in range(nRun):

    # Optimize (the other PSO parameters have always their default values)
    layout.optimize(nPts=nPts, ns=ns, nPop=nPop, epochs=epochs,
                    f_interp=f_interp, Xinit=Xinit)

    # Save run
    paths[run] = deepcopy(layout)

    # Print results
    L = layout.sol[1]               # Length
    count = layout.sol[2]           # Number of violated obstacles
    print("\nrun={0:d}, L={1:.2f}, count={2:d}"
          .format(run+1, L, count), end='', flush=True)

    # Save if best result (regardless the violations)
    if (L < best_L):
        best_L = L
        best_run = run
        best_count = count

    # Only for example 3 (move start, goal, and circular obstacle)
    if (example == '3'):

        # Print the current start, goal, and circle centroid
        print(", s=[{0:.2f},{1:.2f}], g=[{2:.2f},{3:.2f}], c=[{4:.2f},{5:.2f}]"
              .format(layout.start[0], layout.start[1], layout.goal[0],
              layout.goal[1], layout.obs[3][1], layout.obs[3][2]), end='',
              flush=True)

        # Path coordinates
        Px = layout.sol[3].flatten()
        Py = layout.sol[4].flatten()

        # Move the start position along the tangential direction of the
        # current path with a"speed" of 0.1
        vel_s = 0.1
        theta_s = np.arctan2(Py[1]-Py[0], Px[1]-Px[0])
        x_s = layout.start[0] + vel_s * np.cos(theta_s)
        y_s = layout.start[1] + vel_s * np.sin(theta_s)
        layout.set_start(x_s, y_s)

        # Move the goal position along a straight line with a "speed" of 0.15
        vel_g = 0.15
        theta_g = -165.0 * np.pi / 180.0
        x_g = layout.goal[0] + vel_g * np.cos(theta_g)
        y_g = layout.goal[1] + vel_g * np.sin(theta_g)
        layout.set_goal(x_g, y_g)

        # Move the circular obstacle along a straight line with a "speed" of 0.1
        vel_o = 0.1
        if (run > 25):
            vel_o = -vel_o          # Invert direction after 25 runs
        theta_o = +150.0 * np.pi / 180.0
        x_o = layout.obs[3][1] + vel_o * np.cos(theta_o)
        y_o = layout.obs[3][2] + vel_o * np.sin(theta_o)
        layout.remove_obs(3)
        layout.add_circle(x_o, y_o, r=1.2, Kv=100)

        # Initialize the solution with the last optimal path
        Xinit = layout.sol[0]

# Plots for examples 1 and 2
if (example == '1' or example == '2'):

    print("\n\nBest:", end='')
    print(" run={0:d}, L={1:.2f}, count={2:d}"
          .format(best_run+1, best_L, best_count))

    fig, axs = plt.subplots(3, 5)
    axs = axs.flatten()
    for run in range(nRun):

        layout = paths[run]         # Layout to plot
        ax = axs[run]               # Subplot

        L = layout.sol[1]           # Length
        count = layout.sol[2]       # Number of violated obstacles

        # Text position on the plots (lower left corner)
        xt = layout.limits[0] + 0.05 * (layout.limits[1] - layout.limits[0])
        yt = layout.limits[2] + 0.05 * (layout.limits[3] - layout.limits[2])

        layout.plot_obs(ax)         # Plot obstacles
        layout.plot_path(ax)        # Plot path

        # Plot run, length, and count
        title = "run=" + str(run+1) + ", L=" + str("{:.2f}".format(L)) + \
                ", count=" + str(count)
        ax.text(xt, yt, title, fontsize=10)

    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
    plt.show()

# Animation for example 3
elif (example == '3'):

    def animate(row):

        # Path x and y coordinates
        path.set_xdata(Px[row, :])
        path.set_ydata(Py[row, :])

        # Start position x and y coordinates
        Ps.set_xdata(start[row, 0])
        Ps.set_ydata(start[row, 1])

        # Goal position x and y coordinates
        Pg.set_xdata(goal[row, 0])
        Pg.set_ydata(goal[row, 1])

        # Text with the path length
        L_text.set_text(lengths[row])

        # Obstacle and its centroid
        obs.set_center((XYc[row, 0], XYc[row, 1]))
        XYobs.set_xdata(XYc[row, 0])
        XYobs.set_ydata(XYc[row, 1])

    # Elements to be animated
    Px = np.zeros((nRun, ns))           # Path x coordinates
    Py = np.zeros((nRun, ns))           # Path y coordinates
    start = np.zeros((nRun, 2))         # Start position
    goal = np.zeros((nRun, 2))          # Goal position
    lengths = [None] * nRun             # Length (text)
    XYc = np.zeros((nRun, 2))           # Obstacle (circle) centroid

    # Build the arrays needed for the animation
    for run in range(nRun):

        layout = paths[run]
        Px[run, :] = layout.sol[3].flatten()
        Py[run, :] = layout.sol[4].flatten()
        start[run, :] = layout.start
        goal[run, :] = layout.goal
        lengths[run] = "run=" + str(run+1) \
                       + ", L=" + str("{:.2f}".format(layout.sol[1]))
        XYc[run, :] = layout.obs[3][1:3]

    # Plot obstacles (except circle)
    fig, ax = plt.subplots()
    paths[0].remove_obs(3)
    paths[0].plot_obs(ax)

    # Path
    path = ax.plot(Px[0, :], Py[0, :], lw=0.5, c='r')[0]

    # Start and goal
    Ps = ax.plot(start[0, 0], start[0, 1], 'o', ms=6, c='k')[0]
    Pg = ax.plot(goal[0, 0], goal[0, 1], '*', ms=8, c='k')[0]

    # Length (text)
    xt = paths[0].limits[0] + 0.05 * (paths[0].limits[1] - paths[0].limits[0])
    yt = paths[0].limits[2] + 0.05 * (paths[0].limits[3] - paths[0].limits[2])
    L_text = ax.text(xt, yt, lengths[0], fontsize=10)

    # Obstacle (circle)
    r = 1.2
    obs = ax.add_patch(Circle((XYc[0, 0], XYc[0, 1]), r, fc='wheat', ec=None))
    XYobs = ax.plot(XYc[0, 0], XYc[0, 1], 'x', ms=4, c='orange')[0]

    # Animate and save a copy
    anim = FuncAnimation(fig, animate, interval=200, frames=nRun-1)
    print("\n")
    anim.save("Results_Example_3.gif")
    plt.show()
