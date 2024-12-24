"""
Example control_approx_linearization.py (Modified for QVEX APSC 103 Software Project)
Author: Joshua A. Marshall <joshua.marshall@queensu.ca>
GitHub: https://github.com/botprof/agv-examples
"""

# %%
# SIMULATION SETUP


import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from math import pi
from mobotpy.models import DiffDrive
from mobotpy.integration import rk_four
from apsc_103.path_model import *

# %%
# COMPUTE THE PATH
path = [
    LineSegment(0, 0, 1, 0),
    ArcSegment(1, 1, 1, -pi/2, 0),
]

# %%
# COMPUTE THE REFERENCE TRAJECTORY

# Set the simulation time [s] and the sample period [s]
SIM_TIME = 20.0
T = 0.04

# Create an array of time values [s]
t = np.arange(0.0, SIM_TIME, T)
N = np.size(t)

# Radius of the circle [m]
R = 10

# Angular rate [rad/s] at which to traverse the circle
OMEGA = 0.1

# Pre-compute the desired trajectory
x_d = np.zeros((3, N))
u_d = np.zeros((2, N))
segment_time = 0.0
"""
for k in range(0, N):
    x_d[0, k] = R * np.sin(OMEGA * t[k]) #x
    x_d[1, k] = R * (1 - np.cos(OMEGA * t[k])) #y
    x_d[2, k] = OMEGA * t[k] #orientation?
    u_d[0, k] = R * OMEGA #forward velocity
    u_d[1, k] = OMEGA #angular velocity
"""
# Loop to iterate over segments of specified line and arc segment: 
for segment in path: 
    if isinstance(segment, LineSegment):
        segment_duration = segment.length / segment.velocity_at_point()
        for k in range (int(segment_time/T), int((segment_time + segment_duration)/T)):
            x_d[0, k], x_d[1, k] = segment.position_at_point((k - (segment_time/T))*T, segment.velocity_at_point()) # x and y coords
            x_d[2, k] = 0 # Orientation
            u_d[0, k] = segment.velocity_at_point() # Forward velocity
            u_d[1, k] = 0 # Angular velocity
        segment_time += segment_duration
    elif isinstance(segment, ArcSegment): 
        segment_duration = segment.length / segment.velocity_at_point()
        angular_velocity = segment.velocity_at_point() / segment.radius
        for k in range (int(segment_time/T), int((segment_time + segment_duration)/T)):
            x_d[0, k], x_d[1, k] = segment.position_at_point((k - (segment_time/T))*T, angular_velocity)
            x_d[2, k] = segment.start_angle + angular_velocity * ((k - int(segment_time / T)) * T)
            u_d[0, k] = segment.velocity_at_point()
            u_d[1, k] = angular_velocity
        segment_time += segment_duration


# %%
# VEHICLE SETUP

# Set the track length of the vehicle [m]
ELL = 1.0

# Create a vehicle object of type DiffDrive
vehicle = DiffDrive(ELL)

# %%
# SIMULATE THE CLOSED-LOOP SYSTEM

# Initial conditions
x_init = np.zeros(3)
x_init[0] = 0.0
x_init[1] = 5.0
x_init[2] = 0.0

# Setup some arrays
x = np.zeros((3, N))
u = np.zeros((2, N))
x[:, 0] = x_init

for k in range(1, N):

    # Simulate the differential drive vehicle motion
    x[:, k] = rk_four(vehicle.f, x[:, k - 1], u[:, k - 1], T)

    # Compute the approximate linearization
    A = np.array(
        [
            [0, 0, -u_d[0, k - 1] * np.sin(x_d[2, k - 1])],
            [0, 0, u_d[0, k - 1] * np.cos(x_d[2, k - 1])],
            [0, 0, 0],
        ]
    )
    B = np.array([[np.cos(x_d[2, k - 1]), 0], [np.sin(x_d[2, k - 1]), 0], [0, 1]])

    try:
        # Compute the gain matrix to place poles of (A - BK) at p
        p = np.array([-1.0, -2.0, -0.5])
        K = signal.place_poles(A, B, p) #FIX HERE!!!! - poles can't be placed
    except ValueError as err:
        print(f"k = {k}")
        print(f"A = {A}")
        print(f"B = {B}")
        raise err

    # Compute the controls (v, omega) and convert to wheel speeds (v_L, v_R)
    u_unicycle = -K.gain_matrix @ (x[:, k - 1] - x_d[:, k - 1]) + u_d[:, k]
    u[:, k] = vehicle.uni2diff(u_unicycle)

# %%
# MAKE PLOTS

# Change some plot settings (optional)
# plt.rc("text", usetex=True)
# plt.rc("text.latex", preamble=r"\usepackage{cmbright,amsmath,bm}")
plt.rc("savefig", format="pdf")
plt.rc("savefig", bbox="tight")

# Plot the states as a function of time
fig1 = plt.figure(1)
fig1.set_figheight(6.4)
ax1a = plt.subplot(411)
plt.plot(t, x_d[0, :], "C1--")
plt.plot(t, x[0, :], "C0")
plt.grid(color="0.95")
plt.ylabel("x [m]")
plt.setp(ax1a, xticklabels=[])
plt.legend(["Desired", "Actual"])
ax1b = plt.subplot(412)
plt.plot(t, x_d[1, :], "C1--")
plt.plot(t, x[1, :], "C0")
plt.grid(color="0.95")
plt.ylabel("y [m]")
plt.setp(ax1b, xticklabels=[])
ax1c = plt.subplot(413)
plt.plot(t, x_d[2, :] * 180.0 / np.pi, "C1--")
plt.plot(t, x[2, :] * 180.0 / np.pi, "C0")
plt.grid(color="0.95")
plt.ylabel("θ [deg]")
plt.setp(ax1c, xticklabels=[])
ax1d = plt.subplot(414)
plt.step(t, u[0, :], "C2", where="post", label="$v_L$")
plt.step(t, u[1, :], "C3", where="post", label="$v_R$")
plt.grid(color="0.95")
plt.ylabel("u [m/s]")
plt.xlabel("t [s]")
plt.legend()

# Save the plot
# plt.savefig("../agv-book/figs/ch4/control_approx_linearization_fig1.pdf")

# Plot the position of the vehicle in the plane
fig2 = plt.figure(2)
plt.plot(x_d[0, :], x_d[1, :], "C1--", label="Desired")
plt.plot(x[0, :], x[1, :], "C0", label="Actual")
plt.axis("equal")
X_L, Y_L, X_R, Y_R, X_B, Y_B, X_C, Y_C = vehicle.draw(x[0, 0], x[1, 0], x[2, 0])
plt.fill(X_L, Y_L, "k")
plt.fill(X_R, Y_R, "k")
plt.fill(X_C, Y_C, "k")
plt.fill(X_B, Y_B, "C2", alpha=0.5, label="Start")
X_L, Y_L, X_R, Y_R, X_B, Y_B, X_C, Y_C = vehicle.draw(
    x[0, N - 1], x[1, N - 1], x[2, N - 1]
)
plt.fill(X_L, Y_L, "k")
plt.fill(X_R, Y_R, "k")
plt.fill(X_C, Y_C, "k")
plt.fill(X_B, Y_B, "C3", alpha=0.5, label="End")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.legend()

# Save the plot
# plt.savefig("../agv-book/figs/ch4/control_approx_linearization_fig2.pdf")

# Show the plots to the screen
# plt.show()

# %%
# MAKE AN ANIMATION

# Create the animation
ani = vehicle.animate_trajectory(x, x_d, T)

# Create and save the animation
# ani = vehicle.animate_trajectory(
#     x, x_d, T, True, "../agv-book/gifs/ch4/control_approx_linearization.gif"
# )

# Show all the plots to the screen
plt.show()

# Show animation in HTML output if you are using IPython or Jupyter notebooks
# from IPython.display import display

# plt.rc("animation", html="jshtml")
# display(ani)
# plt.close()

# %%
