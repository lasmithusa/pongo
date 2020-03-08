import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# from matplotlib.patches import Arrow

def calc_var_r_mag(wheel_r_mag, offset_r_mag, theta):
    # takes the magnitudes of the wheel radius, R, and the offset radius, r, and the current eccentric wheel angle (positive = ccw from BDC)
    # returns the magnitude of the variable radius, R*, as a double

    return (math.sqrt(math.pow(wheel_r_mag, 2) - math.pow(offset_r_mag, 2)*math.pow(math.sin(theta), 2)) - offset_r_mag*math.cos(theta))

def calc_var_r_vec(var_r_mag, theta):
    # takes the magnitude of the variable radius, R*, and the current eccentric wheel angle (positive = ccw from BDC)
    # returns a vector (numpy array) of the variable radius, R*, in [x, y] form

    return np.array([0, var_r_mag])

def calc_offset_r_vec(offset_r_mag, theta):
    # takes the magnitude of the offset radius, r, and the current eccentric wheel angle (positive = ccw from BDC)
    # returns a vector (numpy array) of offset radius, r, in [x, y] form

    return np.array([offset_r_mag*math.sin(theta), -offset_r_mag*math.cos(theta)])

def calc_wheel_r_vec(var_r_vec, offset_r_vec):
    # takes the variable radius vector, R*, and the offset radius vector, r, and returns the wheel radius vector, R
    # all inputs are numpy arrays
    # returns a vector (numpy array)

    return np.subtract(var_r_vec, offset_r_vec)

def calc_load_vec(load_mag, wheel_r_vec):
    # takes the magnitude of the load, |F_L|, and the wheel radius vector, R
    # returns the load vector, F_L
    
    # calculate unit vector of F_L
    load_unit_vec = np.negative(wheel_r_vec)/np.sqrt(np.sum(wheel_r_vec**2))

    return load_mag*load_unit_vec

def calc_fric_vec(load_mag, mu, wheel_r_vec):
    # takes the magnitude of the load, |F_L|, the coefficient of friction, mu, and the wheel radius vector, R
    # returns the frictional load vecotr, F_f

    # calculate unit vector of F_f
    wheel_unit_vec = wheel_r_vec/np.sqrt(np.sum(wheel_r_vec**2))
    fric_unit_vec = np.array([wheel_unit_vec[1], np.negative(wheel_unit_vec[0])])

    return load_mag*mu*fric_unit_vec

# set the magnitude of wheel radius, R, and offset radius, r, in mm
wheel_r_mag = 14
offset_r_mag = 10

# set force representation
show_forces = False
# set the magnitude of the load, F_L, in grams
load_mag = 25
# set the friction coeffiecient, mu
mu = 0.5

# set radial speed in rad / s
# 0.1 rev / s
omega = 2*math.pi / 2
# set theta increment
# 15 deg increment per frame
theta_inc = 2*math.pi / 35
# calculate necessary frames / s
framerate = omega / theta_inc
# set theta values
theta_vals = np.arange(0.0, 2*math.pi, theta_inc)

# generate figure and axes objects
fig, ax = plt.subplots()

# set figure axes
ax.axis([-30, 30, -30, 30])
plt.gca().set_aspect('equal')

# plot motor shaft center
motor_center, = plt.plot(0, 0, 'ok')

# plot first variable radius, R*, line
var_r_mag = calc_var_r_mag(wheel_r_mag, offset_r_mag, 0)
var_r_vec = calc_var_r_vec(var_r_mag, 0)
var_r, = plt.plot([0, var_r_vec[0]], [0, var_r_vec[1]])

# plot first offset radius, r, line
offset_r_vec = calc_offset_r_vec(offset_r_mag, 0)
offset_r, = plt.plot([0, offset_r_vec[0]], [0, offset_r_vec[1]])

# plot first wheel radius, R, line
wheel_r_vec = calc_wheel_r_vec(var_r_vec, offset_r_vec)
wheel_x, wheel_y = offset_r_vec
wheel_r, = plt.plot([wheel_x, wheel_x + wheel_r_vec[0]], [wheel_y, wheel_y + wheel_r_vec[1]])

# plot first wheel
wheel = plt.Circle((wheel_x, wheel_y), radius = wheel_r_mag, fill=False)
plt.gcf().gca().add_artist(wheel)

# plot first wheel center
wheel_center, = plt.plot(wheel_x, wheel_y, 'x')

if show_forces:
    # plot first load vector, F_L
    load_vec = calc_load_vec(load_mag, wheel_r_vec)
    load, = plt.plot([var_r_vec[0] - load_vec[0], load_vec[0]], [var_r_vec[1] - load_vec[1], load_vec[1]])

    # plot first friction vector, F_f
    fric_vec = calc_fric_vec(load_mag, mu, wheel_r_vec)
    fric, = plt.plot([var_r_vec[0], var_r_vec[0] + fric_vec[0]], [var_r_vec[1], var_r_vec[1] + fric_vec[1]])

def animate_wheel(i):
    # recalculate variable radius, R*, vector
    var_r_mag = calc_var_r_mag(wheel_r_mag, offset_r_mag, i)
    var_r_vec = calc_var_r_vec(var_r_mag, i)
    var_r.set_data([0, var_r_vec[0]], [0, var_r_vec[1]])

    # recalculate offset radius, r, vector
    offset_r_vec = calc_offset_r_vec(offset_r_mag, i)
    offset_r.set_data([0, offset_r_vec[0]], [0, offset_r_vec[1]])

    # recalculate wheel radius, R, vector
    wheel_r_vec = calc_wheel_r_vec(var_r_vec, offset_r_vec)
    wheel_x, wheel_y = offset_r_vec
    wheel_r.set_data([wheel_x, wheel_x + wheel_r_vec[0]], [wheel_y, wheel_y + wheel_r_vec[1]])

    # recalculate wheel position and centermark
    wheel.set_center([wheel_x, wheel_y])
    wheel_center.set_data([wheel_x, wheel_y])

    if show_forces:
        # recalculate load vector, F_L
        load_vec = calc_load_vec(load_mag, wheel_r_vec)
        load.set_data([var_r_vec[0] - load_vec[0], var_r_vec[0]], [var_r_vec[1] - load_vec[1], var_r_vec[1]])

        # recalculate friction vector, F_f
        fric_vec = calc_fric_vec(load_mag, mu, wheel_r_vec)
        fric.set_data([var_r_vec[0], var_r_vec[0] + fric_vec[0]], [var_r_vec[1], var_r_vec[1] + fric_vec[1]])

        return var_r, offset_r, wheel_r, wheel, load, fric

    return var_r, offset_r, wheel_r, wheel

# animate graphic
my_animation = animation.FuncAnimation(fig, animate_wheel, frames=theta_vals, interval=1000/framerate, repeat=True)
plt.show()

# # save graphic
# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=framerate, bitrate=1800)
# my_animation.save('EccentricWheel.mp4', writer=writer)
