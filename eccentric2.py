import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import transforms
import matplotlib.animation as animation


# TODO
# Define friction vector
# Set masses and moments of inertia
# Define coefficient of friction (constant initially, function later?)
# Calculate needed spring constant / spring force
# Define spring force function
# Define linear velocity and acceleration functions

# Vector magnitude definitions
def calc_C_mag(R_w_mag, r_w_mag, r_o_mag, theta):
    # takes the magnitudes of vectors R_w, r_w, and r_o and of angle theta (in degrees)
    # returns the magnitude of vector C

    return np.sqrt((R_w_mag + r_w_mag)**2 - r_o_mag**2 * np.sin(np.radians(theta))**2) - r_o_mag * np.cos(np.radians(theta))
    # return np.sqrt(np.square(np.sum(R_w_mag, r_w_mag)) - np.prod(np.square(r_o_mag), np.square(np.sin(np.radians(theta)))) - np.prod(r_o_mag, np.cos(np.radians(theta)))

def calc_R_star_mag(R_w_mag, r_o_mag, z_ang):
    # takes the magnitudes of vectors R_w and r_o and the magnitude of angle z
    # returns the magnitude of R_star
    # TODO: add theta input mode
        # takes z_ang or theta as the fourth input

    return np.sqrt(R_w_mag**2 + r_o_mag**2 - 2 * R_w_mag * r_o_mag * np.cos(np.radians(z_ang)))

def calc_C_dot_mag(R_w_mag, r_w_mag, r_o_mag, theta):
    return (r_o_mag * np.sin(np.radians(theta)) - \
        ((r_o_mag**2 * np.sin(np.radians(theta)) * np.cos(np.radians(theta))) / \
            np.sqrt((R_w_mag + r_w_mag)**2 - r_o_mag**2 * np.sin(np.radians(theta))**2)))

def calc_C_ddot_mag(R_w_mag, r_w_mag, r_o_mag, theta):
    return (((r_o_mag**2 * np.sin(np.radians(theta))**2) / \
        np.sqrt((R_w_mag + r_w_mag)**2 - \
            r_o_mag**2 * np.sin(np.radians(theta))**2)) - \
                ((r_o_mag**2 * (R_w_mag + r_w_mag)**2 * np.cos(np.radians(theta))**2) / \
                    np.power(((R_w_mag + r_w_mag)**2 - \
                        r_o_mag**2 * np.sin(np.radians(theta))**2), 3/2)) + \
                            r_o_mag * np.cos(np.radians(theta)))

# Kinematic vector calculations
def origin_vec():
    # returns the origin vector

    return np.array([0, 0])

def calc_C_vec(C_mag):
    # takes the magnitude of the vector C
    # returns the vector C

    return np.column_stack((np.zeros(np.size(C_mag)), C_mag))

def calc_R_star_vec(r_o_vec, R_w_vec):
    # takes the vectors r_o and R_w
    # returns the R_star vector

    return r_o_vec + R_w_vec

def calc_r_o_vec(r_o_mag, theta):
    # takes the magnitude of r_o and the angle theta
    # returns the r_o vector

    return r_o_mag * np.column_stack((np.sin(np.radians(theta)), -np.cos(np.radians(theta))))

def calc_R_w_vec(R_w_mag, theta, z_ang):
    # takes the magnitude of the vector R_w and the angles theta and z
    # returns the vector R_w
    # TODO: add theta input mode
        # z_ang optional (will calculate if not included)

    return R_w_mag * np.column_stack((-np.sin(np.radians(calc_chi_ang(theta, z_ang))), np.cos(np.radians(calc_chi_ang(theta, z_ang)))))

def calc_r_w_vec(r_w_mag, theta, z_ang):
    # takes the magnitude of the vector r_w and the angles theta and z
    # returns the vector r_w
    # TODO: add theta input mode
        # z_ang optional (will calculate if not included)

    return r_w_mag * np.column_stack((np.sin(np.radians(calc_chi_ang(theta, z_ang))), -np.cos(np.radians(calc_chi_ang(theta, z_ang)))))

def calc_v_mag(R_w_mag, r_w_mag, r_o_mag, theta, omega):
    # takes the vectors R_w_mag, r_w_mag, r_o_mag, and 
    # system angle, theta, and the motor speed, omega
    # returns the cam follower velocity
    C_dot_mag = calc_C_dot_mag(R_w_mag, r_w_mag, r_o_mag, theta)
    return C_dot_mag * (omega * (2*np.pi/360))

def calc_a_mag(R_w_mag, r_w_mag, r_o_mag, theta, omega, alpha):
    # takes the vectors R_w_mag, r_w_mag, r_o_mag, and 
    # system angle, theta, and the motor speed, omega
    # and motor acceleration, alpha
    # returns the cam follower velocity

    C_dot_mag = calc_C_dot_mag(R_w_mag, r_w_mag, r_o_mag, theta)
    C_ddot_mag = calc_C_ddot_mag(R_w_mag, r_w_mag, r_o_mag, theta)  
    return C_ddot_mag * (omega * (2*np.pi/360))**2 + C_dot_mag * alpha

# Angle calculations
def calc_z_angle(R_w_mag, r_w_mag, r_o_mag, C_mag):
    # takes the magnitudes of vectors R_w, r_w, r_o, and C
    # returns the angle z in degrees
    # TODO add theta input mode
        # takes either C_mag or theta as the fourth input, returns the angle z in degrees

    return np.degrees(np.arccos(((R_w_mag + r_w_mag)**2 + r_o_mag**2 - C_mag**2) / (2 * (R_w_mag + r_w_mag) * r_o_mag)))
    # return np.arccos(np.divide(np.square(np.sum(R_w_mag, r_w_mag)) + np.square(r_o_mag) - np.square(C_mag), np.prod(2, np.sum(R_w_mag, r_w_mag), r_o_mag)))

def calc_chi_ang(theta, z_ang):
    return np.where([theta <= 180], theta - z_ang, (theta + z_ang) - 360)[0]

# Calculate general vector properties
def calc_vec_mag(input_vec):
    # computes the magnitude of the input vector
    # takes a numpy array, tuple, or list as an input vector
    # returns the magnitude of the input

    return np.sqrt(np.sum(np.square(input_vec)))

def calc_unit_vec(input_vec):
    # takes a numpy array, tuple, or list as an input vector
    # returns the unit vector of the input

    return np.array(np.divide(input_vec, calc_vec_mag(input_vec)))

def calc_cw_orth(input_vec):
    # computes the orthogonal vector resulting from a negative (cw) 90deg rotation of the input vector
    # takes a numpy array, tuple, or list as an input vector
    # returns the input rotated clock-wise 90 degrees as a numpy array

    output_vec = np.zeros([100, 2])
    output_vec[:, 0], output_vec[:, 1] = input_vec[:, 1], -input_vec[:, 0]
    return output_vec

def calc_vec_ang(input_vec):
    # computes the angle (from the positive x-axis) of the input vector
    # takes a numpy array, tuple, or list as an input vector
    # returns the angle of the input vector in degrees

    return np.degrees(np.arctan(input_vec[1]/input_vec[0]))

# Calculate eccentric wheel position
def calc_ecc_wheel(r_o_vec, R_w_mag):
    # takes the r_o vector and the magnitude of vector R_w
    # returns a wheel object
    # TODO: create theta input mode
        # take either the r_o_vector or theta

    return plt.Circle(r_o_vec, R_w_mag, fill = False, label='eccentric wheel')

# FORCES AND TORQUES
# Force on eccentric wheel due to gravity
def calc_eccen_force_mag(m_eccen, g):
    return m_eccen * g

# Normal / contact force
def calc_norm_force_mag(m_load, g, accel, spring_force, theta, z_ang):
    omega_ang = 180 - (180 - theta) - z_ang
    omega_ang = omega_ang[0:omega_ang.size//2]
    omega_ang = np.append(omega_ang, omega_ang)
    return (m_load * (g + accel) + spring_force) / np.cos(np.radians(omega_ang))

# Friction force
def calc_fric_force_mag(mu, normal_force_mag):
    return mu * normal_force_mag

# Torque due to rotational inertia
def calc_inertial_torq_mag(i_motor, i_wheel, alpha):
    # computes the torque load due to rotational inertia
    return (i_motor + i_wheel)*alpha

# Spring force
def calc_spring_force(k_spring, preload, C_mag):
    return k_spring * (C_mag - C_mag[0]) + preload

# Eccentric gravity force vector
def calc_eccen_force_vec(eccen_force_mag):
    return np.array([0, -eccen_force_mag])

def calc_norm_force_vec(norm_force_mag, r_w_vec):
    output_vec = np.zeros([100, 2])
    output_vec[:, 0], output_vec[:, 1] = norm_force_mag * calc_unit_vec(r_w_vec)[:, 0], norm_force_mag * calc_unit_vec(r_w_vec)[:, 1]
    return output_vec

def calc_fric_force_vec(fric_force_mag, R_w_vec):
    output_vec = np.zeros([100, 2])
    output_vec[:, 0], output_vec[:, 1] = fric_force_mag * calc_cw_orth(calc_unit_vec(R_w_vec))[:, 0], fric_force_mag * calc_cw_orth(calc_unit_vec(R_w_vec))[:, 1]
    return output_vec

# Z_ang range calculator
def calc_z_ang_range(R_w_mag, r_o_mag, r_w_mag, theta):
    C_mag = calc_C_mag(R_w_mag, r_w_mag, r_o_mag, theta)
    z_ang = calc_z_angle(R_w_mag, r_w_mag, r_o_mag, C_mag)

    return z_ang

# Vector state calculator
def calc_vector_state(R_w_mag, r_o_mag, r_w_mag, theta):
    # takes the magnitudes of vectors R_w, r_o, and r_w and the angle theta
    # returns vectors C, R_star, r_o, R_w, and r_w
    
    C_mag = calc_C_mag(R_w_mag, r_w_mag, r_o_mag, theta)
    z_ang = calc_z_angle(R_w_mag, r_w_mag, r_o_mag, C_mag)

    C_vec = calc_C_vec(C_mag)
    r_o_vec = calc_r_o_vec(r_o_mag, theta)
    R_w_vec = calc_R_w_vec(R_w_mag, theta, z_ang)
    R_star_vec = calc_R_star_vec(r_o_vec, R_w_vec)
    r_w_vec = calc_r_w_vec(r_w_mag, theta, z_ang)

    return C_vec, R_star_vec, r_o_vec, R_w_vec, r_w_vec

# Kinematic state calculator
def calc_knmtc_state(R_w_mag, r_o_mag, r_w_mag, theta, omega):
    # returns position, velocity, and acceleration of cam follower (alpha = 0)
    
    C_mag = calc_C_mag(R_w_mag, r_w_mag, r_o_mag, theta)
    z_ang = calc_z_angle(R_w_mag, r_w_mag, r_o_mag, C_mag)

    r_o_vec = calc_r_o_vec(r_o_mag, theta)
    R_w_vec = calc_R_w_vec(R_w_mag, theta, z_ang)
    
    R_star_y = calc_R_star_vec(r_o_vec, R_w_vec)[:, 1]
    velocity = calc_v_mag(R_w_mag, r_w_mag, r_o_mag, theta, omega)
    acceleration = calc_a_mag(R_w_mag, r_w_mag, r_o_mag, theta, omega, 0)

    return R_star_y, velocity, acceleration

# Kinetic state calculator
def calc_kntc_force_mag_state(i_motor, i_wheel, alpha, m_eccen, m_load, g, accel, spring_force, theta, z_ang, mu):
    inertial_torq = calc_inertial_torq_mag(i_motor, i_wheel, alpha)
    weight_mag = calc_eccen_force_mag(m_eccen, g)
    norm_force_mag = calc_norm_force_mag(m_load, g, accel, spring_force, theta, z_ang)
    fric_force_mag = calc_fric_force_mag(mu, norm_force_mag)

    return inertial_torq, weight_mag, norm_force_mag, fric_force_mag

def calc_kntc_force_vec_state(inertial_torq, weight_mag, norm_force_mag, fric_force_mag, r_o_vec, R_star_vec):
    weight_vec = calc_eccen_force_vec(weight_mag)
    norm_force_vec = calc_norm_force_vec(norm_force_mag, r_w_vec)
    fric_force_vec = calc_fric_force_vec(fric_force_mag, R_w_vec)

    return inertial_torq, weight_vec, norm_force_vec, fric_force_vec

def calc_kntc_moment_state(inertial_torq, weight_vec, norm_force_vec, fric_force_vec, r_o_vec, R_star_vec):
    weight_moment = np.cross(r_o_vec, weight_vec)
    norm_moment = np.cross(R_star_vec, norm_force_vec)
    fric_moment = np.cross(R_star_vec, fric_force_vec)

    return intertial_torq, weight_moment, norm_moment, fric_moment

# Arrow transformation
def transform_arrow(arrow, x, y, dx, dy):
    # takes pyplot Arrow object and new coordinates for x, y, dx, and dy
    # transforms the Arrow object

        L = np.hypot(dx, dy)

        if L != 0:
            cx = dx / L
            sx = dy / L
        else:
            # Account for division by zero
            cx, sx = 0, 1

        trans1 = transforms.Affine2D().scale(L, arrow.get_linewidth())
        trans2 = transforms.Affine2D.from_values(cx, sx, -sx, cx, 0.0, 0.0)
        trans3 = transforms.Affine2D().translate(x, y)
        trans = trans1 + trans2 + trans3
        arrow._patch_transform = trans.frozen()

# Set dimensional magnitudes
R_w_mag = 14
r_o_mag = 8
r_w_mag = 6

# set wheel radius and offset radius
wheel_r_mag = 15
offset_r_mag = wheel_r_mag - 4

# set omega
omega = 1666*360/60

# set theta interval
delta_theta = 360*0.01

# calculate framerate
framerate = omega / delta_theta

# generate theta values
theta_vals = np.arange(0, 360, delta_theta)

# calculate the vectors C, R_star, r_o, R_w, and r_w 
C_vec, R_star_vec, r_o_vec, R_w_vec, r_w_vec = calc_vector_state(R_w_mag, r_o_mag, r_w_mag, theta_vals)

# calculate the position, velocity, and acceleration of the cam follower relative to the motor shaft
pos_all, vel_all, accel_all = calc_knmtc_state(R_w_mag, r_o_mag, r_w_mag, theta_vals, omega)
accel_all = accel_all / 1000

# calculate initial vector objects
C = plt.Arrow(*origin_vec(), *C_vec[0])
R_star = plt.Arrow(*origin_vec(), *R_star_vec[0])
r_o = plt.Arrow(*origin_vec(), *r_o_vec[0])
R_w = plt.Arrow(*r_o_vec[0], *R_w_vec[0])
r_w = plt.Arrow(*C_vec[0], *r_w_vec[0])
wheel = calc_ecc_wheel(r_o_vec[0], R_w_mag)

# Create figure and axes objects for the wheel and kinematic plots
fig, (wheel_ax, knmtc_ax, torq_ax) = plt.subplots(1, 3)
fig.set_size_inches(12, 8)
knmtc_ax_2 = knmtc_ax.twinx()

# determine appropriate axes limits for the wheel plot
wheel_ax_lim = (wheel_r_mag + offset_r_mag) * 1.1
wheel_ax_y_max = wheel_ax_lim * 1.5 / 1.1

# set the axes limits for the wheel and kinematic plots
wheel_ax.set(xlim=(-wheel_ax_lim, wheel_ax_lim), ylim=(-wheel_ax_lim, wheel_ax_lim))
wheel_ax.set(aspect='equal')

# add initial vector drawings
wheel_ax.add_artist(C)
wheel_ax.add_artist(R_star)
wheel_ax.add_artist(r_o)
wheel_ax.add_artist(R_w)
wheel_ax.add_artist(r_w)
wheel_ax.add_artist(wheel)

# add initial kinematic plots
pos_plot = knmtc_ax_2.plot(theta_vals, pos_all, linestyle='--', label='paddle position [mm]')[0]
vel_plot = knmtc_ax.plot(theta_vals, vel_all, label='paddle velocity [mm/s]')[0]
accel_plot = knmtc_ax_2.plot(theta_vals, accel_all, label='paddle acceleration [m/s^2]')[0]

knmtc_ax.legend(loc='lower left')
knmtc_ax_2.legend(loc='upper left')

def animate(i):

    # animate vectors
    transform_arrow(C, *origin_vec(), *C_vec[i])
    transform_arrow(R_star, *origin_vec(), *R_star_vec[i])
    transform_arrow(r_o, *origin_vec(), *r_o_vec[i])
    transform_arrow(R_w, *r_o_vec[i], *R_w_vec[i])
    transform_arrow(r_w, *C_vec[i], *r_w_vec[i])

    # animate wheel
    wheel.set_center(r_o_vec[i])

anim = animation.FuncAnimation(fig, animate, frames = len(theta_vals), \
    interval = 1000/framerate, repeat = True)

k_spring = 970 # mN/mm
spring_preload = 100 # mN
C_mag = calc_C_mag(R_w_mag, r_w_mag, r_o_mag, theta_vals)
spring_force = calc_spring_force(k_spring, spring_preload, C_mag)
print(spring_force.max())

m_load = 40
m_eccen = 20
mu = 0.30

z_ang = calc_z_ang_range(R_w_mag, r_o_mag, r_w_mag, theta_vals)

inertial_torq_mag, weight_mag, norm_force_mag, fric_force_mag = calc_kntc_force_mag_state(0, 0, np.zeros(theta_vals.size), m_eccen, m_load, 9.81, accel_all, spring_force, theta_vals, z_ang, mu)
intertial_torq, weight_vec, norm_force_vec, fric_force_vec = calc_kntc_force_vec_state(inertial_torq_mag, weight_mag, norm_force_mag, fric_force_mag, r_o_vec, R_star_vec)
intertial_torq, weight_moment, norm_force_moment, fric_force_moment = calc_kntc_moment_state(intertial_torq, weight_vec, norm_force_vec, fric_force_vec, r_o_vec, R_star_vec)

motor_torq = intertial_torq - weight_moment - norm_force_moment - fric_force_moment

torq_ax.plot(theta_vals, motor_torq / 98.0665, label='required torque [g-cm]')
torq_ax.legend()

req_spring_force_index = norm_force_mag.argmin()
req_spring_force = norm_force_mag[req_spring_force_index]
spring_force_ang = theta_vals[req_spring_force_index]
spring_force_label = '{0} mN needed at {1} degrees'.format(int(-req_spring_force), int(spring_force_ang))

fig2, spring_ax = plt.subplots(1)
spring_ax.plot(theta_vals, norm_force_mag, label='normal force mag [mN]')
spring_ax.plot(spring_force_ang, req_spring_force, 'x', label=spring_force_label)
spring_ax.legend()

plt.show()

# save animation
# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=30, bitrate=1800)
# anim.save('Kinematics2.mp4', writer=writer)