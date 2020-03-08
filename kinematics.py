import numpy as np

def calc_var_r_mag(wheel_r_mag, offset_r_mag, theta):
    # takes the magnitudes of the wheel radius, R, and the offset radius, r, and the current eccentric wheel angle (positive = ccw from BDC)
    # returns the magnitude of the variable radius, R*, as a double

    return (np.sqrt(wheel_r_mag**2 - offset_r_mag**2 * np.sin(theta)**2) - offset_r_mag*np.cos(theta))

def calc_var_r_vec(var_r_mag, theta):
    # takes the magnitude of the variable radius, R*, and the current eccentric wheel angle (positive = ccw from BDC)
    # returns a vector (numpy array) of the variable radius, R*, in [x, y] form

    return np.array([0, var_r_mag])

def calc_offset_r_vec(offset_r_mag, theta):
    # takes the magnitude of the offset radius, r, and the current eccentric wheel angle (positive = ccw from BDC)
    # returns a vector (numpy array) of offset radius, r, in [x, y] form

    return np.array([offset_r_mag*np.sin(theta), -offset_r_mag*np.cos(theta)])

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

def calc_y(wheel_r_mag, offset_r_mag, theta, omega):
    return calc_var_r_mag(wheel_r_mag, offset_r_mag, theta) # - calc_var_r_mag(wheel_r_mag, offset_r_mag, 0)

def calc_y_dot(wheel_r_mag, offset_r_mag, theta, omega):
    # returns the paddle velocity (y dot) in mm/s
    return (offset_r_mag*np.sin(theta) * \
        (1 - ((offset_r_mag*np.cos(theta))/(np.sqrt(wheel_r_mag**2 - offset_r_mag**2*np.sin(theta)**2))))) * \
            omega

def calc_y_ddot(wheel_r_mag, offset_r_mag, theta, omega):
    # returns the paddle acceleration (y double dot) in mm/s^2
    return (offset_r_mag*np.cos(theta) * \
        (1 - ((offset_r_mag*np.cos(theta))/(np.sqrt(wheel_r_mag**2 - offset_r_mag**2*np.sin(theta)**2)))) + \
            offset_r_mag*np.sin(theta) * \
        ((offset_r_mag*np.sin(theta))/(np.sqrt(wheel_r_mag**2 - offset_r_mag**2*np.sin(theta)**2)) - \
            (offset_r_mag**3*np.sin(theta)*np.cos(theta)**2)/(wheel_r_mag**2 - offset_r_mag**2*np.sin(theta)**2)**(3/2))) * \
                omega

