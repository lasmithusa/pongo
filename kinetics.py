import math
g = 9.81

def calc_force_w(mass_w):
    # convention - positive downward
    return mass_w*g

def calc_force_a(mass_w, accel):
    # convention - positive downward
    return mass_w*accel

def calc_force_b(mu_b, force_c):
    # convention - positive downward
    # opposes motion
    return mu*force_c

def calc_mu_b(y_dot):
    # takes the sliding speed in mm/s
        # preserve the sign of the speed
    # returns the coefficient of friction of Rulon J
    abs_y_dot = abs(y_dot)
    if abs_y_dot > 500:
        return -(y_dot/abs_y_dot)*(0.1641*abs_y_dot**0.0594)
    else:
        return -(y_dot/abs_y_dot)*0.0568*math.log(abs_y_dot) - 0.114

def calc_force_fric(mu, force_n):
    # takes the coefficient of friction of the direct drive interface
    # and the magnitude of the normal force
    # assumes motor rotation is always ccw
    return mu*force_n