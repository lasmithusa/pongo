import numpy as np
import matplotlib.pyplot as plt

# set sprung system masses [grams]
mass_load = 35
mass_spring = 0

# set required paddle velocity [mm/s]
vel_paddle = 1600

# generate range of delta_r's and delta_o's [mm]
delta_r = np.linspace(8, 30, 10)

delta_o = np.linspace(4, 8, 10)

print(delta_r)

# generate range of l_r's, l_o's, and l's [mm]
max_l_r = 60
l_r_step = 1
l_r = np.arange(0, max_l_r + l_r_step, l_r_step, dtype='float64')

max_l_o = 20
l_o_step = max_l_o / l_r.size
l_o = np.arange(0, max_l_o + l_o_step, l_o_step, dtype='float64')

max_l = 20
l_step = max_l / l_r.size
l = np.arange(0, max_l + l_step, l_step, dtype='float64')

# calculate required k_spring [mN/m] provided a range of spring extensions, delta_r and delta_o
def calc_k_spring_deltas(mass_load, mass_spring, vel_paddle, delta_r, delta_o):
    #
    return ((mass_load + mass_spring) * vel_paddle**2) / \
        (delta_r_mesh**2 - delta_o_mesh**2)

# calculate required k_spring [mN/m] provided a range of unstretched, preload, and release spring lengths l, l_o, and l_r
def calc_k_spring_lengths(mass_load, mass_spring, vel_paddle, l_r, l_o, l):
    # 
    return ((mass_load + mass_spring) * vel_paddle**2) / \
        ((l_r - l)**2 - (l_o - l)**2)

# calculate required potential delta [mN-m] to produce necessary kinetic energy
def calc_potential_delta(mass_load, mass_spring, vel_paddle):
        return (1/2 * ((mass_load + mass_spring) * vel_paddle**2)) / (10**6)

# convert from mN/m to lbs/in and lbs/mm (more common spring rate units)
def mNPerM_to_lbsPerIn(mNPerM):
    return mNPerM / 175126.929214

def mNPerM_to_lbsPerMm(mNPerM):
    return mNPerM / 4448221.6

# generate X and Y grids via delta_o and delta_r ranges
delta_o_mesh, delta_r_mesh = np.meshgrid(delta_o, delta_r)
# calculate k_spring [mN/m] (Z value on 2D contour, color value on 3D contour) at each system configuration
k_spring = calc_k_spring_deltas(mass_load, mass_spring, vel_paddle, delta_r_mesh, delta_o_mesh)

# replace non-finite, non-positive values with NaNs
k_spring[np.invert(np.isfinite(k_spring))] = np.nan
k_spring[k_spring < 0] = np.nan

# convert spring rate to lbs/in
k_spring = mNPerM_to_lbsPerIn(k_spring)

# calculate max spring force in lbs
mm_to_in = 1 / 25.4
max_force = k_spring * delta_r_mesh * mm_to_in

# generate contour levels
rate_levels = np.append(np.arange(0, 1, 0.05), np.logspace(0, 15, 20, base=1.2))

# create figure and axis objects
fig = plt.figure()
ax = fig.add_subplot(111)

# generate gradient plot
plt.imshow(k_spring, extent=[delta_o.min(), delta_o.max(), delta_r.min(), delta_r.max()], origin='lower', cmap='RdGy')
plt.colorbar()

# generate contour plot
cplot = plt.contour(delta_o_mesh, delta_r_mesh, k_spring, levels=rate_levels, colors='black')
plt.clabel(cplot, inline=True, fontsize=8)

# set title and axis labels
plt.title('Required K_spring [lbs/in] vs Release and Preload Spring Extensions')
plt.xlabel('delta_o [mm]')
plt.ylabel('delta_r [mm]')

# set aspect ratio
ax.set_aspect(aspect='auto')

# generate contour levels
force_levels = np.append(np.arange(0, 1, 0.05), np.logspace(0, 8, 15, base=1.2))

# create figure and axis objects
fig = plt.figure()
ax = fig.add_subplot(111)

# generate gradient plot
plt.imshow(max_force, extent=[delta_o.min(), delta_o.max(), delta_r.min(), delta_r.max()], origin='lower', cmap='RdGy')
plt.colorbar()

# generate contour plot
cplot = plt.contour(delta_o_mesh, delta_r_mesh, max_force, levels=force_levels, colors='black')
plt.clabel(cplot, inline=True, fontsize=8)

# set title and axis labels
plt.title('Max Spring Force [lbs] vs Release and Preload Spring Extensions')
plt.xlabel('delta_o [mm]')
plt.ylabel('delta_r [mm]')

# set aspect ratio
ax.set_aspect(aspect='auto')

# print required potential energy delta ([mN-m] converted to [lbs-in])
print(calc_potential_delta(mass_load, mass_spring, vel_paddle) / 112.985)

# show plots
plt.show()

# l = 38.1
# l_o = l - delta_o
# l_r = l - delta_r
