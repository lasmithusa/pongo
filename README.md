# eccentric-wheel-drive
Collection of tools used in the design of a 2019 oscillating drive mechanism project meant for striking a bouncing ball.

Included modules:

Eccentric drive concepts -
* eccentric.py - Generates an eccentric wheel animation (see eccentric.mp4 for example output).
* eccentric2.py - Generates an eccentric wheel animation and plots showing the kinematic and kinetic behavior of the system (see eccentric2.mp4 for example output).
  * Kinematic plot shows output shaft position, velocity, and acceleration
  * Kinetic plot shows required torque given the friction factor between output shaft and drive wheel and torque due to wheel acceleration

Sprung paddle concepts -
* powerspring.py - Generates contour plots showing required spring rate and max force for given spring "charge" and "contact" deltas
  * "charge" delta = difference in spring length between the relaxed position and fully compressed position
  * "contact" delta = difference in length between the relaxed position and ball contact position

Helper functions -
* kinetics.py - Includes helper functions for performing kinetics calculations
* kinematics.py - Includes helper functions for performing kinematics calculations
