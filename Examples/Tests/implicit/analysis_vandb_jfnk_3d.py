#!/usr/bin/env python3

# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL
#
# This is a script that analyses the simulation results from the script
# `inputs_test_3d_theta_implicit_jfnk_vandb`.
# This simulates a 3D periodic uniform plasma using the theta-implicit
# solver with direct deposition and shape factor 2, using the full mass
# matrices for the Jacobian and the curl-curl preconditioner with the
# diagonal response from the mass matrices.

import numpy as np

newton_solver = np.loadtxt("diags/reduced_files/newton_solver.txt", skiprows=1)
num_steps = newton_solver[-1, 0]
total_newton_iters = newton_solver[-1, 3]
total_gmres_iters = newton_solver[-1, 7]

field_energy = np.loadtxt("diags/reduced_files/field_energy.txt", skiprows=1)
particle_energy = np.loadtxt("diags/reduced_files/particle_energy.txt", skiprows=1)

total_energy = field_energy[:, 2] + particle_energy[:, 2]

delta_E = (total_energy - total_energy[0]) / total_energy[0]
max_delta_E = np.abs(delta_E).max()

# This case should have near machine precision conservation of energy
tolerance_rel_energy = 2.0e-14

print(f"max change in energy: {max_delta_E}")
print(f"tolerance: {tolerance_rel_energy}")

assert max_delta_E < tolerance_rel_energy

# check that the number of gmres iterations per newton iteration is below
# tolerance; this is sensitive to the quality of the mass-matrices-based
# preconditioner
gmres_iters_tol = 5.0
print(f"gmres iters per newton: {total_gmres_iters / total_newton_iters}")
print(f"gmres iters tolerance: {gmres_iters_tol}")
assert total_gmres_iters / total_newton_iters <= gmres_iters_tol

# check that the number of newton iterations per step is below tolerance;
# this is sensitive to the quality of the mass-matrices-based Jacobian
newton_iters_tol = 10.0
print(f"newton iters per time step: {total_newton_iters / num_steps}")
print(f"newton iters tolerance: {newton_iters_tol}")
assert total_newton_iters / num_steps <= newton_iters_tol
