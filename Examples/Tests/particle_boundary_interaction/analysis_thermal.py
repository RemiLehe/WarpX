#!/usr/bin/env python
"""
This script tests the ``Thermal`` value of ``boundary.particle_eb``.

A cold, low-density beam of electrons is launched at a planar embedded boundary
(a wall at x = x_wall, with the physics region at x < x_wall). Each electron
that reaches the wall is re-emitted with a thermalized velocity, as from a fully
accommodating diffuse wall: the two components tangential to the wall (y, z) are
sampled from a Gaussian of standard deviation u_th, and the component along the
inward normal (-x) from a Gaussian-flux (Rayleigh-like) distribution with the
same standard deviation.

By the end of the simulation all electrons have re-emitted (and none has had time
to reach the far x boundary), so the recorded velocity distribution is entirely
set by the thermal boundary. This script checks that:
  - all electrons move away from the wall (ux < 0),
  - the tangential components uy, uz are zero-mean Gaussians of std u_th,
  - the normal component |ux| follows a Gaussian-flux distribution, whose mean is
    u_th * sqrt(pi/2).
Here (ux, uy, uz) are the proper velocities gamma*v normalized by c, which is the
convention in which u_th is expressed.
"""

import sys

sys.path.append("../../../Tools/Parser/")

import numpy as np
from input_file_parser import parse_input_file
from openpmd_viewer import OpenPMDTimeSeries

# Open plotfile specified in command line
filename = sys.argv[1]
ts = OpenPMDTimeSeries(filename)

# Read the thermal velocity used by the boundary from the input file, so that the
# reference values below automatically follow any change of the input parameter.
input_dict = parse_input_file("./warpx_used_inputs")
uth = float(input_dict["my_constants.uth_e"][0])

# Velocities of the re-emitted electrons at the final iteration. openPMD-viewer
# returns ux, uy, uz as the proper velocity gamma*v normalized by c.
it = ts.iterations
ux, uy, uz = ts.get_particle(["ux", "uy", "uz"], species="electrons", iteration=it[-1])

n = ux.size
print(f"Number of re-emitted electrons: {n}")
assert n > 1000, "Not enough particles for a meaningful statistical test"

# All electrons must move away from the wall, i.e. along the inward normal (-x).
frac_into_domain = np.mean(ux < 0.0)
print(f"Fraction of electrons moving into the domain (ux < 0): {frac_into_domain:.4f}")
assert frac_into_domain > 0.999

# Statistical moments of the re-emitted velocity distribution.
mean_abs_ux = np.mean(np.abs(ux))
expected_mean_abs_ux = uth * np.sqrt(np.pi / 2.0)  # mean of a Gaussian-flux dist.
std_uy = np.std(uy)
std_uz = np.std(uz)
mean_uy = np.mean(uy)
mean_uz = np.mean(uz)

print(f"mean(|ux|) = {mean_abs_ux:.6e} (expected {expected_mean_abs_ux:.6e})")
print(f"std(uy)    = {std_uy:.6e} (expected {uth:.6e})")
print(f"std(uz)    = {std_uz:.6e} (expected {uth:.6e})")
print(f"mean(uy)   = {mean_uy:.6e} (expected 0)")
print(f"mean(uz)   = {mean_uz:.6e} (expected 0)")

# Generous tolerances: the statistical error on these moments is well below 2% for
# the ~1e4 particles used here, so a 15% tolerance leaves a large safety margin.
tol = 0.15
assert abs(mean_abs_ux - expected_mean_abs_ux) / expected_mean_abs_ux < tol
assert abs(std_uy - uth) / uth < tol
assert abs(std_uz - uth) / uth < tol
assert abs(mean_uy) < 0.1 * uth
assert abs(mean_uz) < 0.1 * uth

print("\nTest particle_boundary_interaction_thermal_eb passed")
