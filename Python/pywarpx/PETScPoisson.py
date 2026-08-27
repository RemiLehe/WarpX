# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

from .Bucket import Bucket

# Options of the PETSc Krylov solver that is used for the Poisson equation
# when warpx.poisson_solver = petsc (requires compiling with -DWarpX_PETSC=ON)
petsc_poisson = Bucket("petsc_poisson")
