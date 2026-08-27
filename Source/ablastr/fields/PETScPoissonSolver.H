/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef ABLASTR_FIELDS_PETSC_POISSON_SOLVER_H
#define ABLASTR_FIELDS_PETSC_POISSON_SOLVER_H

#include <AMReX_Config.H>
#include <AMReX_Geometry.H>
#include <AMReX_MLMG.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>


namespace ablastr::fields {

/** Settings of the PETSc Krylov solver that can be used for the Poisson equation
 *
 * This is the electrostatic counterpart of the PETSc interface that the implicit
 * electromagnetic solvers use for the curl-curl equation (@see PETScKSP in
 * `Source/NonlinearSolvers/PETScKSP_Wrapper.H`).
 */
struct PETScPoissonOptions
{
    /** Solve the Poisson equation with PETSc's GMRES instead of MLMG */
    bool use_petsc_ksp = false;

    /** Use MLMG V-cycles as the preconditioner (through PETSc's `PCShell`).
     *  When false, GMRES runs unpreconditioned, which is usually much slower
     *  and is mostly useful for debugging.
     */
    bool use_mlmg_preconditioner = true;

    /** Number of MLMG V-cycles per application of the preconditioner */
    int precond_num_iters = 1;

    /** Verbosity: 0 is silent, 1 prints the exit status of every solve,
     *  2 additionally prints the residual at every Krylov iteration.
     */
    int verbosity = 0;
};

#ifdef AMREX_USE_PETSC

/** \brief Solve the Poisson equation of one MR level with PETSc's GMRES
 *
 * The linear system that is solved is the very same one that `mlmg` would
 * solve: the operator is handed to PETSc as a matrix-free `MatShell` whose
 * action is computed by the AMReX linear operator, and the AMReX multigrid
 * V-cycles are handed to PETSc as a right `PCShell` preconditioner. This
 * mirrors `amrex::GMRESMLMG`, with PETSc's GMRES in place of the AMReX GMRES
 * implementation. Further PETSc customization (e.g. `-ksp_type`) is available
 * through PETSc's own runtime options, which take precedence.
 *
 * As in `amrex::GMRESMLMG`, GMRES does not iterate on the potential itself but
 * on a correction: the residual of the initial guess is computed with the
 * inhomogeneous operator, so that non-zero Dirichlet boundary values stored in
 * `phi` are accounted for, and the correction vanishes on the Dirichlet nodes.
 *
 * \param[in] mlmg the multigrid solver that defines the linear operator; it is
 *                 used both for the operator and for the preconditioner
 * \param[inout] phi on input the initial guess, which must hold the Dirichlet
 *                   boundary values; on output the computed potential
 * \param[in] rho the (already scaled) right-hand side
 * \param[in] geom the geometry of the MR level that is solved
 * \param[in] relative_tolerance the relative convergence threshold
 * \param[in] absolute_tolerance the absolute convergence threshold
 * \param[in] max_iters the maximum number of Krylov iterations
 * \param[in] options the settings of the PETSc solver
 */
void
petscPoissonSolve (amrex::MLMG & mlmg,
                   amrex::MultiFab & phi,
                   amrex::MultiFab const & rho,
                   amrex::Geometry const & geom,
                   amrex::Real relative_tolerance,
                   amrex::Real absolute_tolerance,
                   int max_iters,
                   PETScPoissonOptions const & options);

#endif // AMREX_USE_PETSC

} // namespace ablastr::fields

#endif // ABLASTR_FIELDS_PETSC_POISSON_SOLVER_H
