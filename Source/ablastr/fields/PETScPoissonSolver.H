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
#include <AMReX_INT.H>
#include <AMReX_MLMG.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>
#include <AMReX_iMultiFab.H>

#include <memory>
#include <string>


namespace ablastr::fields {

/** Settings of the PETSc Krylov solver that can be used for the Poisson equation
 *
 * This is the electrostatic counterpart of the PETSc interface that the implicit
 * electromagnetic solvers use for the curl-curl equation (@see PETScKSP in
 * `Source/NonlinearSolvers/PETScKSP_Wrapper.H`): the linear system is handed to a
 * PETSc `KSP` object as a matrix-free operator, and the AMReX geometric multigrid
 * is used as a preconditioner through PETSc's `PCShell` interface.
 */
struct PETScPoissonOptions
{
    /** Solve the Poisson equation with a PETSc Krylov solver instead of MLMG */
    bool use_petsc_ksp = false;

    /** PETSc `KSP` type, e.g. "gmres", "fgmres", "bcgs", "cg" */
    std::string ksp_type = "gmres";

    /** Restart length, only used by the GMRES variants (PETSc default: 30) */
    int restart_length = 30;

    /** Use MLMG V-cycles as the preconditioner (through PETSc's `PCShell`).
     *  When false, the Krylov solver runs unpreconditioned, which is usually
     *  much slower and is mostly useful for debugging.
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

// The PETSc objects are wrapped in these structs, which are only defined in
// PETScPoissonSolver.cpp, so that the PETSc headers do not have to be included
// here (they conflict with some of the WarpX and AMReX names).
namespace petsc_poisson
{
    struct KSPObj;
    struct MatObj;
    struct VecObj;
}

/** \brief Solve the Poisson equation of one MR level with a PETSc Krylov solver
 *
 * The linear system that is solved is the very same one that `amrex::MLMG` would
 * solve, i.e. it is defined by the `amrex::MLNodeLinOp` that the `amrex::MLMG`
 * object passed to the constructor was built from. The operator is never
 * assembled into a sparse matrix: it is passed to PETSc as a `MatShell` whose
 * action is computed by the AMReX linear operator, and the AMReX multigrid
 * V-cycles are passed to PETSc as a `PCShell`. This mirrors `amrex::GMRESMLMG`,
 * with PETSc's Krylov solvers in place of the AMReX GMRES implementation.
 *
 * As in `amrex::GMRESMLMG`, the Krylov solver does not iterate on the potential
 * itself but on a correction: the residual of the initial guess is computed with
 * the inhomogeneous operator, so that non-zero Dirichlet boundary values stored
 * in `phi` are accounted for, and the correction that the Krylov solver computes
 * vanishes on the Dirichlet nodes.
 *
 * The degrees of freedom of the PETSc vectors are the nodes that are owned by a
 * box (nodes shared between boxes, or with a periodic image, appear only once)
 * and that are not Dirichlet nodes.
 */
class PETScPoissonSolver
{
public:

    /** Construct the solver and create the PETSc objects
     *
     * \param[in] mlmg the multigrid solver that defines the linear operator; it
     *                 is used both for the operator and for the preconditioner,
     *                 and must outlive this object
     * \param[in] phi_prototype a nodal MultiFab with the layout of the unknowns
     * \param[in] geom the geometry of the MR level that is solved
     * \param[in] options the settings of the PETSc solver
     */
    PETScPoissonSolver (amrex::MLMG & mlmg,
                        amrex::MultiFab const & phi_prototype,
                        amrex::Geometry const & geom,
                        PETScPoissonOptions const & options);

    ~PETScPoissonSolver ();

    // Prohibit Move and Copy operations
    PETScPoissonSolver (PETScPoissonSolver const &) = delete;
    PETScPoissonSolver & operator= (PETScPoissonSolver const &) = delete;
    PETScPoissonSolver (PETScPoissonSolver &&) noexcept = delete;
    PETScPoissonSolver & operator= (PETScPoissonSolver &&) noexcept = delete;

    /** Solve the Poisson equation
     *
     * \param[inout] phi on input the initial guess, which must hold the Dirichlet
     *                   boundary values; on output the computed potential
     * \param[in] rho the (already scaled) right-hand side
     * \param[in] relative_tolerance the relative convergence threshold
     * \param[in] absolute_tolerance the absolute convergence threshold
     * \param[in] max_iters the maximum number of Krylov iterations
     */
    void solve (amrex::MultiFab & phi,
                amrex::MultiFab const & rho,
                amrex::Real relative_tolerance,
                amrex::Real absolute_tolerance,
                int max_iters);

    //! Number of Krylov iterations of the most recent solve
    [[nodiscard]] int getNumIters () const { return m_num_iters; }

    //! Residual norm of the most recent solve
    [[nodiscard]] amrex::Real getResidualNorm () const { return m_residual_norm; }

    /** Apply the linear operator: `out = L(in)`
     *
     * This is called back by PETSc, with `in` and `out` pointing to the local
     * part of the PETSc vectors.
     */
    void applyOperator (amrex::Real * out, amrex::Real const * in);

    /** Apply the multigrid preconditioner: `out = P^{-1}(in)`
     *
     * This is called back by PETSc, with `in` and `out` pointing to the local
     * part of the PETSc vectors.
     */
    void applyPreconditioner (amrex::Real * out, amrex::Real const * in);

private:

    //! Number the degrees of freedom that this MPI rank owns
    void buildDOFMap (amrex::MultiFab const & phi_prototype);

    //! Gather the degrees of freedom of `mf` into the PETSc array `arr`
    void copyToArray (amrex::MultiFab const & mf, amrex::Real * arr) const;

    //! Scatter the PETSc array `arr` into `mf`, and make `mf` consistent
    void copyFromArray (amrex::MultiFab & mf, amrex::Real const * arr) const;

    //! Multigrid solver that provides the operator and the preconditioner
    amrex::MLMG * m_mlmg = nullptr;
    //! Geometry of the MR level that is solved
    amrex::Geometry m_geom;
    //! Settings of the PETSc solver
    PETScPoissonOptions m_options;

    //! Local index of the degree of freedom of each node, -1 if it is not one
    std::unique_ptr<amrex::iMultiFab> m_dof;
    //! Number of degrees of freedom owned by this MPI rank
    amrex::Long m_ndofs_local = 0;
    //! Total number of degrees of freedom
    amrex::Long m_ndofs_global = 0;

    //! Work arrays for the operator (`_in` also holds the ghost nodes)
    amrex::MultiFab m_op_in, m_op_out;
    //! Work arrays for the preconditioner
    amrex::MultiFab m_pc_in, m_pc_out;
    //! Residual of the initial guess, and the correction that is solved for
    amrex::MultiFab m_res, m_cor;

    //! Matrix-free linear operator
    std::unique_ptr<petsc_poisson::MatObj> m_A;
    //! Solution vector
    std::unique_ptr<petsc_poisson::VecObj> m_x;
    //! Right-hand-side vector
    std::unique_ptr<petsc_poisson::VecObj> m_b;
    //! Krylov solver
    std::unique_ptr<petsc_poisson::KSPObj> m_ksp;

    //! Number of iterations of the most recent solve
    int m_num_iters = -1;
    //! Residual norm of the most recent solve
    amrex::Real m_residual_norm = amrex::Real(-1.0);
};

/** Solve the Poisson equation of one MR level with a PETSc Krylov solver
 *
 * This is a convenience wrapper that creates a @see PETScPoissonSolver, solves,
 * and destroys it again. The PETSc objects are cheap to create compared to the
 * `amrex::MLMG` object that `mlmg` refers to, which `computePhi` also recreates
 * at every call.
 *
 * \param[in] mlmg the multigrid solver that defines the linear operator
 * \param[inout] phi on input the initial guess (holding the Dirichlet boundary
 *                   values), on output the computed potential
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
