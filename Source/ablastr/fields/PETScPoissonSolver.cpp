/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include <AMReX_Config.H>

#ifdef AMREX_USE_PETSC

#include <ablastr/profiler/ProfilerWrapper.H>
#include <ablastr/utils/TextMsg.H>

#include <AMReX.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_GpuControl.H>
#include <AMReX_GpuDevice.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_IntVect.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <AMReX_Scan.H>

#include <string>
#include <utility>

// The PETSc headers must be included before PETScPoissonSolver.H, see the
// comment in Source/NonlinearSolvers/WarpX_PETSc.cpp
#include <petscksp.h>
#include <petscmat.h>
#include <petscpc.h>
#include <petscvec.h>

#include <ablastr/fields/PETScPoissonSolver.H>


namespace ablastr::fields {

namespace petsc_poisson {

//! Wrapper for a PETSc KSP object
struct KSPObj
{
    KSPObj () = default;
    ~KSPObj () { if (obj) { KSPDestroy(&obj); } }
    KSPObj (KSPObj const &) = delete;
    KSPObj (KSPObj &&) = delete;
    KSPObj & operator= (KSPObj const &) = delete;
    KSPObj & operator= (KSPObj &&) = delete;
    KSP obj = nullptr;
};

//! Wrapper for a PETSc Mat object
struct MatObj
{
    MatObj () = default;
    ~MatObj () { if (obj) { MatDestroy(&obj); } }
    MatObj (MatObj const &) = delete;
    MatObj (MatObj &&) = delete;
    MatObj & operator= (MatObj const &) = delete;
    MatObj & operator= (MatObj &&) = delete;
    Mat obj = nullptr;
};

//! Wrapper for a PETSc Vec object
struct VecObj
{
    VecObj () = default;
    ~VecObj () { if (obj) { VecDestroy(&obj); } }
    VecObj (VecObj const &) = delete;
    VecObj (VecObj &&) = delete;
    VecObj & operator= (VecObj const &) = delete;
    VecObj & operator= (VecObj &&) = delete;
    Vec obj = nullptr;
};

//! Apply the matrix-free linear operator, called back by PETSc
PetscErrorCode applyOperator (Mat a_A, Vec a_in, Vec a_out)
{
    PetscFunctionBeginUser;

    PETScPoissonSolver * solver = nullptr;
    PetscCall(MatShellGetContext(a_A, &solver));

    PetscScalar const * in_arr = nullptr;
    PetscScalar * out_arr = nullptr;
    PetscCall(VecGetArrayRead(a_in, &in_arr));
    PetscCall(VecGetArrayWrite(a_out, &out_arr));

    solver->applyOperator( static_cast<amrex::Real*>(out_arr),
                           static_cast<amrex::Real const*>(in_arr) );

    PetscCall(VecRestoreArrayWrite(a_out, &out_arr));
    PetscCall(VecRestoreArrayRead(a_in, &in_arr));

    PetscFunctionReturn(PETSC_SUCCESS);
}

//! Apply the multigrid preconditioner, called back by PETSc
PetscErrorCode applyPreconditioner (PC a_pc, Vec a_in, Vec a_out)
{
    PetscFunctionBeginUser;

    PETScPoissonSolver * solver = nullptr;
    PetscCall(PCShellGetContext(a_pc, &solver));

    PetscScalar const * in_arr = nullptr;
    PetscScalar * out_arr = nullptr;
    PetscCall(VecGetArrayRead(a_in, &in_arr));
    PetscCall(VecGetArrayWrite(a_out, &out_arr));

    solver->applyPreconditioner( static_cast<amrex::Real*>(out_arr),
                                 static_cast<amrex::Real const*>(in_arr) );

    PetscCall(VecRestoreArrayWrite(a_out, &out_arr));
    PetscCall(VecRestoreArrayRead(a_in, &in_arr));

    PetscFunctionReturn(PETSC_SUCCESS);
}

//! Print the residual of every Krylov iteration
PetscErrorCode printResidual (KSP a_ksp, PetscInt a_n, PetscReal a_rnorm, void * a_ctxt)
{
    PetscFunctionBeginUser;
    amrex::ignore_unused(a_ksp, a_ctxt);
    amrex::Print() << "Poisson (PETSc KSP): iter = " << a_n
                   << ", residual = " << a_rnorm << "\n";
    PetscFunctionReturn(PETSC_SUCCESS);
}

//! Is `a_type` one of the GMRES variants of PETSc?
bool isGMRES (std::string const & a_type)
{
    return (a_type == "gmres") || (a_type == "fgmres")
        || (a_type == "lgmres") || (a_type == "dgmres")
        || (a_type == "pgmres") || (a_type == "pipefgmres");
}

} // namespace petsc_poisson


PETScPoissonSolver::PETScPoissonSolver (amrex::MLMG & mlmg,
                                        amrex::MultiFab const & phi_prototype,
                                        amrex::Geometry const & geom,
                                        PETScPoissonOptions const & options)
    : m_mlmg(&mlmg), m_geom(geom), m_options(options)
{
    ABLASTR_PROFILE("PETScPoissonSolver::PETScPoissonSolver()");

    // This builds the multigrid hierarchy and the masks of the linear operator,
    // which the operator, the preconditioner and buildDOFMap() below all need.
    m_mlmg->preparePrecond();

    buildDOFMap(phi_prototype);

    // The work arrays are created by the linear operator itself, so that they
    // have the right layout and factory. The inputs of the operator and of the
    // preconditioner need one layer of ghost nodes, as in
    // amrex::GMRESMLMG::makeVecLHS().
    auto & linop = m_mlmg->getLinOp();
    m_op_in = linop.make(0, 0, amrex::IntVect(1));
    m_op_out = linop.make(0, 0, amrex::IntVect(0));
    m_pc_in = linop.make(0, 0, amrex::IntVect(1));
    m_pc_out = linop.make(0, 0, amrex::IntVect(1));
    m_res = linop.make(0, 0, amrex::IntVect(1));
    m_cor = linop.make(0, 0, amrex::IntVect(1));

    m_A = std::make_unique<petsc_poisson::MatObj>();
    m_x = std::make_unique<petsc_poisson::VecObj>();
    m_b = std::make_unique<petsc_poisson::VecObj>();
    m_ksp = std::make_unique<petsc_poisson::KSPObj>();

    // Vectors
    VecCreate(PETSC_COMM_WORLD, &m_x->obj);
#ifdef AMREX_USE_GPU
#   if defined(AMREX_USE_CUDA)
    VecSetType(m_x->obj, VECCUDA);
#   elif defined(AMREX_USE_HIP)
    VecSetType(m_x->obj, VECHIP);
#   else
    ABLASTR_ABORT_WITH_MESSAGE(
        "The PETSc Poisson solver is not yet implemented for non-CUDA/HIP GPUs");
#   endif
#else
    VecSetType(m_x->obj, VECSTANDARD);
#endif
    auto const ndofs_local = static_cast<PetscInt>(m_ndofs_local);
    auto const ndofs_global = static_cast<PetscInt>(m_ndofs_global);
    VecSetSizes(m_x->obj, ndofs_local, ndofs_global);
    VecSetFromOptions(m_x->obj);
    VecDuplicate(m_x->obj, &m_b->obj);

    // Matrix-free linear operator
    MatCreateShell( PETSC_COMM_WORLD,
                    ndofs_local, ndofs_local,
                    ndofs_global, ndofs_global,
                    this, &m_A->obj );
    MatShellSetOperation( m_A->obj, MATOP_MULT,
                          (void(*)())petsc_poisson::applyOperator ); // NOLINT
    MatSetUp(m_A->obj);

    // Krylov solver
    KSPCreate(PETSC_COMM_WORLD, &m_ksp->obj);
    KSPSetType(m_ksp->obj, m_options.ksp_type.c_str());
    KSPSetOperators(m_ksp->obj, m_A->obj, m_A->obj);
    if (petsc_poisson::isGMRES(m_options.ksp_type)) {
        KSPGMRESSetRestart(m_ksp->obj, m_options.restart_length);
        // Right preconditioning, so that the residual that PETSc monitors and
        // uses for its convergence test is the residual of the actual system
        KSPSetPCSide(m_ksp->obj, PC_RIGHT);
        KSPSetNormType(m_ksp->obj, KSP_NORM_UNPRECONDITIONED);
    }

    PC pc = nullptr;
    KSPGetPC(m_ksp->obj, &pc);
    if (m_options.use_mlmg_preconditioner) {
        PCSetType(pc, PCSHELL);
        PCShellSetApply(pc, petsc_poisson::applyPreconditioner);
        PCShellSetContext(pc, this);
        PCShellSetName(pc, "AMReX MLMG");
    } else {
        PCSetType(pc, PCNONE);
    }

    if (m_options.verbosity > 1) {
        KSPMonitorSet(m_ksp->obj, petsc_poisson::printResidual, nullptr, nullptr);
    }
    // Command-line and input-file PETSc options take precedence over the above
    KSPSetFromOptions(m_ksp->obj);

    if (m_options.verbosity > 0) {
        amrex::Print() << "PETScPoissonSolver: using PETSc's KSP (" << m_options.ksp_type
                       << ") with "
                       << (m_options.use_mlmg_preconditioner ? "the AMReX MLMG" : "no")
                       << " preconditioner (total DOFs = " << m_ndofs_global << ").\n";
    }
}

PETScPoissonSolver::~PETScPoissonSolver () = default;

void PETScPoissonSolver::buildDOFMap (amrex::MultiFab const & phi_prototype)
{
    ABLASTR_PROFILE("PETScPoissonSolver::buildDOFMap()");

    using namespace amrex::literals;

    // The nodes that sit on the boundary between two boxes (or on the boundary
    // between a box and the periodic image of another one) belong to the valid
    // region of both boxes, but they are a single unknown of the linear system:
    // only the node of the "owner" box is a degree of freedom.
    auto const owner_mask = amrex::OwnerMask(phi_prototype, m_geom.periodicity());

    // The nodes on which a Dirichlet boundary condition is applied are not
    // unknowns of the linear system either. `setDirichletNodesToZero` is the
    // public interface through which the AMReX linear operator exposes them.
    auto & linop = m_mlmg->getLinOp();
    amrex::MultiFab dirichlet_indicator = linop.make(0, 0, amrex::IntVect(0));
    dirichlet_indicator.setVal(1._rt);
    linop.setDirichletNodesToZero(0, 0, dirichlet_indicator);

    m_dof = std::make_unique<amrex::iMultiFab>(phi_prototype.boxArray(),
                                               phi_prototype.DistributionMap(), 1, 0);
    m_dof->setVal(-1);

    m_ndofs_local = 0;
    for (amrex::MFIter mfi(*m_dof); mfi.isValid(); ++mfi)
    {
        amrex::Box const & bx = mfi.validbox();
        auto const npts = static_cast<int>(bx.numPts());
        amrex::BoxIndexer const box_indexer(bx);

        auto const & owner_arr = owner_mask->const_array(mfi);
        auto const & dirichlet_arr = dirichlet_indicator.const_array(mfi);
        auto const & dof_arr = m_dof->array(mfi);
        auto const first_dof = static_cast<int>(m_ndofs_local);

        auto const ndofs = amrex::Scan::PrefixSum<int>(
            npts,
            [=] AMREX_GPU_DEVICE (int offset) -> int
            {
                auto const [i,j,k] = box_indexer(offset);
                return (owner_arr(i,j,k) && (dirichlet_arr(i,j,k) > 0.5_rt)) ? 1 : 0;
            },
            [=] AMREX_GPU_DEVICE (int offset, int ps)
            {
                auto const [i,j,k] = box_indexer(offset);
                if (owner_arr(i,j,k) && (dirichlet_arr(i,j,k) > 0.5_rt)) {
                    dof_arr(i,j,k) = ps + first_dof;
                }
            },
            amrex::Scan::Type::exclusive, amrex::Scan::retSum);

        m_ndofs_local += ndofs;
    }

    m_ndofs_global = m_ndofs_local;
    amrex::ParallelDescriptor::ReduceLongSum(m_ndofs_global);

    ABLASTR_ALWAYS_ASSERT_WITH_MESSAGE(m_ndofs_global > 0,
        "PETScPoissonSolver: the linear system has no degree of freedom");
}

void PETScPoissonSolver::copyToArray (amrex::MultiFab const & mf, amrex::Real * arr) const
{
    ABLASTR_PROFILE("PETScPoissonSolver::copyToArray()");

    for (amrex::MFIter mfi(*m_dof); mfi.isValid(); ++mfi)
    {
        amrex::Box const & bx = mfi.validbox();
        auto const & mf_arr = mf.const_array(mfi);
        auto const & dof_arr = m_dof->const_array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            int const dof = dof_arr(i,j,k);
            if (dof >= 0) { arr[dof] = mf_arr(i,j,k); }
        });
    }
    amrex::Gpu::streamSynchronize();
}

void PETScPoissonSolver::copyFromArray (amrex::MultiFab & mf, amrex::Real const * arr) const
{
    ABLASTR_PROFILE("PETScPoissonSolver::copyFromArray()");

    using namespace amrex::literals;

    // The nodes that are not degrees of freedom (Dirichlet nodes, and the nodes
    // that another box owns) are set to zero here, and the ones that another box
    // owns are then filled from their owner by `OverrideSync` below.
    mf.setVal(0._rt);

    for (amrex::MFIter mfi(*m_dof); mfi.isValid(); ++mfi)
    {
        amrex::Box const & bx = mfi.validbox();
        auto const & mf_arr = mf.array(mfi);
        auto const & dof_arr = m_dof->const_array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            int const dof = dof_arr(i,j,k);
            if (dof >= 0) { mf_arr(i,j,k) = arr[dof]; }
        });
    }
    amrex::Gpu::streamSynchronize();

    // `OverrideSync` uses the same `amrex::OwnerMask` as `buildDOFMap` above,
    // so the nodes that are shared between boxes are filled from the very box
    // whose node was numbered as a degree of freedom.
    mf.OverrideSync(m_geom.periodicity());
    mf.FillBoundary(m_geom.periodicity());
}

void PETScPoissonSolver::applyOperator (amrex::Real * out, amrex::Real const * in)
{
    ABLASTR_PROFILE("PETScPoissonSolver::applyOperator()");

    copyFromArray(m_op_in, in);
    // `applyPrecond` applies the operator with homogeneous boundary conditions,
    // which is the operator that the correction equation solved here uses
    m_mlmg->applyPrecond({&m_op_out}, {&m_op_in});
    m_mlmg->getLinOp().setDirichletNodesToZero(0, 0, m_op_out);
    copyToArray(m_op_out, out);
}

void PETScPoissonSolver::applyPreconditioner (amrex::Real * out, amrex::Real const * in)
{
    ABLASTR_PROFILE("PETScPoissonSolver::applyPreconditioner()");

    using namespace amrex::literals;

    copyFromArray(m_pc_in, in);
    m_mlmg->setPrecondIter(m_options.precond_num_iters);
    m_pc_out.setVal(0._rt);
    m_mlmg->precond({&m_pc_out}, {&m_pc_in}, 0._rt, 0._rt);
    copyToArray(m_pc_out, out);
}

void PETScPoissonSolver::solve (amrex::MultiFab & phi,
                                amrex::MultiFab const & rho,
                                amrex::Real relative_tolerance,
                                amrex::Real absolute_tolerance,
                                int max_iters)
{
    ABLASTR_PROFILE("PETScPoissonSolver::solve()");

    using namespace amrex::literals;

    auto & linop = m_mlmg->getLinOp();

    // MLMG is only used as a preconditioner here, so its bottom solve must be
    // cheap and linear; this mirrors what amrex::GMRESMLMG does.
    auto const bottom_solver = m_mlmg->getBottomSolver();
    auto const mlmg_verbose = m_mlmg->getVerbose();
    auto const mlmg_bottom_verbose = m_mlmg->getBottomVerbose();
    if (bottom_solver != amrex::BottomSolver::smoother &&
        bottom_solver != amrex::BottomSolver::hypre &&
        bottom_solver != amrex::BottomSolver::petsc)
    {
        m_mlmg->setBottomSolver(amrex::BottomSolver::smoother);
    }
    m_mlmg->setVerbose(0);
    m_mlmg->setBottomVerbose(0);

    // Residual of the initial guess: res = L(phi) - rho. Note that `apply` uses
    // the inhomogeneous operator, so that the Dirichlet values that `phi` holds
    // contribute to the residual.
    m_res.setVal(0._rt);
    m_mlmg->apply({&m_res}, {&phi});

    amrex::MultiFab scaled_rho;
    amrex::MultiFab const * rhs = &rho;
    if (linop.scaleRHS(0, nullptr)) {
        scaled_rho.define(rho.boxArray(), rho.DistributionMap(), 1, 0);
        amrex::MultiFab::Copy(scaled_rho, rho, 0, 0, 1, 0);
        auto const scaled = linop.scaleRHS(0, &scaled_rho);
        amrex::ignore_unused(scaled);
        rhs = &scaled_rho;
    }
    amrex::MultiFab::Saxpy(m_res, -1._rt, *rhs, 0, 0, 1, amrex::IntVect(0));
    linop.setDirichletNodesToZero(0, 0, m_res);

    // Solve L(cor) = res for the correction, with PETSc's Krylov solver
    {
        PetscScalar * b_arr = nullptr;
        VecGetArrayWrite(m_b->obj, &b_arr);
        copyToArray(m_res, static_cast<amrex::Real*>(b_arr));
        VecRestoreArrayWrite(m_b->obj, &b_arr);
    }
    VecZeroEntries(m_x->obj);

    KSPSetTolerances( m_ksp->obj,
                      relative_tolerance,
                      absolute_tolerance,
                      PETSC_CURRENT,
                      (max_iters > 0 ? max_iters : PETSC_CURRENT) );
    KSPSolve(m_ksp->obj, m_b->obj, m_x->obj);

    {
        PetscScalar const * x_arr = nullptr;
        VecGetArrayRead(m_x->obj, &x_arr);
        copyFromArray(m_cor, static_cast<amrex::Real const*>(x_arr));
        VecRestoreArrayRead(m_x->obj, &x_arr);
    }

    // phi = phi - cor
    amrex::MultiFab::Saxpy(phi, -1._rt, m_cor, 0, 0, 1, amrex::IntVect(0));
    phi.FillBoundary(m_geom.periodicity());

    // Report on the solve
    PetscInt niters = -1;
    KSPGetIterationNumber(m_ksp->obj, &niters);
    m_num_iters = static_cast<int>(niters);
    PetscReal norm = -1;
    KSPGetResidualNorm(m_ksp->obj, &norm);
    m_residual_norm = static_cast<amrex::Real>(norm);

    KSPConvergedReason reason;
    KSPGetConvergedReason(m_ksp->obj, &reason);
    char const * reason_string = nullptr;
    KSPGetConvergedReasonString(m_ksp->obj, &reason_string);

    if (m_options.verbosity > 0) {
        amrex::Print() << "Poisson (PETSc KSP): " << m_num_iters << " iterations, exited due to \""
                       << reason_string << "\" (abs. norm = " << m_residual_norm << ").\n";
    }
    ABLASTR_ALWAYS_ASSERT_WITH_MESSAGE(reason > 0,
        std::string("The PETSc Poisson solver failed to converge: ") + reason_string);

    // Restore the settings of the multigrid solver
    m_mlmg->setBottomSolver(bottom_solver);
    m_mlmg->setVerbose(mlmg_verbose);
    m_mlmg->setBottomVerbose(mlmg_bottom_verbose);
}

void
petscPoissonSolve (amrex::MLMG & mlmg,
                   amrex::MultiFab & phi,
                   amrex::MultiFab const & rho,
                   amrex::Geometry const & geom,
                   amrex::Real relative_tolerance,
                   amrex::Real absolute_tolerance,
                   int max_iters,
                   PETScPoissonOptions const & options)
{
    PETScPoissonSolver solver(mlmg, phi, geom, options);
    solver.solve(phi, rho, relative_tolerance, absolute_tolerance, max_iters);
}

} // namespace ablastr::fields

#endif // AMREX_USE_PETSC
