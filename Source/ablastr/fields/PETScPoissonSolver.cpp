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
#include <AMReX_GpuControl.H>
#include <AMReX_GpuDevice.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_IntVect.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <AMReX_Scan.H>
#include <AMReX_iMultiFab.H>

#include <memory>
#include <string>

// The PETSc headers must be included before PETScPoissonSolver.H, see the
// comment in Source/NonlinearSolvers/WarpX_PETSc.cpp
#include <petscksp.h>
#include <petscmat.h>
#include <petscpc.h>
#include <petscvec.h>

#include <ablastr/fields/PETScPoissonSolver.H>


namespace ablastr::fields {

namespace {

//! RAII wrapper for a PETSc KSP object
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

//! RAII wrapper for a PETSc Mat object
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

//! RAII wrapper for a PETSc Vec object
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

/** Data shared between petscPoissonSolve() and the PETSc callbacks
 *
 * The degrees of freedom of the PETSc vectors are the nodes that this MPI rank
 * owns: nodes shared between boxes, or with a periodic image, appear only once.
 * Dirichlet nodes are kept as degrees of freedom; they simply remain zero
 * throughout the Krylov solve, since both the right-hand side and the operator
 * output are zeroed on them (this is also how `amrex::GMRESMLMG` treats them).
 */
struct PoissonCtx
{
    amrex::MLMG * mlmg = nullptr;
    amrex::Geometry geom;
    PETScPoissonOptions options;

    //! Local index of the degree of freedom of each node, -1 if it is not one
    std::unique_ptr<amrex::iMultiFab> dof;
    //! Number of degrees of freedom owned by this MPI rank / in total
    amrex::Long ndofs_local = 0;
    amrex::Long ndofs_global = 0;

    //! Work arrays (one ghost layer), reused by the operator and the
    //! preconditioner callbacks, and by petscPoissonSolve() itself
    amrex::MultiFab work_in;
    amrex::MultiFab work_out;

    //! Number the degrees of freedom that this MPI rank owns
    void buildDOFMap (amrex::MultiFab const & phi)
    {
        ABLASTR_PROFILE("petsc_poisson::buildDOFMap()");

        // Owner is the box with the lowest index containing the node; the same
        // convention is used by OverrideSync in copyFromArray() below.
        auto const owner_mask = amrex::OwnerMask(phi, geom.periodicity());

        dof = std::make_unique<amrex::iMultiFab>(phi.boxArray(),
                                                 phi.DistributionMap(), 1, 0);
        dof->setVal(-1);

        for (amrex::MFIter mfi(*dof); mfi.isValid(); ++mfi)
        {
            amrex::Box const & bx = mfi.validbox();
            auto const npts = static_cast<int>(bx.numPts());
            amrex::BoxIndexer const box_indexer(bx);

            auto const & owner_arr = owner_mask->const_array(mfi);
            auto const & dof_arr = dof->array(mfi);
            auto const first_dof = static_cast<int>(ndofs_local);

            auto const ndofs = amrex::Scan::PrefixSum<int>(
                npts,
                [=] AMREX_GPU_DEVICE (int offset) -> int
                {
                    auto const [i,j,k] = box_indexer(offset);
                    return owner_arr(i,j,k) ? 1 : 0;
                },
                [=] AMREX_GPU_DEVICE (int offset, int ps)
                {
                    auto const [i,j,k] = box_indexer(offset);
                    if (owner_arr(i,j,k)) {
                        dof_arr(i,j,k) = ps + first_dof;
                    }
                },
                amrex::Scan::Type::exclusive, amrex::Scan::retSum);

            ndofs_local += ndofs;
        }

        ndofs_global = ndofs_local;
        amrex::ParallelDescriptor::ReduceLongSum(ndofs_global);
    }

    /** Gather the degrees of freedom of `mf` into the PETSc array `arr`
     *
     * Note that on GPUs `arr` comes from `VecGetArray`, i.e. it is the host-side
     * array of the PETSc vector (PETSc copies it back to the device before the
     * next device operation). The kernel below writes it from device code, which
     * assumes that this allocation is addressable from the device. This is the
     * same assumption that `WarpXSolverVec::copyTo/copyFrom` makes for the
     * curl-curl solver, see Source/FieldSolver/ImplicitSolvers/WarpXSolverVec.cpp.
     */
    void copyToArray (amrex::MultiFab const & mf, amrex::Real * arr) const
    {
        ABLASTR_PROFILE("petsc_poisson::copyToArray()");

        for (amrex::MFIter mfi(*dof); mfi.isValid(); ++mfi)
        {
            auto const & mf_arr = mf.const_array(mfi);
            auto const & dof_arr = dof->const_array(mfi);
            amrex::ParallelFor(mfi.validbox(),
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                int const idx = dof_arr(i,j,k);
                if (idx >= 0) { arr[idx] = mf_arr(i,j,k); }
            });
        }
        amrex::Gpu::streamSynchronize();
    }

    //! Scatter the PETSc array `arr` into `mf`, and make `mf` consistent
    void copyFromArray (amrex::MultiFab & mf, amrex::Real const * arr) const
    {
        ABLASTR_PROFILE("petsc_poisson::copyFromArray()");

        using namespace amrex::literals;

        mf.setVal(0._rt);
        for (amrex::MFIter mfi(*dof); mfi.isValid(); ++mfi)
        {
            auto const & mf_arr = mf.array(mfi);
            auto const & dof_arr = dof->const_array(mfi);
            amrex::ParallelFor(mfi.validbox(),
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                int const idx = dof_arr(i,j,k);
                if (idx >= 0) { mf_arr(i,j,k) = arr[idx]; }
            });
        }
        amrex::Gpu::streamSynchronize();

        // Fill the nodes owned by another box from their owner (OverrideSync
        // uses the same OwnerMask convention as buildDOFMap), then the ghosts
        mf.OverrideSync(geom.periodicity());
        mf.FillBoundary(geom.periodicity());
    }
};

//! Apply the matrix-free linear operator, called back by PETSc
PetscErrorCode applyOperator (Mat a_A, Vec a_in, Vec a_out)
{
    PetscFunctionBeginUser;

    PoissonCtx * ctx = nullptr;
    PetscCall(MatShellGetContext(a_A, &ctx));

    PetscScalar const * in_arr = nullptr;
    PetscScalar * out_arr = nullptr;
    PetscCall(VecGetArrayRead(a_in, &in_arr));
    PetscCall(VecGetArrayWrite(a_out, &out_arr));

    ctx->copyFromArray(ctx->work_in, static_cast<amrex::Real const*>(in_arr));
    // `applyPrecond` applies the operator with homogeneous boundary conditions,
    // which is the operator that the correction equation uses
    ctx->mlmg->applyPrecond({&ctx->work_out}, {&ctx->work_in});
    ctx->mlmg->getLinOp().setDirichletNodesToZero(0, 0, ctx->work_out);
    ctx->copyToArray(ctx->work_out, static_cast<amrex::Real*>(out_arr));

    PetscCall(VecRestoreArrayWrite(a_out, &out_arr));
    PetscCall(VecRestoreArrayRead(a_in, &in_arr));

    PetscFunctionReturn(PETSC_SUCCESS);
}

//! Apply the multigrid preconditioner, called back by PETSc
PetscErrorCode applyPreconditioner (PC a_pc, Vec a_in, Vec a_out)
{
    PetscFunctionBeginUser;

    using namespace amrex::literals;

    PoissonCtx * ctx = nullptr;
    PetscCall(PCShellGetContext(a_pc, &ctx));

    PetscScalar const * in_arr = nullptr;
    PetscScalar * out_arr = nullptr;
    PetscCall(VecGetArrayRead(a_in, &in_arr));
    PetscCall(VecGetArrayWrite(a_out, &out_arr));

    ctx->copyFromArray(ctx->work_in, static_cast<amrex::Real const*>(in_arr));
    ctx->mlmg->setPrecondIter(ctx->options.precond_num_iters);
    ctx->work_out.setVal(0._rt);
    ctx->mlmg->precond({&ctx->work_out}, {&ctx->work_in}, 0._rt, 0._rt);
    ctx->copyToArray(ctx->work_out, static_cast<amrex::Real*>(out_arr));

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

} // anonymous namespace

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
    ABLASTR_PROFILE("petscPoissonSolve()");

    using namespace amrex::literals;

    ABLASTR_ALWAYS_ASSERT_WITH_MESSAGE(phi.nGrowVect().allGE(amrex::IntVect(1)),
        "petscPoissonSolve: phi must have (at least) one ghost layer");

    // This builds the multigrid hierarchy and the masks of the linear operator,
    // which the operator, the preconditioner and the DOF map all need
    mlmg.preparePrecond();
    auto & linop = mlmg.getLinOp();

    PoissonCtx ctx;
    ctx.mlmg = &mlmg;
    ctx.geom = geom;
    ctx.options = options;
    ctx.buildDOFMap(phi);
    // The work arrays share the layout (and factory) of `phi`; their ghost
    // layer is needed by the AMReX operators (as in amrex::GMRESMLMG::makeVecLHS).
    // Note that `phi` must be nodal with one ghost layer, like the vectors that
    // amrex::MLMG::solve would create internally.
    ctx.work_in.define(phi.boxArray(), phi.DistributionMap(), 1, 1,
                       amrex::MFInfo(), phi.Factory());
    ctx.work_out.define(phi.boxArray(), phi.DistributionMap(), 1, 1,
                        amrex::MFInfo(), phi.Factory());

    // PETSc vectors and matrix-free operator
    VecObj x, b;
    MatObj A;
    KSPObj ksp;
    auto const ndofs_l = static_cast<PetscInt>(ctx.ndofs_local);
    auto const ndofs_g = static_cast<PetscInt>(ctx.ndofs_global);
    VecCreate(PETSC_COMM_WORLD, &x.obj);
#ifdef AMREX_USE_GPU
#   if defined(AMREX_USE_CUDA)
    VecSetType(x.obj, VECCUDA);
#   elif defined(AMREX_USE_HIP)
    VecSetType(x.obj, VECHIP);
#   else
    ABLASTR_ABORT_WITH_MESSAGE(
        "The PETSc Poisson solver is not yet implemented for non-CUDA/HIP GPUs");
#   endif
#else
    VecSetType(x.obj, VECSTANDARD);
#endif
    VecSetSizes(x.obj, ndofs_l, ndofs_g);
    VecSetFromOptions(x.obj);
    VecDuplicate(x.obj, &b.obj);
    MatCreateShell(PETSC_COMM_WORLD, ndofs_l, ndofs_l, ndofs_g, ndofs_g,
                   &ctx, &A.obj);
    MatShellSetOperation(A.obj, MATOP_MULT, (void (*)(void))applyOperator);
    MatSetUp(A.obj);

    // GMRES, right-preconditioned so that the monitored residual is the
    // residual of the actual system
    KSPCreate(PETSC_COMM_WORLD, &ksp.obj);
    KSPSetType(ksp.obj, KSPGMRES);
    KSPSetOperators(ksp.obj, A.obj, A.obj);
    KSPSetPCSide(ksp.obj, PC_RIGHT);
    KSPSetNormType(ksp.obj, KSP_NORM_UNPRECONDITIONED);
    PC pc = nullptr;
    KSPGetPC(ksp.obj, &pc);
    if (options.use_mlmg_preconditioner) {
        PCSetType(pc, PCSHELL);
        PCShellSetApply(pc, applyPreconditioner);
        PCShellSetContext(pc, &ctx);
        PCShellSetName(pc, "AMReX MLMG");
    } else {
        PCSetType(pc, PCNONE);
    }
    KSPSetTolerances(ksp.obj, relative_tolerance, absolute_tolerance,
                     PETSC_CURRENT, (max_iters > 0 ? max_iters : PETSC_CURRENT));
    if (options.verbosity > 1) {
        KSPMonitorSet(ksp.obj, printResidual, nullptr, nullptr);
    }
    // PETSc runtime options (e.g. -ksp_type) take precedence over the above
    KSPSetFromOptions(ksp.obj);

    if (options.verbosity > 0) {
        amrex::Print() << "Poisson (PETSc KSP): "
                       << (options.use_mlmg_preconditioner ? "MLMG-preconditioned"
                                                           : "unpreconditioned")
                       << " solve, total DOFs = " << ctx.ndofs_global << ".\n";
    }

    // MLMG is only used as a preconditioner here, so its bottom solve must be
    // cheap and linear; this mirrors what amrex::GMRESMLMG does
    auto const bottom_solver = mlmg.getBottomSolver();
    auto const mlmg_verbose = mlmg.getVerbose();
    auto const mlmg_bottom_verbose = mlmg.getBottomVerbose();
    if (bottom_solver != amrex::BottomSolver::smoother &&
        bottom_solver != amrex::BottomSolver::hypre &&
        bottom_solver != amrex::BottomSolver::petsc)
    {
        mlmg.setBottomSolver(amrex::BottomSolver::smoother);
    }
    mlmg.setVerbose(0);
    mlmg.setBottomVerbose(0);

    // Residual of the initial guess: res = L(phi) - rho. Note that `apply` uses
    // the inhomogeneous operator, so that the Dirichlet values that `phi` holds
    // contribute to the residual. `work_in` is free until KSPSolve starts.
    amrex::MultiFab & res = ctx.work_in;
    res.setVal(0._rt);
    mlmg.apply({&res}, {&phi});

    amrex::MultiFab scaled_rho;
    amrex::MultiFab const * rhs = &rho;
    if (linop.scaleRHS(0, nullptr)) {
        scaled_rho.define(rho.boxArray(), rho.DistributionMap(), 1, 0);
        amrex::MultiFab::Copy(scaled_rho, rho, 0, 0, 1, 0);
        auto const scaled = linop.scaleRHS(0, &scaled_rho);
        amrex::ignore_unused(scaled);
        rhs = &scaled_rho;
    }
    amrex::MultiFab::Saxpy(res, -1._rt, *rhs, 0, 0, 1, amrex::IntVect(0));
    linop.setDirichletNodesToZero(0, 0, res);

    // Solve L(cor) = res for the correction
    {
        PetscScalar * b_arr = nullptr;
        VecGetArrayWrite(b.obj, &b_arr);
        ctx.copyToArray(res, static_cast<amrex::Real*>(b_arr));
        VecRestoreArrayWrite(b.obj, &b_arr);
    }
    KSPSolve(ksp.obj, b.obj, x.obj);

    // phi = phi - cor. `work_in` is free again once KSPSolve has returned.
    amrex::MultiFab & cor = ctx.work_in;
    {
        PetscScalar const * x_arr = nullptr;
        VecGetArrayRead(x.obj, &x_arr);
        ctx.copyFromArray(cor, static_cast<amrex::Real const*>(x_arr));
        VecRestoreArrayRead(x.obj, &x_arr);
    }
    amrex::MultiFab::Saxpy(phi, -1._rt, cor, 0, 0, 1, amrex::IntVect(0));

    // `amrex::MLMG::solve` ends with this; for the embedded-boundary operator it
    // writes the prescribed potential into the nodes that the EB covers
    linop.postSolve({&phi});
    phi.FillBoundary(geom.periodicity());

    // Report on the solve, and abort if it failed (as MLMG does)
    PetscInt niters = -1;
    KSPGetIterationNumber(ksp.obj, &niters);
    PetscReal norm = -1;
    KSPGetResidualNorm(ksp.obj, &norm);
    KSPConvergedReason reason;
    KSPGetConvergedReason(ksp.obj, &reason);
    char const * reason_string = nullptr;
    KSPGetConvergedReasonString(ksp.obj, &reason_string);
    if (options.verbosity > 0) {
        amrex::Print() << "Poisson (PETSc KSP): " << niters
                       << " iterations, exited due to \"" << reason_string
                       << "\" (abs. norm = " << norm << ").\n";
    }
    ABLASTR_ALWAYS_ASSERT_WITH_MESSAGE(reason > 0,
        std::string("The PETSc Poisson solver failed to converge: ") + reason_string);

    // Restore the settings of the multigrid solver
    mlmg.setBottomSolver(bottom_solver);
    mlmg.setVerbose(mlmg_verbose);
    mlmg.setBottomVerbose(mlmg_bottom_verbose);
}

} // namespace ablastr::fields

#endif // AMREX_USE_PETSC
