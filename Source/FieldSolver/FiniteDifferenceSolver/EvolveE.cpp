/* Copyright 2020 Remi Lehe
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "WarpXAlgorithmSelection.H"
#include "FiniteDifferenceSolver.H"
#ifdef WARPX_DIM_RZ
#   include "FiniteDifferenceAlgorithms/CylindricalYeeAlgorithm.H"
#else
#   include "FiniteDifferenceAlgorithms/YeeAlgorithm.H"
#   include "FiniteDifferenceAlgorithms/CKCAlgorithm.H"
#   include "FiniteDifferenceAlgorithms/NodalAlgorithm.H"
#endif
#include "WarpXConst.H"
#include <AMReX_Gpu.H>

using namespace amrex;

/**
 * \brief Update the E field, over one timestep
 */
void FiniteDifferenceSolver::EvolveE (
    std::array< std::unique_ptr<amrex::MultiFab>, 3 >& Efield,
    std::array< std::unique_ptr<amrex::MultiFab>, 3 > const& Bfield,
    std::array< std::unique_ptr<amrex::MultiFab>, 3 > const& Jfield,
    amrex::Real const dt ) {

   // Select algorithm (The choice of algorithm is a runtime option,
   // but we compile code for each algorithm, using templates)
#ifdef WARPX_DIM_RZ
    if (m_fdtd_algo == MaxwellSolverAlgo::Yee){

        EvolveECylindrical <CylindricalYeeAlgorithm> ( Efield, Bfield, Jfield, dt );

#else
    if (m_do_nodal) {

        EvolveECartesian <NodalAlgorithm> ( Efield, Bfield, Jfield, dt );

    } else if (m_fdtd_algo == MaxwellSolverAlgo::Yee) {

        EvolveECartesian <YeeAlgorithm> ( Efield, Bfield, Jfield, dt );

    } else if (m_fdtd_algo == MaxwellSolverAlgo::CKC) {

        EvolveECartesian <CKCAlgorithm> ( Efield, Bfield, Jfield, dt );

#endif
    } else {
        amrex::Abort("Unknown algorithm");
    }

}


#ifndef WARPX_DIM_RZ

template<typename T_Algo>
void FiniteDifferenceSolver::EvolveECartesian (
    std::array< std::unique_ptr<amrex::MultiFab>, 3 >& Efield,
    std::array< std::unique_ptr<amrex::MultiFab>, 3 > const& Bfield,
    std::array< std::unique_ptr<amrex::MultiFab>, 3 > const& Jfield,
    amrex::Real const dt ) {

    // Loop through the grids, and over the tiles within each grid
#ifdef _OPENMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(*Efield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {

        // Extract field data for this grid/tile
        auto const& Ex = Efield[0]->array(mfi);
        auto const& Ey = Efield[1]->array(mfi);
        auto const& Ez = Efield[2]->array(mfi);
        auto const& Bx = Bfield[0]->array(mfi);
        auto const& By = Bfield[1]->array(mfi);
        auto const& Bz = Bfield[2]->array(mfi);
        auto const& jx = Jfield[0]->array(mfi);
        auto const& jy = Jfield[1]->array(mfi);
        auto const& jz = Jfield[2]->array(mfi);

        // Extract stencil coefficients
        Real const* AMREX_RESTRICT coefs_x = stencil_coefs_x.dataPtr();
        int const n_coefs_x = stencil_coefs_x.size();
        Real const* AMREX_RESTRICT coefs_y = stencil_coefs_y.dataPtr();
        int const n_coefs_y = stencil_coefs_y.size();
        Real const* AMREX_RESTRICT coefs_z = stencil_coefs_z.dataPtr();
        int const n_coefs_z = stencil_coefs_z.size();

        // Extract tileboxes for which to loop
        const Box& tex  = mfi.tilebox(Efield[0]->ixType().ixType());
        const Box& tey  = mfi.tilebox(Efield[1]->ixType().ixType());
        const Box& tez  = mfi.tilebox(Efield[2]->ixType().ixType());

        Real constexpr c2 = PhysConst::c * PhysConst::c;

        // Loop over the cells and update the fields
        amrex::ParallelFor(tex, tey, tez,

            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                Ex(i, j, k) += c2 * dt * (
                    - T_Algo::DownwardDz(By, coefs_z, n_coefs_z, i, j, k)
                    + T_Algo::DownwardDy(Bz, coefs_y, n_coefs_y, i, j, k)
                    - PhysConst::mu0 * jx(i, j, k) );
            },

            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                Ey(i, j, k) += c2 * dt * (
                    - T_Algo::DownwardDx(Bz, coefs_x, n_coefs_x, i, j, k)
                    + T_Algo::DownwardDz(Bx, coefs_z, n_coefs_z, i, j, k)
                    - PhysConst::mu0 * jy(i, j, k) );
            },

            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                Ez(i, j, k) += c2 * dt * (
                    - T_Algo::DownwardDy(Bx, coefs_y, n_coefs_y, i, j, k)
                    + T_Algo::DownwardDx(By, coefs_x, n_coefs_x, i, j, k)
                    - PhysConst::mu0 * jz(i, j, k) );
            }

        );

    }

}

#else // corresponds to ifndef WARPX_DIM_RZ

template<typename T_Algo>
void FiniteDifferenceSolver::EvolveECylindrical (
    std::array< std::unique_ptr<amrex::MultiFab>, 3 >& Efield,
    std::array< std::unique_ptr<amrex::MultiFab>, 3 > const& Bfield,
    amrex::Real const dt ) {

    // Loop through the grids, and over the tiles within each grid
#ifdef _OPENMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(*Efield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {

        // Extract field data for this grid/tile
        auto const& Er = Efield[0]->array(mfi);
        auto const& Et = Efield[1]->array(mfi);
        auto const& Ez = Efield[2]->array(mfi);
        auto const& Br = Bfield[0]->array(mfi);
        auto const& Bt = Bfield[1]->array(mfi);
        auto const& Bz = Bfield[2]->array(mfi);
        auto const& jr = Jfield[0]->array(mfi);
        auto const& jt = Jfield[1]->array(mfi);
        auto const& jz = Jfield[2]->array(mfi);

        // Extract stencil coefficients
        Real const* AMREX_RESTRICT coefs_r = stencil_coefs_r.dataPtr();
        int const n_coefs_r = stencil_coefs_r.size();
        Real const* AMREX_RESTRICT coefs_z = stencil_coefs_z.dataPtr();
        int const n_coefs_z = stencil_coefs_z.size();

        // Extract cylindrical specific parameters
        Real const dr = m_dr;
        int const nmodes = m_nmodes;
        Real const rmin = m_rmin;

        // Extract tileboxes for which to loop
        const Box& ter  = mfi.tilebox(Efield[0]->ixType().ixType());
        const Box& tet  = mfi.tilebox(Efield[1]->ixType().ixType());
        const Box& tez  = mfi.tilebox(Efield[2]->ixType().ixType());

        Real const c2 = PhysConst::c * PhysConst::c;

        // Loop over the cells and update the fields
        amrex::ParallelFor(ter, tet, tez,

            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                Real const r = rmin + (i + 0.5)*dr; // r on cell-centered point (Er is cell-centered in r)
                Er(i, j, 0, 0) +=  c2 * dt*(
                    - T_Algo::DownwardDz(Bt, coefs_z, n_coefs_z, i, j, 0, 0)
                    - PhysConst::mu0 * jr(i, j, 0, 0) ); // Mode m=0
                for (int m=1; m<nmodes; m++) { // Higher-order modes
                    Er(i, j, 0, 2*m-1) += c2 * dt*(
                        - T_Algo::DownwardDz(Bt, coefs_z, n_coefs_z, i, j, 0, 2*m-1)
                        + m * Bz(i, j, 0, 2*m  )/r
                        - PhysConst::mu0 * jr(i, j, 0, 2*m-1) );  // Real part
                    Er(i, j, 0, 2*m  ) += c2 * dt*(
                        - T_Algo::DownwardDz(Bt, coefs_z, n_coefs_z, i, j, 0, 2*m  )
                        - m * Bz(i, j, 0, 2*m-1)/r
                        - PhysConst::mu0 * jr(i, j, 0, 2*m  ) ); // Imaginary part
                }
            },

            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                Et(i, j, 0, 0) += c2 * dt*(
                    - T_Algo::DownwardDr(Bz, coefs_r, n_coefs_r, i, j, 0, 0)
                    + T_Algo::DownwardDz(Br, coefs_z, n_coefs_z, i, j, 0, 0)
                    - PhysConst::mu0 * jt(i, j, 0, 0 ) ); // Mode m=0
                for (int m=1 ; m<nmodes ; m++) { // Higher-order modes
                    Et(i, j, 0, 2*m-1) += c2 * dt*(
                        - T_Algo::DownwardDr(Bz, coefs_r, n_coefs_r, i, j, 0, 2*m-1)
                        + T_Algo::DownwardDz(Br, coefs_z, n_coefs_z, i, j, 0, 2*m-1)
                        - PhysConst::mu0 * jt(i, j, 0, 2*m-1) ); // Real part
                    Et(i, j, 0, 2*m  ) += c2 * dt*(
                        - T_Algo::DownwardDr(Bz, coefs_r, n_coefs_r, i, j, 0, 2*m  )
                        + T_Algo::DownwardDz(Br, coefs_z, n_coefs_z, i, j, 0, 2*m  )
                        - PhysConst::mu0 * jt(i, j, 0, 2*m  ) ); // Imaginary part
                }
                // TODO: Modify on-axis condition
            },

            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                Real const r = rmin + i*dr; // r on a nodal grid (Bz is nodal in r)
                Ez(i, j, 0, 0) += c2 * dt*(
                   T_Algo::DownwardDrr_over_r(Bt, r, dr, coefs_r, n_coefs_r, i, j, 0, 0)
                    - PhysConst::mu0 * jz(i, j, 0, 0  ) );
                for (int m=1 ; m<nmodes ; m++) { // Higher-order modes
                    Ez(i, j, 0, 2*m-1) += c2 * dt *(
                        - m * Br(i, j, 0, 2*m  )/r
                        T_Algo::DownwardDrr_over_r(Bt, r, dr, coefs_r, n_coefs_r, i, j, 0, 2*m-1)
                        - PhysConst::mu0 * jz(i, j, 0, 2*m-1) ); // Real part
                    Ez(i, j, 0, 2*m  ) += c2 * dt *(
                        m * Br(i, j, 0, 2*m-1)/r
                        T_Algo::DownwardDrr_over_r(Bt, r, dr, coefs_r, n_coefs_r, i, j, 0, 2*m  )
                        - PhysConst::mu0 * jz(i, j, 0, 2*m  ) ); // Imaginary part
                }
                // TODO: Modify on-axis condition
            }

        );

    }

}

#endif // corresponds to ifndef WARPX_DIM_RZ
