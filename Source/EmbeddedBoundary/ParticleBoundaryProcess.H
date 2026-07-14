/* Copyright 2021 Andrew Myers
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PARTICLEBOUNDARYPROCESS_H_
#define WARPX_PARTICLEBOUNDARYPROCESS_H_

#include "EmbeddedBoundary/DistanceToEB.H"
#include "Particles/Pusher/UpdatePosition.H"
#include "Particles/WarpXParticleContainer.H"

#include <ablastr/particles/NodalFieldGather.H>

#include <AMReX_Algorithm.H>
#include <AMReX_Array4.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_Particle.H>
#include <AMReX_REAL.H>
#include <AMReX_RealVect.H>
#include <AMReX_Random.H>


namespace ParticleBoundaryProcess {

struct NoOp {
    template <typename PData>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void operator() (const PData& /*ptd*/, int /*i*/,
                     amrex::ParticleReal /*xp*/, amrex::ParticleReal /*yp*/, amrex::ParticleReal /*zp*/,
                     amrex::Array4<const amrex::Real> const& /*phi*/,
                     amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& /*plo*/,
                     amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& /*dxi*/,
                     amrex::RandomEngine const& /*engine*/) const noexcept
    {}
};

struct Absorb {
    template <typename PData>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void operator() (PData& ptd, int i,
                     amrex::ParticleReal /*xp*/, amrex::ParticleReal /*yp*/, amrex::ParticleReal /*zp*/,
                     amrex::Array4<const amrex::Real> const& /*phi*/,
                     amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& /*plo*/,
                     amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& /*dxi*/,
                     amrex::RandomEngine const& /*engine*/) const noexcept
    {
        amrex::ParticleIDWrapper{ptd.m_idcpu[i]}.make_invalid();
    }
};

/** \brief Specular reflection off the embedded boundary using bisection.
 *
 *  For each particle that has crossed into the EB (phi < 0), this functor:
 *  1. Bisects along the particle trajectory to find the contact point on
 *     the EB surface (where phi = 0).
 *  2. Computes the surface normal at the exact contact point.
 *  3. Performs specular velocity reflection: v_new = v - 2*(v . n)*n.
 *  4. Advances the particle from the contact point with the reflected
 *     velocity for the remaining fraction of the timestep.
 *
 *  This handles curved EB surfaces correctly, unlike the simpler phi-based
 *  approximation (x_new = x - 2*phi*n) which is only exact for planar
 *  surfaces.
 */
struct Reflect {
    amrex::Real m_dt;
    amrex::ParticleReal m_mass;

    template <typename PData>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void operator() (PData& ptd, int i,
                     amrex::ParticleReal xp, amrex::ParticleReal yp, amrex::ParticleReal zp,
                     amrex::Array4<const amrex::Real> const& phi,
                     amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& plo,
                     amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dxi,
                     amrex::RandomEngine const& /*engine*/) const noexcept
    {
        amrex::ParticleReal const ux = ptd.m_rdata[PIdx::ux][i];
        amrex::ParticleReal const uy = ptd.m_rdata[PIdx::uy][i];
        amrex::ParticleReal const uz = ptd.m_rdata[PIdx::uz][i];

        // Step 1: Bisect along trajectory to find the fraction of dt
        // to walk backward from end-of-step to reach the EB surface.
        amrex::Real const dt = m_dt;
        amrex::ParticleReal const mass = m_mass;
        amrex::Real const dt_fraction = amrex::bisect(0.0, 1.0,
            [=] (amrex::Real dt_frac) {
                int bi, bj, bk;
                amrex::Real bW[AMREX_SPACEDIM][2];
                amrex::ParticleReal x_temp = xp, y_temp = yp, z_temp = zp;
                UpdatePosition(x_temp, y_temp, z_temp,
                               ux, uy, uz, -dt_frac * dt, mass);
                ablastr::particles::compute_weights<amrex::IndexType::NODE>(
                    x_temp, y_temp, z_temp, plo, dxi, bi, bj, bk, bW);
                return ablastr::particles::interp_field_nodal(
                    bi, bj, bk, bW, phi);
            });

        // Step 2: Move to contact point on the EB surface
        amrex::ParticleReal x_c = xp, y_c = yp, z_c = zp;
        UpdatePosition(x_c, y_c, z_c, ux, uy, uz, -dt_fraction * dt, mass);

        // Step 3: Compute normal at the contact point
        int ci, cj, ck;
        amrex::Real cW[AMREX_SPACEDIM][2];
        ablastr::particles::compute_weights<amrex::IndexType::NODE>(
            x_c, y_c, z_c, plo, dxi, ci, cj, ck, cW);
        int cci, ccj, cck;
        amrex::Real ccW[AMREX_SPACEDIM][2];
        ablastr::particles::compute_weights<amrex::IndexType::CELL>(
            x_c, y_c, z_c, plo, dxi, cci, ccj, cck, ccW);
        amrex::RealVect n = DistanceToEB::interp_normal(
            ci, cj, ck, cW, cci, ccj, cck, ccW, phi, dxi);
        DistanceToEB::normalize(n);

        // Step 4: Specular velocity reflection  v = v - 2(v.n)n
        // Normal is in reduced-dimension coordinates;
        // velocity is always 3D Cartesian.
#if defined(WARPX_DIM_3D)
        amrex::ParticleReal const v_dot_n =
            ux * amrex::ParticleReal(n[0]) +
            uy * amrex::ParticleReal(n[1]) +
            uz * amrex::ParticleReal(n[2]);
        amrex::ParticleReal const ux_refl =
            ux - amrex::ParticleReal(2.0) * v_dot_n * amrex::ParticleReal(n[0]);
        amrex::ParticleReal const uy_refl =
            uy - amrex::ParticleReal(2.0) * v_dot_n * amrex::ParticleReal(n[1]);
        amrex::ParticleReal const uz_refl =
            uz - amrex::ParticleReal(2.0) * v_dot_n * amrex::ParticleReal(n[2]);
#elif defined(WARPX_DIM_XZ)
        // normal is (n_x, n_z). uy is out-of-plane, unaffected.
        amrex::ParticleReal const v_dot_n =
            ux * amrex::ParticleReal(n[0]) +
            uz * amrex::ParticleReal(n[1]);
        amrex::ParticleReal const ux_refl =
            ux - amrex::ParticleReal(2.0) * v_dot_n * amrex::ParticleReal(n[0]);
        amrex::ParticleReal const uy_refl = uy;
        amrex::ParticleReal const uz_refl =
            uz - amrex::ParticleReal(2.0) * v_dot_n * amrex::ParticleReal(n[1]);
#elif defined(WARPX_DIM_RZ)
        // normal is (n_r, n_z). Convert to 3D Cartesian using
        // the particle theta at the contact point.
        amrex::ParticleReal const theta_c = std::atan2(y_c, x_c);
        amrex::ParticleReal const cos_tc = std::cos(theta_c);
        amrex::ParticleReal const sin_tc = std::sin(theta_c);
        amrex::ParticleReal const n3d_x =
            amrex::ParticleReal(n[0]) * cos_tc;
        amrex::ParticleReal const n3d_y =
            amrex::ParticleReal(n[0]) * sin_tc;
        auto const n3d_z = amrex::ParticleReal(n[1]);
        amrex::ParticleReal const v_dot_n =
            ux * n3d_x + uy * n3d_y + uz * n3d_z;
        amrex::ParticleReal const ux_refl =
            ux - amrex::ParticleReal(2.0) * v_dot_n * n3d_x;
        amrex::ParticleReal const uy_refl =
            uy - amrex::ParticleReal(2.0) * v_dot_n * n3d_y;
        amrex::ParticleReal const uz_refl =
            uz - amrex::ParticleReal(2.0) * v_dot_n * n3d_z;
#elif defined(WARPX_DIM_RCYLINDER)
        // normal is radial n_r. Convert to 3D Cartesian using
        // the particle theta at the contact point.
        amrex::ParticleReal const theta_c = std::atan2(y_c, x_c);
        amrex::ParticleReal const cos_tc = std::cos(theta_c);
        amrex::ParticleReal const sin_tc = std::sin(theta_c);
        amrex::ParticleReal const n3d_x =
            amrex::ParticleReal(n[0]) * cos_tc;
        amrex::ParticleReal const n3d_y =
            amrex::ParticleReal(n[0]) * sin_tc;
        amrex::ParticleReal const n3d_z = amrex::ParticleReal(0.0);
        amrex::ParticleReal const v_dot_n =
            ux * n3d_x + uy * n3d_y + uz * n3d_z;
        amrex::ParticleReal const ux_refl =
            ux - amrex::ParticleReal(2.0) * v_dot_n * n3d_x;
        amrex::ParticleReal const uy_refl =
            uy - amrex::ParticleReal(2.0) * v_dot_n * n3d_y;
        amrex::ParticleReal const uz_refl = uz;
#elif defined(WARPX_DIM_RSPHERE)
        // normal is radial n_r. Convert to 3D Cartesian using
        // the particle position at the contact point.
        amrex::ParticleReal const r_c =
            std::sqrt(x_c*x_c + y_c*y_c + z_c*z_c);
        amrex::ParticleReal const inv_r_c =
            amrex::ParticleReal(1.0) / r_c;
        amrex::ParticleReal const n3d_x =
            amrex::ParticleReal(n[0]) * x_c * inv_r_c;
        amrex::ParticleReal const n3d_y =
            amrex::ParticleReal(n[0]) * y_c * inv_r_c;
        amrex::ParticleReal const n3d_z =
            amrex::ParticleReal(n[0]) * z_c * inv_r_c;
        amrex::ParticleReal const v_dot_n =
            ux * n3d_x + uy * n3d_y + uz * n3d_z;
        amrex::ParticleReal const ux_refl =
            ux - amrex::ParticleReal(2.0) * v_dot_n * n3d_x;
        amrex::ParticleReal const uy_refl =
            uy - amrex::ParticleReal(2.0) * v_dot_n * n3d_y;
        amrex::ParticleReal const uz_refl =
            uz - amrex::ParticleReal(2.0) * v_dot_n * n3d_z;
#elif defined(WARPX_DIM_1D_Z)
        amrex::ParticleReal const v_dot_n =
            uz * amrex::ParticleReal(n[0]);
        amrex::ParticleReal const ux_refl = ux;
        amrex::ParticleReal const uy_refl = uy;
        amrex::ParticleReal const uz_refl =
            uz - amrex::ParticleReal(2.0) * v_dot_n * amrex::ParticleReal(n[0]);
#endif

        // Step 5: Store reflected velocity
        ptd.m_rdata[PIdx::ux][i] = ux_refl;
        ptd.m_rdata[PIdx::uy][i] = uy_refl;
        ptd.m_rdata[PIdx::uz][i] = uz_refl;

        // Step 6: Advance from contact point with reflected velocity
        // for the remaining fraction of the timestep
        amrex::ParticleReal x_new = x_c, y_new = y_c, z_new = z_c;
        UpdatePosition(x_new, y_new, z_new, ux_refl, uy_refl, uz_refl,
                       dt_fraction * dt, mass);

        // Step 7: Write final position back to particle (manual RZ conversion)
#if defined(WARPX_DIM_3D)
        ptd.m_rdata[PIdx::x][i] = x_new;
        ptd.m_rdata[PIdx::y][i] = y_new;
        ptd.m_rdata[PIdx::z][i] = z_new;
#elif defined(WARPX_DIM_XZ)
        ptd.m_rdata[PIdx::x][i] = x_new;
        ptd.m_rdata[PIdx::z][i] = z_new;
#elif defined(WARPX_DIM_RZ)
        ptd.m_rdata[PIdx::r][i] = std::sqrt(x_new*x_new + y_new*y_new);
        ptd.m_rdata[PIdx::z][i] = z_new;
        ptd.m_rdata[PIdx::theta][i] = std::atan2(y_new, x_new);
#elif defined(WARPX_DIM_RCYLINDER)
        ptd.m_rdata[PIdx::r][i] = std::sqrt(x_new*x_new + y_new*y_new);
        ptd.m_rdata[PIdx::theta][i] = std::atan2(y_new, x_new);
#elif defined(WARPX_DIM_RSPHERE)
        amrex::ParticleReal const rxy_new =
            std::sqrt(x_new*x_new + y_new*y_new);
        ptd.m_rdata[PIdx::r][i] =
            std::sqrt(x_new*x_new + y_new*y_new + z_new*z_new);
        ptd.m_rdata[PIdx::theta][i] = std::atan2(y_new, x_new);
        ptd.m_rdata[PIdx::phi][i] = std::atan2(z_new, rxy_new);
#elif defined(WARPX_DIM_1D_Z)
        ptd.m_rdata[PIdx::z][i] = z_new;
#endif
    }
};

}

#endif //WARPX_PARTICLEBOUNDARYPROCESS_H_
