/* Copyright 2021 Andrew Myers
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PARTICLEBOUNDARYPROCESS_H_
#define WARPX_PARTICLEBOUNDARYPROCESS_H_

#include "EmbeddedBoundary/DistanceToEB.H"
#include "Initialization/SampleGaussianFluxDistribution.H"
#include "Particles/Pusher/GetAndSetPosition.H"
#include "Particles/Pusher/UpdatePosition.H"
#include "Particles/WarpXParticleContainer.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "Utils/WarpXConst.H"

#include <ablastr/particles/NodalFieldGather.H>

#include <AMReX_Algorithm.H>
#include <AMReX_Array4.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_Particle.H>
#include <AMReX_REAL.H>
#include <AMReX_RealVect.H>
#include <AMReX_Random.H>

#include <cmath>


namespace ParticleBoundaryProcess {

struct NoOp {
    template <typename PData>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void operator() (const PData& /*ptd*/, int /*i*/,
                     const SetParticlePosition<PIdx>& /*setPosition*/,
                     amrex::ParticleReal /*xp*/, amrex::ParticleReal /*yp*/, amrex::ParticleReal /*zp*/,
                     amrex::Array4<const amrex::Real> const& /*distance_to_eb*/,
                     amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& /*plo*/,
                     amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& /*dxi*/,
                     amrex::RandomEngine const& /*engine*/) const noexcept
    {}
};

struct Absorb {
    template <typename PData>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void operator() (PData& ptd, int i,
                     const SetParticlePosition<PIdx>& /*setPosition*/,
                     amrex::ParticleReal /*xp*/, amrex::ParticleReal /*yp*/, amrex::ParticleReal /*zp*/,
                     amrex::Array4<const amrex::Real> const& /*distance_to_eb*/,
                     amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& /*plo*/,
                     amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& /*dxi*/,
                     amrex::RandomEngine const& /*engine*/) const noexcept
    {
        amrex::ParticleIDWrapper{ptd.m_idcpu[i]}.make_invalid();
    }
};

/** \brief Particle interaction with the embedded boundary, for the
 *         ``Reflecting`` and ``Thermal`` values of ``boundary.particle_eb``.
 *
 *  For each particle that has crossed into the EB (distance_to_eb < 0), this functor:
 *   - Bisects along the particle trajectory to find the contact point on
 *     the EB surface (where distance_to_eb = 0).
 *   - Computes the surface normal at the exact contact point.
 *   - Sets the new velocity of the particle, depending on m_boundary_type:
 *     - Reflecting: specular reflection, u_new = u - 2*(u . n)*n.
 *     - Thermal: re-emission from a wall Maxwellian (fully accommodating
 *       diffuse wall): a half-Maxwellian flux distribution along the
 *       (outward, into the domain) normal, and a full Maxwellian in the
 *       tangential plane, with thermal speed m_uth (same convention as the
 *       domain thermal particle boundary, see
 *       ApplyParticleBoundaries::thermalize_boundary_particle).
 *   - Advances the particle from the contact point with the new
 *     velocity for the remaining fraction of the timestep.
 */
struct ParticleBoundaryInteraction {
    amrex::Real m_dt;
    amrex::ParticleReal m_mass;
    ParticleBoundaryType m_boundary_type;
    amrex::ParticleReal m_uth = 0._prt;

    template <typename PData>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void operator() (PData& ptd, int i,
                     const SetParticlePosition<PIdx>& setPosition,
                     amrex::ParticleReal xp, amrex::ParticleReal yp, amrex::ParticleReal zp,
                     amrex::Array4<const amrex::Real> const& distance_to_eb,
                     amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& plo,
                     amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dxi,
                     amrex::RandomEngine const& engine) const noexcept
    {
        using namespace amrex::literals;

        amrex::ParticleReal const ux = ptd.m_rdata[PIdx::ux][i];
        amrex::ParticleReal const uy = ptd.m_rdata[PIdx::uy][i];
        amrex::ParticleReal const uz = ptd.m_rdata[PIdx::uz][i];

        // Find the point of contat with the EB surface
        // Bisect along trajectory to find the fraction of dt
        // to walk backward from end-of-step to reach distance_to_eb = 0.
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
                    bi, bj, bk, bW, distance_to_eb);
            });
        // Move to contact point on the EB surface
        amrex::ParticleReal x_c = xp, y_c = yp, z_c = zp;
        UpdatePosition(x_c, y_c, z_c, ux, uy, uz, -dt_fraction * dt, mass);

        // Compute the surface normal vector at the contact point, in 3D
        // Cartesian coordinates (as is the velocity below). The normal
        // points away from the EB solid, into the domain (gas region).
        auto const n3d = DistanceToEB::interp_normal(x_c, y_c, z_c, plo, dxi, distance_to_eb);
        amrex::ParticleReal const n3d_x = n3d[0];
        amrex::ParticleReal const n3d_y = n3d[1];
        amrex::ParticleReal const n3d_z = n3d[2];

        if (m_boundary_type == ParticleBoundaryType::Thermal) {
            // Degenerate level-set gradient at a sharp corner makes the
            // (internally normalized) normal non-finite: absorb the particle
            // rather than re-emit it in an ill-defined direction.
            if (!(std::isfinite(n3d_x) && std::isfinite(n3d_y) && std::isfinite(n3d_z))) {
                amrex::ParticleIDWrapper{ptd.m_idcpu[i]}.make_invalid();
                return;
            }

            // Build an orthonormal frame (n, t1, t2) out of the surface normal,
            // to re-emit the particle in the (half-Maxwellian normal, full
            // Maxwellian tangential) wall distribution.
            amrex::ParticleReal e0 = 0._prt, e1 = 0._prt, e2 = 0._prt;
            if (std::abs(n3d_x) <= std::abs(n3d_y) && std::abs(n3d_x) <= std::abs(n3d_z)) {
                e0 = 1._prt;
            } else if (std::abs(n3d_y) <= std::abs(n3d_z)) {
                e1 = 1._prt;
            } else {
                e2 = 1._prt;
            }
            amrex::ParticleReal const edn = e0*n3d_x + e1*n3d_y + e2*n3d_z;
            amrex::ParticleReal t10 = e0 - edn*n3d_x;
            amrex::ParticleReal t11 = e1 - edn*n3d_y;
            amrex::ParticleReal t12 = e2 - edn*n3d_z;
            amrex::ParticleReal const t1n = 1._prt/std::sqrt(t10*t10 + t11*t11 + t12*t12);
            t10 *= t1n; t11 *= t1n; t12 *= t1n;
            amrex::ParticleReal const t20 = n3d_y*t12 - n3d_z*t11;
            amrex::ParticleReal const t21 = n3d_z*t10 - n3d_x*t12;
            amrex::ParticleReal const t22 = n3d_x*t11 - n3d_y*t10;

            auto const cc = static_cast<amrex::ParticleReal>(PhysConst::c);
            amrex::ParticleReal const u_norm =
                cc * static_cast<amrex::ParticleReal>(generateGaussianFluxDist(0._rt, m_uth, engine));
            amrex::ParticleReal const u_tang1 =
                cc * static_cast<amrex::ParticleReal>(amrex::RandomNormal(0._rt, m_uth, engine));
            amrex::ParticleReal const u_tang2 =
                cc * static_cast<amrex::ParticleReal>(amrex::RandomNormal(0._rt, m_uth, engine));
            amrex::ParticleReal const ux_new = u_norm*n3d_x + u_tang1*t10 + u_tang2*t20;
            amrex::ParticleReal const uy_new = u_norm*n3d_y + u_tang1*t11 + u_tang2*t21;
            amrex::ParticleReal const uz_new = u_norm*n3d_z + u_tang1*t12 + u_tang2*t22;
            ptd.m_rdata[PIdx::ux][i] = ux_new;
            ptd.m_rdata[PIdx::uy][i] = uy_new;
            ptd.m_rdata[PIdx::uz][i] = uz_new;

            // Advance from the contact point, with the new velocity,
            // for the remaining fraction of the timestep
            amrex::ParticleReal x_new = x_c, y_new = y_c, z_new = z_c;
            UpdatePosition(x_new, y_new, z_new, ux_new, uy_new, uz_new,
                           dt_fraction * dt, mass);

            // If re-emission still lands inside the solid (sharp concave
            // corner), absorb the particle rather than let it accumulate
            // inside the EB.
            int ri, rj, rk;
            amrex::Real rW[AMREX_SPACEDIM][2];
            ablastr::particles::compute_weights<amrex::IndexType::NODE>(
                x_new, y_new, z_new, plo, dxi, ri, rj, rk, rW);
            if (ablastr::particles::interp_field_nodal(ri, rj, rk, rW, distance_to_eb) < 0.0) {
                amrex::ParticleIDWrapper{ptd.m_idcpu[i]}.make_invalid();
                return;
            }
            setPosition(i, x_new, y_new, z_new);
        } else {
            // Specular reflection: u_refl = u - 2 (u.n) n
            amrex::ParticleReal const u_dot_n = ux*n3d_x + uy*n3d_y + uz*n3d_z;
            amrex::ParticleReal const ux_refl = ux - 2.0_prt * u_dot_n * n3d_x;
            amrex::ParticleReal const uy_refl = uy - 2.0_prt * u_dot_n * n3d_y;
            amrex::ParticleReal const uz_refl = uz - 2.0_prt * u_dot_n * n3d_z;
            // Store reflected velocity
            ptd.m_rdata[PIdx::ux][i] = ux_refl;
            ptd.m_rdata[PIdx::uy][i] = uy_refl;
            ptd.m_rdata[PIdx::uz][i] = uz_refl;

            // Advance from the contact point, with the reflected velocity,
            // for the remaining fraction of the timestep
            amrex::ParticleReal x_new = x_c, y_new = y_c, z_new = z_c;
            UpdatePosition(x_new, y_new, z_new, ux_refl, uy_refl, uz_refl,
                           dt_fraction * dt, mass);
            // SetParticlePosition handles the conversion to the stored
            // (reduced-dimension) coordinates.
            setPosition(i, x_new, y_new, z_new);
        }
    }
};

}

#endif //WARPX_PARTICLEBOUNDARYPROCESS_H_
