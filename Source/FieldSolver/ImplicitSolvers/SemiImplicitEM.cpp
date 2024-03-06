/* Copyright 2024 Justin Angus
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "SemiImplicitEM.H"
#include "WarpX.H"

void SemiImplicitEM::Define ( WarpX*  a_WarpX )
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_is_defined,
        "SemiImplicitEM object is already defined!");

    // Retain a pointer back to main WarpX class
    m_WarpX = a_WarpX;

    // Define E vectors
    m_E.Define( m_WarpX->getEfield_fp_vec() );
    m_Eold.Define( m_WarpX->getEfield_fp_vec() );

    // Need to define the WarpXSolverVec owned dot_mask to do dot
    // product correctly for linear and nonlinear solvers
    const amrex::Vector<amrex::Geometry>& Geom = m_WarpX->Geom();
    m_E.SetDotMask(Geom);

    // Parse implicit solver parameters
    const amrex::ParmParse pp("implicit_evolve");

    std::string nlsolver_type_str;
    pp.query("nonlinear_solver", nlsolver_type_str);
    if (nlsolver_type_str=="picard") {
        m_nlsolver_type = NonlinearSolverType::Picard;
        m_max_particle_iterations = 1;
        m_particle_tolerance = 0.0;
        m_nlsolver = std::make_unique<PicardSolver<WarpXSolverVec,SemiImplicitEM>>();
    }
    else if (nlsolver_type_str=="newton") {
        m_nlsolver_type = NonlinearSolverType::Newton;
        pp.query("max_particle_iterations", m_max_particle_iterations);
        pp.query("particle_tolerance", m_particle_tolerance);
        m_nlsolver = std::make_unique<NewtonSolver<WarpXSolverVec,SemiImplicitEM>>();
    }
    else {
        WARPX_ABORT_WITH_MESSAGE(
            "invalid nonlinear_solver specified. Valid options are picard and newton.");
    }

    // Define the nonlinear solver
    m_nlsolver->Define(m_E, this);

    m_is_defined = true;
}

void SemiImplicitEM::PrintParameters () const
{
    if (!m_WarpX->Verbose()) { return; }
    amrex::Print() << std::endl;
    amrex::Print() << "-----------------------------------------------------------" << std::endl;
    amrex::Print() << "----------- SEMI IMPLICIT EM SOLVER PARAMETERS ------------" << std::endl;
    amrex::Print() << "-----------------------------------------------------------" << std::endl;
    amrex::Print() << "max particle iterations:    " << m_max_particle_iterations << std::endl;
    amrex::Print() << "particle tolerance:         " << m_particle_tolerance << std::endl;
    if (m_nlsolver_type==NonlinearSolverType::Picard) {
        amrex::Print() << "Nonlinear solver type:      Picard" << std::endl;
    }
    else if (m_nlsolver_type==NonlinearSolverType::Newton) {
        amrex::Print() << "Nonlinear solver type:      Newton" << std::endl;
    }
    m_nlsolver->PrintParams();
    amrex::Print() << "-----------------------------------------------------------" << std::endl;
    amrex::Print() << std::endl;
}

void SemiImplicitEM::OneStep ( amrex::Real  a_time,
                               amrex::Real  a_dt,
                               int          a_step )
{
    using namespace amrex::literals;
    amrex::ignore_unused(a_step);

    // Fields have E^{n}, B^{n-1/2]
    // Particles have p^{n} and x^{n}.

    // Save the values at the start of the time step,
    m_WarpX->SaveParticlesAtImplicitStepStart ( );

    // Save the fields at the start of the step
    m_Eold.Copy( m_WarpX->getEfield_fp_vec() );
    m_E = m_Eold; // initial guess for E

    // Compute Bfield at time n+1/2
    m_WarpX->EvolveB(a_dt, DtType::Full);
    m_WarpX->ApplyMagneticFieldBCs();

    const amrex::Real half_time = a_time + 0.5_rt*a_dt;

    // Solve nonlinear system for E at t_{n+1/2}
    // Particles will be advanced to t_{n+1/2}
    m_nlsolver->Solve( m_E, m_Eold, half_time, a_dt );

    // update WarpX owned Efield_fp and Bfield_fp to t_{n+1/2}
    UpdateWarpXState( m_E, half_time, a_dt );

    // Update field boundary probes prior to updating fields to t_{n+1}
    //m_fields->updateBoundaryProbes( a_dt )

    // Advance particles from time n+1/2 to time n+1
    m_WarpX->FinishImplicitParticleUpdate();

    // Advance E fields from time n+1/2 to time n+1
    // Eg^{n+1} = 2.0*E_g^{n+1/2} - E_g^n
    m_E.linComb( 2._rt, m_E, -1._rt, m_Eold );
    const amrex::Real new_time = a_time + a_dt;
    UpdateWarpXState( m_E, new_time, a_dt );

}

void SemiImplicitEM::PreRHSOp ( const WarpXSolverVec&  a_E,
                                amrex::Real            a_time,
                                amrex::Real            a_dt,
                                int                    a_nl_iter,
                                bool                   a_from_jacobian )
{
    amrex::ignore_unused(a_E);

    // update derived variable B and then update WarpX owned Efield_fp and Bfield_fp
    UpdateWarpXState( a_E, a_time, a_dt );

    // Advance the particle positions by 1/2 dt,
    // particle velocities by dt, then take average of old and new v,
    // deposit currents, giving J at n+1/2 used in ComputeRHSE below
    m_WarpX->PreRHSOp( a_time, a_dt, a_nl_iter, a_from_jacobian );

}

void SemiImplicitEM::ComputeRHS ( WarpXSolverVec&  a_Erhs,
                            const WarpXSolverVec&  a_E,
                                  amrex::Real      a_time,
                                  amrex::Real      a_dt )
{
    amrex::ignore_unused(a_E, a_time);
    using namespace amrex::literals;
    m_WarpX->ComputeRHSE(0.5_rt*a_dt, a_Erhs);
}

void SemiImplicitEM::UpdateWarpXState ( const WarpXSolverVec&  a_E,
                                        amrex::Real            a_time,
                                        amrex::Real            a_dt )
{
    using namespace amrex::literals;
    amrex::ignore_unused(a_time,a_dt);

    // Update Efield_fp owned by WarpX
    m_WarpX->SetElectricFieldAndApplyBCs( a_E );

    // The B field update needs. Talk to DG about this. Only needed when B updates?
    if (WarpX::num_mirrors>0){
        m_WarpX->applyMirrors(a_time);
        // E : guard cells are NOT up-to-date from the mirrors
        // B : guard cells are NOT up-to-date from the mirrors
    }

}
