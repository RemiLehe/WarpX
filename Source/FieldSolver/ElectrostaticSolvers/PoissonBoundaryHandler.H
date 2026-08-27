/* Copyright 2024 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Roelof Groenewald (TAE Technologies)
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_BOUNDARYHANDLER_H_
#define WARPX_BOUNDARYHANDLER_H_

#include "Utils/Parser/ParserUtils.H"

#include <AMReX_Array.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MLLinOp.H>
#include <AMReX_MLMG.H>
#include <AMReX_MLNodeLinOp.H>
#include <AMReX_REAL.H>
#include <AMReX_Parser.H>

#include <array>
#include <string>
#include <utility>


class PoissonBoundaryHandler
{
public:
    PoissonBoundaryHandler (); // constructor

    /** Read runtime parameters. Called in constructor. */
    void ReadParameters ();

    /**
     * \brief Read the input settings and set the boundary conditions used
     * on each domain boundary for the Poisson solver.
     */
    void DefinePhiBCs (const amrex::Geometry& geom);

    /**
     * \brief Initialize amrex::Parser objects to get the boundary potential
     * values at specified times.
     */
    void BuildParsers ();
    void BuildParsersEB ();

    /**
     * \brief Sets the EB potential string and updates the function parser.
     *
     * \param [in] potential The string value of the potential
     */
    void setPotentialEB(const std::string& potential) {
        potential_eb_str = potential;
        BuildParsersEB();
    }

    struct PhiCalculatorEB {

        amrex::Real t;
        amrex::ParserExecutor<4> potential_eb;

        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        amrex::Real operator()(const amrex::Real x, const amrex::Real z) const noexcept {
            using namespace amrex::literals;
            return potential_eb(x, 0.0_rt, z, t);
        }

        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        amrex::Real operator()(const amrex::Real x, const amrex::Real y, const amrex::Real z) const noexcept {
            return potential_eb(x, y, z, t);
        }
    };

    [[nodiscard]] PhiCalculatorEB
    getPhiEB(amrex::Real t) const noexcept
    {
        return PhiCalculatorEB{t, potential_eb};
    }

    bool m_boundary_potential_specified = false;

    // set default potentials to zero in order for current tests to pass
    // but forcing the user to specify a potential might be better
    std::string potential_xlo_str = "0";
    std::string potential_xhi_str = "0";
    std::string potential_ylo_str = "0";
    std::string potential_yhi_str = "0";
    std::string potential_zlo_str = "0";
    std::string potential_zhi_str = "0";
    std::string potential_eb_str = "0";

    amrex::ParserExecutor<1> potential_xlo;
    amrex::ParserExecutor<1> potential_xhi;
    amrex::ParserExecutor<1> potential_ylo;
    amrex::ParserExecutor<1> potential_yhi;
    amrex::ParserExecutor<1> potential_zlo;
    amrex::ParserExecutor<1> potential_zhi;
    amrex::ParserExecutor<1> potential_eb_t;
    amrex::ParserExecutor<4> potential_eb;

    amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> lobc, hibc;
    std::array<bool, AMREX_SPACEDIM * 2> dirichlet_flag;
    bool has_non_periodic = false;
    bool phi_EB_only_t = true;

private:

    amrex::Parser potential_xlo_parser;
    amrex::Parser potential_xhi_parser;
    amrex::Parser potential_ylo_parser;
    amrex::Parser potential_yhi_parser;
    amrex::Parser potential_zlo_parser;
    amrex::Parser potential_zhi_parser;
    amrex::Parser potential_eb_parser;
};

/** use amrex to directly calculate the electric field since with EB's the
 *
 * simple finite difference scheme in WarpX::computeE sometimes fails
 */
class EBCalcEfromPhiPerLevel {
    private:
    amrex::Vector<
        amrex::Array<amrex::MultiFab *, AMREX_SPACEDIM>
    > m_e_field;

    public:
    explicit EBCalcEfromPhiPerLevel(amrex::Vector<amrex::Array<amrex::MultiFab *, AMREX_SPACEDIM> > e_field)
            : m_e_field(std::move(e_field)) {}

    void operator()(amrex::MLMG & mlmg, int const lev) {
        using namespace amrex::literals;

        mlmg.getGradSolution({m_e_field[lev]});
        for (auto &field: m_e_field[lev]) {
            field->mult(-1._rt);
        }
    }

    /** \brief Same, but computed directly from the linear operator and `phi`
     *
     * `amrex::MLMG::getGradSolution` takes the potential from the internal state
     * that `amrex::MLMG::solve` leaves behind, so it can only be used right after
     * an MLMG solve. This overload is for the solvers that compute `phi` without
     * going through `amrex::MLMG::solve`, such as the PETSc Poisson solver.
     *
     * \param[in] linop the linear operator of the Poisson solve
     * \param[inout] phi the potential; only its ghost nodes are modified
     * \param[in] lev the mesh refinement level
     */
    void operator()(amrex::MLNodeLinOp & linop, amrex::MultiFab & phi, int const lev) {
        using namespace amrex::literals;

        // Fill the ghost nodes of `phi`, in particular the ones that lie outside
        // of a Neumann boundary, which `compGrad` reads. `amrex::MLMG::solve`
        // does this internally on the potential that it hands to `compGrad`.
        linop.applyBC(0, 0, phi, amrex::MLNodeLinOp::BCMode::Inhomogeneous,
                      amrex::MLNodeLinOp::StateMode::Solution);
        linop.compGrad(0, m_e_field[lev], phi, amrex::MLNodeLinOp::Location::FaceCenter);
        for (auto &field: m_e_field[lev]) {
            field->mult(-1._rt);
        }
    }
};

#endif // WARPX_BOUNDARYHANDLER_H_
