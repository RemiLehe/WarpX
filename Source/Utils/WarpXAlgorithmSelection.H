/* Copyright 2019 David Grote, Luca Fedeli, Remi Lehe
 * Yinjian Zhao
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_UTILS_WARPXALGORITHMSELECTION_H_
#define WARPX_UTILS_WARPXALGORITHMSELECTION_H_

#include <AMReX_BaseFwd.H>
#include <AMReX_Enum.H>
#include <ablastr/utils/Enums.H>

#include <string>

using namespace ablastr::utils::enums;  // NOLINT(google-global-names-in-headers)


/**
  * \brief struct to determine the computational medium, i.e., vacuum or material/macroscopic
           default is vacuum.
  */
AMREX_ENUM(MediumForEM,
           Vacuum,
           Macroscopic,
           Default = Vacuum);

/**
  * \brief struct to select the overall evolve scheme
  */
AMREX_ENUM(EvolveScheme,
           Explicit,
           Theta_Implicit_EM,
           Semi_Implicit_EM,
           Strang_Implicit_Spectral_EM,
           Semi_Implicit_Darwin,
           Default = Explicit);

/**
  * \brief struct to select algorithm for macroscopic Maxwell solver
           LaxWendroff (semi-implicit) represents sigma*E = sigma*0.5*(E^(n) + E^(n+1))
           Backward Euler (fully-implicit) represents sigma*E = sigma*E^(n+1)
           default is Backward Euler as it is more robust.
  */
AMREX_ENUM(MacroscopicSolverAlgo,
           BackwardEuler,
           LaxWendroff,
           Default = BackwardEuler);

AMREX_ENUM(ElectromagneticSolverAlgo,
           None,
           Yee,
           CKC,
           PSATD,
           ECT,
           HybridPIC,
           hybrid = HybridPIC,
           Default = Yee);

AMREX_ENUM(ElectrostaticSolverAlgo,
           None,
           Relativistic,
           LabFrameElectroMagnetostatic,
           LabFrame,
           LabFrameEffectivePotential,
           Default = None);

AMREX_ENUM(PoissonSolverAlgo,
           Multigrid,
           IntegratedGreenFunction,
           fft = IntegratedGreenFunction,
           Default = Multigrid);

AMREX_ENUM(ParticlePusherAlgo,
           Boris,
           Vay,
           HigueraCary,
           higuera = HigueraCary,
           Default = Boris);

AMREX_ENUM(CurrentDepositionAlgo,
           Esirkepov,
           Direct,
           Vay,
           Villasenor,
           Default = Esirkepov);

AMREX_ENUM(ChargeDepositionAlgo,
           Standard,
           Default = Standard);

AMREX_ENUM(GatheringAlgo,
           EnergyConserving,
           MomentumConserving,
           Default = EnergyConserving);

AMREX_ENUM(PSATDSolutionType,
           FirstOrder,
           SecondOrder,
           Default = SecondOrder);

AMREX_ENUM(TimeDependencyJ,
           Constant,
           Linear,
           Quadratic,
           Default = Constant);

AMREX_ENUM(TimeDependencyRho,
           Constant,
           Linear,
           Quadratic,
           Default = Linear);

/** Strategy to compute weights for use in load balance.
 */
AMREX_ENUM(LoadBalanceCostsUpdateAlgo,
           Timers,     //!< load balance according to in-code timer-based weights (i.e., with  `costs`)
           Heuristic,  /**< load balance according to weights computed from number of cells
                          and number of particles per box (i.e., with `costs_heuristic`) */
           Default = Timers);

/** Field boundary conditions at the domain boundary
 */
AMREX_ENUM(FieldBoundaryType,
           PML,
           Periodic,
           PEC,      //!< perfect electric conductor (PEC) with E_tangential=0
           PMC,      //!< perfect magnetic conductor (PMC) with B_tangential=0
           Neumann = PMC, // For electrostatic, the normal E is set to zero
           Damped,   // Fields in the guard cells are damped for PSATD
                     //in the moving window direction
           Absorbing_Silver_Mueller, // Silver-Mueller boundary condition
           None,    // The fields values at the boundary are not updated. This is
                    // useful for RZ simulations, at r=0.
           Open,    // Used in the Integrated Green Function Poisson solver
                    // Note that the solver implicitely assumes open BCs:
                    // no need to enforce them separately
           PEC_Insulator, // Mixed boundary with PEC and insulator
           Default = PML);

/** Particle boundary conditions at the domain boundary
 */
AMREX_ENUM(ParticleBoundaryType,
           Absorbing,     //!< particles crossing domain boundary are removed
           Open,          //!< particles cross domain boundary leave with damped j
           Reflecting,    //!< particles are reflected
           Periodic,      //!< particles are introduced from the periodic boundary
           Thermal,
           None,          //!< For r=0 boundary with RZ simulations
           Default = Absorbing);

/** MPI reductions
 */
AMREX_ENUM(ReductionType,
           Maximum,
           Minimum,
           Sum,
           Integral = Sum);

/** \brief Subcycling half selector */
AMREX_ENUM(SubcyclingHalf,
           FirstHalf,
           SecondHalf,
           None,
           Default = None);

/** \brief Particle push scheme */
AMREX_ENUM(PushType,
           Explicit,  //!< Standard leap-frog scheme
           Implicit,  //!< Crank-Nicolson scheme (e.g., equations (15)-(18) in (Chen et al, JCP 407, 2020))
           Default = Explicit);

/** \brief For advanced collision algorithms that split the particle push in substeps */
AMREX_ENUM(PositionPushType,
           None,
           Full,
           Default = Full);

/** \brief For advanced collision algorithms that split the particle push in substeps */
AMREX_ENUM(MomentumPushType,
           Full,
           FirstHalf,
           SecondHalf,
           Default = Full);

/** \brief How the Boris rotation obtains the Lorentz factor gamma_bar.
 *
 * The semi-implicit Darwin scheme splits one Boris push into a predictor
 * (electrostatic field only) and a corrector (inductive field only, applied to
 * zeroed velocities). That split is exact only if both halves rotate with the
 * *same* gamma_bar, which must also be the gamma_bar used to build the mass
 * matrix, otherwise the mass matrix is not the Jacobian of the push that is
 * actually applied and the energy-conservation proof does not close.
 *
 * Freezing gamma_bar at a value that does not depend on the inductive field is
 * also what keeps the implicit system linear in E_sol.
 *
 * Compute       : usual behaviour, gamma is recomputed inside the pusher.
 * ComputeAndStore: compute gamma as usual and save it to the `inv_gamma_bar`
 *                  particle attribute (used by the Darwin predictor push).
 * UseStored     : rotate with the previously stored `inv_gamma_bar` (used by
 *                 the Darwin corrector push, where the velocities are zeroed
 *                 and an internally computed gamma would be meaningless).
 */
AMREX_ENUM(BorisGammaMode,
           Compute,
           ComputeAndStore,
           UseStored,
           Default = Compute);

/** \brief Particle shuffling method for binary collisions */
AMREX_ENUM(
    ParticleShufflingMethod,
    None,
    FisherYates,
    Modulus,
    Standard,
    Default = FisherYates
);

/** \brief For binary collision algorithms, the strategy to determine
 *  the scattering angle in the center of mass frame */
AMREX_ENUM(ScatteringAngleModel,
           Forward,    /**< Scattering angle is zero, i.e., products have the same direction as the incident particle in the center of mass frame */
           Backward,   /**< Scattering angle is pi, i.e., products have the opposite direction of the incident particle in the center of mass frame */
           Isotropic,  /**< Scattering angle is random, i.e., products are scattered isotropically in the center of mass frame */
           Default = Isotropic  /**< Isotropic */
);

#endif // WARPX_UTILS_WARPXALGORITHMSELECTION_H_
