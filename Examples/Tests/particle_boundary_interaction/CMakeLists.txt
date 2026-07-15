# Add tests (alphabetical order) ##############################################
#

if(WarpX_EB)
    add_warpx_test(
        test_rz_particle_boundary_interaction_picmi  # name
        RZ  # dims
        2  # nprocs
        inputs_test_rz_particle_boundary_interaction_picmi.py  # inputs
        "analysis.py diags/diag1/"  # analysis
        "analysis_default_regression.py --path diags/diag1/"  # checksum
        OFF  # dependency
    )
    add_warpx_test(
        test_rz_particle_boundary_interaction_reflecting  # name
        RZ  # dims
        2  # nprocs
        inputs_test_rz_particle_boundary_interaction_reflecting  # inputs
        "analysis.py diags/diag1/"  # analysis
        "analysis_default_regression.py --path diags/diag1/"  # checksum
        OFF  # dependency
    )
    add_warpx_test(
        test_3d_particle_boundary_interaction_thermal_eb  # name
        3  # dims
        1  # nprocs
        inputs_test_3d_particle_boundary_interaction_thermal_eb  # inputs
        "analysis_thermal.py diags/diag1/"  # analysis
        OFF  # checksum (statistical test; validated by analysis_thermal.py)
        OFF  # dependency
    )
endif()
