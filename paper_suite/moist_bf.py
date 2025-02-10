"""
The moist rising bubble test from Bryan & Fritsch, 2002:
``A Benchmark Simulation for Moist Nonhydrostatic Numerical Models'', GMD.

The test simulates a rising thermal in a cloudy atmosphere, which is fueled by
latent heating from condensation.

This setup uses a vertical slice with the order 1 finite elements.
"""
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from firedrake import (
    PeriodicIntervalMesh, ExtrudedMesh, SpatialCoordinate, conditional, cos, pi,
    sqrt, NonlinearVariationalProblem, NonlinearVariationalSolver, TestFunction,
    dx, TrialFunction, Function, as_vector, LinearVariationalProblem,
    LinearVariationalSolver, Constant
)

from gusto import (
    Domain, CompressibleEulerEquations, IO, CompressibleParameters, DGUpwind,
    SSPRK3, TrapeziumRule, SemiImplicitQuasiNewton, EmbeddedDGOptions,
    WaterVapour, CloudWater, OutputParameters, Theta_e, SaturationAdjustment,
    ForwardEuler, saturated_hydrostatic_balance, thermodynamics, Recoverer,
    CompressibleSolver, Timestepper, split_continuity_form,
    IMEXRungeKutta, time_derivative, transport, implicit, explicit, physics_label,
    IMEX_Euler, SDC, SplitPhysicsTimestepper, IMEX_SSP3, source_label, SUPGOptions,
    horizontal, vertical, Split_DGUpwind, split_hv_advective_form
)

import numpy as np

import time

moist_bryan_fritsch_defaults = {
    'ncolumns': 100,
    'nlayers': 100,
    'dt': 1.0,
    'tmax': 1.0,
    'dumpfreq': 125,
    'dirname': 'moist_bryan_fritsch_imex_sdc_nonsplit'
}


def moist_bryan_fritsch(
        ncolumns=moist_bryan_fritsch_defaults['ncolumns'],
        nlayers=moist_bryan_fritsch_defaults['nlayers'],
        dt=moist_bryan_fritsch_defaults['dt'],
        tmax=moist_bryan_fritsch_defaults['tmax'],
        dumpfreq=moist_bryan_fritsch_defaults['dumpfreq'],
        dirname=moist_bryan_fritsch_defaults['dirname']
):

    # ------------------------------------------------------------------------ #
    # Parameters for test case
    # ------------------------------------------------------------------------ #
    domain_width = 10000.     # domain width, in m
    domain_height = 10000.    # domain height, in m
    zc = 2000.                # vertical centre of bubble, in m
    rc = 2000.                # radius of bubble, in m
    Tdash = 2.0               # strength of temperature perturbation, in K
    Tsurf = 320.0             # background theta_e value, in K
    total_water = 0.02        # total moisture mixing ratio, in kg/kg

    # ------------------------------------------------------------------------ #
    # Our settings for this set up
    # ------------------------------------------------------------------------ #
    element_order = 1
    u_eqn_type = 'vector_advection_form'

    # ------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #

    # Domain
    base_mesh = PeriodicIntervalMesh(ncolumns, domain_width)
    mesh = ExtrudedMesh(
        base_mesh, layers=nlayers, layer_height=domain_height/nlayers
    )
    domain = Domain(mesh, dt, 'CG', element_order)

    # Equation
    params = CompressibleParameters()
    tracers = [WaterVapour(), CloudWater()]
    eqns = CompressibleEulerEquations(
        domain, params, active_tracers=tracers, u_transport_option=u_eqn_type)

    eqns = split_continuity_form(eqns)
    eqns = split_hv_advective_form(eqns, "rho")
    eqns = split_hv_advective_form(eqns, "theta")


    opts =SUPGOptions(suboptions={"theta": [transport],
                                "water_vapour":[transport],
                                "cloud_water": [transport]})
    # Check number of optimal cores
    print("Opt Cores:", eqns.X.function_space().dim()/50000.)
    # I/O
    output = OutputParameters(
        dirname=dirname, dumpfreq=dumpfreq, dump_vtus=False, dump_nc=True
    )
    diagnostic_fields = [Theta_e(eqns)]
    io = IO(domain, output, diagnostic_fields=diagnostic_fields)

    transport_methods = [
        DGUpwind(eqns, "u"), Split_DGUpwind(eqns, "rho"), Split_DGUpwind(eqns, "theta", ibp=SUPGOptions.ibp),
        DGUpwind(eqns, "water_vapour", ibp=SUPGOptions.ibp), DGUpwind(eqns, "cloud_water", ibp=SUPGOptions.ibp) 
    ]

    nl_solver_parameters = {
    "snes_converged_reason": None,
    "snes_lag_preconditioner_persists":None,
    "snes_lag_preconditioner":-2,
    "snes_lag_jacobian": -2,
    "snes_lag_jacobian_persists": None,
    'ksp_ew': None,
    'ksp_ew_version': 1,
    "ksp_ew_threshold": 1e-2,
    "ksp_ew_rtol0": 1e-3,
    "mat_type": "matfree",
    "ksp_type": "gmres",
    "ksp_converged_reason": None,
    "ksp_atol": 1e-4,
    "ksp_rtol": 1e-4,
    "snes_atol": 1e-4,
    "snes_rtol": 1e-4,
    "ksp_max_it": 400,
    "pc_type": "python",
    "pc_python_type": "firedrake.AssembledPC","assembled": {
        "pc_type": "python",
        "pc_python_type": "firedrake.ASMStarPC",
        "pc_star": {
            "construct_dim": 0,
            "sub_sub": {
                "pc_type": "lu",
                "pc_factor_mat_ordering_type": "rcm",
                "pc_factor_reuse_ordering": None,
                "pc_factor_reuse_fill": None,
                "pc_factor_fill": 1.2
            }
        },
    },}
    physics_schemes = [SaturationAdjustment(eqns)]
    eqns.label_terms(lambda t: not any(t.has_label(time_derivative, transport, source_label)), implicit)
    eqns.label_terms(lambda t: t.has_label(transport) and t.has_label(horizontal), explicit)
    eqns.label_terms(lambda t: t.has_label(transport) and t.has_label(vertical), implicit)
    eqns.label_terms(lambda t: t.has_label(transport) and not any(t.has_label(horizontal, vertical)), explicit)
    base_scheme = IMEX_Euler(domain, options=opts, nonlinear_solver_parameters=nl_solver_parameters)
    node_type = "LEGENDRE"
    qdelta_exp = "FE"
    quad_type = "GAUSS"
    M = 2
    k = 3
    qdelta_imp = "LU"
    scheme =SDC(base_scheme, domain, M, k, quad_type, node_type, qdelta_imp,
                        qdelta_exp, formulation="Z2N", options=opts, nonlinear_solver_parameters=nl_solver_parameters,final_update=True, initial_guess="copy")
    #scheme = IMEX_SSP3(domain, nonlinear_solver_parameters=nl_solver_parameters)
    # Time stepper
    stepper = Timestepper(eqns, scheme, io, transport_methods, physics_parametrisations=physics_schemes)


    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    u0 = stepper.fields("u")
    rho0 = stepper.fields("rho")
    theta0 = stepper.fields("theta")
    water_v0 = stepper.fields("water_vapour")
    water_c0 = stepper.fields("cloud_water")

    # spaces
    Vt = domain.spaces("theta")
    Vr = domain.spaces("DG")
    x, z = SpatialCoordinate(mesh)
    quadrature_degree = (4, 4)
    dxp = dx(degree=(quadrature_degree))

    # Define constant theta_e and water_t
    theta_e = Function(Vt).assign(Tsurf)
    water_t = Function(Vt).assign(total_water)

    # Calculate hydrostatic fields
    saturated_hydrostatic_balance(eqns, stepper.fields, theta_e, water_t)

    # make mean fields
    theta_b = Function(Vt).assign(theta0)
    rho_b = Function(Vr).assign(rho0)
    water_vb = Function(Vt).assign(water_v0)
    water_cb = Function(Vt).assign(water_t - water_vb)

    # define perturbation
    xc = domain_width / 2
    r = sqrt((x - xc) ** 2 + (z - zc) ** 2)
    theta_pert = Function(Vt).interpolate(
        conditional(
            r > rc,
            0.0,
            Tdash * (cos(pi * r / (2.0 * rc))) ** 2
        )
    )

    # define initial theta
    theta0.interpolate(theta_b * (theta_pert / 300.0 + 1.0))

    # find perturbed rho
    gamma = TestFunction(Vr)
    rho_trial = TrialFunction(Vr)
    a = gamma * rho_trial * dxp
    L = gamma * (rho_b * theta_b / theta0) * dxp
    rho_problem = LinearVariationalProblem(a, L, rho0)
    rho_solver = LinearVariationalSolver(rho_problem)
    rho_solver.solve()

    # find perturbed water_v
    w_v = Function(Vt)
    phi = TestFunction(Vt)
    rho_averaged = Function(Vt)
    rho_recoverer = Recoverer(rho0, rho_averaged)
    rho_recoverer.project()

    exner = thermodynamics.exner_pressure(eqns.parameters, rho_averaged, theta0)
    p = thermodynamics.p(eqns.parameters, exner)
    T = thermodynamics.T(eqns.parameters, theta0, exner, r_v=w_v)
    w_sat = thermodynamics.r_sat(eqns.parameters, T, p)

    w_functional = (phi * w_v * dxp - phi * w_sat * dxp)
    w_problem = NonlinearVariationalProblem(w_functional, w_v)
    w_solver = NonlinearVariationalSolver(w_problem)
    w_solver.solve()

    water_v0.assign(w_v)
    water_c0.assign(water_t - water_v0)

    # wind initially zero
    u0.project(as_vector(
        [Constant(0.0, domain=mesh), Constant(0.0, domain=mesh)]
    ))

    stepper.set_reference_profiles(
        [
            ('rho', rho_b),
            ('theta', theta_b),
            ('water_vapour', water_vb),
            ('cloud_water', water_cb)
        ]
    )

    # ------------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------------ #
    initial_time = time.time()
    stepper.run(t=0, tmax=tmax)
    end_time=time.time()
    print("Time taken:", end_time-initial_time)
    print("Total KSP iterations: ", scheme.linear_iterations)

# ---------------------------------------------------------------------------- #
# MAIN
# ---------------------------------------------------------------------------- #


if __name__ == "__main__":

    parser = ArgumentParser(
        description=__doc__,
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--ncolumns',
        help="The number of columns in the vertical slice mesh.",
        type=int,
        default=moist_bryan_fritsch_defaults['ncolumns']
    )
    parser.add_argument(
        '--nlayers',
        help="The number of layers for the mesh.",
        type=int,
        default=moist_bryan_fritsch_defaults['nlayers']
    )
    parser.add_argument(
        '--dt',
        help="The time step in seconds.",
        type=float,
        default=moist_bryan_fritsch_defaults['dt']
    )
    parser.add_argument(
        "--tmax",
        help="The end time for the simulation in seconds.",
        type=float,
        default=moist_bryan_fritsch_defaults['tmax']
    )
    parser.add_argument(
        '--dumpfreq',
        help="The frequency at which to dump field output.",
        type=int,
        default=moist_bryan_fritsch_defaults['dumpfreq']
    )
    parser.add_argument(
        '--dirname',
        help="The name of the directory to write to.",
        type=str,
        default=moist_bryan_fritsch_defaults['dirname']
    )
    args, unknown = parser.parse_known_args()

    moist_bryan_fritsch(**vars(args))
