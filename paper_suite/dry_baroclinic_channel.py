"""
The dry baroclinic wave in a channel from the appendix of Ullrich, Reed &
Jablonowski, 2015:
``Analytical initial conditions and an analysis of baroclinic instability waves
in f - and β-plane 3D channel models'', QJRMS.

This test emulates a dry baroclinic wave in a channel

The setup here is for the order 1 finite elements, in a 3D slice which is
periodic in the x direction but with rigid walls in the y direction.
"""
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from firedrake import (
    PeriodicRectangleMesh, ExtrudedMesh, SpatialCoordinate, conditional, cos,
    sin, pi, sqrt, ln, exp, Constant, Function, as_vector, errornorm, norm
)
from firedrake.fml import subject, drop
from gusto import (
    Domain, CompressibleParameters, CompressibleSolver, WaterVapour, CloudWater,
    CompressibleEulerEquations, OutputParameters, IO, logger, SSPRK3,
    DGUpwind, SemiImplicitQuasiNewton, compressible_hydrostatic_balance,
    Perturbation, SaturationAdjustment, ForwardEuler, thermodynamics,
    Temperature, Exner, Pressure, Split_DGUpwind,
    time_derivative, transport, implicit, explicit, split_continuity_form,
    horizontal, vertical,
    SDC, IMEX_Euler, Timestepper, IMEX_SSP3, SpongeLayerParameters, IMEX_ARK2,
    XComponent, YComponent, ZComponent, ImplicitMidpoint, prognostic, coriolis,
    split_hv_advective_form, TrapeziumRule, IMEXRungeKutta, MixedFSLimiter,
    ThetaLimiter, SUPGOptions, split_hv_continuity_form)
import numpy as np
import time

dry_baroclinic_channel_defaults = {
    'nx': 100,                  # number of columns in x-direction
    'ny': 15,                  # number of columns in y-direction
    'nlayers': 15,             # number of layers in mesh
    'dt': 1800,                # 30 minutes
    'tmax': 24*60*60*12,            # 15 days
    'dumpfreq': 48,           # Corresponds to every 1 day with default opts
    'dirname': 'dry_baro_channel_exner'  # output directory
}


def dry_baroclinic_channel(
        nx=dry_baroclinic_channel_defaults['nx'],
        ny=dry_baroclinic_channel_defaults['ny'],
        nlayers=dry_baroclinic_channel_defaults['nlayers'],
        dt=dry_baroclinic_channel_defaults['dt'],
        tmax=dry_baroclinic_channel_defaults['tmax'],
        dumpfreq=dry_baroclinic_channel_defaults['dumpfreq'],
        dirname=dry_baroclinic_channel_defaults['dirname']
):

    # ------------------------------------------------------------------------ #
    # Parameters for test case
    # ------------------------------------------------------------------------ #
    Lx = 4.0e7                   # length of domain in x direction, in m
    Ly = 6.0e6                   # width of domain in y direction, in m
    H = 3.0e4                    # height of domain, in m
    omega = Constant(7.292e-5)   # planetary rotation rate, in 1/s
    phi0 = Constant(pi/4)        # latitude of centre of channel, in radians
    a = Constant(6.371229e6)     # radius of earth, in m
    b = Constant(2)              # vertical width parameter, dimensionless
    T0 = Constant(288.)          # reference temperature, in K
    u0 = Constant(35.)           # reference zonal wind speed, in m/s
    Gamma = Constant(0.005)      # lapse rate, in K/m
    beta0 = Constant(0.0)        # beta-plane parameter, in 1/s
    q0 = Constant(0.016)         # specific humidity parameter, in kg/kg
    xc = 2.0e6                   # x coordinate for centre of perturbation, in m
    yc = 2.5e6                   # y coordinate for centre of perturbation, in m
    Lp = 6.0e5                   # width parameter for perturbation, in m
    up = Constant(1.0)           # strength of wind perturbation, in m/s
    sponge_depth = 6000.0   # depth of sponge layer, in m

    # ------------------------------------------------------------------------ #
    # Our settings for this set up
    # ------------------------------------------------------------------------ #
    element_order = 1
    u_eqn_type = 'vector_advection_form'
    max_iterations = 40          # max num of iterations for finding eta coords
    tolerance = 1e-10            # tolerance of error in finding eta coords

    # ------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #

    # Domain
    base_mesh = PeriodicRectangleMesh(nx, ny, Lx, Ly, "x", quadrilateral=True)
    mesh = ExtrudedMesh(base_mesh, layers=nlayers, layer_height=H/nlayers)
    domain = Domain(mesh, dt, "RTCF", element_order)
    x, y, z = SpatialCoordinate(mesh)

    # Equation
    params = CompressibleParameters(Omega=omega*sin(phi0))
    eqns = CompressibleEulerEquations(
        domain, params, u_transport_option=u_eqn_type,
        no_normal_flow_bc_ids=[1, 2]
    )

    # Check number of optimal cores
    print("Opt Cores:", eqns.X.function_space().dim()/50000.)

    eqns = split_continuity_form(eqns)
    eqns = split_hv_advective_form(eqns, "rho")
    eqns = split_hv_advective_form(eqns, "theta")
    #eqns = split_hv_advective_form(eqns, "u")
    eqns.label_terms(lambda t: not any(t.has_label(time_derivative, transport)), implicit)
    eqns.label_terms(lambda t: t.has_label(transport) and t.has_label(horizontal), explicit)
    eqns.label_terms(lambda t: t.has_label(transport) and t.has_label(vertical), implicit)
    eqns.label_terms(lambda t: t.has_label(transport) and not any(t.has_label(horizontal, vertical)), explicit)
    # eqns.label_terms(lambda t: t.has_label(coriolis), explicit)
    Vt = domain.spaces('theta')
    theta_limiter = MixedFSLimiter(
                    eqns,
                    {'theta': ThetaLimiter(Vt)})

    opts =SUPGOptions(suboptions={"theta":[transport]})

    # I/O
    dirname = 'dry_baroclinic_channel_imex_sdc_dt1800'
    output = OutputParameters(
        dirname=dirname, dumpfreq=dumpfreq, dump_nc=True, dump_vtus=False
    )
    diagnostic_fields = [Perturbation('theta'), Temperature(eqns), Pressure(eqns), XComponent('u'), YComponent('u'), ZComponent('u')]
    io = IO(domain, output, diagnostic_fields=diagnostic_fields)

    transport_methods = [DGUpwind(eqns, "u"),
                     Split_DGUpwind(eqns, "rho"),
                     Split_DGUpwind(eqns, "theta", ibp=SUPGOptions.ibp)]

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

    linear_solver_parameters = {'snes_type': 'ksponly',
                                'ksp_rtol': 1e-5,
                                'ksp_rtol': 1e-7,
                                'ksp_type': 'cg',
                                'pc_type': 'bjacobi',
                                'sub_pc_type': 'ilu'}

    # IMEX time stepper

    base_scheme = IMEX_Euler(domain, limiter = theta_limiter, options = opts, nonlinear_solver_parameters=nl_solver_parameters, linear_solver_parameters=linear_solver_parameters)
    node_type = "LEGENDRE"
    qdelta_imp = "LU"
    qdelta_exp = "FE"
    quad_type = "GAUSS"
    M = 2
    k = 3
    qdelta_imp = "LU"
    scheme =SDC(base_scheme, domain, M, k, quad_type, node_type, qdelta_imp,
                        qdelta_exp, formulation="Z2N", limiter = theta_limiter, options = opts,  nonlinear_solver_parameters=nl_solver_parameters,final_update=True,
                        linear_solver_parameters=linear_solver_parameters, initial_guess="copy")

    #scheme = IMEX_SSP3(domain, limiter = theta_limiter, options=opts, nonlinear_solver_parameters=nl_solver_parameters, linear_solver_parameters=linear_solver_parameters)
    #Time stepper
    stepper = Timestepper(eqns, scheme, io, transport_methods)

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    # Physical parameters
    beta0 = 2 * omega * cos(phi0) / a
    Rd = params.R_d
    Rv = params.R_v
    f0 = 2 * omega * sin(phi0)
    y0 = Constant(Ly / 2)
    g = params.g
    p0 = params.p_0

    # Initial conditions
    u = stepper.fields("u")
    rho = stepper.fields("rho")
    theta = stepper.fields("theta")
    # exner = stepper.fields("exner")

    # spaces
    Vu = u.function_space()
    Vt = theta.function_space()
    Vr = rho.function_space()
    # Ve = exner.function_space()

    # set up background state expressions
    eta = Function(Vt).interpolate(Constant(1e-7))
    Phi = Function(Vt).interpolate(g * z)
    T = Function(Vt)
    Phi_prime = u0 / 2 * (
        (f0 - beta0 * y0) * (y - (Ly / 2) - (Ly / (2 * pi)) * sin(2*pi*y/Ly))
        + beta0 / 2*(
            y**2 - (Ly * y / pi) * sin(2*pi*y/Ly)
            - (Ly**2 / (2 * pi**2)) * cos(2*pi*y/Ly) - (Ly**2 / 3)
            - (Ly**2 / (2 * pi**2))
        )
    )
    Phi_expr = (
        T0 * g / Gamma * (1 - eta ** (Rd * Gamma / g))
        + Phi_prime * ln(eta) * exp(-(ln(eta) / b) ** 2)
    )

    Tv_expr = (
        T0 * eta ** (Rd * Gamma / g) + Phi_prime / Rd * exp(-(ln(eta) / b)**2)
        * ((2 / b**2) * (ln(eta)) ** 2 - 1)
    )
    u_expr = as_vector(
        [-u0 * (sin(pi*y/Ly))**2 * ln(eta) * eta ** (-ln(eta) / b ** 2),
         0.0, 0.0]
    )
    T_expr = Tv_expr

    # do Newton method to obtain eta
    eta_new = Function(Vt)
    F = -Phi + Phi_expr
    dF = -Rd * Tv_expr / eta
    for _ in range(max_iterations):
        eta_new.interpolate(eta - F/dF)
        if errornorm(eta_new, eta) / norm(eta) < tolerance:
            eta.assign(eta_new)
            break
        eta.assign(eta_new)

    # make mean u and theta
    u.project(u_expr)
    T.interpolate(T_expr)
    theta.interpolate(
        thermodynamics.theta(params, T_expr, p0 * eta)
    )
    Phi_test = Function(Vt).interpolate(Phi_expr)
    logger.info(
        f"Error-norm for setting up p: {errornorm(Phi_test, Phi) / norm(Phi)}"
    )

    # Calculate hydrostatic fields
    compressible_hydrostatic_balance(
        eqns, theta, rho, solve_for_rho=True
    )

    # make mean fields
    rho_b = Function(Vr).assign(rho)
    u_b = stepper.fields("ubar", space=Vu, dump=False).project(u)
    theta_b = Function(Vt).assign(theta)

    # define perturbation
    r = sqrt((x - xc) ** 2 + (y - yc) ** 2)
    u_pert = Function(Vu).project(as_vector([up * exp(-(r / Lp)**2), 0.0, 0.0]))

    # define initial u
    u.assign(u_b+u_pert)

    #theta_w3 = Function(Vr).interpolate(theta)

    #exner.interpolate(thermodynamics.exner_pressure(params, rho, theta_w3))

    # initialise fields
    stepper.set_reference_profiles(
        [('rho', rho_b), ('theta', theta_b)]
    )

    # ------------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------------ #
    start_time = time.time()
    stepper.run(t=0, tmax=tmax)
    end_time = time.time()
    print("Time taken: ", end_time - start_time)


# ---------------------------------------------------------------------------- #
# MAIN
# ---------------------------------------------------------------------------- #


if __name__ == "__main__":

    parser = ArgumentParser(
        description=__doc__,
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--nx',
        help="The number of columns in the vertical slice mesh.",
        type=int,
        default=dry_baroclinic_channel_defaults['nx']
    )
    parser.add_argument(
        '--ny',
        help="The number of columns in the vertical slice mesh.",
        type=int,
        default=dry_baroclinic_channel_defaults['ny']
    )
    parser.add_argument(
        '--nlayers',
        help="The number of layers for the mesh.",
        type=int,
        default=dry_baroclinic_channel_defaults['nlayers']
    )
    parser.add_argument(
        '--dt',
        help="The time step in seconds.",
        type=float,
        default=dry_baroclinic_channel_defaults['dt']
    )
    parser.add_argument(
        "--tmax",
        help="The end time for the simulation in seconds.",
        type=float,
        default=dry_baroclinic_channel_defaults['tmax']
    )
    parser.add_argument(
        '--dumpfreq',
        help="The frequency at which to dump field output.",
        type=int,
        default=dry_baroclinic_channel_defaults['dumpfreq']
    )
    parser.add_argument(
        '--dirname',
        help="The name of the directory to write to.",
        type=str,
        default=dry_baroclinic_channel_defaults['dirname']
    )
    args, unknown = parser.parse_known_args()

    dry_baroclinic_channel(**vars(args))
