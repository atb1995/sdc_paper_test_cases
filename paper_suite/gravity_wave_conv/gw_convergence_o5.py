"""
The non-linear gravity wave test case of Skamarock and Klemp (1994).

Potential temperature is transported using SUPG.
"""

from petsc4py import PETSc
PETSc.Sys.popErrorHandler()
from gusto import *
import itertools
from firedrake import (as_vector, SpatialCoordinate, PeriodicIntervalMesh,
                       ExtrudedMesh, exp, sin, Function, pi, COMM_WORLD)
import numpy as np
import sys

# ---------------------------------------------------------------------------- #
# Test case parameters
# ---------------------------------------------------------------------------- #

L = 3.0e5  # Domain length
H = 1.0e4  # Height position of the model top

if '--running-tests' in sys.argv:
    nlayers = 5
    columns = 30
    tmax = dt
    dumpfreq = 1
else:
    nlayers = 10
    #columns = [ 600, 300, 150, 75, 50, ]
    #columns = [50, 25, 10]
    #columns = [225., 150., 75.]
    # dts = [0.9375, 1.875]
    # columns = [60., 30.]
    columns = [120., 15.]

    dts = [0.46875, 3,75]
    #dts = [1.2, 1.2, 1.2]
    # columns = [900., 800., 600., 400.]
    # columns = [6000., 3000.]
    #columns=[900, 450]
    #columns = [75, 100]
    #columns=columns[::-1]
    # dts = [0.3, 0.75, 1.5, 3.0, 6.0]
    # #dts = [1.2, 1.2, 1.2, 1.2, 1.2]
    # dts = [1.2, 3.0, 6.0, 12.0, 24.0]
    # dts = [0.5, 2.5, 5.0 ,10.0, 20.0]
    # dts = [10.0/3., 20./3.]
    # #dts = [0.6, 3.0, 6.0, 12.0, 24.0]
    # dts = dts[::-1]
    cfl = 0.0015
    u0_val = 20.0
    tmax = 3000.0
for column, dt in zip(columns,dts):
    dx = float(float(L)/float(column))
    # dt = float(dx*cfl/u0_val)
    #cfl = u0_val*dt/dx
    #dx*(cfl/u0_val)
    print(tmax/dt)
    print("dx, dt, cfl:", dx, dt, cfl)
    print(tmax/dt)
    # ---------------------------------------------------------------------------- #
    # Set up model objects
    # ---------------------------------------------------------------------------- #
    dumpfreq = int(tmax / (dt))
    # print("Dumpfreq:", dumpfreq)
    # Domain -- 3D volume mesh
    m = PeriodicIntervalMesh(column, L)
    mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)
    domain = Domain(mesh, dt, "CG", 5)
    u_eqn_type = 'vector_advection_form'

    # Equation
    Tsurf = 300.
    parameters = CompressibleParameters()
    eqns = CompressibleEulerEquations(domain, parameters, u_transport_option=u_eqn_type)
    eqns = split_continuity_form(eqns)
    eqns = split_hv_advective_form(eqns, "rho")
    eqns = split_hv_advective_form(eqns, "theta")
    opts =SUPGOptions(suboptions={"theta": [transport]})
    print("Opt Cores:", eqns.X.function_space().dim()/50000.)

    # I/O
    points_x = np.linspace(0., L, 100)
    points_z = [H/2.]
    points = np.array([p for p in itertools.product(points_x, points_z)])
    deltax = L/column
    dirname = 'gravity_wave_imex_sdc_paper_o5_dx_%s_dt_%s' %(deltax,dt)

    # # Dumping point data using legacy PointDataOutput is not supported in parallel
    # if COMM_WORLD.size == 1:
    #     output = OutputParameters(
    #         dirname=dirname,
    #         dumpfreq=dumpfreq,
    #         pddumpfreq=dumpfreq,
    #         dumplist=['u'],
    #         point_data=[('theta_perturbation', points)],
    #     )
    # else:
    #     logger.warning(
    #         'Dumping point data using legacy PointDataOutput is not'
    #         ' supported in parallel\nDisabling PointDataOutput'
    #     )
    #     output = OutputParameters(
    #         dirname=dirname,
    #         dumpfreq=dumpfreq,
    #         pddumpfreq=dumpfreq,
    #         dumplist=['u'],
    #     )

    diagnostic_fields = [CourantNumber(), Gradient('u'), Perturbation('theta'),
                        Gradient('theta_perturbation'), Perturbation('rho'),
                        RichardsonNumber('theta', parameters.g/Tsurf), Gradient('theta')]
    output = OutputParameters(dirname=dirname,
                            dumpfreq=dumpfreq,
                            checkpoint=True,
                            dump_nc=True,
                            dump_vtus=False,
                            checkpoint_method="checkpointfile",
                            chkptfreq=dumpfreq,
                            dumplist=['u','theta','rho'])
    io = IO(domain, output, diagnostic_fields=diagnostic_fields)

    # Transport schemes
    transport_methods = [DGUpwind(eqns, "u"),
                        Split_DGUpwind(eqns, "rho"),
                        Split_DGUpwind(eqns, "theta", ibp=SUPGOptions.ibp)]

    # Linear solver
    # linear_solver = CompressibleSolver(eqns, alpha = 0.8)
    # nl_solver_parameters = {
    #     "snes_converged_reason": None,
    #     "mat_type": "matfree",
    #     "ksp_type": "gmres",
    #     "ksp_converged_reason": None,
    #     "ksp_atol": 1e-5,
    #     "ksp_rtol": 1e-5,
    #     "ksp_max_it": 400,
    #     "pc_type": "python",
    #     "pc_python_type": "firedrake.AssembledPC",
    #     "assembled_pc_type": "python",
    #     "assembled_pc_python_type": "firedrake.ASMStarPC",
    #     "assembled_pc_star_construct_dim": 0,
    #     "assembled_pc_star_sub_sub_pc_factor_mat_ordering_type": "rcm",
    #     "assembled_pc_star_sub_sub_pc_factor_reuse_ordering": None,
    #     "assembled_pc_star_sub_sub_pc_factor_reuse_fill": None,
    #     "assembled_pc_star_sub_sub_pc_factor_fill": 1.2}

    # gungho_solver_parameters = {"ksp_type": "gmres",
                                
    #                             "ksp_converged_reason": None,
    #                             "mat_type":"aij",
    #                             "ksp_atol": 1e-5,
    #                             "ksp_rtol": 1e-5,
    #                             "ksp_max_it": 400,
    #                             "pc_type": "fieldsplit",
    #                             "pc_fieldsplit_type": "schur",
    #                             # first split contains first two fields, second
    #                             # contains the third
    #                             "pc_fieldsplit_0_fields": "0, 1",
    #                             "pc_fieldsplit_1_fields": "2",
    #                             # Multiplicative fieldsplit for first field
    #                             "fieldsplit_0_pc_type": "fieldsplit",
    #                             "fieldsplit_0_pc_fieldsplit_type": "multiplicative",
    #                             # LU on each field
    #                             "fieldsplit_0_fieldsplit_0_ksp_type": "pre_only",
    #                             "fieldsplit_0_fieldsplit_1_ksp_type": "pre_only",
    #                             "fieldsplit_0_fieldsplit_0_pc_type": "lu",
    #                             "fieldsplit_0_fieldsplit_1_pc_type": "lu",
    #                             # ILU on the schur complement block
    #                             "fieldsplit_1_ksp_type": "preonly",
    #                             "fieldsplit_1_pc_type": 'gamg',
    #                             'fieldsplit_1_pc_gamg_sym_graph': None,
    #                             'fieldsplit_1_mg_levels': {'ksp_type': 'preonly',
    #                                         'pc_type': 'jacobi'}}

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
    eqns.label_terms(lambda t: not any(t.has_label(time_derivative, transport)), implicit)
    eqns.label_terms(lambda t: t.has_label(transport), explicit)
    eqns.label_terms(lambda t: t.has_label(transport) and t.has_label(horizontal), explicit)
    eqns.label_terms(lambda t: t.has_label(transport) and t.has_label(vertical), implicit)
    eqns.label_terms(lambda t: t.has_label(transport) and not any(t.has_label(horizontal, vertical)), explicit)
    # scheme = ImplicitMidpoint(domain, solver_parameters=nl_solver_parameters)
    base_scheme = IMEX_Euler(domain, options=opts,nonlinear_solver_parameters=nl_solver_parameters)
    node_type = "LEGENDRE"
    qdelta_exp = "FE"
    quad_type = "RADAU-RIGHT"
    M = 4
    k = 7
    qdelta_imp = "LU"
    scheme =SDC(base_scheme, domain, M, k, quad_type, node_type, qdelta_imp,
                        qdelta_exp, formulation="Z2N", options=opts, nonlinear_solver_parameters=nl_solver_parameters,final_update=False, initial_guess="copy")
    # # Time stepper
    stepper = Timestepper(eqns, scheme, io, transport_methods)

    # # Time stepper
    # stepper = Timestepper(eqns, scheme, io,
    #                                 transport_methods)

    # stepper = SemiImplicitQuasiNewton(eqns, io, transported_fields,
    #                               transport_methods,
    #                               linear_solver=linear_solver, alpha=0.8)
    # ---------------------------------------------------------------------------- #
    # Initial conditions
    # ---------------------------------------------------------------------------- #

    u0 = stepper.fields("u")
    rho0 = stepper.fields("rho")
    theta0 = stepper.fields("theta")

    # spaces
    Vu = domain.spaces("HDiv")
    Vt = domain.spaces("theta")
    Vr = domain.spaces("DG")

    # Thermodynamic constants required for setting initial conditions
    # and reference profiles
    g = parameters.g
    N = parameters.N

    x, z = SpatialCoordinate(mesh)

    # N^2 = (g/theta)dtheta/dz => dtheta/dz = theta N^2g => theta=theta_0exp(N^2gz)
    thetab = Tsurf*exp(N**2*z/g)

    theta_b = Function(Vt).interpolate(thetab)
    rho_b = Function(Vr)

    # Calculate hydrostatic exner
    compressible_hydrostatic_balance(eqns, theta_b, rho_b)

    a = 5.0e3
    deltaTheta = 1.0e-2
    theta_pert = deltaTheta*sin(pi*z/H)/(1 + (x - L/2)**2/a**2)
    theta0.interpolate(theta_b + theta_pert)
    rho0.assign(rho_b)
    u0.project(as_vector([20.0, 0.0]))

    stepper.set_reference_profiles([('rho', rho_b),
                                    ('theta', theta_b)])

    # ---------------------------------------------------------------------------- #
    # Run
    # ---------------------------------------------------------------------------- #

    stepper.run(t=0, tmax=tmax)
