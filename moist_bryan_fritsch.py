"""
The moist rising bubble test from Bryan & Fritsch (2002), in a cloudy
atmosphere.

The rise of the thermal is fueled by latent heating from condensation.
"""

from gusto import *
from gusto import thermodynamics
from firedrake import *
import sys


# ---------------------------------------------------------------------------- #
# Test case parameters
# ---------------------------------------------------------------------------- #

dt = 1.0
L = 10000.
H = 10000.

if '--running-tests' in sys.argv:
    deltax = 1000.
    tmax = 5.
else:
    deltax = 100
    tmax = 1000.
# ---------------------------------------------------------------------------- #
# Set up model objects
# ---------------------------------------------------------------------------- #

# Domain
nlayers = int(H/deltax)
ncolumns = int(L/deltax)

m = PeriodicIntervalMesh(ncolumns, L)
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)
degree = 1
domain = Domain(mesh, dt, 'CG', degree)

# Equation
params = CompressibleParameters()
tracers = [WaterVapour(), CloudWater()]
eqns = CompressibleEulerEquations(domain, params, active_tracers=tracers)
eqns = split_continuity_form(eqns)
eqns.label_terms(lambda t: not any(t.has_label(time_derivative, transport, physics_label)), implicit)
eqns.label_terms(lambda t: t.has_label(transport), explicit)

print("Opt Cores:", eqns.X.function_space().dim()/50000.)


# I/O
dirname = 'moist_bryan_fritsch_sdc_paper_100m_edg'
output = OutputParameters(dirname=dirname,
                          dumpfreq=int(tmax / (5*dt)),
                          dump_nc=True,
                          dump_vtus=False,
                          dumplist=['u'])
diagnostic_fields = [Theta_e(eqns)]
io = IO(domain, output, diagnostic_fields=diagnostic_fields)

# Transport schemes
# transported_fields = [SSPRK3(domain, "rho"),
#                       SSPRK3(domain, "theta", options=EmbeddedDGOptions()),
#                       SSPRK3(domain, "water_vapour", options=EmbeddedDGOptions()),
#                       SSPRK3(domain, "cloud_water", options=EmbeddedDGOptions()),
#                       TrapeziumRule(domain, "u")]
#theta_opts = SUPGOptions()
suboptions = {'theta': EmbeddedDGOptions(),
              'water_vapour': EmbeddedDGOptions(),
              'cloud_water': EmbeddedDGOptions()}
#suboptions = {}
opts = MixedFSOptions(suboptions=suboptions)
transport_methods = [DGUpwind(eqns, "u"), DGUpwind(eqns, "rho"),DGUpwind(eqns, "theta"),
                      DGUpwind(eqns, "water_vapour"), DGUpwind(eqns, "cloud_water")]

# Linear solver
# linear_solver = CompressibleSolver(eqns)

# Physics schemes (condensation/evaporation)
physics_schemes = [(SaturationAdjustment(eqns), ForwardEuler(domain))]

nl_solver_parameters = {
    "snes_converged_reason": None,
    "snes_lag_jacobian_persists":True,
    "snes_lag_jacobian":15,
    "snes_lag_preconditioner_persists":True,
    "snes_lag_preconditioner":4, 
    'ksp_ew': None,
    'ksp_ew_version': 1,
    'ksp_ew_threshold': 1e-2,
    "mat_type": "matfree",
    "ksp_type": "gmres",
    "ksp_converged_reason": None,
    "ksp_atol": 1e-5,
    "ksp_rtol": 1e-5,
    "ksp_max_it": 400,
    "pc_type": "python",
    "pc_python_type": "firedrake.AssembledPC",
    "assembled_pc_type": "python",
    "assembled_pc_python_type": "firedrake.ASMStarPC",
    "assembled_pc_star_construct_dim": 0,
    "assembled_pc_star_sub_sub_pc_factor_mat_ordering_type": "rcm",
    "assembled_pc_star_sub_sub_pc_factor_reuse_ordering": None,
    "assembled_pc_star_sub_sub_pc_factor_reuse_fill": None,
    "assembled_pc_star_sub_sub_pc_factor_fill": 1.2}

nl_solver_parameters = {
    "snes_converged_reason": None,
    "snes_lag_jacobian_persists":True,
    "snes_lag_jacobian":15,
    "snes_lag_preconditioner_persists":True,
    "snes_lag_preconditioner":4, 
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

base_scheme = IMEX_Euler(domain, solver_parameters=nl_solver_parameters)
node_type = "LEGENDRE"
qdelta_imp = "BE"
qdelta_exp = "FE"
quad_type = "GAUSS"
M = 2
k = 3
qdelta_imp = "LU"
scheme =SDC(base_scheme, domain, M, k, quad_type, node_type, qdelta_imp,
                    qdelta_exp, options=opts, formulation="Z2N", nonlinear_solver_parameters=nl_solver_parameters,final_update=True, initial_guess="copy")
scheme = SSPRK3(domain, options=opts, solver_parameters=nl_solver_parameters, increment_form=False)
# Time stepper
stepper = Timestepper(eqns, scheme, io, transport_methods,
                                  physics_schemes=physics_schemes)

# ---------------------------------------------------------------------------- #
# Initial conditions
# ---------------------------------------------------------------------------- #

u0 = stepper.fields("u")
rho0 = stepper.fields("rho")
theta0 = stepper.fields("theta")
water_v0 = stepper.fields("water_vapour")
water_c0 = stepper.fields("cloud_water")

# spaces
Vu = domain.spaces("HDiv")
Vt = domain.spaces("theta")
Vr = domain.spaces("DG")
x, z = SpatialCoordinate(mesh)
quadrature_degree = (4, 4)
dxp = dx(degree=(quadrature_degree))

# Define constant theta_e and water_t
Tsurf = 320.0
total_water = 0.02
theta_e = Function(Vt).assign(Tsurf)
water_t = Function(Vt).assign(total_water)

# Calculate hydrostatic fields
saturated_hydrostatic_balance(eqns, stepper.fields, theta_e, water_t)

# make mean fields
theta_b = Function(Vt).assign(theta0)
rho_b = Function(Vr).assign(rho0)
water_vb = Function(Vt).assign(water_v0)
water_cb = Function(Vt).assign(water_t - water_vb)
exner_b = thermodynamics.exner_pressure(eqns.parameters, rho_b, theta_b)
Tb = thermodynamics.T(eqns.parameters, theta_b, exner_b, r_v=water_vb)

# define perturbation
xc = L / 2
zc = 2000.
rc = 2000.
Tdash = 2.0
r = sqrt((x - xc) ** 2 + (z - zc) ** 2)
theta_pert = Function(Vt).interpolate(
    conditional(r > rc,
                0.0,
                Tdash * (cos(pi * r / (2.0 * rc))) ** 2))

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

stepper.set_reference_profiles([('rho', rho_b),
                                ('theta', theta_b),
                                ('water_vapour', water_vb),
                                ('cloud_water', water_cb)])

# ---------------------------------------------------------------------------- #
# Run
# ---------------------------------------------------------------------------- #

stepper.run(t=0, tmax=tmax)
