from gusto import *
from firedrake import (PeriodicRectangleMesh, ExtrudedMesh,
                       SpatialCoordinate, conditional, cos, sin, pi, sqrt,
                       ln, exp, Constant, Function, DirichletBC, as_vector,
                       FunctionSpace, BrokenElement, VectorFunctionSpace,
                       errornorm, norm, cross, grad)
from firedrake.slope_limiter.vertex_based_limiter import VertexBasedLimiter
import sys

# ---------------------------------------------------------------------------- #
# Test case parameters
# ---------------------------------------------------------------------------- #

days = 12 # suggested is 15
dt = 150.0
Lx = 4.0e7  # length
Ly = 6.0e6  # width
H = 3.0e4  # height
degree = 1
omega = Constant(7.292e-5)
phi0 = Constant(pi/4)

if '--running-tests' in sys.argv:
    tmax = 5*dt
    deltax = 2.0e6
    deltay = 1.0e6
    deltaz = 6.0e3
    dumpfreq = 5
else:
    tmax = days * 24 * 60 * 60
    deltax = 10e5
    deltay = deltax
    deltaz = 3.e3
    dumpfreq = int(tmax / (3 * days * dt))

# ---------------------------------------------------------------------------- #
# Set up model objects
# ---------------------------------------------------------------------------- #

# Domain
nlayers = int(H/deltaz)
ncolumnsx = int(Lx/deltax)
ncolumnsy = int(Ly/deltay)
m = PeriodicRectangleMesh(ncolumnsx, ncolumnsy, Lx, Ly, "x", quadrilateral=True)
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)
domain = Domain(mesh, dt, "RTCF", degree)
x,y,z = SpatialCoordinate(mesh)

# Equation
params = CompressibleParameters(Omega=omega)
coriolis = 2*omega*sin(phi0)*domain.k
eqns = CompressibleEulerEquations(domain, params, no_normal_flow_bc_ids=[1, 2])
print("Opt Cores:", eqns.X.function_space().dim()/50000.)
eqns = split_continuity_form(eqns)
eqns.label_terms(lambda t: not any(t.has_label(time_derivative, transport)), implicit)
eqns.label_terms(lambda t: t.has_label(transport), explicit)

# I/O
dirname = 'dry_baroclinic_channel_sdc'
output = OutputParameters(dirname=dirname, dumpfreq=dumpfreq, dump_nc=True)
diagnostic_fields = [Perturbation('theta')]
io = IO(domain, output, diagnostic_fields=diagnostic_fields)

# Transport schemes
transported_fields = [SSPRK3(domain, "u"),
                      SSPRK3(domain, "rho"),
                      SSPRK3(domain, "theta")]
transport_methods = [DGUpwind(eqns, "u"), 
                     DGUpwind(eqns, "rho"),
                     DGUpwind(eqns, "theta")]
# # Linear solver
# linear_solver = CompressibleSolver(eqns)

# # Time stepper
# stepper = SemiImplicitQuasiNewton(eqns, io, transported_fields,
#                                   transport_methods,
#                                   linear_solver=linear_solver)
nl_solver_parameters= {
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
    "ksp_atol": 1e-8,
    "ksp_rtol": 1e-8,
    "snes_atol": 1e-8,
    "snes_rtol": 1e-8,
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
qdelta_imp = "LU"
qdelta_exp = "FE"
quad_type = "RADAU-RIGHT"
M = 2
k = 2
qdelta_imp = "LU"
scheme =SDC(base_scheme, domain, M, k, quad_type, node_type, qdelta_imp,
                     qdelta_exp, formulation="Z2N", nonlinear_solver_parameters=nl_solver_parameters,final_update=True, initial_guess="copy")

stepper = Timestepper(eqns, scheme, io, 
                                 transport_methods)
# ---------------------------------------------------------------------------- #
# Initial conditions
# ---------------------------------------------------------------------------- #

# Physical parameters
a = Constant(6.371229e6)  # radius of earth
b = Constant(2)  # vertical width parameter
beta0 = 2 * omega * cos(phi0) / a
T0 = Constant(288)
Ts = Constant(260)
u0 = Constant(35)
Gamma = Constant(0.005)
Rd = params.R_d
Rv = params.R_v
f0 = 2 * omega * sin(phi0)
y0 = Constant(Ly / 2)
g = params.g
p0 = Constant(100000.)
beta0 = Constant(0)
eta_w = Constant(0.3)
deltay_w = Constant(3.2e6)
q0 = Constant(0.016)
cp = params.cp

# Initial conditions
u = stepper.fields("u")
rho = stepper.fields("rho")
theta = stepper.fields("theta")

# spaces
Vu = u.function_space()
Vt = theta.function_space()
Vr = rho.function_space()

# set up background state expressions
eta = Function(Vt).interpolate(Constant(1e-7))
Phi = Function(Vt).interpolate(g * z)
q = Function(Vt)
T = Function(Vt)
Phi_prime = u0 / 2 * ((f0 - beta0 * y0) *(y - (Ly/2) - (Ly/(2*pi))*sin(2*pi*y/Ly))
                       + beta0/2*(y**2 - (Ly*y/pi)*sin(2*pi*y/Ly)
                                  - (Ly**2/(2*pi**2))*cos(2*pi*y/Ly) - (Ly**2/3) - (Ly**2/(2*pi**2))))
Phi_expr = (T0 * g / Gamma * (1 - eta ** (Rd * Gamma / g))
            + Phi_prime * ln(eta) * exp(-(ln(eta) / b) ** 2))

Tv_expr = T0 * eta ** (Rd * Gamma / g) + Phi_prime / Rd * ((2/b**2) * (ln(eta)) ** 2 - 1) * exp(-(ln(eta)/b)**2)
u_expr = as_vector([-u0 * (sin(pi*y/Ly))**2 * ln(eta) * eta ** (-ln(eta) / b ** 2), 0.0, 0.0])
T_expr = Tv_expr

# do Newton method to obtain eta
eta_new = Function(Vt)
F = -Phi + Phi_expr
dF = -Rd * Tv_expr / eta
max_iterations = 40
tolerance = 1e-10
for i in range(max_iterations):
    eta_new.interpolate(eta - F/dF)
    if errornorm(eta_new, eta) / norm(eta) < tolerance:
        eta.assign(eta_new)
        break
    eta.assign(eta_new)

# make mean u and theta
u.project(u_expr)
T.interpolate(T_expr)
theta.interpolate(thermodynamics.theta(params, T_expr, p0 * eta) )
Phi_test = Function(Vt).interpolate(Phi_expr)
print("Error in setting up p:", errornorm(Phi_test, Phi) / norm(Phi))

# Calculate hydrostatic fields
compressible_hydrostatic_balance(eqns, theta, rho, solve_for_rho=True)

# make mean fields
rho_b = Function(Vr).assign(rho)
u_b = stepper.fields("ubar", space=Vu, dump=False).project(u)
theta_b = Function(Vt).assign(theta)

# define perturbation
xc = 2.0e6
yc = 2.5e6
Lp = 6.0e5
up = Constant(1.0)
r = sqrt((x - xc) ** 2 + (y - yc) ** 2)
u_pert = Function(Vu).project(as_vector([up * exp(-(r/Lp)**2), 0.0, 0.0]))

# define initial u
u.assign(u_b + u_pert)

# initialise fields
stepper.set_reference_profiles([('rho', rho_b),
                                ('theta', theta_b)])

# ---------------------------------------------------------------------------- #
# Run
# ---------------------------------------------------------------------------- #

stepper.run(t=0, tmax=tmax)