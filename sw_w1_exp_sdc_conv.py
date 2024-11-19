"""
The Williamson 1 test case (advection of gaussian hill), solved with a
discretisation of the non-linear advection equations.

This uses an icosahedral mesh of the sphere, and runs a series of resolutions to find convergence.
"""

from re import L
from gusto import *
from firedrake import (CubedSphereMesh, SpatialCoordinate,
                       as_vector, pi, sqrt, min_value, errornorm, norm, cos, sin,
                       acos, grad, curl, div, conditional)
import sys
import matplotlib.pyplot as plt
import numpy as np
import netCDF4 as nc

# ---------------------------------------------------------------------------- #
# Test case parameters
# ---------------------------------------------------------------------------- #

day = 24.*60.*60.
# setup resolution and timestepping parameters for convergence test
dts = [  1800., 1200., 900.,600., 450.]

dts = [  2400.]
tmax = 1*day
ndumps = 1
# setup shallow water parameters
R = 6371220.
H = 5960.
dt_true = 0.5

ref_level= 5
degree = 1

mesh = CubedSphereMesh(radius=R,
                             refinement_level=ref_level, degree=2)

x = SpatialCoordinate(mesh)

# Domain
domain = Domain(mesh, dt_true, 'RTCF', degree)
# Equation
V = domain.spaces('DG')
eqns = AdvectionEquation(domain, V, "D")

# I/O
dirname = "williamson_1_true_paper7_ref%s_dt%s_deg%s" % (ref_level, dt_true, degree)
dumpfreq = int(tmax / (ndumps*dt_true))
output = OutputParameters(dirname=dirname,
                        dumpfreq=dumpfreq,
                        checkpoint=True,
                        dump_nc=True,
                        dump_vtus=False,
                        checkpoint_method="checkpointfile",
                        chkptfreq=dumpfreq,
                        dumplist_latlon=['D'])
io = IO(domain, output)
solver_parameters = {'snes_type': 'ksponly',
                                       'ksp_type': 'cg',
                                       'pc_type': 'bjacobi',
                                       'sub_pc_type': 'ilu'}
scheme = SSPRK3(domain, solver_parameters=solver_parameters)
transport_methods = [ DGUpwind(eqns, "D")]
stepper = PrescribedTransport(eqns, scheme, io, prescribed_transporting_velocity=False, transport_method=transport_methods)

# ------------------------------------------------------------------------ #
# Initial conditions
# ------------------------------------------------------------------------ #
u0 = stepper.fields('u')
D0 = stepper.fields('D')

u_max = 2*pi*R/(12*day)  # Maximum amplitude of the zonal wind (m/s)
D_max = 1000.
#theta, lamda, _ = lonlatr_from_xyz(x[0], x[1], x[2])
lamda, theta, _ = lonlatr_from_xyz(x[0], x[1], x[2])
lamda_c=3.*pi/2.
theta_c=0.
alpha=0.

# Intilising the velocity field
CG2 = FunctionSpace(mesh, 'CG', degree+1)
psi = Function(CG2)
psiexpr = -R*u_max*(sin(theta)*cos(alpha)-cos(alpha)*cos(theta)*sin(alpha))
psi.interpolate(psiexpr)
uexpr = domain.perp(grad(psi))
c_dist=R*acos(sin(theta_c)*sin(theta) + cos(theta_c)*cos(theta)*cos(lamda-lamda_c))

Dexpr = conditional(c_dist < R/3., 0.5*D_max*(1.+cos(3.*pi*c_dist/R)), 0.0)

u0.project(uexpr)
D0.interpolate(Dexpr)
# ------------------------------------------------------------------------ #
# Run
# ------------------------------------------------------------------------ #

#stepper.run(t=0, tmax=tmax)

scheme_index= [0,1,2]
for dt in dts:

        #     ------------------------------------------------------------------------ #
        #     Set up model objects
        #     ------------------------------------------------------------------------ #

        #     Domain
        x = SpatialCoordinate(mesh)
        domain = Domain(mesh, dt, 'RTCF', degree)

        # Equation
        V = domain.spaces('DG')
        eqns = AdvectionEquation(domain, V, "D")
        # eqns = split_continuity_form(eqns)
        # # Label terms are implicit and explicit
        # eqns.label_terms(lambda t: not any(t.has_label(time_derivative, transport)), implicit)
        # eqns.label_terms(lambda t: t.has_label(transport), explicit)
        eqns.label_terms(lambda t: not t.has_label(time_derivative), explicit)
        ik = 0
        for s in scheme_index:

                # I/O
                dirname = "williamson_1_EX_SDC_paper7_ref%s_dt%s_k%s_deg%s" % (ref_level, dt, s, degree)
                dumpfreq = int(tmax / (ndumps*dt))
                print(dumpfreq)
                output = OutputParameters(dirname=dirname,
                                        dumpfreq=dumpfreq,
                                        checkpoint=True,
                                        dump_nc=True,
                                        dump_vtus=False,
                                        checkpoint_method="checkpointfile",
                                        chkptfreq=dumpfreq,
                                        dumplist_latlon=['D'])
                io = IO(domain, output)
                node_dist = "LEGENDRE"
                qdelta_imp="BE"
                qdelta_exp="FE"
                solver_parameters = {'snes_type': 'newtonls',
                                      "ksp_atol": 1e-4,
                                     "ksp_rtol": 1e-3,
                                                        'ksp_type': 'gmres',
                                                        'pc_type': 'bjacobi',
                                                        'sub_pc_type': 'ilu'}
                
                solver_parameters = {'snes_type': 'ksponly',
                                       'ksp_type': 'cg',
                                       'pc_type': 'bjacobi',
                                       'sub_pc_type': 'ilu'}


                # Time stepper
                if (s==0):
                        node_type="GAUSS"
                        M = 4
                        k = 7
                        base_scheme=ForwardEuler(domain,solver_parameters=solver_parameters)
                        scheme = SDC(base_scheme, domain, M, k, node_type, node_dist, qdelta_imp, qdelta_exp, nonlinear_solver_parameters=solver_parameters, formulation="Z2N", final_update=True, initial_guess="copy")
                elif(s==1):
                        node_type="GAUSS"
                        M=2
                        k=3
                        base_scheme=ForwardEuler(domain,solver_parameters=solver_parameters)
                        scheme = SDC(base_scheme, domain, M, k, node_type, node_dist, qdelta_imp, qdelta_exp, nonlinear_solver_parameters=solver_parameters, formulation="Z2N", final_update=True  , initial_guess="copy")
                elif (s==2):
                        node_type="GAUSS"
                        M = 3
                        k = 5
                        base_scheme=ForwardEuler(domain,solver_parameters=solver_parameters)
                        scheme = SDC(base_scheme, domain, M, k, node_type, node_dist, qdelta_imp, qdelta_exp, nonlinear_solver_parameters=solver_parameters, formulation="Z2N", final_update=True, initial_guess="copy")
                transport_methods = [ DGUpwind(eqns, "D")]

                stepper = PrescribedTransport(eqns, scheme, io, prescribed_transporting_velocity=False, transport_method=transport_methods)


                u0 = stepper.fields("u")
                D0 = stepper.fields("D")

                # ------------------------------------------------------------------------ #
                # Initial conditions
                # ------------------------------------------------------------------------ #

                x = SpatialCoordinate(mesh)
                u_max = 2*pi*R/(12*day)  # Maximum amplitude of the zonal wind (m/s)
                D_max = 1000.
                lamda, theta, _ = lonlatr_from_xyz(x[0], x[1], x[2])
                lamda_c=3.*pi/2.
                theta_c=0.
                alpha=0.

                # Intilising the velocity field
                CG2 = FunctionSpace(mesh, 'CG', degree+1)
                psi = Function(CG2)
                psiexpr = -R*u_max*(sin(theta)*cos(alpha)-cos(alpha)*cos(theta)*sin(alpha))
                psi.interpolate(psiexpr)
                uexpr = domain.perp(grad(psi))
                c_dist=R*acos(sin(theta_c)*sin(theta) + cos(theta_c)*cos(theta)*cos(lamda-lamda_c))

                Dexpr = conditional(c_dist < R/3., 0.5*D_max*(1.+cos(3.*pi*c_dist/R)), 0.0)

                u0.project(uexpr)
                D0.interpolate(Dexpr)

                # ------------------------------------------------------------------------ #
                # Run
                # ------------------------------------------------------------------------ #

                stepper.run(t=0, tmax=tmax)

                u = stepper.fields('u')
                D = stepper.fields('D')