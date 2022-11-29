"""
Dedalus script simulating Boussinesq convection in a spherical shell. This script
demonstrates solving an initial value problem in the shell. It can be ran serially
or in parallel, and uses the built-in analysis framework to save data snapshots
to HDF5 files. The `plot_shell.py` script can be used to produce plots from the
saved data. The simulation should take about 20 cpu-minutes to run.

The problem is non-dimensionalized using the shell thickness and freefall time, so
the resulting thermal diffusivity and viscosity are related to the Prandtl
and Rayleigh numbers as:

    kappa = (Rayleigh * Prandtl)**(-1/2)
    nu = (Rayleigh / Prandtl)**(-1/2)

For incompressible hydro with two boundaries, we need two tau terms for each the
velocity and buoyancy. Here we choose to use a first-order formulation, putting
one tau term each on auxiliary first-order gradient variables and the others in
the PDE, and lifting them all to the first derivative basis. This formulation puts
a tau term in the divergence constraint, as required for this geometry.

To run and plot using e.g. 4 processes:
    $ mpiexec -n 4 python3 shell_convection.py
    $ mpiexec -n 4 python3 plot_shell.py snapshots/*.h5
"""

import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)

# Parameters
Ri, Ro = 0.4, 1.0

Nmodes=32
Nphi, Ntheta, Nr = Nmodes*2, Nmodes, Nmodes
Rayleigh = 1e6
MagneticPrandtl=1.0
Prandtl = 1
dealias = 3/2
stop_sim_time = 300
timestepper = d3.SBDF2
max_timestep = 1e-1
dtype = np.float64
mesh =None# [2,2]

MHD_ON=True
perfectlyconducting_BC=0
potential_BC=1

# Bases
coords = d3.SphericalCoordinates('phi', 'theta', 'r')
dist = d3.Distributor(coords, dtype=dtype, mesh=mesh)
shell = d3.ShellBasis(coords, shape=(Nphi, Ntheta, Nr), radii=(Ri, Ro), dealias=dealias, dtype=dtype)
sphere = shell.outer_surface

# Fields
p = dist.Field(name='p', bases=shell)
T = dist.Field(name='T', bases=shell)
Phi = dist.Field(name='Phi', bases=shell)
u = dist.VectorField(coords, name='u', bases=shell)
A = dist.VectorField(coords, name='A', bases=shell)
tau_p = dist.Field(name='tau_p')
tau_T1 = dist.Field(name='tau_T1', bases=sphere)
tau_T2 = dist.Field(name='tau_T2', bases=sphere)
tau_u1 = dist.VectorField(coords, name='tau_u1', bases=sphere)
tau_u2 = dist.VectorField(coords, name='tau_u2', bases=sphere)
tau_A1 = dist.VectorField(coords, name='tau_A1', bases=sphere)
tau_A2 = dist.VectorField(coords, name='tau_A2', bases=sphere)
tau_Phi = dist.Field(name='tau_Phi')



# Substitutions
kappa = (Rayleigh * Prandtl)**(-1/2)
nu = (Rayleigh / Prandtl)**(-1/2)
eta = nu / MagneticPrandtl
phi, theta, r = dist.local_grids(shell)

er = dist.VectorField(coords, bases=shell)
er['g'][2] = 1
rvec = dist.VectorField(coords, bases=shell.radial_basis)
rvec['g'][2] = r

lift_basis = shell.derivative_basis(1)
lift = lambda G: d3.Lift(G, lift_basis, -1)
grad_u = d3.grad(u) + rvec*lift(tau_u1) # First-order reduction
grad_A = d3.grad(A) + rvec*lift(tau_A1) # First-order reduction
grad_T = d3.grad(T) + rvec*lift(tau_T1) # First-order reduction
strain_rate = d3.grad(u) + d3.trans(d3.grad(u))
shear_stress_Ri = d3.angular(d3.radial(strain_rate(r=Ri), index=1))
shear_stress_Ro = d3.angular(d3.radial(strain_rate(r=Ro), index=1))
ell_func = lambda ell: ell+1
A_potential_outerbc = d3.radial(d3.grad(A)(r=Ro)) + d3.SphericalEllProduct(A, coords, ell_func)(r=Ro)/Ro
A_potential_innerbc = d3.radial(d3.grad(A)(r=Ri)) + d3.SphericalEllProduct(A, coords, ell_func)(r=Ri)/Ri


# Problem
if MHD_ON:
    B=d3.curl(A)
    J=-d3.Laplacian(A)
    if perfectlyconducting_BC:
        problem = d3.IVP([p, u, T, A, Phi, tau_p, tau_T1, tau_T2, tau_u1, tau_u2, tau_A1, tau_A2, tau_Phi], namespace=locals())
        problem.add_equation("dt(u) - nu*div(grad_u) + grad(p) - T*rvec + lift(tau_u2) = - u@grad(u) ")#+ cross(J,B)
        problem.add_equation("dt(A) + eta*div(grad_A) +grad(Phi) + lift(tau_A2) = cross(u,B)")
        problem.add_equation("trace(grad_A)+tau_Phi = 0")  # Coulumb gauge
        
        problem.add_equation("angular(A(r=Ri)) = 0")
        problem.add_equation("angular(A(r=Ro)) = 0") 
        problem.add_equation("Phi(r=Ro) = 0")  
        problem.add_equation("Phi(r=Ri) = 0")  
    if potential_BC:
        problem = d3.IVP([p, u, T, A, Phi, tau_p, tau_T1, tau_T2, tau_u1, tau_u2, tau_A1, tau_A2, tau_Phi], namespace=locals())
        problem.add_equation("dt(u) - nu*div(grad_u) + grad(p) - T*rvec + lift(tau_u2) = - u@grad(u) ")#+ cross(J,B)
        problem.add_equation("dt(A) + eta*div(grad_A) +grad(Phi) + lift(tau_A2) = cross(u,B)")
        problem.add_equation("trace(grad_A)+tau_Phi = 0")  # Coulumb gauge

        problem.add_equation("A_potential_innerbc=0")
        problem.add_equation("A_potential_outerbc=0") 
else:
    problem = d3.IVP([p, T, u, tau_p, tau_T1, tau_T2, tau_u1, tau_u2], namespace=locals())
    problem.add_equation("dt(u) - nu*div(grad_u) + grad(p) - T*rvec + lift(tau_u2) = - u@grad(u)")
problem.add_equation("trace(grad_u) + tau_p = 0")
problem.add_equation("dt(T) - kappa*div(grad_T) + lift(tau_T2) = - u@grad(T)")
problem.add_equation("T(r=Ri) = 1")
problem.add_equation("T(r=Ro) = 0")
problem.add_equation("u(r=Ri) = 0")
problem.add_equation("u(r=Ro) = 0")
#problem.add_equation("radial(u(r=Ri)) = 0")
#problem.add_equation("radial(u(r=Ro)) = 0")
#problem.add_equation("shear_stress_Ri = 0")
#problem.add_equation("shear_stress_Ro = 0")
problem.add_equation("integ(p) = 0") # Pressure gauge


# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions
if MHD_ON:
    A.fill_random('g', seed=42, distribution='normal', scale=1e-6) # Random noise
    A['g'] *= (r - Ri) * (Ro - r) # Damp noise at walls
T.fill_random('g', seed=42, distribution='normal', scale=1e-3) # Random noise
T['g'] *= (r - Ri) * (Ro - r) # Damp noise at walls
T['g'] += (Ri - Ri*Ro/r) / (Ri - Ro) # Add linear background

# Analysis
rad_flux = er @ (-kappa*d3.grad(T) )
conv_flux = er @ (u*T)
flux = er @ (-kappa*d3.grad(T)+u*T)
file_handler_mode = 'overwrite'
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=10, max_writes=10, mode=file_handler_mode)
snapshots.add_task(T(r=(Ri+Ro)/2), scales=dealias, name='bmid')
snapshots.add_task(flux(r=Ro), scales=dealias, name='flux_r_outer')
snapshots.add_task(flux(r=Ri), scales=dealias, name='flux_r_inner')
snapshots.add_task(flux(phi=0), scales=dealias, name='flux_phi_start')
snapshots.add_task(flux(phi=3*np.pi/2), scales=dealias, name='flux_phi_end')

snapshots.add_task((u@er)(r=Ro), scales=dealias, name='ur_r_outer')
snapshots.add_task((u@er)(r=Ri), scales=dealias, name='ur_r_inner')
snapshots.add_task((u@er)(phi=0), scales=dealias, name='ur_phi_start')
snapshots.add_task((u@er)(phi=3*np.pi/2), scales=dealias, name='ur_phi_end')

snapshots.add_task(T(r=Ro), scales=dealias, name='T_r_outer')
snapshots.add_task(T(r=Ri), scales=dealias, name='T_r_inner')
snapshots.add_task(T(phi=0), scales=dealias, name='T_phi_start')
snapshots.add_task(T(phi=3*np.pi/2), scales=dealias, name='T_phi_end')

series = solver.evaluator.add_file_handler('series', sim_dt=10.0, mode=file_handler_mode)
series.add_task(d3.Integrate( 0.5*u@u ), name='KE')
series.add_task(d3.ave( er @ (-kappa*d3.grad(T) ),coords.S2coordsys), scales=dealias, name='rad_flux')
series.add_task(d3.ave( er @ (u*T) ,coords.S2coordsys), scales=dealias, name='conv_flux')
series.add_task(d3.ave(T,coords.S2coordsys), scales=dealias, name='T_r_profile')
series.add_task(d3.ave(np.sqrt((u@er)**2),coords.S2coordsys), scales=dealias, name='ur_rms_r_profile')


if MHD_ON:
    series.add_task(d3.Integrate( 0.5*B@B ), name='ME')

# CFL
CFL = d3.CFL(solver, initial_dt=max_timestep/10.0, cadence=10, safety=0.5, threshold=0.1,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)
if MHD_ON:
    CFL.add_velocity(B)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(np.sqrt(u@u)/nu, name='Re')
flow.add_property(np.sqrt(u@u), name='u')
if MHD_ON:
    flow.add_property(np.sqrt(B@B), name='b')

# Main loop
try:
    logger.info('Starting main loop')
    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        if (solver.iteration-1) % 10 == 0:
            max_Re = flow.max('Re')
            max_u  = flow.max('u')
            if MHD_ON:
                max_B = np.sqrt(flow.max('b'))
                logger.info("Iteration=%i, Time=%e, dt=%e, max(u)=%e, max(Re)=%.0f, max(B)=%e" %(solver.iteration, solver.sim_time, timestep, max_u, max_Re, max_B))
            else:
                logger.info('Iteration=%i, Time=%e, dt=%e, max(u)=%f, max(Re)=%.0f' %(solver.iteration, solver.sim_time, timestep, max_u, max_Re))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()
