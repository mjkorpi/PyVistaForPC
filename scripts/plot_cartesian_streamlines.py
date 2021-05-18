# plot_cartesian_streamlines.py
'''
Plot the streamlines in 3d using pyvista.
'''

import pencil as pc
import pyvista as pv 
import numpy as np
import time

# Read the data.
var = pc.read.var(trimall=True, magic='bb')
grid = pc.read.grid(trim=True)

# Prepare the streamline integration parameters.
params = pc.diag.TracersParameterClass()
params.dx = var.dx
params.dy = var.dy
params.dz = var.dz
params.Lx = var.x[-1] - var.x[0]
params.Ly = var.y[-1] - var.y[0]
params.Lz = var.z[-1] - var.z[0]
params.nx = var.x.shape[0]
params.ny = var.y.shape[0]
params.nz = var.z.shape[0]
params.Ox = var.x[0]
params.Oy = var.y[0]
params.Oz = var.z[0]
dxyz = np.array([params.dx, params.dy, params.dz])
oxyz = np.array([params.Ox, params.Oy, params.Oz])
nxyz = np.array([params.nx, params.ny, params.nz])

# Optional parameters.
params.interpolation = 'trilinear'
params.rtol = 1e-5
params.atol = 1e-5
t = np.linspace(0, 40, 200)

# Perform streamline integration.
xx = np.array([np.random.random()*params.Lx + params.Ox,
               np.random.random()*params.Ly + params.Oy,
               np.random.random()*params.Lz + params.Oz])
xx = np.array([var.x[128], var.y[200], var.z[200]])
ti = time.time()
stream = pc.calc.Stream(var.bb, params, xx=xx, time=t)
print('time streamlines = {0}'.format(time.time() - ti))

# Compute the scalar field on the streamline points.
scalar = np.zeros(stream.tracers.shape[0])
for idx in range(scalar.shape[0]):
    scalar[idx] = pc.math.vec_int(stream.tracers[idx], var.bb, dxyz, oxyz, nxyz)[0]

# Plot all streamlines as pyvista splines.
poly = pv.PolyData()
poly.points = stream.tracers
cells = np.full((len(stream.tracers)-1, 3), 2, dtype=np.int_)
cells[:, 1] = np.arange(0, len(stream.tracers)-1, dtype=np.int_)
cells[:, 2] = np.arange(1, len(stream.tracers), dtype=np.int_)
poly.lines = cells
poly["scalars"] = scalar
tube = poly.tube(radius=0.1)
tube.plot(smooth_shading=True)


