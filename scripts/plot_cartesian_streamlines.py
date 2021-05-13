import pencil as pc
import pyvista as pv 
import numpy as np

var = pc.read.var(trimall=True,precision="half")
grid=pc.read.grid(trim=True)

ux = var.ux#[6:-6]
uy = var.uy#[6:-6]
uz = var.uz#[6:-6]

nx = ux.shape[2]
ny = ux.shape[1]
nz = ux.shape[0]
origin = (-grid.Lx, -grid.Ly, -grid.Lz)
spacing = (2*grid.Lx/(nx-1), 2*grid.Ly/(ny-1), 2*grid.Lz/(nz-1))
mesh = pv.UniformGrid((nx, ny, nx), spacing, origin)

#ux=np.transpose(ux, (2,1,0)).flatten()
#uy=np.transpose(uy, (2,1,0)).flatten()
#uz=np.transpose(uz, (2,1,0)).flatten()

ux = ux.flatten()
uy = uy.flatten()
uz = uz.flatten()

x = mesh.points[:, 0]
y = mesh.points[:, 1]
z = mesh.points[:, 2]

vectors = np.empty((mesh.n_points, 3))
vectors[:, 0] = ux
vectors[:, 1] = uy
vectors[:, 2] = uz
mesh['vectors'] = vectors

stream, src = mesh.streamlines('vectors', return_source=True,
                               terminal_speed=0.0, n_points=100,
                               source_radius=0.1)

stream.tube(radius=.1).plot()

