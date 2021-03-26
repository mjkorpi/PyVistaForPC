#Plot PC data in spherical coordinates
#MJK March 2021

import pyvista as pv
import numpy as np
import pencil as pc

def _cell_bounds(points, bound_position=0.5):
    """
    Calculate coordinate cell boundaries.

    Parameters
    ----------
    points: numpy.array
        One-dimensional array of uniformly spaced values of shape (M,)
    bound_position: bool, optional
        The desired position of the bounds relative to the position
        of the points.

    Returns
    -------
    bounds: numpy.array
        Array of shape (M+1,)

    Examples
    --------
    >>> a = np.arange(-1, 2.5, 0.5)
    >>> a
    array([-1. , -0.5,  0. ,  0.5,  1. ,  1.5,  2. ])
    >>> cell_bounds(a)
    array([-1.25, -0.75, -0.25,  0.25,  0.75,  1.25,  1.75,  2.25])
    """
    assert points.ndim == 1, "Only 1D points are allowed"
    diffs = np.diff(points)
    delta = diffs[0] * bound_position
    bounds = np.concatenate([[points[0] - delta], points + delta])
    return bounds


# First, read in the data
var = pc.read.var(trimall=True,precision="half")
grid=pc.read.grid(trim=True)


# Determine the radial position of the spherical projections
rbot=10 #Gridpoints from surface
rtop=20 #Gridpoints from bottom
nr=(grid.x.shape)[0]
rmid=int(nr/2)
nt=(grid.y.shape)[0]
nphi=(grid.z.shape)[0]
TOP = grid.x[nr-rtop]
BOT = grid.x[rbot]
plot_level=grid.x[rmid]

#Grids to project over spherical surfaces
#Wedge?
ntiles=int(2.*np.pi/grid.z[nphi-1])
ntileso=2 # Number of tiles to open up
#Original phi array
phiw=180.*grid.z/np.pi
#Replicating over the full longitudinal range
phi=360.*np.arange(ntiles*nphi)/(ntiles*nphi-1)
#Onion type plotting needs to be done differently for wedges and full spheres
if ntiles > 1:
	#Removing ntileso wedge tile to show the interior in an onion type plot
	phi2=((ntiles-ntileso)*360./(ntiles))*np.arange((ntiles-ntileso)*nphi)/((ntiles-ntileso)*nphi-1)
else:
	#The same here, but ntiles cannot be used
	#Note that in the case of a full 2pi run, the data will become distorted in the surface plot, as it is squeezed by a factor of 3./4.
	phi2=((nphi-ntileso*90.)*360./(nphi))*np.arange((ntiles-ntileso*90.)*nphi)/((nphi-ntileso*90.)*nphi-1)
	
#lat=180.-grid.y*180./np.pi
lat=grid.y*180./np.pi
r=grid.x

#2D arrays for projections of coordinates
xx, yy = np.meshgrid(phi, lat) #On a sphere
xx2, yy = np.meshgrid(phi2, lat) #On a sphere
zz, yy = np.meshgrid(r, lat) #On meridional slices


# Scalar data
#scalartop=np.tile(var.uu[0,:,:,nr-rtop].T,ntiles)
#scalartop2=np.tile(var.uu[0,:,:,nr-rtop].T,ntiles-1)
#scalarbot=np.tile(var.uu[1,:,:,rbot].T,ntiles)
#scalarmer1=var.uu[2,0,:,:]
#scalarmer2=var.uu[2,nphi-1,:,:]
scalartop=np.tile(var.ux[:,:,nr-rtop].T,ntiles)
scalartop2=np.tile(var.ux[:,:,nr-rtop].T,ntiles-ntileso)
scalarbot=np.tile(var.ux[:,:,rbot].T,ntiles)
scalarmer1=var.uy[0,:,:]
scalarmer2=var.uy[int(nphi/2),:,:]

### Create arrays of grid cell boundaries, which have shape of (x.shape[0] + 1)
xx_bounds = _cell_bounds(phi)
xx_bounds2 = _cell_bounds(phi2)
yy_bounds = _cell_bounds(lat)
zz_bounds = _cell_bounds(r)

# Meshes
levels = [TOP * 1.01]
levels2 = [BOT + 0.01]
level3 = [zz[0]]

# Grid on the sphere
grid_scalartop = pv.grid_from_sph_coords(xx_bounds, yy_bounds, levels)
grid_scalartop2 = pv.grid_from_sph_coords(xx_bounds2, yy_bounds, levels)
grid_scalarbot = pv.grid_from_sph_coords(xx_bounds, yy_bounds, levels2)
# Meridional grid
grid_scalarmer1 = pv.grid_from_sph_coords(0., yy_bounds, zz_bounds)
grid_scalarmer2 = pv.grid_from_sph_coords(-ntileso*360./(ntiles), yy_bounds, zz_bounds)

# And fill its cell arrays with the scalar data
grid_scalartop.cell_arrays["Ur top"] = np.array(scalartop).swapaxes(-2, -1).ravel("C")
grid_scalartop2.cell_arrays["Ur top"] = np.array(scalartop2).swapaxes(-2, -1).ravel("C")
grid_scalarbot.cell_arrays["Ur bottom"] = np.array(scalarbot).swapaxes(-2, -1).ravel("C")
#
grid_scalarmer1.cell_arrays["Meridional cut of Uphi"] = np.array(scalarmer1).swapaxes(-2, -1).ravel("C")
grid_scalarmer2.cell_arrays["Meridional cut of Uphi"] = np.array(scalarmer2).swapaxes(-2, -1).ravel("C")

# Onion style plot with scalar data only
#p = pv.Plotter(shape=(2,2))
#p.subplot(0, 0)
p = pv.Plotter()
#p.add_mesh(pv.Sphere(radius=BOT))
p.add_mesh(grid_scalartop2, opacity=1.0, cmap="rainbow",show_scalar_bar=False)
p.add_mesh(grid_scalarbot, opacity=1.0, cmap="rainbow",show_scalar_bar=False)
p.add_mesh(grid_scalarmer1, opacity=1.0, cmap="rainbow",show_scalar_bar=False)
p.add_mesh(grid_scalarmer2, opacity=1.0, cmap="rainbow",show_scalar_bar=False)
p.show()

###

