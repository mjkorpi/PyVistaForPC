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
# Minimize memory usage by reading in double precision floats as single
var = pc.read.var(trimall=True,precision="half")
#Pencil Code axis will be defined in radians. vtk eats degrees.
grid=pc.read.grid(trim=True,precision="half")


# Determine the radial positions of the spherical projections
# Dimensions of the grid
nr=(grid.x.shape)[0]
nt=(grid.y.shape)[0]
nphi=(grid.z.shape)[0]
# Bottom slice placed relative to the grid spacing, now 10% lowest
rbot=int(0.1*nr)
# Top slice also relative, now 90%
rtop=int(0.9*nr)
# Middle slice would also be placed middle
rmid=int(0.5*nr)
# Values from the grid read into special variables
TOP = grid.x[rtop]
BOT = grid.x[rbot]
plot_level=grid.x[rmid]
# 

#Grids to project over spherical surfaces
# Have the computations been done over a wedge? ntiles will tell you how many
# times you need to replicate the data to cover a full sphere
ntiles=int(2.*np.pi/grid.z[nphi-1])
# For onion layers, you want to leave the top open. This parameter tells how many 
# wedge tiles you want to leave open for the largest onion layer
ntileso=2
#Original phi array, transformed into degrees
phiw=180.*grid.z/np.pi
# Replicating over the full longitudinal range; to be used with data that you want 
# to be shown over a full sphere; in the onion-type plot this will be the low
phi=360.*np.arange(ntiles*nphi)/(ntiles*nphi-1)
#Onion type plotting needs to be done differently for wedges and full spheres
if ntiles > 1:
	#Removing ntileso tiles to make the next layer of onion peeling; should be repeated,
        #if more layers are wanted, but is trivial.
	phi2=((ntiles-ntileso)*360./(ntiles))*np.arange((ntiles-ntileso)*nphi)/((ntiles-ntileso)*nphi-1)
else:
	#The same here, but ntiles cannot be used
	#Note that in the case of a full 2pi run, the data will become distorted in the surface plot, as it is currently squeezed. Could be done more intelligently, but now no time.
#	phi2=((nphi-1)*360./(nphi))*np.arange((ntiles-1)*nphi)/((nphi-1)*nphi-1)
	phi2=180.*grid.z/np.pi/ntileso
# Co-latitude grid in degrees
lat=grid.y*180./np.pi
# Radius
r=grid.x

#2D arrays for projections of coordinates
xx, yy = np.meshgrid(phi, lat) #On a sphere
xx2, yy = np.meshgrid(phi2, lat) #On a sphere
zz, yy = np.meshgrid(r, lat) #On meridional slices


# Scalar data for surface contour plots on spheres
scalartop=np.tile(var.uu[0,:,:,rtop].T,ntiles)
scalartop2=np.tile(var.uu[0,:,:,rtop].T,ntiles-ntileso)
scalarbot=np.tile(var.uu[0,:,:,rbot].T,ntiles)
scalarmer1=var.uu[0,0,:,:]
scalarmer2=var.uu[0,int(nphi/2),:,:]

# Create arrays of grid cell boundaries, which have shape of (x.shape[0] + 1)
xx_bounds = _cell_bounds(phi)
xx_bounds2 = _cell_bounds(phi2)
yy_bounds = _cell_bounds(lat)
zz_bounds = _cell_bounds(r)

# Mesh building starts here
levels = [TOP * 1.01]
levels2 = [BOT + 0.01]

# Creating Cartesian grid that describes spherical surfaces at TOP and BOT levels,
# either full and partial depending on where on the onion you are...
# ... and meridional surfaces 0. and the desired azimuthal "opening of the cone"
grid_scalartop = pv.grid_from_sph_coords(xx_bounds, yy_bounds, levels)
grid_scalartop2 = pv.grid_from_sph_coords(xx_bounds2, yy_bounds, levels)
grid_scalarbot = pv.grid_from_sph_coords(xx_bounds, yy_bounds, levels2)
# ... and meridional surfaces 0. and the desired azimuthal "opening of the cone"
grid_scalarmer1 = pv.grid_from_sph_coords(0., yy_bounds, zz_bounds)
grid_scalarmer2 = pv.grid_from_sph_coords(-ntileso*360./ntiles, yy_bounds, zz_bounds)

# And fill its cell arrays with the scalar data
grid_scalartop.cell_arrays["Ur top"] = np.array(scalartop).swapaxes(-2, -1).ravel("C")
grid_scalartop2.cell_arrays["Ur top"] = np.array(scalartop2).swapaxes(-2, -1).ravel("C")
grid_scalarbot.cell_arrays["Ur bottom"] = np.array(scalarbot).swapaxes(-2, -1).ravel("C")
#
grid_scalarmer1.cell_arrays["Meridional cut of Ur"] = np.array(scalarmer1).swapaxes(-2, -1).ravel("C")
grid_scalarmer2.cell_arrays["Meridional cut of Ur"] = np.array(scalarmer2).swapaxes(-2, -1).ravel("C")

# Onion style plot with scalar data only
p = pv.Plotter(shape=(2,2))
p.subplot(0, 0)
#p.add_mesh(pv.Sphere(radius=BOT))
p.add_mesh(grid_scalartop2, opacity=0.7, cmap="rainbow",show_scalar_bar=False)
p.add_mesh(grid_scalarbot, opacity=1.0, cmap="rainbow",show_scalar_bar=False)
p.add_mesh(grid_scalarmer1, opacity=0.7, cmap="rainbow",show_scalar_bar=False)
p.add_mesh(grid_scalarmer2, opacity=0.7, cmap="rainbow",show_scalar_bar=False)

########################## "Hairy ball vectors" plot  ################################


# Vector data in 2D
u_vec=np.tile(var.uu[2,:,:,rbot].T,ntiles).T
v_vec=np.tile(var.uu[1,:,:,rbot].T,ntiles).T
#Removing any systematic radial motion
w_vec=np.tile(var.uu[0,:,:,rbot].T,ntiles).T - np.mean(var.uu[0,:,:,rbot])

# Transform of the 2D arrays to Cartesian coordinates to create hairy ball
vectors = pv.transform_vectors_sph_to_cart(phi,lat,plot_level,u_vec,-v_vec,w_vec)

# Create a grid for the vectors
grid_vec = pv.grid_from_sph_coords(phi, lat, plot_level)

# Add vectors to the grid
vectors2d=np.reshape(np.transpose(vectors,(1,2,0)),[nt*nphi*ntiles,3])

# Scaling to the vectors to make them visible; could be done more elegantly, but I do not have time now.
vectors2d *= 100000.

grid_vec.point_arrays["example"] = vectors2d

p.subplot(0, 1)
p.add_mesh(grid_scalarbot, opacity=1.0, cmap="rainbow",show_scalar_bar=False)
p.add_mesh(grid_vec.glyph(orient="example", scale="example", tolerance=0.01),cmap="rainbow",show_scalar_bar=False)

####### Streamlines ###############################################

# Previously, we had 2D vectors, but now we use 3D for streamlines
u_vec=np.transpose(np.tile(var.uu[2,:,:,:],ntiles),(2,1,0))
v_vec=np.transpose(np.tile(var.uu[1,:,:,:],ntiles),(2,1,0))
w_vec=np.transpose(np.tile(var.uu[0,:,:,:],ntiles),(2,1,0))

# Transform vectors to cartesian coordinates

vectors2 = pv.transform_vectors_sph_to_cart(phi,lat,r,u_vec,v_vec,w_vec)
vectors = np.transpose(np.reshape(vectors2,[3,nr*nt*nphi*ntiles]),(1,0))

# Scaling to the vectors to make them visible; could be done more elegantly, but I do not have time now.
vectors *= 100000.

# Create mesh and streamlines
mesh = pv.grid_from_sph_coords(phi, lat, r)
mesh['vectors'] = vectors
stream, src = mesh.streamlines('vectors', return_source=True,
                               terminal_speed=0.0, n_points=1000,initial_step_length=0.001,
                               source_radius=1.0)

# Plot the streamlines without anything else, for fun
p.subplot(1, 0)
#p.add_mesh(mesh.outline(), color="k")
#p.add_mesh(src)
p.add_mesh(stream.tube(radius=0.005), lighting=False,cmap="rainbow")

##### Show all together #######################
p.subplot(1, 1)
p.add_mesh(grid_scalartop2, opacity=0.7, cmap="rainbow",show_scalar_bar=False)
p.add_mesh(grid_scalarbot, opacity=0.9, cmap="rainbow",show_scalar_bar=False)
p.add_mesh(grid_scalarmer1, opacity=0.9, cmap="rainbow",show_scalar_bar=False)
p.add_mesh(grid_scalarmer2, opacity=0.9, cmap="rainbow",show_scalar_bar=False)
p.add_mesh(stream.tube(radius=0.01), lighting=False, cmap="rainbow")

p.show()
###

