"""
Functions for checking for collisions with ports and other obstacles.
"""

import numpy as np
from abc import ABC, abstractmethod
from .._core.dev import SimsoptRequires

try:
    from pyevtk.hl import unstructuredGridToVTK
    from pyevtk.vtk import VtkTriangle
except ImportError:
    unstructuredGridToVTK = None

__all__ = ['PortSet', 'Port', 'CircularPort', 'RectangularPort']

contig = np.ascontiguousarray


class PortSet(object):
    """
    Handles sets of port-like objects.
    """

    def __init__(self, ports=None, file=None, port_type=None):
        """
        Initializes a list of ports by loading parameters from a file and/or
        incorporating input port class instances.

        Parameters
        ----------
            ports: list of port class instances (optional)
                Ports to be added to the set
            file: string or list of strings (optional)
                Name(s) of the file(s) from which to load port data of type 
                specified by `port_type`
            port_type: string or list of strings (optional)
                Type(s) of ports to load from file(s) if `file` is specified. 
                Supported inputs currently include 'circular', 'rectangular',
                and the corresponding Python type instances.
        """

        self.n_ports = 0
        self.ports = []

        if ports is not None:
            self.add_ports(ports)

        if file is not None:

            if isinstance(file, str):
                file = [file]

            if port_type is None:
                raise ValueError('port_type must be specified if loading '
                                 + 'from file')
            elif isinstance(port_type, str):
                port_type = [port_type]
            elif not hasattr(port_type, '__len__'):
                port_type = [port_type]

            if len(file) > 1 and len(port_type) != len(file):
                raise ValueError('port_type must have one element for every '
                                 'element of file')

            for i in range(len(file)):

                if port_type[i] == 'circular' or port_type[i] == CircularPort:
                    self.load_circular_ports_from_file(file[i])
                elif port_type[i] == 'rectangular' or \
                        port_type[i] == RectangularPort:
                    self.load_rectangular_ports_from_file(file[i])
                else:
                    raise ValueError('Input port_type is not supported')

    def add_ports(self, ports):
        """
        Add ports to the set

        Parameters
        ----------
            ports: list of port class instances
        """

        if not hasattr(ports, '__len__'):
            raise ValueError('Input `ports` must be a list or array')

        for port in ports:
            if not (isinstance(port, CircularPort) or
                    isinstance(port, RectangularPort)):
                raise ValueError('Element %d of `ports` is not a supported '
                                 + 'port class')

        for port in ports:
            self.ports.append(port)
            self.n_ports = self.n_ports + 1

    def __add__(self, p):
        """
        Allow new port sets to be built with the "+" operator
        """

        if isinstance(p, PortSet):
            return PortSet(ports=(self.ports + p.ports))

        elif isinstance(p, RectangularPort) or isinstance(p, CircularPort):
            p_out = PortSet(ports=self.ports)
            p_out.add_ports([p])
            return p_out

        else:
            raise ValueError('Addition with PortSet class instances is only '
                             'supported for Port and PortSet class instances.')

    def __getitem__(self, key):
        """
        Allow member ports to be accessed with "[]" directly from the class
        instance
        """

        return self.ports[key]

    def load_circular_ports_from_file(self, file):
        """
        Loads a set of circular ports from a file. The file must have the CSV
        (comma-separated values) format and exactly one header line (the 
        first line in the file will be ignored).

        The columns must have the following parameters in order (see docstring 
        for CircularPort for definitions):
        ox, oy, oz, ax, ay, az, ir, thick, l0, l1

        Parameters
        ----------
            file: string
                Name of the file from which to load the port parameters
        """

        portdata = np.loadtxt(file, delimiter=',', skiprows=1)

        if portdata.shape[1] != 10:
            raise ValueError('Circular ports input file must have 10 columns '
                             + 'with the following data:\n'
                             + 'ox, oy, oz, ax, ay, az, ir, thick, l0, l1')

        for i in range(portdata.shape[0]):

            self.ports.append(
                CircularPort(ox=portdata[i, 0], oy=portdata[i, 1],
                             oz=portdata[i, 2], ax=portdata[i, 3],
                             ay=portdata[i, 4], az=portdata[i, 5],
                             ir=portdata[i, 6], thick=portdata[i, 7],
                             l0=portdata[i, 8], l1=portdata[i, 9]))

            self.n_ports = self.n_ports + 1

    def load_rectangular_ports_from_file(self, file):
        """
        Loads a set of rectangular ports from a file. The file must have the CSV
        (comma-separated values) format and exactly one header line (the 
        first line in the file will be ignored).

        The columns must have the following parameters in order (see docstring 
        for RectangularPort for definitions):
        ox, oy, oz, ax, ay, az, wx, wy, wz, iw, ih, thick, l0, l1

        Parameters
        ----------
            file: string
                Name of the file from which to load the port parameters
        """

        portdata = np.loadtxt(file, delimiter=',', skiprows=1)

        if portdata.shape[1] != 14:
            raise ValueError('Rectangular ports input file must have 17 '
                             + 'columns with the following data:\n'
                             + 'ox, oy, oz, ax, ay, az, wx, wy, wz, '
                             + 'iw, ih, thick, l0, l1')

        for i in range(portdata.shape[0]):

            self.ports.append(
                RectangularPort(ox=portdata[i, 0], oy=portdata[i, 1],
                                oz=portdata[i, 2], ax=portdata[i, 3],
                                ay=portdata[i, 4], az=portdata[i, 5],
                                wx=portdata[i, 6], wy=portdata[i, 7],
                                wz=portdata[i, 8], iw=portdata[i, 9],
                                ih=portdata[i, 10], thick=portdata[i, 11],
                                l0=portdata[i, 12], l1=portdata[i, 13]))

            self.n_ports = self.n_ports + 1

    def save_ports_to_file(self, fname):
        """
        Save data on ports to csv-formatted files. Separate files will be 
        created for each port type. Circular ports will be saved to a file
        ending in "_circ.csv"; rectangular ports will be saved to a file 
        ending in "_rect.csv".

        Parameters
        ----------
            fname: string
                Name of the file to save, not including the suffix
        """

        # Group the ports by type
        circ_ports = [p for p in self.ports if isinstance(p, CircularPort)]
        rect_ports = [p for p in self.ports if isinstance(p, RectangularPort)]

        # Save the circular ports if any exist
        if len(circ_ports) > 0:
            lines = [('%.16e,'*9 + '%.16e') % (p.ox, p.oy, p.oz, p.ax, p.ay,
                                               p.az, p.ir, p.thick, p.l0, p.l1) for p in circ_ports]
            all_lines = '\n'.join(lines)
            print('Saving circular ports to ' + fname + '_circ.csv')
            with open(fname + '_circ.csv', 'w') as f:
                f.write('ox,oy,oz,ax,ay,az,ir,thick,l0,l1\n')
                f.write(all_lines)

        # Save the rectangular ports if any exist
        if len(rect_ports) > 0:
            lines = [('%.16e,'*13 + '%.16e') % (p.ox, p.oy, p.oz, p.ax, p.ay,
                                                p.az, p.wx, p.wy, p.wz, p.iw, p.ih, p.thick,
                                                p.l0, p.l1) for p in rect_ports]
            all_lines = '\n'.join(lines)
            print('Saving rectangular ports to ' + fname + '_rect.csv')
            with open(fname + '_rect.csv', 'w') as f:
                f.write('ox,oy,oz,ax,ay,az,wx,wy,wz,iw,ih,thick,l0,l1\n')
                f.write(all_lines)

    def collides(self, x, y, z, gap=0.0):
        """
        Determines if the user input point(s) collide with any port in the set.

        Parameters
        ----------
            x, y, z: float arrays
                Cartesian x, y, and z coordinates of the point(s) to be 
                assessed. Dimensions must be the same.
            gap: float or array-like (optional)
                Minimum gap spacing(s) required between the point and the 
                exteriors of each port. If scalar, the same gap will be applied
                to all ports; if a list or array, the n-th element will be
                assigned to the n-th port in the set. Default is a scalar 0.

        Returns
        -------
            colliding: logical array
                Array of logical values indicating whether each of the input
                points collide with any of the ports in the set.
        """

        if np.isscalar(gap):
            gaparr = gap * np.ones(self.n_ports)
        elif np.size(gap) == 1:
            gaparr = np.squeeze(gap)[0] * np.ones(self.n_ports)
        else:
            gaparr = np.array(gap).reshape((-1))
            if len(gaparr) != self.n_ports:
                raise ValueError('Input gap must be scalar or have one '
                                 'element per port in the set')

        colliding = np.full(x.shape, False)
        for i in range(self.n_ports):
            colliding_i = self.ports[i].collides(x, y, z, gap=gaparr[i])
            colliding = np.logical_or(colliding, colliding_i)

        return colliding

    def repeat_via_symmetries(self, nfp, stell_sym):
        """
        Creates a new set that contains the ports in the initial set plus 
        additional equivalent ports the remaining field periods/half-periods.

        Parameters
        ----------
            nfp: integer
                Number of toroidal field periods
            stell_sym: logical
                If true, stellarator symmetry will be assumed and equivalent
                ports will be created for every half-period

        Returns
        -------
            ports_out: PortSet class instance
                A set of all the symmetric ports, including the ones represented
                by the calling PortSet class instance
        """

        ports_out = PortSet()

        for port in self.ports:

            ports_out += port.repeat_via_symmetries(nfp, stell_sym)

        return ports_out

    def plot(self, n_edges=100, **kwargs):
        """
        Places representations of the ports on a 3D plot. Currently only
        works with the mayavi plotting package.

        Parameters
        ----------
            n_edges: integer (optional)
                Number of edges for the polygons used to approximate the 
                cross-section for any ports that are circular. Default is 100.
            **kwargs
                Keyword arguments to be passed to the mayavi surface module
                for each port

        Returns
        -------
            surfs: list of mayavi.modules.surface.Surface class instance
                References to the plotted ports
        """

        surfs = []
        for port in self.ports:
            if isinstance(port, CircularPort):
                surfs.append(port.plot(n_edges=n_edges, **kwargs))
            elif isinstance(port, RectangularPort):
                surfs.append(port.plot(**kwargs))
            else:
                raise RuntimeError('Should not get here!')

        return surfs

    @SimsoptRequires(unstructuredGridToVTK is not None,
                     "to_vtk method requires pyevtk module")
    def to_vtk(self, filename, n_edges=100):
        """
        Export a mesh representation of the port set to a VTK file, which can 
        be read with ParaView.

        In the VTK file, each mesh point will be associated with an integer
        value "index" corresponding to the index of its respective port
        within the port set.

        Note: This function requires the ``pyevtk`` python package, which can 
        be installed using ``pip install pyevtk``.

        Parameters
        ----------
            filename: string
                Name of the VTK file, without extension, to create.
            n_edges: integer (optional)
                Number of edges for the polygon used to approximate the 
                circular cross-section of any circular ports. Default is 100.
        """

        x_list = [[]]*self.n_ports
        y_list = [[]]*self.n_ports
        z_list = [[]]*self.n_ports
        triangles_list = [[]]*self.n_ports
        index_list = [[]]*self.n_ports
        nPoints = 0

        for i in range(self.n_ports):

            if isinstance(self.ports[i], CircularPort):
                x_list[i], y_list[i], z_list[i], triangles_list[i] \
                    = self.ports[i].mesh_representation(n_edges=n_edges)
            else:
                x_list[i], y_list[i], z_list[i], triangles_list[i] \
                    = self.ports[i].mesh_representation()

            # Offset vertex indices by # of vertices in previous ports
            triangles_list[i] += nPoints
            nPoints += x_list[i].shape[0]

            index_list[i] = np.full(x_list[i].size, i)

        x = contig(np.concatenate(x_list, axis=0))
        y = contig(np.concatenate(y_list, axis=0))
        z = contig(np.concatenate(z_list, axis=0))
        triangles = np.concatenate(triangles_list, axis=0)
        connectivity = contig(triangles.reshape((-1)))
        offsets = contig(3*np.arange(triangles.shape[0])+3)
        index = contig(np.concatenate(index_list))

        unstructuredGridToVTK(filename, x, y, z, connectivity, offsets,
                              contig(np.full(offsets.shape, VtkTriangle.tid)),
                              pointData={'index': index})


class Port(ABC):
    """
    Abstract base class for ports
    """

    @abstractmethod
    def collides(self):
        """
        Determines if the user input point(s) collide with the port, with an
        optional gap spacing enforced around the port's exterior.
        """
        pass

    @abstractmethod
    def repeat_via_symmetries(self):
        """
        Returns a PortSet class instance containing the calling Port class
        instances as well ports with parameters transformed to equivalent
        locations in other periods (and half-periods if stellarator symmetry
        is assumed).
        """
        pass

    @abstractmethod
    def mesh_representation(self):
        """
        Constructs a triangular mesh representation of the port for plotting
        and visualization.

        Returns
        -------
            x, y, z: 1D arrays
                Cartesian x, y, and z coordinates of the vertices of the mesh.
            triangles: 2D array
                Indices of the vertices (as listed in the x, y, and z arrays)
                surrounding each triangular face within the mesh. Each row
                (dimension 1) represents a face.
        """
        pass

    @abstractmethod
    def plot(self):
        """
        Returns handle to a three-dimensional plot with a visual depiction
        of the port.
        """
        pass

    def __add__(self, p):

        if isinstance(p, Port):
            return PortSet(ports=[self, p])

        elif isinstance(p, PortSet):
            return PortSet(ports=([self] + p.ports))

        else:
            raise ValueError('Addition with Port class instances is only '
                             'supported for Port and PortSet class instances.')


class CircularPort(Port):
    """
    Class representing a port with cylindrical geometry; specifically, with
    a circular cross section.
    """

    def __init__(self, ox=1.0, oy=0.0, oz=0.0, ax=1.0, ay=0.0, az=0.0,
                 ir=0.5, thick=0.01, l0=0.0, l1=0.5):
        """
        Initializes the CircularPort class according to the geometric port
        parameters.

        Parameters
        ----------
            ox, oy, oz: floats
                Cartesian x, y, and z coordinates of the origin point of the 
                port in 3d space
            ax, ay, az: floats
                Cartesian x, y, and z components of a vector in the direction
                of the port's axis
            ir: float
                Inner radius of the port
            thick: float
                Wall thickness of the port
            l0, l1: double
                Locations of the ends of the port, specified by the signed
                distance along the port axis from the origin point
        """

        self.ox = ox
        self.oy = oy
        self.oz = oz

        # Ensure that the axis vector is nonzero
        mod_a = np.linalg.norm([ax, ay, az])
        if mod_a == 0:
            raise ValueError('Vector given by ax, ay, and az has zero length')

        self.ax = ax/mod_a
        self.ay = ay/mod_a
        self.az = az/mod_a

        self.ir = ir
        self.thick = thick
        self.l0 = l0
        self.l1 = l1

    def collides(self, x, y, z, gap=0.0):
        """
        Determines if the user input point(s) collide with the port

        Parameters
        ----------
            x, y, z: float arrays
                Cartesian x, y, and z coordinates of the point(s) to be 
                assessed. Dimensions must be the same.
            gap: float (optional)
                Minimum gap spacing required between the point and the exterior
                of the port. Default is 0.

        Returns
        -------
            colliding: logical array
                Array of logical values indicating whether each of the input
                points collide with the port.
        """

        xarr = np.array(x)
        yarr = np.array(y)
        zarr = np.array(z)
        if (xarr.size != yarr.size or xarr.size != zarr.size):
            raise ValueError('Input x, y, and z arrays must have the same size')

        # Projected positions of the points along the port axis
        l_proj = (xarr - self.ox) * self.ax + (yarr - self.oy) * self.ay \
                                            + (zarr - self.oz) * self.az
        proj_x = l_proj * self.ax + self.ox
        proj_y = l_proj * self.ay + self.oy
        proj_z = l_proj * self.az + self.oz

        # Radial distance from the axis
        rvec_x = xarr - proj_x
        rvec_y = yarr - proj_y
        rvec_z = zarr - proj_z
        r2 = rvec_x * rvec_x + rvec_y * rvec_y + rvec_z * rvec_z

        # Outer radius within which a point is considered colliding
        out_r = self.ir + self.thick + gap
        out_r2 = out_r * out_r

        l_start = np.min([self.l0, self.l1])
        l_stop = np.max([self.l0, self.l1])
        in_axial_bounds = np.logical_and(l_proj >= l_start - gap,
                                         l_proj <= l_stop + gap)
        in_radial_bounds = r2 <= out_r2

        return np.logical_and(in_axial_bounds, in_radial_bounds)

    def repeat_via_symmetries(self, nfp, stell_sym, include_self=True):
        """
        Generates a set of equivalent ports to uphold toroidal and/or 
        stellarator symmetry.

        Parameters
        ----------
            nfp: integer
                Number of toroidal field periods
            stell_sym: logical
                If true, stellarator symmetry will be assumed and equivalent
                ports will be created for every half-period
            include_self: logical (optional)
                If true, the port instance calling this method will be included
                in the returned set of ports (see below). Default is True.

        Returns
        -------
            ports: PortSet class instance
                A set of all the symmetric ports, including the one represented
                by the calling Port class instance if ``include_self==True``.
        """

        dphi = 2.*np.pi/nfp

        ports = [self] if include_self else []

        for i in range(nfp):

            # Rotate origin and axis vectors to the given field period
            oxi = self.ox*np.cos(i*dphi) - self.oy*np.sin(i*dphi)
            oyi = self.ox*np.sin(i*dphi) + self.oy*np.cos(i*dphi)
            ozi = self.oz

            axi = self.ax*np.cos(i*dphi) - self.ay*np.sin(i*dphi)
            ayi = self.ax*np.sin(i*dphi) + self.ay*np.cos(i*dphi)
            azi = self.az

            if i > 0:

                ports.append(CircularPort(ox=oxi, oy=oyi, oz=ozi,
                                          ax=axi, ay=ayi, az=azi, ir=self.ir, thick=self.thick,
                                          l0=self.l0, l1=self.l1))

            if stell_sym:

                ports.append(CircularPort(ox=oxi, oy=-oyi, oz=-ozi,
                                          ax=-axi, ay=ayi, az=azi, ir=self.ir, thick=self.thick,
                                          l0=-self.l1, l1=-self.l0))

        return PortSet(ports=ports)

    def mesh_representation(self, n_edges=100):
        """
        Constructs a triangular mesh representation of the port for plotting
        and visualization.


        Parameters
        ----------
            n_edges: integer (optional)
                Number of edges for the polygon used to approximate the 
                circular cross-section. Default is 100.

        Returns
        -------
            x, y, z: 1D arrays
                Cartesian x, y, and z coordinates of the vertices of the mesh.
            triangles: 2D array
                Indices of the vertices (as listed in the x, y, and z arrays)
                surrounding each triangular face within the mesh. Each row
                (dimension 1) represents a face.
        """

        # Compute two radial vectors normal to the cylinder axis
        a_theta = np.arctan2(np.sqrt(self.ax**2 + self.ay**2), self.az)
        a_phi = np.arctan2(self.ay, self.ax)
        b_theta = a_theta + 0.5*np.pi
        b_phi = a_phi
        bx = np.sin(b_theta)*np.cos(b_phi)
        by = np.sin(b_theta)*np.sin(b_phi)
        bz = np.cos(b_theta)
        cx = self.ay * bz - self.az * by
        cy = -self.ax * bz + self.az * bx
        cz = self.ax * by - self.ay * bx

        # Points at each end of the cylinder
        x0 = self.ox + self.l0*self.ax
        y0 = self.oy + self.l0*self.ay
        z0 = self.oz + self.l0*self.az
        x1 = self.ox + self.l1*self.ax
        y1 = self.oy + self.l1*self.ay
        z1 = self.oz + self.l1*self.az

        # Radial unit vector extending from the axis at different angles
        phi = np.linspace(0, 2.*np.pi, n_edges, endpoint=False).reshape((1, -1))
        rx = bx*np.cos(phi) + cx*np.sin(phi)
        ry = by*np.cos(phi) + cy*np.sin(phi)
        rz = bz*np.cos(phi) + cz*np.sin(phi)

        # Coordinates of points on the inner and outer edges of the cylinder
        ri = self.ir
        ro = self.ir + self.thick
        x = np.concatenate((x0 + ro*rx, x0 + ri*rx, x1 + ri*rx, x1 + ro*rx),
                           axis=0).reshape((-1, 1))
        y = np.concatenate((y0 + ro*ry, y0 + ri*ry, y1 + ri*ry, y1 + ro*ry),
                           axis=0).reshape((-1, 1))
        z = np.concatenate((z0 + ro*rz, z0 + ri*rz, z1 + ri*rz, z1 + ro*rz),
                           axis=0).reshape((-1, 1))

        # Index arrays
        upper_left = np.arange(4*n_edges).reshape((4, n_edges))
        lower_left = np.roll(upper_left, 1, axis=1)
        upper_right = np.roll(upper_left, 1, axis=0)
        lower_right = np.roll(upper_right, 1, axis=1)
        ul_col = upper_left.reshape((-1, 1))
        ll_col = lower_left.reshape((-1, 1))
        ur_col = upper_right.reshape((-1, 1))
        lr_col = lower_right.reshape((-1, 1))
        triangles1 = np.concatenate((ul_col, ll_col, lr_col), axis=1)
        triangles2 = np.concatenate((ul_col, lr_col, ur_col), axis=1)
        triangles = np.concatenate((triangles1, triangles2), axis=0)

        return x, y, z, triangles

    def plot(self, n_edges=100, **kwargs):
        """
        Places a representation of the port on a 3D plot. Currently only
        works with the mayavi plotting package.

        Parameters
        ----------
            n_edges: integer (optional)
                Number of edges for the polygon used to approximate the 
                circular cross-section. Default is 100.
            **kwargs
                Keyword arguments to be passed to the mayavi surface module
                for the port

        Returns
        -------
            surf: mayavi.modules.surface.Surface class instance
                Reference to the plotted port
        """

        from mayavi import mlab

        # Obtain data for the mesh representation of the port
        x, y, z, triangles = self.mesh_representation(n_edges=n_edges)

        # Generate a mayavi surface instance
        if 'color' not in kwargs.keys():
            kwargs['color'] = (0.75, 0.75, 0.75)
        return mlab.triangular_mesh(x, y, z, triangles, **kwargs)

    @SimsoptRequires(unstructuredGridToVTK is not None,
                     "to_vtk method requires pyevtk module")
    def to_vtk(self, filename, n_edges=100):
        """
        Export a mesh representation of the port to a VTK file, which can be 
        read with ParaView.

        Note: This function requires the ``pyevtk`` python package, which can 
        be installed using ``pip install pyevtk``.

        Parameters
        ----------
            filename: string
                Name of the VTK file, without extension, to create.
            n_edges: integer (optional)
                Number of edges for the polygon used to approximate the 
                circular cross-section. Default is 100.
        """

        # Obtain data for the mesh representation of the port
        x, y, z, triangles = self.mesh_representation(n_edges=n_edges)
        x = contig(x)
        y = contig(y)
        z = contig(z)
        triangles = contig(triangles)

        # Convert triangles array data to unstructured representation
        connectivity = contig(triangles.reshape((-1)))
        offsets = contig(3*np.arange(triangles.shape[0]) + 3)

        # Save to file
        unstructuredGridToVTK(filename, x, y, z, connectivity, offsets,
                              contig(np.full(offsets.shape, VtkTriangle.tid)))


class RectangularPort(Port):
    """
    Class representing a port with a rectangular cross-section.
    """

    def __init__(self, ox=1.0, oy=0.0, oz=0.0, ax=1.0, ay=0.0, az=0.0,
                 wx=0.0, wy=1.0, wz=0.0, iw=0.5, ih=0.5, thick=0.01,
                 l0=0.0, l1=0.5):
        """
        Initializes the RectangularPort class according to the geometric port
        parameters.

        Parameters
        ----------
            ox, oy, oz: floats
                Cartesian x, y, and z coordinates of the origin point of the 
                port in 3d space
            ax, ay, az: floats
                Cartesian x, y, and z components of a vector in the direction
                of the port's axis
            wx, wy, wz: floats
                Cartesian x, y, and z components of a vector in the direction
                spanning the width of the cross-section, assumed perpendicular
                to the axis
            iw: float
                Inner width of the cross-section, i.e. the dimension spanned by
                the vector (wx, wy, wz)
            ih: float
                Inner height of the cross-section, i.e. the dimension spanned by
                the vector (hx, hy, hz)
            thick: float
                Thickness of the port wall
            l0, l1: double
                Locations of the ends of the port, specified by the signed
                distance along the port axis from the origin point
        """

        self.ox = ox
        self.oy = oy
        self.oz = oz

        # Ensure that the axis and orientation vectors are nonzero
        mod_a = np.linalg.norm([ax, ay, az])
        mod_w = np.linalg.norm([wx, wy, wz])
        if mod_a == 0 or mod_w == 0:
            raise ValueError('Vectors given by (ax, ay, az),  (wx, wy, wz), '
                             + 'and (hx, hy, hz) must have nonzero length')

        # Ensure that the axis vectors are mutually perpendiclar
        self.ax = ax/mod_a
        self.ay = ay/mod_a
        self.az = az/mod_a
        self.wx = wx/mod_w
        self.wy = wy/mod_w
        self.wz = wz/mod_w
        tol = 1e-12
        if np.abs(self.ax*self.wx + self.ay*self.wy + self.az*self.wz) > tol:
            raise ValueError('Vectors given by (ax, ay, az),  (wx, wy, wz), '
                             + 'and (hx, hy, hz) must be mutually perpendicuar')

        # Determine the third axis (height) from the two suppied axes
        self.hx = self.ay*self.wz - self.az*self.wy
        self.hy = -self.ax*self.wz + self.az*self.wx
        self.hz = self.ax*self.wy - self.ay*self.wx
        assert np.abs(np.sqrt(self.hx**2 + self.hy**2 + self.hz**2) - 1) < tol

        self.iw = iw
        self.ih = ih
        self.thick = thick
        self.l0 = l0
        self.l1 = l1

    def collides(self, x, y, z, gap=0.0):
        """
        Determines if the user input point(s) collide with the port

        Parameters
        ----------
            x, y, z: float arrays
                Cartesian x, y, and z coordinates of the point(s) to be 
                assessed. Dimensions must be the same.
            gap: float (optional)
                Minimum gap spacing required between the point and the exterior
                of the port. Default is 0.

        Returns
        -------
            colliding: logical array
                Array of logical values indicating whether each of the input
                points collide with the port.
        """

        xarr = np.array(x)
        yarr = np.array(y)
        zarr = np.array(z)
        if (xarr.size != yarr.size or xarr.size != zarr.size):
            raise ValueError('Input x, y, and z arrays must have the same size')

        # Projected positions of the points along the port axis
        l_proj = (xarr - self.ox) * self.ax + (yarr - self.oy) * self.ay \
                                            + (zarr - self.oz) * self.az

        # Projected positions of the points along the width-spanning axis
        w_proj = (xarr - self.ox) * self.wx + (yarr - self.oy) * self.wy \
                                            + (zarr - self.oz) * self.wz

        # Projected positions of the points along the height-spanning axis
        h_proj = (xarr - self.ox) * self.hx + (yarr - self.oy) * self.hy \
                                            + (zarr - self.oz) * self.hz

        l_start = np.min([self.l0, self.l1])
        l_stop = np.max([self.l0, self.l1])
        in_axial_bounds = np.logical_and(l_proj >= l_start - gap,
                                         l_proj <= l_stop + gap)
        in_cross_section = \
            np.logical_and(np.abs(w_proj) < 0.5*self.iw + self.thick + gap,
                           np.abs(h_proj) < 0.5*self.ih + self.thick + gap)

        return np.logical_and(in_axial_bounds, in_cross_section)

    def repeat_via_symmetries(self, nfp, stell_sym, include_self=True):
        """
        Generates a set of equivalent ports to uphold toroidal and/or 
        stellarator symmetry.

        Parameters
        ----------
            nfp: integer
                Number of toroidal field periods
            stell_sym: logical
                If true, stellarator symmetry will be assumed and equivalent
                ports will be created for every half-period
            include_self: logical (optional)
                If true, the port instance calling this method will be included
                in the returned list of ports (see below). Default is True.

        Returns
        -------
            ports: PortSet class instance
                PortSet containing the equivalent ports, including the one
                represented by the calling instance if include_self == True.
        """

        dphi = 2.*np.pi/nfp

        ports = [self] if include_self else []

        for i in range(nfp):

            # Rotate origin and axis vectors to the given field period
            oxi = self.ox*np.cos(i*dphi) - self.oy*np.sin(i*dphi)
            oyi = self.ox*np.sin(i*dphi) + self.oy*np.cos(i*dphi)
            ozi = self.oz

            axi = self.ax*np.cos(i*dphi) - self.ay*np.sin(i*dphi)
            ayi = self.ax*np.sin(i*dphi) + self.ay*np.cos(i*dphi)
            azi = self.az

            wxi = self.wx*np.cos(i*dphi) - self.wy*np.sin(i*dphi)
            wyi = self.wx*np.sin(i*dphi) + self.wy*np.cos(i*dphi)
            wzi = self.wz

            if i > 0:

                ports.append(RectangularPort(ox=oxi, oy=oyi, oz=ozi,
                                             ax=axi, ay=ayi, az=azi, wx=wxi, wy=wyi, wz=wzi,
                                             iw=self.iw, ih=self.ih, thick=self.thick,
                                             l0=self.l0, l1=self.l1))

            if stell_sym:

                ports.append(RectangularPort(ox=oxi, oy=-oyi, oz=-ozi,
                                             ax=-axi, ay=ayi, az=azi, wx=-wxi, wy=wyi, wz=wzi,
                                             iw=self.iw, ih=self.ih, thick=self.thick,
                                             l0=-self.l1, l1=-self.l0))

        return PortSet(ports=ports)

    def mesh_representation(self):
        """
        Constructs a triangular mesh representation of the port for plotting
        and visualization.

        Returns
        -------
            x, y, z: 1D arrays
                Cartesian x, y, and z coordinates of the vertices of the mesh.
            triangles: 2D array
                Indices of the vertices (as listed in the x, y, and z arrays)
                surrounding each triangular face within the mesh. Each row
                (dimension 1) represents a face.
        """

        # Points at each end of the port
        x0 = self.ox + self.l0*self.ax
        y0 = self.oy + self.l0*self.ay
        z0 = self.oz + self.l0*self.az
        x1 = self.ox + self.l1*self.ax
        y1 = self.oy + self.l1*self.ay
        z1 = self.oz + self.l1*self.az

        # Unit vectors in the cross-sectional dimensions
        wxxs = self.wx * np.array([[1, 1, -1, -1]])
        wyxs = self.wy * np.array([[1, 1, -1, -1]])
        wzxs = self.wz * np.array([[1, 1, -1, -1]])
        hxxs = self.hx * np.array([[-1, 1, 1, -1]])
        hyxs = self.hy * np.array([[-1, 1, 1, -1]])
        hzxs = self.hz * np.array([[-1, 1, 1, -1]])

        # Coordinates of points on the inner and outer edges of the cylinder
        ihw = 0.5*self.iw
        ohw = 0.5*self.iw + self.thick
        ihh = 0.5*self.ih
        ohh = 0.5*self.ih + self.thick
        x = np.concatenate((x0 + ihw*wxxs + ihh*hxxs, x0 + ohw*wxxs + ohh*hxxs,
                            x1 + ohw*wxxs + ohh*hxxs, x1 + ihw*wxxs + ihh*hxxs),
                           axis=0).reshape((-1, 1))
        y = np.concatenate((y0 + ihw*wyxs + ihh*hyxs, y0 + ohw*wyxs + ohh*hyxs,
                            y1 + ohw*wyxs + ohh*hyxs, y1 + ihw*wyxs + ihh*hyxs),
                           axis=0).reshape((-1, 1))
        z = np.concatenate((z0 + ihw*wzxs + ihh*hzxs, z0 + ohw*wzxs + ohh*hzxs,
                            z1 + ohw*wzxs + ohh*hzxs, z1 + ihw*wzxs + ihh*hzxs),
                           axis=0).reshape((-1, 1))

        # Index arrays
        upper_left = np.arange(16).reshape((4, 4))
        lower_left = np.roll(upper_left, 1, axis=1)
        upper_right = np.roll(upper_left, 1, axis=0)
        lower_right = np.roll(upper_right, 1, axis=1)
        ul_col = upper_left.reshape((-1, 1))
        ll_col = lower_left.reshape((-1, 1))
        ur_col = upper_right.reshape((-1, 1))
        lr_col = lower_right.reshape((-1, 1))
        triangles1 = np.concatenate((ul_col, ll_col, lr_col), axis=1)
        triangles2 = np.concatenate((ul_col, lr_col, ur_col), axis=1)
        triangles = np.concatenate((triangles1, triangles2), axis=0)

        return x, y, z, triangles

    def plot(self, **kwargs):
        """
        Places a representation of the port on a 3D plot. Currently only
        works with the mayavi plotting package.

        Parameters
        ----------
            **kwargs
                Keyword arguments to be passed to the mayavi surface module
                for the port

        Returns
        -------
            surf: mayavi.modules.surface.Surface class instance
                Reference to the plotted port
        """

        from mayavi import mlab

        # Obtain data for the mesh representation of the port
        x, y, z, triangles = self.mesh_representation()

        # Generate a mayavi surface instance
        if 'color' not in kwargs.keys():
            kwargs['color'] = (0.75, 0.75, 0.75)
        return mlab.triangular_mesh(x, y, z, triangles, **kwargs)

    @SimsoptRequires(unstructuredGridToVTK is not None,
                     "to_vtk method requires pyevtk module")
    def to_vtk(self, filename):
        """
        Export a mesh representation of the port to a VTK file, which can be 
        read with ParaView.

        Note: This function requires the ``pyevtk`` python package, which can 
        be installed using ``pip install pyevtk``.

        Parameters
        ----------
            filename: string
                Name of the VTK file, without extension, to create.
        """

        # Obtain data for the mesh representation of the port
        x, y, z, triangles = self.mesh_representation()
        x = contig(x)
        y = contig(y)
        z = contig(z)
        triangles = contig(triangles)

        # Convert triangles array data to unstructured representation
        connectivity = contig(triangles.reshape((-1)))
        offsets = contig(3*np.arange(triangles.shape[0]) + 3)

        # Save to file
        unstructuredGridToVTK(filename, x, y, z, connectivity, offsets,
                              contig(np.full(offsets.shape, VtkTriangle.tid)))
