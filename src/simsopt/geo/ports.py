"""
ports.py

Functions for checking for collisions with ports and other obstacles.
"""

import numpy as np

__all__ = ['PortSet', 'CircularPort', 'RectangularPort']

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

        self.nPorts = 0
        self.ports = []

        if ports is not None:
            self.add_ports(ports)

        if file is not None:

            if isinstance(file, str):
                file = [file]

            if port_type is None:
                raise ValueError('port_type must be specified if loading ' \
                                 + 'from file')
            elif not hasattr(port_type, '__len__'):
                port_type = [port_type]

            if len(file) > 1 and len(port_type) != len(file):
                raise ValueError('port_type must have one element for every ' \
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
            self.nPorts = self.nPorts + 1

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
            raise ValueError('Circular ports input file must have 10 columns ' \
                             + 'with the following data:\n' \
                             + 'ox, oy, oz, ax, ay, az, ir, thick, l0, l1')
        
        for i in range(portdata.shape[0]):

            self.ports.append( \
                CircularPort(xo = portdata[i,0], yo = portdata[i,1], \
                             zo = portdata[i,2], ax = portdata[i,3], \
                             ay = portdata[i,4], az = portdata[i,5], \
                             ir = portdata[i,6], thick = portdata[i,7], \
                             l0 = portdata[i,8], l1 = portdata[i,9]))

            self.nPorts = self.nPorts + 1

    def load_rectangular_ports_from_file(self, file):
        """
        Loads a set of rectangular ports from a file. The file must have the CSV
        (comma-separated values) format and exactly one header line (the 
        first line in the file will be ignored).

        The columns must have the following parameters in order (see docstring 
        for RectangularPort for definitions):
          ox, oy, oz, ax, ay, az, wx, wy, wz, hx, hy, hz, iw, ih, thick, l0, l1

        Parameters
        ----------
            file: string
                Name of the file from which to load the port parameters
        """

        portdata = np.loadtxt(file, delimiter=',', skiprows=1)

        if portdata.shape[1] != 17:
            raise ValueError('Circular ports input file must have 10 columns ' \
                             + 'with the following data:\n' \
                             + 'ox, oy, oz, ax, ay, az, wx, wy, wz, ' \
                             + 'hx, hy, hz, iw, ih, thick, l0, l1')
        
        for i in range(portdata.shape[0]):

            self.ports.append( \
                RectangularPort(xo = portdata[i, 0], yo = portdata[i, 1], \
                                zo = portdata[i, 2], ax = portdata[i, 3], \
                                ay = portdata[i, 4], az = portdata[i, 5], \
                                wx = portdata[i, 6], wy = portdata[i, 7], \
                                wz = portdata[i, 8], hx = portdata[i, 9], \
                                hy = portdata[i,10], hz = portdata[i,11], \
                                iw = portdata[:,12], ih = portdata[i,13], \
                                thick = portdata[i,14], \
                                l0 = portdata[i,15], l1 = portdata[i,16]))

            self.nPorts = self.nPorts + 1

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
            gaparr = gap * np.ones(self.nPorts)
        elif np.size(gap) == 1:
            gaparr = np.squeeze(gap)[0] * np.ones(self.nPorts)
        else:
            gaparr = np.array(gap).reshape((-1))
            if len(gaparr) != self.nPorts:
                raise ValueError('Input gap must be scalar or have one ' \
                                 'element per port in the set')

        colliding = np.full(x.shape, False)
        for i in range(self.nPorts):
            colliding_i = self.ports[i].collides(x, y, z, gap=gaparr[i])
            colliding = np.logical_or(colliding, colliding_i)

        return colliding

    def repeat_via_symmetries(self, nfp, stell_sym):
        """
        Adds ports that are equivalent to the existing ports to uphold
        toroidal and/or stellarator symmetry.

        Parameters
        ----------
            nfp: integer
                Number of toroidal field periods
            stell_sym: logical
                If true, stellarator symmetry will be assumed and equivalent
                ports will be created for every half-period
        """

        for i in range(self.nPorts):
            newPorts = self.ports[i].repeat_via_symmetries(nfp, stell_sym, \
                                                           include_self=False)
            self.ports = self.ports + newPorts
            self.nPorts += len(newPorts)

    def plot(self, nEdges=100, **kwargs):
        """
        Places representations of the ports on a 3D plot. Currently only
        works with the mayavi plotting package.

        Parameters
        ----------
            nEdges: integer (optional)
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
                surfs.append(port.plot(nEdges=nEdges, **kwargs))
            elif isinstance(port, RectangularPort):
                surfs.append(port.plot(**kwargs))
            else:
                raise RuntimeError('Should not get here!')

        return surfs

class CircularPort(object):
    """
    Class representing a port with cylindrical geometry; specifically, with
    a circular cross section.
    """

    def __init__(self, ox=1.0, oy=0.0, oz=0.0, ax=1.0, ay=0.0, az=0.0, \
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

        lStart = np.min([self.l0, self.l1])
        lStop  = np.max([self.l0, self.l1])
        in_axial_bounds = np.logical_and(l_proj >= lStart - gap, \
                                         l_proj <= lStop  + gap)
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
                in the returned list of ports (see below). Default is True.

        Returns
        -------
            ports: list of CircularPort class instances
                The equivalent ports, including the calling instance if 
                include_self == True.
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

            if stell_sym:

                ports.append(CircularPort(ox=oxi, oy=-oyi, oz=-ozi, \
                    ax=-axi, ay=ayi, az=azi, ir=self.ir, thick=self.thick, \
                    l0=-self.l1, l1=-self.l0))

            if i > 0:

                ports.append(CircularPort(ox=oxi, oy=oyi, oz=ozi, \
                    ax=axi, ay=ayi, az=azi, ir=self.ir, thick=self.thick, \
                    l0=self.l0, l1=self.l1))

        return ports

    def plot(self, nEdges=100, **kwargs):
        """
        Places a representation of the port on a 3D plot. Currently only
        works with the mayavi plotting package.

        Parameters
        ----------
            nEdges: integer (optional)
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

        # Compute two radial vectors normal to the cylinder axis
        a_theta = np.arctan2(np.sqrt(self.ax**2 + self.ay**2), self.az)
        a_phi = np.arctan2(self.ay, self.ax)
        b_theta = a_theta + 0.5*np.pi
        b_phi = a_phi
        bx = np.sin(b_theta)*np.cos(b_phi)
        by = np.sin(b_theta)*np.sin(b_phi)
        bz = np.cos(b_theta)
        cx =  self.ay * bz - self.az * by
        cy = -self.ax * bz + self.az * bx
        cz =  self.ax * by - self.ay * bx

        # Points at each end of the cylinder
        x0 = self.ox + self.l0*self.ax
        y0 = self.oy + self.l0*self.ay
        z0 = self.oz + self.l0*self.az
        x1 = self.ox + self.l1*self.ax
        y1 = self.oy + self.l1*self.ay
        z1 = self.oz + self.l1*self.az

        # Radial unit vector extending from the axis at different angles
        phi = np.linspace(0, 2.*np.pi, nEdges, endpoint=False).reshape((1,-1))
        Rx = bx*np.cos(phi) + cx*np.sin(phi)
        Ry = by*np.cos(phi) + cy*np.sin(phi)
        Rz = bz*np.cos(phi) + cz*np.sin(phi)

        # Coordinates of points on the inner and outer edges of the cylinder
        ri = self.ir
        ro = self.ir + self.thick
        x = np.concatenate((x0 + ro*Rx, x0 + ri*Rx, x1 + ri*Rx, x1 + ro*Rx), \
                           axis=0).reshape((-1,1))
        y = np.concatenate((y0 + ro*Ry, y0 + ri*Ry, y1 + ri*Ry, y1 + ro*Ry), \
                           axis=0).reshape((-1,1))
        z = np.concatenate((z0 + ro*Rz, z0 + ri*Rz, z1 + ri*Rz, z1 + ro*Rz), \
                           axis=0).reshape((-1,1))

        # Index arrays
        upper_left = np.arange(4*nEdges).reshape((4,nEdges))
        lower_left = np.roll(upper_left, 1, axis=1)
        upper_right = np.roll(upper_left, 1, axis=0)
        lower_right = np.roll(upper_right, 1, axis=1)
        ul_col = upper_left.reshape((-1,1))
        ll_col = lower_left.reshape((-1,1))
        ur_col = upper_right.reshape((-1,1))
        lr_col = lower_right.reshape((-1,1))
        triangles1 = np.concatenate((ul_col, ll_col, lr_col), axis=1)
        triangles2 = np.concatenate((ul_col, lr_col, ur_col), axis=1)
        triangles = np.concatenate((triangles1, triangles2), axis=0)

        # Generate a mayavi surface instance
        if 'color' not in kwargs.keys():
            kwargs['color'] = (0.75, 0.75, 0.75)
        mesh_source = mlab.pipeline.triangular_mesh_source(x, y, z, triangles)
        return mlab.pipeline.surface(mesh_source, **kwargs)

class RectangularPort(object):
    """
    Class representing a port with a rectangular cross-section.
    """

    def __init__(self, ox=1.0, oy=0.0, oz=0.0, ax=1.0, ay=0.0, az=0.0, \
                 wx=0.0, wy=1.0, wz=0.0, hx=0.0, hy=0.0, hz=1.0, \
                 iw=0.5, ih=0.5, thick=0.01, l0=0.0, l1=0.5):
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
            hx, hy, hz: floats
                Cartesian x, y, and z components of a vector in the direction
                spanning the height of the cross-section, assumed perpendicular
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
        mod_h = np.linalg.norm([hx, hy, hz])
        if mod_a == 0 or mod_w == 0 or mod_h == 0:
            raise ValueError('Vectors given by (ax, ay, az),  (wx, wy, wz), ' \
                             + 'and (hx, hy, hz) must have nonzero length')

        # Ensure that the axis vectors are mutually perpendiclar
        self.ax = ax/mod_a
        self.ay = ay/mod_a
        self.az = az/mod_a
        self.wx = wx/mod_w
        self.wy = wy/mod_w
        self.wz = wz/mod_w
        self.hx = hx/mod_h
        self.hy = hy/mod_h
        self.hz = hz/mod_h
        tol = 1e-12
        if np.abs(self.ax*self.wx + self.ay*self.wy + self.az*self.wz) > tol or\
           np.abs(self.ax*self.hx + self.ay*self.hy + self.az*self.hz) > tol or\
           np.abs(self.wx*self.hx + self.wy*self.hy + self.wz*self.hz) > tol:
            raise ValueError('Vectors given by (ax, ay, az),  (wx, wy, wz), ' \
                             + 'and (hx, hy, hz) must be mutually perpendicuar')

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
        

        lStart = np.min([self.l0, self.l1])
        lStop  = np.max([self.l0, self.l1])
        in_axial_bounds = np.logical_and(l_proj >= lStart - gap, \
                                         l_proj <= lStop  + gap)
        in_cross_section = \
            np.logical_and(np.abs(w_proj) < 0.5*self.iw + self.thick + gap, \
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
            ports: list of RectangularPort class instances
                The equivalent ports, including the calling instance if 
                include_self == True.
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

            hxi = self.hx*np.cos(i*dphi) - self.hy*np.sin(i*dphi)
            hyi = self.hx*np.sin(i*dphi) + self.hy*np.cos(i*dphi)
            hzi = self.hz

            if stell_sym:

                ports.append(RectangularPort(ox=oxi, oy=-oyi, oz=-ozi, \
                    ax=-axi, ay=ayi, az=azi, wx=-wxi, wy=wyi, wz=wzi, \
                    hx=-hxi, hy=hyi, hz=hzi, iw=self.iw, ih=self.ih, \
                    thick=self.thick, l0=-self.l1, l1=-self.l0))

            if i > 0:

                ports.append(RectangularPort(ox=oxi, oy=oyi, oz=ozi, \
                    ax=axi, ay=ayi, az=azi, wx=wxi, wy=wyi, wz=wzi, \
                    hx=hxi, hy=hyi, hz=hzi, iw=self.iw, ih=self.ih, \
                    thick=self.thick, l0=self.l0, l1=self.l1))

        return ports

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

        # Points at each end of the port
        x0 = self.ox + self.l0*self.ax
        y0 = self.oy + self.l0*self.ay
        z0 = self.oz + self.l0*self.az
        x1 = self.ox + self.l1*self.ax
        y1 = self.oy + self.l1*self.ay
        z1 = self.oz + self.l1*self.az

        # Unit vectors in the cross-sectional dimensions
        Wx = self.wx * np.array([[ 1,  1, -1, -1]])
        Wy = self.wy * np.array([[ 1,  1, -1, -1]])
        Wz = self.wz * np.array([[ 1,  1, -1, -1]])
        Hx = self.hx * np.array([[-1,  1,  1, -1]])
        Hy = self.hy * np.array([[-1,  1,  1, -1]])
        Hz = self.hz * np.array([[-1,  1,  1, -1]])

        # Coordinates of points on the inner and outer edges of the cylinder
        ihw = 0.5*self.iw
        ohw = 0.5*self.iw + self.thick
        ihh = 0.5*self.ih
        ohh = 0.5*self.ih + self.thick
        x = np.concatenate((x0 + ihw*Wx + ihh*Hx, x0 + ohw*Wx + ohh*Hx, \
                            x1 + ohw*Wx + ohh*Hx, x1 + ihw*Wx + ihh*Hx), \
                           axis=0).reshape((-1,1))
        y = np.concatenate((y0 + ihw*Wy + ihh*Hy, y0 + ohw*Wy + ohh*Hy, \
                            y1 + ohw*Wy + ohh*Hy, y1 + ihw*Wy + ihh*Hy), \
                           axis=0).reshape((-1,1))
        z = np.concatenate((z0 + ihw*Wz + ihh*Hz, z0 + ohw*Wz + ohh*Hz, \
                            z1 + ohw*Wz + ohh*Hz, z1 + ihw*Wz + ihh*Hz), \
                           axis=0).reshape((-1,1))

        # Index arrays
        upper_left = np.arange(16).reshape((4,4))
        lower_left = np.roll(upper_left, 1, axis=1)
        upper_right = np.roll(upper_left, 1, axis=0)
        lower_right = np.roll(upper_right, 1, axis=1)
        ul_col = upper_left.reshape((-1,1))
        ll_col = lower_left.reshape((-1,1))
        ur_col = upper_right.reshape((-1,1))
        lr_col = lower_right.reshape((-1,1))
        triangles1 = np.concatenate((ul_col, ll_col, lr_col), axis=1)
        triangles2 = np.concatenate((ul_col, lr_col, ur_col), axis=1)
        triangles = np.concatenate((triangles1, triangles2), axis=0)

        # Generate a mayavi surface instance
        if 'color' not in kwargs.keys():
            kwargs['color'] = (0.75, 0.75, 0.75)
        mesh_source = mlab.pipeline.triangular_mesh_source(x, y, z, triangles)
        return mlab.pipeline.surface(mesh_source, **kwargs)


