"""
ports.py

Functions for checking for collisions with ports and other obstacles.
"""

import numpy as np

__all__ = ['PortSet', 'CircularPort']

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
            file: string (optional)
                Name of the file from which to load port data of type specified
                by `port_type`
            port_type: string (optional)
                Type of port to load from a file if `file` is specified. 
                Supported inputs currently include 'circular'.
        """

        self.nPorts = 0
        self.ports = []

        if ports is not None:
            self.add_ports(ports)
        
        if file is not None:
            if port_type is None:
                raise ValueError('port_type must be specified if loading ' \
                                 + 'from file')
            elif port_type == 'circular' or port_type == CircularPort:
                self.load_circular_ports_from_file(file)
            else:
                raise ValueError('port_type %s is not supported' % (port_type))
                

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
            if not isinstance(port, CircularPort):
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

        # Ensure that the axis is a unit vector
        mod_ax = np.linalg.norm([ax, ay, az])
        if mod_ax == 0:
            raise ValueError('Vector given by ax, ay, and az has zero length')

        self.ax = ax/mod_ax
        self.ay = ay/mod_ax
        self.az = az/mod_ax

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

        in_axial_bounds = np.logical_and(l_proj >= self.l0 - gap, \
                                         l_proj <= self.l1 + gap)
        in_radial_bounds = r2 <= out_r2

        return np.logical_and(in_axial_bounds, in_radial_bounds)


