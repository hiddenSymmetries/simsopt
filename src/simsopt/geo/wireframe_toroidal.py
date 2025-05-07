"""
Implementation of the ToroidalWireframe class and associated functions
"""
import numpy as np
import collections
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from .._core.dev import SimsoptRequires

try:
    from pyevtk.hl import linesToVTK
except ImportError:
    linesToVTK = None

__all__ = ['ToroidalWireframe', 'windowpane_wireframe']


class ToroidalWireframe(object):
    """
    ``ToroidalWireframe`` is a wireframe grid whose nodes are placed on a
    toroidal surface as a 2D grid with regular spacing in the poloidal and
    toroidal dimensions.

    Currently only supports surfaces that exhibit stellarator symmetry.

    Parameters
    ----------
        surface: SurfaceRZFourier class instance
            Toroidal surface on which on which the nodes will be placed.
        n_phi: integer
            Number of wireframe nodes per half-period in the toroidal dimension.
            Must be even; if an odd number is provided, it will be incremented
            by one.
        n_theta: integer
            Number of wireframe nodes in the poloidal dimension. Must be even;
            if an odd number is provided, it will be incremented by one.
        constraint_atol, constraint_rtol: double (optional)
            Absolute and relative tolerances against which constraint equations 
            are to be evaluated (see docstring for method check_constraints for 
            more details). Default for each is 1e-10.
    """

    def __init__(self, surface, n_phi, n_theta, constraint_atol=1e-10,
                 constraint_rtol=1e-10):

        if not isinstance(surface, SurfaceRZFourier):
            raise ValueError('Surface must be a SurfaceRZFourier object')

        if not surface.stellsym:
            raise ValueError('Surfaces without stellarator symmetry are not '
                             + 'currently supported in the ToroidalWireframe '
                             + 'class')

        if not isinstance(n_theta, int) or not isinstance(n_phi, int):
            raise ValueError('n_theta and n_phi must be integers')

        if n_theta % 2 or n_phi % 2:
            raise ValueError('n_phi and n_theta must be even.')
        self.n_theta = n_theta
        self.n_phi = n_phi

        # Check the constraint tolerances
        if (not np.isscalar(constraint_atol) and not constraint_atol > 0) \
                or (not np.isscalar(constraint_rtol) and not constraint_rtol > 0):
            raise ValueError('constraint_atol must be a positive scalar')
        self.constraint_atol = constraint_atol
        self.constraint_rtol = constraint_rtol

        # Make copy of surf with quadrature points according to n_theta, n_phi
        qpoints_phi = list(np.linspace(0, 0.5/surface.nfp, n_phi+1))
        qpoints_theta = list(np.linspace(0, 1., n_theta, endpoint=False))
        self.nfp = surface.nfp
        self.surface = SurfaceRZFourier(nfp=surface.nfp, stellsym=True,
                                        mpol=surface.mpol, ntor=surface.ntor,
                                        quadpoints_phi=qpoints_phi,
                                        quadpoints_theta=qpoints_theta,
                                        dofs=surface.dofs)

        # Determine the locations of the node points within a half period
        nodes_surf = self.surface.gamma()
        self.n_nodes = np.prod(nodes_surf.shape[:2])
        nodes_hp = np.ascontiguousarray(np.zeros((self.n_nodes, 3)))
        nodes_hp[:, 0] = nodes_surf[:, :, 0].reshape((-1))
        nodes_hp[:, 1] = nodes_surf[:, :, 1].reshape((-1))
        nodes_hp[:, 2] = nodes_surf[:, :, 2].reshape((-1))
        self.node_inds = np.arange(self.n_nodes).reshape(nodes_surf.shape[:2])

        # Generate list of sets of nodes for each half period
        self.nodes = [[]]*self.nfp*2
        self.seg_signs = [[]]*self.nfp*2
        self.nodes[0] = nodes_hp
        self.seg_signs[0] = 1.0
        self.nodes[1] = np.ascontiguousarray(np.zeros((self.n_nodes, 3)))
        self.nodes[1][:, 0] = self.nodes[0][:, 0]
        self.nodes[1][:, 1] = -self.nodes[0][:, 1]
        self.nodes[1][:, 2] = -self.nodes[0][:, 2]
        self.seg_signs[1] = -1.0
        for i in range(1, self.nfp):

            phi_rot = 2.0*i*np.pi/self.nfp

            self.nodes[2*i] = np.ascontiguousarray(np.zeros((self.n_nodes, 3)))
            self.nodes[2*i+1] = np.ascontiguousarray(np.zeros((self.n_nodes, 3)))

            self.nodes[2*i][:, 0] = np.cos(phi_rot)*self.nodes[0][:, 0] - \
                np.sin(phi_rot)*self.nodes[0][:, 1]
            self.nodes[2*i][:, 1] = np.sin(phi_rot)*self.nodes[0][:, 0] + \
                np.cos(phi_rot)*self.nodes[0][:, 1]
            self.nodes[2*i][:, 2] = self.nodes[0][:, 2]

            self.nodes[2*i+1][:, 0] = np.cos(phi_rot)*self.nodes[1][:, 0] - \
                np.sin(phi_rot)*self.nodes[1][:, 1]
            self.nodes[2*i+1][:, 1] = np.sin(phi_rot)*self.nodes[1][:, 0] + \
                np.cos(phi_rot)*self.nodes[1][:, 1]
            self.nodes[2*i+1][:, 2] = self.nodes[1][:, 2]

            # Positive current direction reverses in reflected half-periods
            self.seg_signs[2*i] = 1.0
            self.seg_signs[2*i+1] = -1.0

        # Define the segments according to the pairs of nodes connecting them
        self.n_tor_segments = n_theta*n_phi
        self.n_pol_segments = n_theta*n_phi
        self.n_segments = self.n_tor_segments + self.n_pol_segments

        # Toroidal segments
        segments_tor = np.zeros((self.n_tor_segments, 2))
        segments_tor[:, 0] = \
            self.node_inds[:-1, :].reshape((self.n_tor_segments))
        segments_tor[:, 1] = \
            self.node_inds[1:, :].reshape((self.n_tor_segments))

        # Map nodes to index in the segment array of segment originating
        # from the respective node
        self.tor_segment_key = -np.ones(nodes_surf.shape[:2]).astype(np.int64)
        self.tor_segment_key[:-1, :] = \
            np.arange(self.n_tor_segments).reshape((n_phi, n_theta))

        # Poloidal segments (on symmetry planes, only include segments for z>0)
        segments_pol = np.zeros((self.n_pol_segments, 2))
        self.pol_segment_key = -np.ones(nodes_surf.shape[:2]).astype(np.int64)
        HalfNTheta = int(n_theta/2)

        segments_pol[:HalfNTheta, 0] = self.node_inds[0, :HalfNTheta]
        segments_pol[:HalfNTheta, 1] = self.node_inds[0, 1:HalfNTheta+1]
        self.pol_segment_key[0, :HalfNTheta] = np.arange(HalfNTheta) \
            + self.n_tor_segments
        for i in range(1, n_phi):
            polInd0 = HalfNTheta + (i-1)*n_theta
            polInd1 = polInd0 + n_theta
            segments_pol[polInd0:polInd1, 0] = self.node_inds[i, :]
            segments_pol[polInd0:polInd1-1, 1] = self.node_inds[i, 1:]
            segments_pol[polInd1-1, 1] = self.node_inds[i, 0]
            self.pol_segment_key[i, :] = np.arange(polInd0, polInd1) \
                + self.n_tor_segments

        segments_pol[-HalfNTheta:, 0] = self.node_inds[-1, :HalfNTheta]
        segments_pol[-HalfNTheta:, 1] = self.node_inds[-1, 1:HalfNTheta+1]
        self.pol_segment_key[-1, :HalfNTheta] = \
            np.arange(self.n_pol_segments-HalfNTheta, self.n_pol_segments) \
            + self.n_tor_segments

        # Join the toroidal and poloidal segments into a single array
        self.segments = \
            np.ascontiguousarray(np.zeros((self.n_segments, 2)).astype(np.int64))
        self.segments[:self.n_tor_segments, :] = segments_tor[:, :]
        self.segments[self.n_tor_segments:, :] = segments_pol[:, :]

        # Initialize currents to zero
        self.currents = np.ascontiguousarray(np.zeros((self.n_segments)))

        #self.nConstraints = self.n_tor_segments - 2

        # Create a matrix listing which segments are connected to each node
        self.determine_connected_segments()

        # Create a matrix listing which segments surround each cell
        self.set_up_cell_key()

        # Add constraints to enforce continuity at each node
        self.initialize_constraints()
        self.add_continuity_constraints()

    def determine_connected_segments(self):
        """
        Determine which segments are connected to each node.
        """

        self.connected_segments = \
            np.ascontiguousarray(np.zeros((self.n_nodes, 4)).astype(np.int64))

        half_n_theta = int(self.n_theta/2)

        for i in range(self.n_phi+1):
            for j in range(self.n_theta):

                # First symmetry plane
                if i == 0:
                    ind_tor_in = \
                        self.tor_segment_key[i, (self.n_theta-j) % self.n_theta]
                    ind_tor_out = self.tor_segment_key[i, j]
                    if j == 0:
                        ind_pol_in = self.pol_segment_key[i, j]
                        ind_pol_out = self.pol_segment_key[i, j]
                    elif j < half_n_theta:
                        ind_pol_in = self.pol_segment_key[i, j-1]
                        ind_pol_out = self.pol_segment_key[i, j]
                    elif j == half_n_theta:
                        ind_pol_in = self.pol_segment_key[i, j-1]
                        ind_pol_out = self.pol_segment_key[i, j-1]
                    else:
                        ind_pol_in = self.pol_segment_key[i, self.n_theta-j]
                        ind_pol_out = self.pol_segment_key[i, self.n_theta-j-1]

                # Between the symmetry planes
                elif i > 0 and i < self.n_phi:
                    ind_tor_in = self.tor_segment_key[i-1, j]
                    ind_tor_out = self.tor_segment_key[i, j]
                    if j == 0:
                        ind_pol_in = self.pol_segment_key[i, self.n_theta-1]
                    else:
                        ind_pol_in = self.pol_segment_key[i, j-1]
                    ind_pol_out = self.pol_segment_key[i, j]

                # Second symmetry plane
                else:
                    ind_tor_in = self.tor_segment_key[i-1, j]
                    ind_tor_out = \
                        self.tor_segment_key[i-1,
                                             (self.n_theta-j) % self.n_theta]
                    if j == 0:
                        ind_pol_in = self.pol_segment_key[i, 0]
                        ind_pol_out = self.pol_segment_key[i, 0]
                    elif j < half_n_theta:
                        ind_pol_in = self.pol_segment_key[i, j-1]
                        ind_pol_out = self.pol_segment_key[i, j]
                    elif j == half_n_theta:
                        ind_pol_in = self.pol_segment_key[i, j-1]
                        ind_pol_out = self.pol_segment_key[i, j-1]
                    else:
                        ind_pol_in = self.pol_segment_key[i, self.n_theta-j]
                        ind_pol_out = self.pol_segment_key[i, self.n_theta-j-1]

                self.connected_segments[self.node_inds[i, j]][:] = \
                    [ind_tor_in, ind_pol_in, ind_tor_out, ind_pol_out]

    def set_up_cell_key(self):
        """
        Set up a matrix giving the indices of the segments forming each 
        cell/loop in the wireframe. Also populate an array giving the 
        indices of the four adjacent cells to each cell.
        """

        n_cells = self.n_theta * self.n_phi
        self.cell_key = np.zeros((n_cells, 4)).astype(np.int64)
        self.cell_neighbors = np.zeros((n_cells, 4)).astype(np.int64)

        half_n_theta = int(self.n_theta/2)
        cell_grid = np.arange(n_cells).reshape((self.n_phi, self.n_theta))

        for i in range(self.n_phi):
            for j in range(self.n_theta):

                # First symmetry plane
                if i == 0:
                    ind_tor1 = self.tor_segment_key[i, j]
                    ind_pol2 = self.pol_segment_key[i+1, j]
                    ind_tor3 = self.tor_segment_key[i, (j+1) % self.n_theta]
                    if j < half_n_theta:
                        ind_pol4 = self.pol_segment_key[i, j]
                    else:
                        ind_pol4 = self.pol_segment_key[i, self.n_theta - j - 1]

                    nbr_npol = cell_grid[i, (j-1) % self.n_theta]
                    nbr_ptor = cell_grid[i+1, j]
                    nbr_ppol = cell_grid[i, (j+1) % self.n_theta]
                    nbr_ntor = cell_grid[i, self.n_theta - j - 1]

                # Between the symmetry planes
                elif i < self.n_phi-1:
                    ind_tor1 = self.tor_segment_key[i, j]
                    ind_pol2 = self.pol_segment_key[i+1, j]
                    ind_tor3 = self.tor_segment_key[i, (j+1) % self.n_theta]
                    ind_pol4 = self.pol_segment_key[i, j]

                    nbr_npol = cell_grid[i, (j-1) % self.n_theta]
                    nbr_ptor = cell_grid[i+1, j]
                    nbr_ppol = cell_grid[i, (j+1) % self.n_theta]
                    nbr_ntor = cell_grid[i-1, j]

                # Second symmetry plane
                else:
                    ind_tor1 = self.tor_segment_key[i, j]
                    if j < half_n_theta:
                        ind_pol2 = self.pol_segment_key[i+1, j]
                    else:
                        ind_pol2 = self.pol_segment_key[i+1,
                                                        self.n_theta - j - 1]
                    ind_tor3 = self.tor_segment_key[i, (j+1) % self.n_theta]
                    ind_pol4 = self.pol_segment_key[i, j]

                    nbr_npol = cell_grid[i, (j-1) % self.n_theta]
                    nbr_ptor = cell_grid[i, self.n_theta - j - 1]
                    nbr_ppol = cell_grid[i, (j+1) % self.n_theta]
                    nbr_ntor = cell_grid[i-1, j]

                self.cell_key[i*self.n_theta + j, :] = \
                    [ind_tor1, ind_pol2, ind_tor3, ind_pol4]

                self.cell_neighbors[i*self.n_theta + j, :] = \
                    [nbr_npol, nbr_ptor, nbr_ppol, nbr_ntor]

    def initialize_constraints(self):

        self.constraints = collections.OrderedDict()
        self.implicits_updated = True

    def add_constraint(self, name, constraint_type, matrix_row, constant):
        """
        Add a linear equality constraint on the currents in the segments
        in the wireframe of the form 

            ``matrix_row * x = constant``,

        where ``x`` is the array of currents in each segment,
        ``matrix_row`` is a 1d array of coefficients for each segment, and
        ``constant`` is the constant appearing on the right-hand side

        Parameters
        ----------
            name: string
                Unique name for the constraint
            constraint_type: string
                Type of constraint 
            matrix_row: 1d double array
                Array of coefficients as described above
            constant: double
                Constant on the right-hand side of the equation above
        """

        if name in self.constraints.keys():
            raise ValueError('Constraint %s already exists' % (name))

        if matrix_row.size != self.n_segments:
            raise ValueError('matrix_row must have one element for every '
                             + 'segment in the wireframe')

        self.constraints[name] = \
            {'type': constraint_type,
             'matrix_row': matrix_row,
             'constant': constant}

        if self.implicits_updated:
            ctype = self.constraints[name]['type']
            if ctype == 'segment' or ctype == 'implicit_segment':
                self.implicits_updated = False

    def remove_constraint(self, names):
        """
        Remove a constraint from the wireframe's set of constraints.

        Parameters
        ----------
            names: string or list of strings
                Name(s) of the constraints to be removed
        """

        if isinstance(names, str):
            names = [names]

        for name in names:

            if name not in self.constraints:
                raise ValueError('Constraint %s does not exist' % (name))

            if self.implicits_updated:
                ctype = self.constraints[name]['type']
                if ctype == 'segment' or ctype == 'implicit_segment':
                    self.implicits_updated = False

            del self.constraints[name]

    def add_poloidal_current_constraint(self, current):
        """
        Add constraint to require the total poloidal current through the 
        inboard midplane to be a certain value (effectively sets the toroidal
        magnetic field). 

        Parameters
        ----------
            current: double
                Total poloidal current; i.e. the sum of the currents in all 
                poloidal segments passing through the inboard midplane.
                A positive poloidal current thereby creates a toroidal field 
                in the negative toroidal direction (clockwise when viewed from 
                above).
        """

        pol_current_per_segment = current/(2.0*self.nfp*self.n_phi)
        pol_current_sum = pol_current_per_segment * self.n_phi * 2

        half_n_theta = int(self.n_theta/2)
        seg_ind0 = self.n_tor_segments + half_n_theta - 1
        seg_ind1a = seg_ind0 + half_n_theta
        seg_ind2a = self.n_segments
        seg_ind1b = seg_ind1a + 1
        seg_ind2b = self.n_segments - self.n_theta + 1

        matrix_row = np.zeros((1, self.n_segments))
        matrix_row[0, seg_ind0] = 1
        matrix_row[0, seg_ind1a:seg_ind2a:self.n_theta] = 1
        matrix_row[0, seg_ind1b:seg_ind2b:self.n_theta] = 1

        self.add_constraint('poloidal_current', 'poloidal_current',
                            matrix_row, pol_current_sum)

    def remove_poloidal_current_constraint(self):

        self.remove_constraint('poloidal_current')

    def set_poloidal_current(self, current):
        """
        Set the constraint requiring the total poloidal current through the 
        inboard midplane to be a certain value (effectively sets the toroidal
        magnetic field). 

        This method will replace an existing poloidal current constraint and
        create one if one does not exist.

        Parameters
        ----------
            current: double
                Total poloidal current; i.e. the sum of the currents in all 
                poloidal segments passing through the inboard midplane.
                A positive poloidal current thereby creates a toroidal field 
                in the negative toroidal direction (clockwise when viewed from 
                above).
        """

        if 'poloidal_current' in self.constraints:
            self.remove_constraint('poloidal_current')

        self.add_poloidal_current_constraint(current)

    def add_toroidal_current_constraint(self, current):
        """
        Add constraint to require the total toroidal current through a poloidal
        cross-section to be a certain value (effectively requires a helical
        current distribution when combined with a poloidal current constraint).

        Parameters
        ----------
            current: double
                Total toroidal current; i.e. the sum of the currents in all 
                toroidal segments passing through a symmetry plane.
                A positive toroidal current thereby creates a dipole moment
                in the positive "z" direction.
        """

        matrix_row = np.zeros((1, self.n_segments))
        matrix_row[0, :self.n_theta] = 1

        self.add_constraint('toroidal_current', 'toroidal_current',
                            matrix_row, current)

    def remove_toroidal_current_constraint(self):

        self.remove_constraint('toroidal_current')

    def set_toroidal_current(self, current):
        """
        Set the constraint requiring the total toroidal current through a 
        poloidal cross-section to be a certain value (effectively requires a 
        helical current distribution when combined with a poloidal current 
        constraint).

        This method will replace an existing toroidal current constraint and
        create one if one does not exist.

        Parameters
        ----------
            current: double
                Total toroidal current; i.e. the sum of the currents in all 
                toroidal segments passing through a symmetry plane.
                A positive toroidal current thereby creates a dipole moment
                in the positive "z" direction.
        """

        if 'toroidal_current' in self.constraints:
            self.remove_constraint('toroidal_current')

        self.add_toroidal_current_constraint(current)

    def add_segment_constraints(self, segments, implicit=False):
        """
        Adds a constraint or constraints requiring the current to be zero in
        one or more given segments.

        Parameters
        ----------
            segments: integer or array/list of integers
                Index of the segmenet or segments to be constrained
            implicit: boolean (optional)
                If true, the constraints will be marked as implicit (i.e. 
                implied by other constraints rather than set by the user).
                Default is false. This option is meant primarily for internal
                use and should not normally be invoked by the user.
        """

        if np.isscalar(segments):
            segments = np.array([segments])
        else:
            segments = np.array(segments)

        if np.any(segments < 0) or np.any(segments >= self.n_segments):
            raise ValueError('Segment indices must be positive and less than '
                             + ' the number of segments in the wireframe')

        if not implicit:
            # Remove implicit constraints on requested segments if they exist
            for i in segments:
                if 'implicit_segment_%d' % (i) in self.constraints:
                    self.remove_segment_constraint(i)
            name = 'segment'
        else:
            name = 'implicit_segment'

        for i in range(len(segments)):

            matrix_row = np.zeros((1, self.n_segments))
            matrix_row[0, segments[i]] = 1

            self.add_constraint(name + '_%d' % (segments[i]), name,
                                matrix_row, 0)

    def remove_segment_constraints(self, segments, implicit=False):
        """
        Removes constraints restricting the currents in given segment(s) to be
        zero.

        Parameters
        ----------
            segments: integer or array/list of integers
                Index of the segmenet or segments for which constraints are to
                be removed
            implicit: boolean (optional)
                If true, the constraints will be marked as implicit (i.e. 
                implied by other constraints rather than set by the user).
                Default is false.
        """

        if np.isscalar(segments):
            segments = np.array([segments])
        else:
            segments = np.array(segments)

        if np.any(segments < 0) or np.any(segments >= self.n_segments):
            raise ValueError('Segment indices must be positive and less than '
                             ' the number of segments in the wireframe')

        name = 'segment' if not implicit else 'implicit_segment'

        for i in range(len(segments)):

            self.remove_constraint(name + '_%d' % (segments[i]))

    def set_segments_constrained(self, segments, implicit=False):
        """
        Ensures that one or more given segments are constrained to have zero
        current.

        Parameters
        ----------
            segments: integer or array/list of integers
                Index of the segmenet or segments to be constrained
            implicit: boolean (optional)
                If true, the constraints will be marked as implicit (i.e. 
                implied by other constraints rather than set by the user).
                Default is false.
        """

        # Free existing constrained segments to avoid conflicts
        self.set_segments_free(segments)

        self.add_segment_constraints(segments, implicit=implicit)

    def constrain_colliding_segments(self, coll_func, pts_per_seg=10, **kwargs):
        """
        Constains segments found to be colliding with external objects or other
        spatial constraints according to a user-provided function. 

        Parameters
        ----------
            coll_func: function
                Function with the following interface:
                ``colliding = coll_func(x, y, z, **kwargs)``.
                Here, x, y, and z are arrays the Cartesian x, y, and z coordinates
                of a set of test points. The function returns a logical array
                (colliding) with the same dimensions of x, y, and z in which
                elements are True if the corresponding input point violates
                the spatial constraint and False otherwise.
            pts_per_seg: integer (optional)
                Number of spatial points to test along each segment for 
                collisions. Must be at least 2; endpoints (i.e. nodes) are
                always included. Default is 10.
            **kwargs:
                Any keyword arguments to be supplied to coll_func.
        """

        if pts_per_seg < 2:
            raise ValueError('pts_per_seg must be at least 2')

        # Coordinates of the test points along each segment
        pos = np.linspace(0.0, 1.0, pts_per_seg).reshape((pts_per_seg, 1, 1))
        point0 = self.nodes[0][self.segments[:, 0], :]
        point1 = self.nodes[0][self.segments[:, 1], :]
        seg_vec = point1 - point0
        test_pts = point0 + pos*seg_vec

        # Check test points for collisions
        coll = coll_func(test_pts[:, :, 0], test_pts[:, :, 1], test_pts[:, :, 2],
                         **kwargs)

        # Identify the segments containing colliding points
        colliding_segs = np.where(np.any(coll, axis=0))[0]

        # Set the colliding segments as constrained
        self.set_segments_constrained(colliding_segs)

    def set_toroidal_breaks(self, n_breaks, width, allow_pol_current=False):
        """
        Imposes segment constraints to prevent current from flowing toroidally
        through planes positioned at a given number of toroidal angles. 

        The spacing  between the breaks will be as even as possible for the
        given grid dimension. For perfectly even spacing, the number of 
        breaks must be an integer factor of the wireframe grid's toroidal
        dimension (n_phi).

        Parameters
        ----------
            n_breaks: integer
                Number of breaks to be inserted
            width: integer 
                Toroidal extent of each break, in terms of the number of 
                constrained toroidal segments (n_breaks*width must be less
                than n_phi for the wireframe)
            allow_pol_current: boolean (optional)
                If True, poloidal current will be permitted to flow in regions
                where toroidal current is forbidden. Otherwise, all poloidal
                segments within toroidal breaks will be constrained. Default is 
                False.
        """

        # Ensure that there aren't excessive breaks
        if n_breaks >= 0.5*self.n_phi:
            raise ValueError('n_breaks must be < half of wireframe n_phi')
        if n_breaks*width > 0.5*self.n_phi:
            raise ValueError('n_breaks*width must be <= half of '
                             + 'wireframe n_phi')

        # Check for existing toroidal current constraint
        if 'toroidal_current' in self.constraints:
            if self.constraints['toroidal_current']['constant'] == 0:
                print('Note: existing constraint for zero net toroidal '
                      + 'current is redundant\nand will be removed.')
                self.remove_constraint('toroidal_current')
            else:
                raise ValueError('Toroidal breaks would conflict with '
                                 + 'existing nonzero toroidal\ncurrent constraint')

        # Indices in the toroidal dimension where breaks are to be centered
        tor_inds = \
            np.ceil(np.linspace(0, self.n_phi, n_breaks, endpoint=False)
                    + 0.5*self.n_phi/n_breaks)

        to_constrain = []

        # Determine indices of segments to constrain
        for i in range(len(tor_inds)):

            itor0 = self.n_theta * (tor_inds[i] - np.ceil(0.5*width))
            itor1 = itor0 + self.n_theta * width

            to_constrain += list(np.arange(itor0, itor1).astype(np.int64))

            if width > 1 and not allow_pol_current:

                ipol0 = self.n_tor_segments + itor0 + int(0.5*self.n_theta)
                ipol1 = ipol0 + self.n_theta*(width - 1)
                assert (ipol0 >= self.n_tor_segments + int(0.5*self.n_theta))
                assert (ipol1 <= self.n_segments - int(0.5*self.n_theta))

                to_constrain += list(np.arange(ipol0, ipol1).astype(np.int64))

        self.set_segments_constrained(to_constrain)

    def set_segments_free(self, segments):
        """
        Ensures that one or more given segments are unconstrained.

        Parameters
        ----------
            segments: integer or array/list of integers
                Index of the segmenet or segments to be unconstrained
        """

        if np.isscalar(segments):
            segments = np.array([segments])
        else:
            segments = np.array(segments)

        if np.any(segments < 0) or np.any(segments >= self.n_segments):
            raise ValueError('Segment indices must be positive and less than '
                             ' the number of segments in the wireframe')

        for i in range(len(segments)):
            if 'segment_%d' % (segments[i]) in self.constraints:
                self.remove_constraint('segment_%d' % (segments[i]))
            elif 'implicit_segment_%d' % (segments[i]) in self.constraints:
                self.remove_constraint('implicit_segment_%d' % (segments[i]))

    def free_all_segments(self):
        """
        Remove any existing constraints that restrict individual segments to
        carry zero current.
        """

        for constr in list(self.constraints.keys()):
            if self.constraints[constr]['type'] == 'segment' \
                    or self.constraints[constr]['type'] == 'implicit_segment':
                self.remove_constraint(constr)

    def update_implicit_constraints(self):
        """
        Determines which segments are are implicitly constrained due to all
        connected segments on one or both ends being constrained.
        """

        # Clear out existing implicit constraints (they may no longer be needed)
        for constr in list(self.constraints.keys()):
            if self.constraints[constr]['type'] == 'implicit_segment':
                self.remove_constraint(constr)

        node_sum = np.zeros((self.n_nodes))

        implicits_remain = True
        while implicits_remain:

            node_sum[:] = 0

            # Tally how many unconstrained segments each node is connected to
            for seg_ind in self.unconstrained_segments(update=False):
                connected_nodes = \
                    np.sum(self.connected_segments == seg_ind, axis=1)
                node_sum += connected_nodes

            # Check for implicitly constrained segments (i.e. free segments
            # connected to a node with no other free segments)
            implicits = np.where(node_sum == 1)[0]
            if len(implicits) > 0:
                for node_ind in implicits:
                    for seg_ind in self.connected_segments[node_ind, :]:
                        if ('segment_%d' % (seg_ind) not in self.constraints) \
                            and ('implicit_segment_%d' % (seg_ind) not in
                                 self.constraints):

                            self.add_segment_constraints(seg_ind, implicit=True)
            else:
                implicits_remain = False

        self.implicits_updated = True

    def constrained_segments(self, include='all', update=True):
        """
        Returns the IDs of the segments that are currently constrained 
        (explicitly or implicitly) to have zero current.

        Parameters
        ----------
            include: string (optional)
                'all':      (default) returns IDs of both explicitly and implicitly constrained segments.
                'explicit': returns IDs only of explicitly constrained segments.
                'implicit': returns IDs only of implicitly constrained segments.
            update: boolean (optional)
                Ensure that the implicit segments are updated before returning.
                Default is True. NOTE: this option is primarily for internal
                use and should not normally be changed by the user.

        Returns
        -------
            segment_ids: list of integers
                IDs of the constrained segments.
        """

        if update and not self.implicits_updated:
            self.update_implicit_constraints()

        expl_keys = [key for key in self.constraints.keys()
                     if self.constraints[key]['type'] == 'segment']
        expl_ids = [int(key.split('_')[1]) for key in expl_keys]

        impl_keys = [key for key in self.constraints.keys()
                     if self.constraints[key]['type'] == 'implicit_segment']
        impl_ids = [int(key.split('_')[2]) for key in impl_keys]

        if include == 'explicit':
            return expl_ids
        elif include == 'implicit':
            return impl_ids
        elif include == 'all':
            return expl_ids + impl_ids
        else:
            raise ValueError('Include must be \'all\', \'explicit\', '
                             + 'or \'implicit\'')

    def unconstrained_segments(self, update=True):
        """
        Returns the IDs of the segments that are unconstrained, explicitly or
        implicitly.

        Parameters
        ----------
            update: boolean (optional)
                Ensure that the implicit segments are updated before returning.
                Default is True. NOTE: this option is primarily for internal
                use and should not normally be changed by the user.
        """

        free_segs = np.full(self.n_segments, True)
        free_segs[self.constrained_segments(update=update)] = False
        return np.where(free_segs)[0]

    def add_continuity_constraints(self):
        """
        Add constraints to ensure current continuity at each node. This is
        called automatically on initialization and doesn't normally need to
        be called by the user.
        """

        for i in range(self.n_phi+1):
            for j in range(self.n_theta):

                if i == 0:
                    if j == 0 or j >= self.n_theta/2:
                        # Constraint automatically satisfied due to symmetry
                        continue

                elif i == self.n_phi:
                    if j == 0 or j >= self.n_theta/2:
                        # Constraint automatically satisfied due to symmetry
                        continue

                ind_tor_in, ind_pol_in, ind_tor_out, ind_pol_out = \
                    list(self.connected_segments[self.node_inds[i, j]])

                self.add_continuity_constraint(self.node_inds[i, j],
                                               ind_tor_in, ind_pol_in, ind_tor_out, ind_pol_out)

    def add_continuity_constraint(self, node_ind, ind_tor_in, ind_pol_in,
                                  ind_tor_out, ind_pol_out):

        name = 'continuity_node_%d' % (node_ind)

        matrix_row = np.zeros((1, self.n_segments))
        matrix_row[0, [ind_tor_in, ind_pol_in]] = -1
        matrix_row[0, [ind_tor_out, ind_pol_out]] = 1

        self.add_constraint(name, 'continuity', matrix_row, 0.0)

    def constraint_matrices(self, remove_redundancies=True,
                            remove_constrained_segments=False,
                            assume_no_crossings=False):
        """
        Return the matrices for the system of equations that define the linear
        equality constraints for the wireframe segment currents. The equations
        have the form
        ``C * x = d``,
        where x is a column vector with the segment currents, C is a matrix of
        coefficients for the segment currents in each equation, and d is a
        column vector of constant terms in each equation.

        The matrices are initially constructed to have one row (equation) per
        constraint. However, if some of the constraints may be redundant.
        By default, this function will check the constraints for certain 
        redundancies and remove rows from the matrix that are found to be 
        redundant. This is necessary, e.g. for some constrained linear 
        least-squares solvers in which the constraint matrix must be full-rank. 
        However, the user may also opt out of row reduction such that the 
        output matrices contain every constraint equation explicitly.

        Note: the function does not put the output matrix into reduced row-
        echelon form or otherwise guarantee that it will be full-rank. Rather, 
        it only checks for cases in which all four segments connected to a node
        are constrained to have zero current, in which case the node's
        continuity constraint is redundant. If there are other redundancies,
        e.g. due to arbitrary constraints introduced by the user that aren't
        of the usual constraint types, these may not be removed.

        Parameters
        ----------
            remove_redundancies: boolean (optional)
                If true (the default option), rows in the matrix found to be
                redundant will be removed. If false, no checks for redundancy
                will be performed and all constraints will be represented in the
                output matrices.
            remove_constrained_segments: boolean (optional)
                If true (default is false), columns in the constraint matrix
                corresponding to constrained segments will be removed, and
                segment constraints will not appear explicitly in the 
                constraints matrix. 
            assume_no_crossings: boolean (optional)
                If true, will apply the assumption that the wireframe contains
                enough segment constraints such that the free segments form
                single-track loops with no forks or crossings. In this case,
                enough constraints will be removed to allow for one degree of
                freedom per loop.

        Returns
        -------
            constraints_C: 2d double array
                The matrix B in the constraint equation
            constraints_d: 1d double array (column vector)
                The column vector on the right-hand side of the constraint 
                equation
        """

        # Update the constraints if necessary
        if not self.implicits_updated:
            self.update_implicit_constraints()

        # Collect names of excluded constraints in a set

        excluded = set()

        if remove_redundancies:
            # Identify redundant continuity constraints
            inactive_nodes = self.find_inactive_nodes(assume_no_crossings)
            excluded.update(['continuity_node_%d' % i for i in inactive_nodes])

        if remove_constrained_segments:
            # Identify all segment constraints
            excluded.update([key for key in self.constraints
                             if self.constraints[key]['type'] == 'segment'
                             or self.constraints[key]['type'] == 'implicit_segment'])

        # Construct the matrix and RHS from non-excluded constraints

        constraints_C = np.ascontiguousarray(
            np.concatenate([self.constraints[key]['matrix_row']
                            for key in self.constraints
                            if key not in excluded], axis=0))

        constraints_d = np.ascontiguousarray(
            np.zeros((constraints_C.shape[0], 1)))

        constraints_d[:] = [[self.constraints[key]['constant']]
                            for key in self.constraints.keys() if key not in excluded]

        if remove_constrained_segments:

            return constraints_C[:, self.unconstrained_segments()], constraints_d

        else:

            return constraints_C, constraints_d

    def find_inactive_nodes(self, assume_no_crossings=False):
        """
        Determines which nodes have no current flowing through them according
        to existing segment constraints (i.e. constraints that require 
        individual segments to have zero current).

        Additionally, if no crossings are assumed, identifies one node per
        loop of current for which the associated continuity constraint should
        be marked as redundant.
        """

        node_sum = np.zeros((self.n_nodes))

        # Tally how many unconstrained segments each node is connected to
        for seg_ind in self.unconstrained_segments():
            connected_nodes = np.sum(self.connected_segments == seg_ind, axis=1)
            node_sum += connected_nodes

        # If all four connected segments are constrained, the continuity
        # constraint is redundant
        assert (np.sum(node_sum == 1) == 0)
        node_inactive = node_sum == 0

        # If no crossings are assumed, remove one constraint per loop
        redundant_for_loop = np.full((self.n_nodes), False)
        if assume_no_crossings:

            # Populate a set with the IDs of the free segments
            free_segs = set()
            for i in range(self.n_segments):
                if 'segment_%d' % (i) not in self.constraints \
                        and 'implicit_segment_%d' % (i) not in self.constraints:
                    free_segs.add(i)

            # Identify each loop
            loop_count = 0
            while len(free_segs) > 0:

                seg_0 = free_segs.pop()

                seg_i = seg_0
                nodes_i = self.segments[seg_i, :]
                constraint_lifted = False

                # Find the segments in the loop using a graph search
                while True:

                    # Remove at most one continuity constraint per loop (note
                    # that some nodes on symmetry planes don't have constraints)
                    if not constraint_lifted:
                        if 'continuity_node_%d' % nodes_i[0] in \
                                self.constraints:
                            redundant_for_loop[nodes_i[0]] = True
                            constraint_lifted = True

                    count = 0
                    for seg in self.connected_segments[nodes_i[0], :]:
                        if seg in free_segs:
                            seg_next = seg
                            free_segs.remove(seg)
                            count += 1
                    if count == 0:
                        for seg in self.connected_segments[nodes_i[1], :]:
                            if seg in free_segs:
                                seg_next = seg
                                free_segs.remove(seg)
                                count += 1
                    if count == 0:
                        loop_count += 1
                        break
                    if count > 1:
                        raise RuntimeError('Closed loop assumption is '
                                           'invalid: %d connected segments found to seg %d'
                                           % (count, seg_i))
                    seg_i = seg_next
                    nodes_i = self.segments[seg_i, :]

        return np.where(np.logical_or(node_inactive, redundant_for_loop))[0]

    def get_cell_key(self):
        """
        Returns a matrix of the segments that border every rectangular cell in 
        the wireframe. There is one row for every cell. The columns are defined
        as follows:

        .. code-block:: text

            Column  Short name  Description
            ------  ----------  ----------------------------------------------------
            1       ind_tor1    ID of toroidal segment with lower poloidal angle
            2       ind_pol2    ID of poloidal segment with higher toroidal angle
            3       ind_tor3    ID of toroidal segment with higher poloidal angle
            4       ind_pol4    ID of poloidal segment with lower toroidal angle

        """

        return np.ascontiguousarray(self.cell_key)

    def get_cell_neighbors(self):
        """
        Returns a matrix of the indices of the four adjacent cells to each cell
        in the wireframe. There is one row for every cell. The columns are 
        defined as follows:

        .. code-block:: text

            Column  Short name  Description
            ------  ----------  ----------------------------------------------------
            1       nbr_npol    ID of neighboring cell, negative poloidal direction
            2       nbr_ptor    ID of neighboring cell, positive toroidal direction
            3       nbr_ppol    ID of neighboring cell, positive poloidal direction
            4       nbr_ntor    ID of neighboring cell, negative toroidal direction

        """

        return np.ascontiguousarray(self.cell_neighbors)

    def get_free_cells(self, form='logical'):
        """
        Returns the indices of the cells that are free; i.e. they do not border
        any constrained segments.

        Parameters
        ----------
            form: string (optional)
                If 'indices', will return an array giving the row indices of 
                the free cells as they appear in the output of get_cell_key().
                If 'logical', will return a logical array with one element per
                cell in which free cells are coded as true. 
                Default is 'logical'.
        """

        constr_cells = np.zeros((self.n_theta*self.n_phi))
        for seg_ind in self.constrained_segments(include='all'):
            constr_cells += \
                np.sum(self.cell_key == seg_ind, axis=1).reshape((-1))

        if form == 'indices':
            return np.where(constr_cells == 0)[0]
        elif form == 'logical':
            return np.ascontiguousarray(constr_cells == 0)
        else:
            raise ValueError('form parameter must be ''indices'' '
                             + 'or ''logical''')

    def check_constraints(self, currents=None, atol=None, rtol=None):
        """
        Verify that every constraint is satisfied by the present values of
        the segment currents. Specifically, for each constraint equation,
        confirm that:

        .. math::

            |B * x - d| < atol + mean(|x|) * rtol,

        where B is a vector of coefficients, x is the array of currents in
        each segment, d is a constant, atol is an absolute tolerance, and
        rtol is a relative tolerance.

        Parameters
        ----------
            currents: double (optional)
                Array of segment currents to check against the constraints.
                If none is supplied, the internal `currents` array of the
                class instance will be checked against the constraints.
            atol, rtol: double (optional)
                Absolute and relative tolerances to apply when checking to 
                ensure that the added currents do not violate any existing 
                constraints. Default is to use the values already stored within 
                the class instance.

        Returns
        -------
            constraints_met: boolean
                True if all constraints are met to within the tolerance; 
                otherwise false
        """

        # Construct a column vector with the currents
        x = np.zeros((self.n_segments, 1))
        if currents is not None:
            if len(currents) != self.n_segments:
                raise ValueError('currents array does not match number of '
                                 + 'segments')
            x[:, 0] = currents.reshape((-1, 1))[:, 0]
        else:
            x[:, 0] = self.currents.reshape((-1, 1))[:, 0]

        # Check the constraint tolerances
        if atol is not None:
            if not np.isscalar(atol) and not atol > 0:
                raise ValueError('atol must be a positive scalar')
        else:
            atol = self.constraint_atol

        if rtol is not None:
            if not np.isscalar(rtol) and not rtol > 0:
                raise ValueError('rtol must be a positive scalar')
        else:
            rtol = self.constraint_rtol

        # Set up the constraint matrices
        constraints_C_full, constraints_d_full = \
            self.constraint_matrices(remove_redundancies=False)

        # Evaluate the residuals of the constraint equations
        residuals = constraints_C_full @ x - constraints_d_full

        # Total tolerance
        tol = atol + rtol*np.mean(np.abs(x))

        # Check the residuals
        if np.any(np.abs(residuals) >= tol):
            return False
        else:
            return True

    def add_tfcoil_currents(self, n_tf, current_per_coil, constraint_atol=None,
                            constraint_rtol=None):
        """
        Adds current to certain poloidal segments in order to form a set of 
        planar TF coils. The loops will be spaced as evenly as possible within
        the wireframe.

        Parameters
        ----------
            n_tf: integer
                Number of TF coils per half-period
            current_per_coil: double
                Current carried by each of the coils; positive current flows
                in the positive toroidal direction, thereby creating a negative 
                toroidal magnetic field
            constraint_atol, constraint_rtol: double (optional)
                Absolute and relative tolerances to apply when checking to 
                ensure that the added currents do not violate any existing 
                constraints. Default is to use the value already stored within 
                the class instance.
        """

        if n_tf > self.n_phi:
            raise ValueError('n_tf must not exceed the wireframe n_phi')

        # Indices in the toroidal dimension where current loops are to be added
        tf_inds = np.ceil(np.linspace(0, self.n_phi, n_tf, endpoint=False)
                          + 0.5*self.n_phi/n_tf)

        tf_currents = np.zeros(self.n_segments)

        # Add current to poloidal segments forming the desired loops
        for i in range(len(tf_inds)):
            if tf_inds[i] == 0:
                ind0 = self.n_phi * self.n_theta
                ind1 = ind0 + int(0.5 * self.n_theta)
            else:
                ind0 = int((self.n_phi + 0.5) * self.n_theta
                           + (tf_inds[i] - 1) * self.n_theta)
                ind1 = ind0 + self.n_theta
            tf_currents[ind0:ind1] = current_per_coil

        # Add new TF coil currents to any existing currents
        new_currents = tf_currents + self.currents

        # Check whether proposed new currents adhere to the constraints
        valid = self.check_constraints(currents=new_currents,
                                       atol=constraint_atol,
                                       rtol=constraint_rtol)

        # Update currents with new values if they are within constraints
        if valid:
            self.currents[:] = new_currents[:]
        else:
            raise ValueError('Constraints not met for desired currents')

    @SimsoptRequires(linesToVTK is not None,
                     "to_vtk method requires pyevtk module")
    def to_vtk(self, filename, extent='torus', extra_node_data=None,
               extra_segment_data=None):
        """
        Export the wireframe data to a VTK file, which can be read with 
        ParaView.

        Note: This function requires the ``pyevtk`` python package, which can 
        be installed using ``pip install pyevtk``.

        The following datasets will be stored in the file:

        ``current``: Current [A] in each segment. Positive currents flow in
        the positive toroidal/poloidal direction.

        ``constrained``: 1 if the segment is constrained to carry no current;
        0 otherwise

        ``constrained_exp``: 1 if the segment is explicitly constrained, 
        i.e. set by the user to carry no current.

        ``constrained_imp``: 1 if the segment is implicitly constrained, 
        i.e. it can carry no current due to neighboring segments
        being constrained.

        Parameters
        ----------
            filename: string
                Name of the VTK file, without extension, to create.
            extent: string (optional)
                Portion of the torus to be represented in the file. Options are 
                'half period', 'field period', and 'torus' (default).
            extra_node_data: dictionary (optional)
                Data values to be associated with each node.
            extra_segment_data: dictionary (optional)
                Data values to be associated with each segment.
        """

        if extent == 'half period':
            n_half_periods = 1
        elif extent == 'field period':
            n_half_periods = 2
        elif extent == 'torus' or extent == 'full torus':
            n_half_periods = self.nfp * 2
        else:
            raise ValueError('extent must be \'half period\', '
                             + '\'field period\', or \'torus\'')

        pl_segments = np.zeros((n_half_periods*self.n_segments, 2, 3))
        pl_currents = np.zeros((n_half_periods*self.n_segments))
        pl_constr_segs = np.zeros((n_half_periods*self.n_segments))
        pl_constr_segs_exp = np.zeros((n_half_periods*self.n_segments))
        pl_constr_segs_imp = np.zeros((n_half_periods*self.n_segments))

        for i in range(n_half_periods):
            ind0 = i*self.n_segments
            ind1 = (i+1)*self.n_segments
            pl_segments[ind0:ind1, :, :] = self.nodes[i][:, :][self.segments[:, :]]
            pl_currents[ind0:ind1] = self.currents[:]
            pl_constr_segs[ind0:ind1][self.constrained_segments()] = 1
            pl_constr_segs_exp[ind0:ind1][self.constrained_segments(
                include='explicit')] = 1
            pl_constr_segs_imp[ind0:ind1][self.constrained_segments(
                include='implicit')] = 1

        x = np.ascontiguousarray(pl_segments[:, :, 0].reshape((-1)))
        y = np.ascontiguousarray(pl_segments[:, :, 1].reshape((-1)))
        z = np.ascontiguousarray(pl_segments[:, :, 2].reshape((-1)))

        segment_data = {'current': pl_currents.reshape((-1)),
                        'constrained': pl_constr_segs.reshape((-1)),
                        'constrained_exp': pl_constr_segs_exp.reshape((-1)),
                        'constrained_imp': pl_constr_segs_imp.reshape((-1))}

        if extra_segment_data is not None:
            segment_data = {**segment_data, **extra_segment_data}

        linesToVTK(filename, x, y, z, cellData=segment_data,
                   pointData=extra_node_data)

    def make_plot_3d(self, ax=None, engine='mayavi', extent='full torus',
                     to_show='all', active_tol=1e-12, tube_radius=0.01,
                     colormap=None, **kwargs):
        """
        Make a 3d plot of the wireframe grid.

        Parameters
        ----------
            engine: string (optional)
                Plotting package to use, either 'mayavi' or 'matplotlib'.
                Default is 'mayavi'. 'matplotlib' is not recommended.
            extent: string (optional)
                Portion of the torus to be plotted. Options are 'half period',
                'field period' (default), and 'full torus'.
            to_show: string (optional)
                If 'all', will plot all segments, including those that carry
                no current. If 'active' will only show segments that carry
                a current of magnitude greater than `active_tol`. Default is
                'all'.
            active_tol: float (optional)
                Minimum magnitude of current carried by a given segment for it
                to be plotted if `to_show` is set to 'active'. Default is
                1e-12.
            tube_radius: float (optional)
                Radius of the tubes used to represent each segment in the 
                mayavi rendering. Only used if `engine` is 'mayavi'. 
                Default is 0.01.
            colormap: 2d array (optional)
                4-column array of red, green, blue, and alpha (opacity) values
                on a scale of 0 to 255 applied to the coil currents
            ax: matplotlib.pyplot.axes class instance (optional)
                Axes on which to make the plot. Only used if `engine` is 
                'matplotlib'. If not provided, a new set of axes will be 
                generated.
            **kwargs
                Additional keyword arguments to be passed to the plotting engine
                (mayavi only)

        Returns
        -------
            ax: matplotlib.pyplot.axes class instance
                Axes on which the plot is generated. Only returned if `engine` 
                is 'matplotlib'. 
        """

        if extent == 'half period':
            n_half_periods = 1
        elif extent == 'field period':
            n_half_periods = 2
        elif extent == 'full torus':
            n_half_periods = self.nfp * 2
        else:
            raise ValueError('extent must be \'half period\', '
                             + '\'field period\', or \'full torus\'')

        pl_segments = np.zeros((n_half_periods*self.n_segments, 2, 3))
        pl_currents = np.zeros((n_half_periods*self.n_segments))

        for i in range(n_half_periods):
            ind0 = i*self.n_segments
            ind1 = (i+1)*self.n_segments
            pl_segments[ind0:ind1, :, :] = self.nodes[i][:, :][self.segments[:, :]]
            pl_currents[ind0:ind1] = self.currents[:]*1e-6

        if to_show == 'active':
            inds = np.where(np.abs(pl_currents) > active_tol)[0]
        elif to_show == 'all':
            inds = np.arange(pl_segments.shape[0])
        else:
            raise ValueError('Parameter show must be ''active'' or ''all''')

        xmin = np.min(pl_segments[:, :, 0], axis=(0, 1))
        xmax = np.max(pl_segments[:, :, 0], axis=(0, 1))
        ymin = np.min(pl_segments[:, :, 1], axis=(0, 1))
        ymax = np.max(pl_segments[:, :, 1], axis=(0, 1))
        zmin = np.min(pl_segments[:, :, 2], axis=(0, 1))
        zmax = np.max(pl_segments[:, :, 2], axis=(0, 1))

        if engine == 'mayavi':

            from mayavi import mlab

            x = pl_segments[inds, :, 0].reshape((-1))
            y = pl_segments[inds, :, 1].reshape((-1))
            z = pl_segments[inds, :, 2].reshape((-1))
            s = np.ones((len(inds), 2))
            s[:, 0] = pl_currents[inds]
            s[:, 1] = pl_currents[inds]
            s = s.reshape((-1))

            pts = mlab.pipeline.scalar_scatter(x, y, z, s)
            connections = np.arange(2*len(inds)).reshape((-1, 2))
            pts.mlab_source.dataset.lines = connections

            tube = mlab.pipeline.tube(pts, tube_radius=tube_radius)
            tube.filter.radius_factor = 1.
            surf = mlab.pipeline.surface(tube, **kwargs)

            # Define a colormap similar to matplotlib's "coolwarm"
            if colormap is None:
                x_cmap = np.linspace(0, 1, 255).reshape((-1, 1))
                alpha = np.ones(x_cmap.shape)
                r_cmap = 1 - np.abs(3*x_cmap - 1.5)
                g_cmap = 1 - np.abs(3*x_cmap - 1.5)
                b_cmap = 1 - np.abs(3*x_cmap - 1.5)
                r_cmap[r_cmap < 0] = 0
                g_cmap[g_cmap < 0] = 0
                b_cmap[b_cmap < 0] = 0
                r_cmap[x_cmap > 0.5] = 1
                r_cmap[x_cmap > 0.75] = -2*x_cmap[x_cmap > 0.75] + 2.5
                b_cmap[x_cmap < 0.5] = 1
                b_cmap[x_cmap < 0.25] = 2*x_cmap[x_cmap < 0.25] + 0.5
                f = 0.9
                cmap = 255 * \
                    np.concatenate((f*r_cmap, f*g_cmap, f*b_cmap, alpha),
                                   axis=1)
            else:
                cmap = colormap

            surf.module_manager.scalar_lut_manager.lut.table = cmap
            if len(inds) > 0:
                curr_lim = np.max(np.abs(pl_currents[inds]))
                surf.module_manager.scalar_lut_manager.data_range = \
                    (-curr_lim, curr_lim)

            return surf

        elif engine == 'matplotlib':

            import matplotlib.pylab as pl
            from mpl_toolkits.mplot3d.art3d import Line3DCollection

            lc = Line3DCollection(pl_segments[inds, :, :])
            lc.set_array(pl_currents[inds])
            lc.set_clim(np.max(np.abs(self.currents*1e-6))*np.array([-1, 1]))
            lc.set_cmap('coolwarm')

            if ax is None:
                fig = pl.figure()
                ax = fig.add_subplot(projection='3d')

                ax.set_xlim([xmin, xmax])
                ax.set_ylim([ymin, ymax])
                ax.set_zlim([zmin, zmax])

                ax.set_aspect('equal')
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_zlabel('z')
                cb = pl.colorbar(lc, ax=ax)
                cb.set_label('Current [MA]')

            ax.add_collection(lc)

            return (ax)

    def make_plot_2d(self, extent='half period', quantity='currents',
                     active_tol=1e-12, ax=None, add_colorbar=True,
                     coordinates='indices', **kwargs):
        """
        Make a 2d plot of the segments in the wireframe grid.

        Parameters
        ----------
            extent: string (optional)
                Portion of the torus to be plotted. Options are 'half period',
                'field period' (default), and 'full torus'.
            quantity: string (optional)
                Quantity to be represented in the color of each segment.
                Options are 'currents' (default), 'nonzero currents' (i.e. show
                only segments with nonzero current), and 'constrained segments'.
            active_tol: float (optional)
                Minimum magnitude of current carried by a given segment for it
                to be plotted if `quantity` is set to 'nonzero currents'. 
                Default is 1e-12.
            ax: instance of the matplotlib.pyplot.Axis class (optional)
                Axis on which to generate the plot. If None, a new plot will
                be created.
            add_colorbar: logical (optional)
                If true, a colorbar will be added to accompany the 2d plot.
                Default is True.
            coordinates: string (optional)
                Coordinates to plot for the nodes of each segment. If 'indices',
                the coordinates will be the indices of the node in each 
                dimension (toroidal and poloidal). If 'radians', the 
                coordinates will be the angles in radians. If 'degrees', the 
                coordinates will be the angles in degrees.
            kwargs: optional keyword arguments
                Keyword arguments to be passed to the LineCollection instance
                used to plot the segments

        Returns
        -------
            ax: instance of the matplotlib.pyplot.Axis class
                Axis instance on which the plot was created.
            lc: instance of the matplotlib.collections.LineCollection class
                LineCollection instance of the plotted segments.
            cb: instance of the matplotlib.colorbar class
                Colorbar instance to go with the plot on `ax`. If `add_colorbar`
                is False, will return None.
        """

        import matplotlib.pyplot as pl
        from matplotlib.collections import LineCollection

        if extent == 'half period':
            n_half_periods = 1
        elif extent == 'field period':
            n_half_periods = 2
        elif extent == 'torus' or extent == 'full torus':
            n_half_periods = self.nfp * 2
        else:
            raise ValueError('extent must be \'half period\', '
                             + '\'field period\', or \'torus\'')

        pl_segments = np.zeros((n_half_periods*self.n_segments, 2, 2))
        pl_quantity = np.zeros((n_half_periods*self.n_segments))

        # Calculate the toroidal and poloidal indices of each segment's
        # endpoints for plotting
        for i in range(n_half_periods):
            ind0 = i*self.n_segments
            ind1 = (i+1)*self.n_segments
            if i % 2 == 0:
                pl_segments[ind0:ind1, 0, 0] = \
                    i*self.n_phi + np.floor(self.segments[:, 0]/self.n_theta)
                pl_segments[ind0:ind1, 0, 1] = self.segments[:, 0] % self.n_theta
                pl_segments[ind0:ind1, 1, 0] = \
                    i*self.n_phi + np.floor(self.segments[:, 1]/self.n_theta)
                pl_segments[ind0:ind1, 1, 1] = self.segments[:, 1] % self.n_theta

                loop_segs = np.where(
                    np.logical_and(pl_segments[ind0:ind1, 0, 1] == self.n_theta-1,
                                   pl_segments[ind0:ind1, 1, 1] == 0))
                pl_segments[ind0+loop_segs[0], 1, 1] = self.n_theta

            else:
                pl_segments[ind0:ind1, 0, 0] = \
                    (i+1)*self.n_phi - np.floor(self.segments[:, 0]/self.n_theta)
                pl_segments[ind0:ind1, 0, 1] = \
                    self.n_theta - (self.segments[:, 0] % self.n_theta)
                pl_segments[ind0:ind1, 1, 0] = \
                    (i+1)*self.n_phi - np.floor(self.segments[:, 1]/self.n_theta)
                pl_segments[ind0:ind1, 1, 1] = \
                    self.n_theta - (self.segments[:, 1] % self.n_theta)

                loop_segs = np.where(
                    np.logical_and(pl_segments[ind0:ind1, 0, 1] == 1,
                                   pl_segments[ind0:ind1, 1, 1] == self.n_theta))
                pl_segments[ind0+loop_segs[0], 1, 1] = 0

            if quantity == 'currents' or quantity == 'nonzero currents':
                pl_quantity[ind0:ind1] = self.currents[:]*1e-6
            elif quantity == 'constrained segments':
                pl_quantity[ind0:ind1][self.constrained_segments(
                    include='explicit')] = 1
                pl_quantity[ind0:ind1][self.constrained_segments(
                    include='implicit')] = -1
            else:
                raise ValueError('Unrecognized quantity for plotting')

        if quantity == 'nonzero currents':
            inds_to_plot = np.where(np.abs(pl_quantity) > active_tol)[0]
        else:
            inds_to_plot = np.arange(len(pl_quantity))

        if coordinates == 'indices':

            lc = LineCollection(pl_segments[inds_to_plot], **kwargs)

            delta_x = 1
            delta_y = 1
            label_x = 'Toroidal index'
            label_y = 'Poloidal index'

        elif coordinates == 'radians' or coordinates == 'degrees':

            if coordinates == 'radians':
                angle_factor = 2.*np.pi
                label_x = r'$\phi_{wf}$ [rad]'
                label_y = r'$\theta_{wf}$ [rad]'
            else:
                angle_factor = 360.
                label_x = r'$\phi_{wf}$ [deg]'
                label_y = r'$\theta_{wf}$ [deg]'

            qp_phi = self.surface.quadpoints_phi
            qp_theta = self.surface.quadpoints_theta

            phi_mod = qp_phi[(pl_segments[:, :, 0] % self.n_phi).astype(int)]
            theta_mod = \
                qp_theta[(pl_segments[:, :, 1] % self.n_theta).astype(int)]
            phi_offs = (0.5/self.nfp)*np.floor(pl_segments[:, :, 0]/self.n_phi)
            theta_offs = np.floor(pl_segments[:, :, 1]/self.n_theta)
            pl_angles = np.zeros(pl_segments.shape)
            pl_angles[:, :, 0] = angle_factor * (phi_mod + phi_offs)
            pl_angles[:, :, 1] = angle_factor * (theta_mod + theta_offs)

            lc = LineCollection(pl_angles[inds_to_plot], **kwargs)

            delta_x = angle_factor * (qp_phi[1] - qp_phi[0])
            delta_y = angle_factor * (qp_theta[1] - qp_theta[0])

        else:

            raise ValueError('Unrecognized value for `dimensions` parameter')

        lc.set_array(pl_quantity[inds_to_plot])
        if quantity == 'currents' or quantity == 'nonzero currents':
            lc.set_clim(np.max(np.abs(self.currents*1e-6))*np.array([-1, 1]))
        elif quantity == 'constrained segments':
            lc.set_clim([-1, 1])
        lc.set_cmap('coolwarm')

        if ax is None:
            fig = pl.figure()
            ax = fig.add_subplot()

        ax.set_xlim((-delta_x, delta_x*(n_half_periods*self.n_phi + 1)))
        ax.set_ylim((-delta_y, delta_y*(self.n_theta + 1)))

        ax.set_xlabel(label_x)
        ax.set_ylabel(label_y)
        if add_colorbar:
            cb = pl.colorbar(lc, ax=ax)
            if quantity == 'currents' or quantity == 'nonzero currents':
                cb.set_label('Current [MA]')
            elif quantity == 'constrained segments':
                cb.set_label('1 = constrained; -1 = implicitly constrained; '
                             + '0 = free')
        else:
            cb = None

        ax.add_collection(lc)

        return ax, lc, cb

    def plot_cells_2d(self, cell_values, value_label=None, ax=None):
        """
        Generates a 2d plot of the cells in one half-period of a wireframe,
        color-coded according to an input array of values.

        Parameters
        ----------
            cell_values: 1d array
                Values that determine the color of each cell.
            value_label: string (optional)
                Label for the colorbar
            ax: instance of the matplotlib.pyplot.Axis class (optional)
                Axis on which to generate the plot. If None, a new plot will
                be created.

        Returns
        -------
            ax: instance of the matplotlib.pyplot.Axis class
                Axis instance on which the plot was created.
        """

        import matplotlib.pyplot as pl
        from matplotlib.collections import PolyCollection

        vertices = np.zeros((self.n_theta*self.n_phi, 4, 2))
        vertices[:, 0, 0] = \
            np.floor(self.segments[self.cell_key[:, 0], 0]/self.n_theta)
        vertices[:, 1, 0] = \
            np.floor(self.segments[self.cell_key[:, 0], 1]/self.n_theta)
        vertices[:, 2, 0] = \
            np.floor(self.segments[self.cell_key[:, 2], 1]/self.n_theta)
        vertices[:, 3, 0] = \
            np.floor(self.segments[self.cell_key[:, 2], 0]/self.n_theta)
        vertices[:, 0, 1] = \
            self.segments[self.cell_key[:, 0], 0] % self.n_theta
        vertices[:, 1, 1] = \
            self.segments[self.cell_key[:, 0], 1] % self.n_theta
        vertices[:, 2, 1] = \
            self.segments[self.cell_key[:, 2], 1] % self.n_theta
        vertices[:, 3, 1] = \
            self.segments[self.cell_key[:, 2], 0] % self.n_theta

        loop_segs = np.where(
            np.logical_and(vertices[:, 0, 1] == self.n_theta-1,
                           vertices[:, 2, 1] == 0))
        vertices[loop_segs, 2:, 1] = self.n_theta

        free_cell_ids = self.get_free_cells()
        all_values = np.zeros((self.cell_key.shape[0]))

        if len(cell_values) == self.cell_key.shape[0]:
            all_values[:] = np.reshape(cell_values, (-1))[:]
        elif len(cell_values) == len(free_cell_ids):
            all_values[free_cell_ids] = np.reshape(cell_values, (-1))[:]
        else:
            raise ValueError('Input cell_values doesn''t have the correct '
                             'number of elements')

        pc = PolyCollection(vertices)
        pc.set_array(all_values)
        pc.set_edgecolor((0, 0, 0))
        pc.set_clim(np.max(np.abs(all_values))*np.array([-1, 1]))
        pc.set_cmap('coolwarm')

        if ax is None:
            fig = pl.figure()
            ax = fig.add_subplot()

        ax.set_xlim((-1, self.n_phi + 1))
        ax.set_ylim((-1, self.n_theta + 1))

        ax.set_xlabel('Toroidal index')
        ax.set_ylabel('Poloidal index')
        cb = pl.colorbar(pc, ax=ax)

        if value_label:
            cb.set_label(value_label)

        ax.add_collection(pc)


def windowpane_wireframe(surface, n_coils_tor, n_coils_pol, size_tor, size_pol,
                         gap_tor, gap_pol, constraint_atol=1e-10,
                         constraint_rtol=1e-10):
    """
    Create a ToroidalWireframe class instance with the current constrained to
    flow only within regularly spaced rectangular loops in the grid, i.e.
    windowpane coils.

    NOTE: the `assume_no_crossings` parameter must always be set to True
    where relevant (e.g. for obtaining constraint matrices or for RCLS solves)

    Parameters
    ----------
        surface: SurfaceRZFourier class instance
            Toroidal surface on which on which the nodes will be placed.
        n_coils_tor, n_coils_pol: integers
            Number of windowpane coils in the toroidal and poloidal dimensions 
        size_tor, size_pol: integers
            Number of wireframe grid cells per coil in the toroidal and 
            poloidal dimensions; must be even
        gap_tor, gap_pol: integers
            Number of wireframe grid cells between adjacent windowpane coils
            in the toroidal and poloidal dimensions; must be even
        constraint_atol, constraint_rtol: double (optional)
            Absolute and relative tolerances against which constraint equations 
            are to be evaluated (see docstring for method check_constraints for 
            more details). Default for each is 1e-10.

    Returns
    -------
        wframe: ToroidalWireframe class instance
            ToroidalWireframe instance with all segments constrained except
            for those that form the windowpane coils
    """

    if not isinstance(n_coils_tor, int) or not isinstance(n_coils_pol, int) or \
       not isinstance(size_tor, int) or not isinstance(size_pol, int) or \
       not isinstance(gap_tor, int) or not isinstance(gap_pol, int):
        raise ValueError('n_coils_tor, n_coils_pol, size_tor, size_pol, '
                         + 'gap_tor, and gap_pol must be integers')

    if size_tor % 2 or size_pol % 2 or gap_tor % 2 or gap_pol % 2:
        raise ValueError('size_[tor,pol] and gap_[tor,pol] must be even.')

    n_phi = n_coils_tor*(size_tor + gap_tor)
    n_theta = n_coils_pol*(size_pol + gap_pol)

    wframe = ToroidalWireframe(surface, n_phi, n_theta,
                               constraint_atol=constraint_atol,
                               constraint_rtol=constraint_rtol)

    unit_pol = size_pol + gap_pol
    unit_tor = size_tor + gap_tor
    unit_offs = unit_tor*n_theta
    inds_coils = np.array([], dtype=int)

    half_gap_tor = int(0.5*gap_tor)
    half_gap_pol = int(0.5*gap_pol)

    # Determine the indices of the toroidal segments constituing the coils
    for i in range(0, n_coils_tor):

        for j in range(half_gap_tor, half_gap_tor+size_tor):

            offs = i*unit_offs + j*n_theta

            inds_bot = np.arange(half_gap_pol, n_theta, unit_pol) + offs
            inds_top = inds_bot + size_pol
            inds_coils = np.concatenate((inds_coils, inds_bot, inds_top))

    # Determine the indices of the poloidal segments constituting the coils
    for i in range(0, n_coils_tor):

        for j in range(0, n_coils_pol):

            offs_left = i*unit_offs + half_gap_tor*n_theta - int(0.5*n_theta) \
                + j*unit_pol + wframe.n_tor_segments
            offs_right = offs_left + size_tor*n_theta

            inds = np.arange(half_gap_pol, half_gap_pol+size_pol)
            inds_left = inds + offs_left
            inds_right = inds + offs_right

            inds_coils = np.concatenate((inds_coils, inds_left, inds_right))

    # Identify segments that do NOT constitute the coils
    unused_segs = np.full(wframe.n_segments, True)
    unused_segs[inds_coils] = False

    # Constrain all segments that are not part of the coils
    wframe.set_segments_constrained(np.where(unused_segs)[0])

    return wframe
