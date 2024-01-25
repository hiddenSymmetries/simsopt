"""
wireframe.py

Definitions for the ToroidalWireframe class
"""

import numpy as np
import collections
from simsopt.geo.surfacerzfourier import SurfaceRZFourier

__all__ = ['ToroidalWireframe']

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
        nPhi: integer
            Number of wireframe nodes per half-period in the toroidal dimension.
            Must be even; if an odd number is provided, it will be incremented
            by one.
        nTheta: integer
            Number of wireframe nodes in the poloidal dimension. Must be even;
            if an odd number is provided, it will be incremented by one.
    """

    def __init__(self, surface, nPhi, nTheta):

        if not isinstance(surface, SurfaceRZFourier):
            raise ValueError('Surface must be a SurfaceRZFourier object')

        if not surface.stellsym:
            raise ValueError('Surfaces without stellarator symmetry are not ' \
                             + 'currently supported in the ToroidalWireframe ' \
                             + 'class')

        if not isinstance(nTheta, int) or not isinstance(nPhi, int):
            raise ValueError('nTheta and nPhi must be integers')

        if nTheta % 2 or nPhi % 2:
            raise ValueError('nPhi and nTheta must be even.')
        self.nTheta = nTheta 
        self.nPhi = nPhi

        # Make copy of surface with quadrature points according to nTheta, nPhi
        qpoints_phi = list(np.linspace(0, 0.5/surface.nfp, nPhi+1))
        qpoints_theta = list(np.linspace(0, 1., nTheta, endpoint=False))
        self.nfp = surface.nfp
        self.surface = SurfaceRZFourier(nfp=surface.nfp, stellsym=True, \
                                        mpol=surface.mpol, ntor=surface.ntor, \
                                        quadpoints_phi=qpoints_phi, \
                                        quadpoints_theta=qpoints_theta, \
                                        dofs=surface.dofs)

        # Determine the locations of the node points within a half period
        nodes_surf = self.surface.gamma()
        self.nNodes = np.prod(nodes_surf.shape[:2])
        nodes_hp = np.ascontiguousarray(np.zeros((self.nNodes, 3)))
        nodes_hp[:, 0] = nodes_surf[:, :, 0].reshape((-1))
        nodes_hp[:, 1] = nodes_surf[:, :, 1].reshape((-1))
        nodes_hp[:, 2] = nodes_surf[:, :, 2].reshape((-1))
        self.node_inds = np.arange(self.nNodes).reshape(nodes_surf.shape[:2])

        # Generate list of sets of nodes for each half period
        self.nodes = [[]]*self.nfp*2
        self.seg_signs = [[]]*self.nfp*2
        self.nodes[0] = nodes_hp
        self.seg_signs[0] = 1.0
        self.nodes[1] = np.ascontiguousarray(np.zeros((self.nNodes, 3)))
        self.nodes[1][:, 0] = self.nodes[0][:, 0]
        self.nodes[1][:, 1] = -self.nodes[0][:, 1]
        self.nodes[1][:, 2] = -self.nodes[0][:, 2]
        self.seg_signs[1] = -1.0
        for i in range(1, self.nfp):

            phi_rot = 2.0*i*np.pi/self.nfp

            self.nodes[2*i]   = np.ascontiguousarray(np.zeros((self.nNodes, 3)))
            self.nodes[2*i+1] = np.ascontiguousarray(np.zeros((self.nNodes, 3)))

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
        self.nTorSegments = nTheta*nPhi
        self.nPolSegments = nTheta*nPhi
        self.nSegments = self.nTorSegments + self.nPolSegments

        # Toroidal segments 
        segments_tor = np.zeros((self.nTorSegments, 2))
        segments_tor[:,0] = self.node_inds[:-1, :].reshape((self.nTorSegments))
        segments_tor[:,1] = self.node_inds[1:,  :].reshape((self.nTorSegments))

        # Map nodes to index in the segment array of segment originating 
        # from the respective node
        self.torSegmentKey = -np.ones(nodes_surf.shape[:2]).astype(np.int64)
        self.torSegmentKey[:-1,:] = \
            np.arange(self.nTorSegments).reshape((nPhi, nTheta))

        # Poloidal segments (on symmetry planes, only include segments for z>0)
        segments_pol = np.zeros((self.nPolSegments, 2))
        self.polSegmentKey = -np.ones(nodes_surf.shape[:2]).astype(np.int64)
        HalfNTheta = int(nTheta/2)

        segments_pol[:HalfNTheta, 0] = self.node_inds[0, :HalfNTheta]
        segments_pol[:HalfNTheta, 1] = self.node_inds[0, 1:HalfNTheta+1]
        self.polSegmentKey[0, :HalfNTheta] = np.arange(HalfNTheta) \
                                             + self.nTorSegments
        for i in range(1, nPhi):
            polInd0 = HalfNTheta + (i-1)*nTheta
            polInd1 = polInd0 + nTheta
            segments_pol[polInd0:polInd1, 0] = self.node_inds[i, :]
            segments_pol[polInd0:polInd1-1, 1] = self.node_inds[i, 1:]
            segments_pol[polInd1-1, 1] = self.node_inds[i, 0]
            self.polSegmentKey[i, :] = np.arange(polInd0, polInd1) \
                                       + self.nTorSegments

        segments_pol[-HalfNTheta:, 0] = self.node_inds[-1, :HalfNTheta]
        segments_pol[-HalfNTheta:, 1] = self.node_inds[-1, 1:HalfNTheta+1]
        self.polSegmentKey[-1, :HalfNTheta] = \
            np.arange(self.nPolSegments-HalfNTheta, self.nPolSegments) \
            + self.nTorSegments

        # Join the toroidal and poloidal segments into a single array
        self.segments = \
            np.ascontiguousarray(np.zeros((self.nSegments, 2)).astype(np.int64))
        self.segments[:self.nTorSegments, :] = segments_tor[:, :]
        self.segments[self.nTorSegments:, :] = segments_pol[:, :]

        # Initialize currents to zero
        self.currents = np.ascontiguousarray(np.zeros((self.nSegments)))

        #self.nConstraints = self.nTorSegments - 2

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
            np.ascontiguousarray(np.zeros((self.nNodes, 4)).astype(np.int64))

        halfNTheta = int(self.nTheta/2)

        for i in range(self.nPhi+1):
            for j in range(self.nTheta):

                # First symmetry plane
                if i == 0:
                    ind_tor_in  = \
                        self.torSegmentKey[i, (self.nTheta-j) % self.nTheta]
                    ind_tor_out = self.torSegmentKey[i, j]
                    if j == 0:
                        ind_pol_in  = self.polSegmentKey[i, j]
                        ind_pol_out = self.polSegmentKey[i, j]
                    elif j < halfNTheta:
                        ind_pol_in  = self.polSegmentKey[i, j-1]
                        ind_pol_out = self.polSegmentKey[i, j]
                    elif j == halfNTheta:
                        ind_pol_in  = self.polSegmentKey[i, j-1]
                        ind_pol_out = self.polSegmentKey[i, j-1]
                    else:
                        ind_pol_in  = self.polSegmentKey[i, self.nTheta-j]
                        ind_pol_out = self.polSegmentKey[i, self.nTheta-j-1]

                # Between the symmetry planes
                elif i > 0 and i < self.nPhi:
                    ind_tor_in  = self.torSegmentKey[i-1, j]
                    ind_tor_out = self.torSegmentKey[i, j]
                    if j == 0:
                        ind_pol_in  = self.polSegmentKey[i, self.nTheta-1]
                    else:
                        ind_pol_in  = self.polSegmentKey[i, j-1]
                    ind_pol_out = self.polSegmentKey[i, j]

                # Second symmetry plane
                else:
                    ind_tor_in  = self.torSegmentKey[i-1, j]
                    ind_tor_out = \
                        self.torSegmentKey[i-1, (self.nTheta-j) % self.nTheta]
                    if j == 0:
                        ind_pol_in  = self.polSegmentKey[i, 0]
                        ind_pol_out = self.polSegmentKey[i, 0]
                    elif j < halfNTheta:
                        ind_pol_in  = self.polSegmentKey[i, j-1]
                        ind_pol_out = self.polSegmentKey[i, j]
                    elif j == halfNTheta:
                        ind_pol_in  = self.polSegmentKey[i, j-1]
                        ind_pol_out = self.polSegmentKey[i, j-1]
                    else:
                        ind_pol_in  = self.polSegmentKey[i, self.nTheta-j]
                        ind_pol_out = self.polSegmentKey[i, self.nTheta-j-1]

                self.connected_segments[self.node_inds[i,j]][:] = \
                    [ind_tor_in, ind_pol_in, ind_tor_out, ind_pol_out]

    def set_up_cell_key(self):
        """
        Set up a matrix giving the indices of the segments forming each 
        cell/loop in the wireframe. Also populate an array giving the 
        indices of the four adjacent cells to each cell.
        """

        nCells = self.nTheta * self.nPhi
        self.cell_key = np.zeros((nCells, 4)).astype(np.int64)
        self.cell_neighbors = np.zeros((nCells, 4)).astype(np.int64)

        halfNTheta = int(self.nTheta/2)
        cell_grid = np.arange(nCells).reshape((self.nPhi, self.nTheta))

        for i in range(self.nPhi):
            for j in range(self.nTheta):

                # First symmetry plane
                if i == 0:
                    ind_tor1 = self.torSegmentKey[i, j]
                    ind_pol2 = self.polSegmentKey[i+1, j]
                    ind_tor3 = self.torSegmentKey[i, (j+1) % self.nTheta]
                    if j < halfNTheta:
                        ind_pol4 = self.polSegmentKey[i, j]
                    else:
                        ind_pol4 = self.polSegmentKey[i, self.nTheta - j - 1]

                    nbr_npol = cell_grid[i, (j-1) % self.nTheta]
                    nbr_ptor = cell_grid[i+1, j]
                    nbr_ppol = cell_grid[i, (j+1) % self.nTheta]
                    nbr_ntor = cell_grid[i, self.nTheta - j - 1]

                # Between the symmetry planes
                elif i < self.nPhi-1:
                    ind_tor1 = self.torSegmentKey[i, j]
                    ind_pol2 = self.polSegmentKey[i+1, j]
                    ind_tor3 = self.torSegmentKey[i, (j+1) % self.nTheta]
                    ind_pol4 = self.polSegmentKey[i, j]

                    nbr_npol = cell_grid[i, (j-1) % self.nTheta]
                    nbr_ptor = cell_grid[i+1, j]
                    nbr_ppol = cell_grid[i, (j+1) % self.nTheta]
                    nbr_ntor = cell_grid[i-1, j]

                # Second symmetry plane
                else:
                    ind_tor1 = self.torSegmentKey[i, j]
                    if j < halfNTheta:
                        ind_pol2 = self.polSegmentKey[i+1, j]
                    else:
                        ind_pol2 = self.polSegmentKey[i+1, self.nTheta - j - 1]
                    ind_tor3 = self.torSegmentKey[i, (j+1) % self.nTheta]
                    ind_pol4 = self.polSegmentKey[i, j]

                    nbr_npol = cell_grid[i, (j-1) % self.nTheta]
                    nbr_ptor = cell_grid[i, self.nTheta - j - 1]
                    nbr_ppol = cell_grid[i, (j+1) % self.nTheta]
                    nbr_ntor = cell_grid[i-1, j]

                self.cell_key[i*self.nTheta + j, :] = \
                    [ind_tor1, ind_pol2, ind_tor3, ind_pol4]

                self.cell_neighbors[i*self.nTheta + j, :] = \
                    [nbr_npol, nbr_ptor, nbr_ppol, nbr_ntor]

    def initialize_constraints(self):

        self.constraints = collections.OrderedDict()

    def add_constraint(self, name, constraint_type, matrix_row, constant):
        """
        Add a linear equality constraint on the currents in the segments
        in the wireframe of the form 

            matrix_row * x = constant, 

            where:
                x is the array of currents in each segment
                matrix_row is a 1d array of coefficients for each segment
                constant is the constant appearing on the right-hand side

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

        if matrix_row.size != self.nSegments:
            raise ValueError('matrix_row must have one element for every ' \
                             + 'segment in the wireframe')

        self.constraints[name] = \
            {'type': constraint_type, \
             'matrix_row': matrix_row, \
             'constant': constant}

    def remove_constraint(self, name):

        if isinstance(name, str):
            del self.constraints[name]
        else:
            for item in name:
                del self.constraints[item]

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

        pol_current_per_segment = current/(2.0*self.nfp*self.nPhi)
        pol_current_sum = pol_current_per_segment * self.nPhi * 2

        halfNTheta = int(self.nTheta/2)
        seg_ind0 = self.nTorSegments + halfNTheta - 1
        seg_ind1a = seg_ind0 + halfNTheta
        seg_ind2a = self.nSegments
        seg_ind1b = seg_ind1a + 1
        seg_ind2b = self.nSegments - self.nTheta + 1

        matrix_row = np.zeros((1, self.nSegments))
        matrix_row[0,seg_ind0] = 1
        matrix_row[0,seg_ind1a:seg_ind2a:self.nTheta] = 1
        matrix_row[0,seg_ind1b:seg_ind2b:self.nTheta] = 1

        self.add_constraint('poloidal_current', 'poloidal_current', \
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

        matrix_row = np.zeros((1, self.nSegments))
        matrix_row[0,:self.nTheta] = 1

        self.add_constraint('toroidal_current', 'toroidal_current', \
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
                Default is false.
        """

        if np.isscalar(segments):
            segments = np.array([segments])
        else:
            segments = np.array(segments)

        if np.any(segments < 0) or np.any(segments >= self.nSegments):
            raise ValueError('Segment indices must be positive and less than ' \
                             + ' the number of segments in the wireframe')

        if not implicit:
            # Remove implicit constraints on requested segments if they exist
            self.set_segments_free(segments)
            name = 'segment'
        else:
            name = 'implicit_segment'

        for i in range(len(segments)):

            matrix_row = np.zeros((1, self.nSegments))
            matrix_row[0,segments[i]] = 1

            self.add_constraint(name + '_%d' % (segments[i]), name, \
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

        if np.any(segments < 0) or np.any(segments >= self.nSegments):
            raise ValueError('Segment indices must be positive and less than ' \
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
                    colliding = coll_func(x, y, z, **kwargs)
                x, y, and z are arrays the Cartesian x, y, and z coordinates
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
        pos = np.linspace(0.0, 1.0, pts_per_seg).reshape((pts_per_seg,1,1))
        point0 = self.nodes[0][self.segments[:,0],:]
        point1 = self.nodes[0][self.segments[:,1],:]
        seg_vec = point1 - point0
        test_pts = point0 + pos*seg_vec

        # Check test points for collisions
        coll = coll_func(test_pts[:,:,0], test_pts[:,:,1], test_pts[:,:,2], \
                         **kwargs)

        # Identify the segments containing colliding points
        colliding_segs = np.where(np.any(coll, axis=0))[0]

        # Set the colliding segments as constrained
        self.set_segments_constrained(colliding_segs)

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

        if np.any(segments < 0) or np.any(segments >= self.nSegments):
            raise ValueError('Segment indices must be positive and less than ' \
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

        for constr in self.constraints:
            if self.constraints[constr]['type'] == 'segment' \
            or self.constraints[constr]['type'] == 'implicit_segment':
                self.remove_constraint(constr)

    def constrained_segments(self, include='all'):
        """
        Returns the IDs of the segments that are currently constrained 
        (explicitly or implicitly) to have zero current.

        Parameters
        ----------
            include: string (optional)
                'all':      (default) returns IDs of both explicitly and 
                            implicitly constrained segments.
                'explicit': returns IDs only of explicitly constrained segments.
                'implicit': returns IDs only of implicitly constrained segments.

        Returns
        -------
            segment_ids: list of integers
                IDs of the constrained segments.
        """  

        expl_keys = [key for key in self.constraints.keys() \
                     if self.constraints[key]['type'] == 'segment']
        expl_ids = [int(key.split('_')[1]) for key in expl_keys]

        impl_keys = [key for key in self.constraints.keys() \
                     if self.constraints[key]['type'] == 'implicit_segment']
        impl_ids = [int(key.split('_')[2]) for key in impl_keys]

        if include == 'explicit':
            return expl_ids
        elif include == 'implicit':
            return impl_ids
        elif include == 'all':
            return expl_ids + impl_ids
        else:
            raise ValueError('Include must be \'all\', \'explicit\', ' \
                             + 'or \'implicit\'')

    def add_continuity_constraints(self):
        """
        Add constraints to ensure current continuity at each node. This is
        called automatically on initialization and doesn't normally need to
        be called by the user.
        """

        for i in range(self.nPhi+1):
            for j in range(self.nTheta):

                if i == 0:
                    if j == 0 or j >= self.nTheta/2:
                        # Constraint automatically satisfied due to symmetry
                        continue

                elif i == self.nPhi:
                    if j == 0 or j >= self.nTheta/2:
                        # Constraint automatically satisfied due to symmetry
                        continue

                ind_tor_in, ind_pol_in, ind_tor_out, ind_pol_out = \
                    list(self.connected_segments[self.node_inds[i,j]])

                self.add_continuity_constraint(self.node_inds[i,j], \
                    ind_tor_in, ind_pol_in, ind_tor_out, ind_pol_out)

    def add_continuity_constraint(self, node_ind, ind_tor_in, ind_pol_in, \
                                  ind_tor_out, ind_pol_out):

        name = 'continuity_node_%d' % (node_ind)

        matrix_row = np.zeros((1, self.nSegments))
        matrix_row[0, [ind_tor_in,  ind_pol_in ]] = -1
        matrix_row[0, [ind_tor_out, ind_pol_out]] = 1
        
        self.add_constraint(name, 'continuity', matrix_row, 0.0)
        

    def constraint_matrices(self, remove_redundancies=True):
        """
        Return the matrices for the system of equations that define the linear
        equality constraints for the wireframe segment currents. The equations
        have the form
            B*x = d,
        where x is a column vector with the segment currents, B is a matrix of
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

        Returns
        -------
            constraints_B: 2d double array
                The matrix B in the constraint equation
            constraints_d: 1d double array (column vector)
                The column vector on the right-hand side of the constraint 
                equation
        """

       
        # If matrix is not full rank, look for redundant continuity constraints
        if remove_redundancies: 

            inactive_nodes = self.find_inactive_nodes()
            inactive_node_names = ['continuity_node_%d' % (i) \
                                   for i in inactive_nodes]

            constraints_B = np.ascontiguousarray( \
                np.concatenate([self.constraints[key]['matrix_row'] \
                                for key in self.constraints.keys() \
                                if key not in inactive_node_names], axis=0))

            constraints_d = np.ascontiguousarray( \
                np.zeros((constraints_B.shape[0], 1)))

            constraints_d[:] = [[self.constraints[key]['constant']] \
                 for key in self.constraints.keys() \
                 if key not in inactive_node_names]

        else:

            constraints_B = np.ascontiguousarray( \
                np.concatenate([constr['matrix_row'] for constr \
                                in self.constraints.values()], axis=0))

            constraints_d = np.ascontiguousarray(\
                            np.zeros((len(self.constraints), 1)))
            constraints_d[:] = \
                [[constr['constant']] for constr in self.constraints.values()]

        return constraints_B, constraints_d

    def find_inactive_nodes(self):
        """
        Determines which nodes have no current flowing through them according
        to existing segment constraints (i.e. constraints that require 
        individual segments to have zero current).
        """

        node_sum = np.zeros((self.nNodes))

        implicits_remain = True
        while implicits_remain:

            node_sum[:] = 0

            # Tally how many inactive segments each node is connected to
            for seg_ind in self.constrained_segments(include='all'):
                connected_nodes = \
                    np.sum(self.connected_segments == seg_ind, axis=1)
                node_sum[connected_nodes > 0] += 1

            # Check for implicitly constrained segments (i.e. free segments
            # connected to a node whose other three segments are constrained)
            implicits = np.where(node_sum == 3)[0]
            if len(implicits) > 0:
                for node_ind in implicits:
                    for seg_ind in self.connected_segments[node_ind,:]:
                        if ('segment_%d' % (seg_ind) not in \
                            self.constraints.keys()) \
                        and ('implicit_segment_%d' % (seg_ind) not in 
                            self.constraints.keys()):

                            self.add_segment_constraints(seg_ind, implicit=True)
            else:
                implicits_remain = False

        # If all four connected segments are constrained, the continuity 
        # constraint is redundant
        return np.where(node_sum >= 4)[0]
            
    def get_cell_key(self):
        """
        Returns a matrix of the segments that border every rectangular cell in 
        the wireframe. There is one row for every cell. The columns are defined
        as follows:

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

        constr_cells = np.zeros((self.nTheta*self.nPhi))
        for seg_ind in self.constrained_segments(include='all'):
            constr_cells += \
                np.sum(self.cell_key == seg_ind, axis=1).reshape((-1))

        if form=='indices':
            return np.where(constr_cells == 0)[0]
        elif form=='logical':
            return np.ascontiguousarray(constr_cells == 0)
        else:
            raise ValueError('form parameter must be ''indices'' ' \
                             + 'or ''logical''')

    def make_plot_3d(self, ax=None, engine='mayavi', to_show='all', \
                     active_tol=1e-12, tube_radius=0.01, **kwargs):
        """
        Make a 3d plot of the wireframe grid.

        Parameters
        ----------
            engine: string (optional)
                Plotting package to use, either 'mayavi' or 'matplotlib'.
                Default is 'mayavi'. 'matplotlib' is not recommended.
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

        pl_segments = np.zeros((2*self.nfp*self.nSegments, 2, 3))
        pl_currents = np.zeros((2*self.nfp*self.nSegments))
   
        for i in range(2*self.nfp):
            ind0 = i*self.nSegments
            ind1 = (i+1)*self.nSegments
            pl_segments[ind0:ind1,:,:] = self.nodes[i][:,:][self.segments[:,:]]
            pl_currents[ind0:ind1] = self.currents[:]*1e-6

        if to_show == 'active':
            inds = np.where(np.abs(pl_currents) > active_tol)[0]
        elif to_show == 'all':
            inds = np.arange(pl_segments.shape[0])
        else:
            raise ValueError('Parameter show must be ''active'' or ''all''')

        xmin = np.min(pl_segments[:,:,0], axis=(0,1))
        xmax = np.max(pl_segments[:,:,0], axis=(0,1))
        ymin = np.min(pl_segments[:,:,1], axis=(0,1))
        ymax = np.max(pl_segments[:,:,1], axis=(0,1))
        zmin = np.min(pl_segments[:,:,2], axis=(0,1))
        zmax = np.max(pl_segments[:,:,2], axis=(0,1))

        if engine == 'mayavi':

            from mayavi import mlab

            x = pl_segments[inds,:,0].reshape((-1))
            y = pl_segments[inds,:,1].reshape((-1))
            z = pl_segments[inds,:,2].reshape((-1))
            s = np.ones((len(inds),2))
            s[:,0] = pl_currents[inds]
            s[:,1] = pl_currents[inds]
            s = s.reshape((-1))

            pts = mlab.pipeline.scalar_scatter(x, y, z, s)
            connections = np.arange(2*len(inds)).reshape((-1,2))
            pts.mlab_source.dataset.lines = connections

            tube = mlab.pipeline.tube(pts, tube_radius=tube_radius)
            tube.filter.radius_factor = 1.
            mlab.pipeline.surface(tube, **kwargs)

            #mlab.axes(extent=[xmin, xmax, ymin, ymax, zmin, zmax])

            #cbar = mlab.colorbar(title='Current [MA]')
 
        elif engine == 'matplotlib':

            import matplotlib.pylab as pl
            from mpl_toolkits.mplot3d.art3d import Line3DCollection
    
            lc = Line3DCollection(pl_segments[inds,:,:])
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
                cb = pl.colorbar(lc)
                cb.set_label('Current [MA]')
    
            ax.add_collection(lc)
    
            return(ax)

    def make_plot_2d(self, extent='field period', quantity='currents', \
                     ax=None, **kwargs):
        """
        Make a 2d plot of the segments in the wireframe grid.

        Parameters
        ----------
            extent: string (optional)
                Portion of the torus to be plotted. Options are 'half period',
                'field period' (default), and 'torus'.
            quantity: string (optional)
                Quantity to be represented in the color of each segment.
                Options are 'currents' (default) and 'constrained segments'.
            ax: instance of the matplotlib.pyplot.Axis class (optional)
                Axis on which to generate the plot. If None, a new plot will
                be created.
            kwargs: optional keyword arguments
                Keyword arguments to be passed to the LineCollection instance
                used to plot the segments

        Returns
        -------
            ax: instance of the matplotlib.pyplot.Axis class
                Axis instance on which the plot was created.
        """

        import matplotlib.pyplot as pl
        from matplotlib.collections import LineCollection

        if extent=='half period':
            nHalfPeriods = 1
        elif extent=='field period':
            nHalfPeriods = 2
        elif extent=='torus':
            nHalfPeriods = self.nfp * 2
        else:
            raise ValueError('extent must be \'half period\', ' \
                             + '\'field period\', or \'torus\'')

        pl_segments = np.zeros((nHalfPeriods*self.nSegments, 2, 2))
        pl_quantity = np.zeros((nHalfPeriods*self.nSegments))
   
        for i in range(nHalfPeriods):
            ind0 = i*self.nSegments
            ind1 = (i+1)*self.nSegments
            if i % 2 == 0:
                pl_segments[ind0:ind1,0,0] = \
                    np.floor(self.segments[:,0]/self.nTheta)
                pl_segments[ind0:ind1,0,1] = self.segments[:,0] % self.nTheta
                pl_segments[ind0:ind1,1,0] = \
                    np.floor(self.segments[:,1]/self.nTheta)
                pl_segments[ind0:ind1,1,1] = self.segments[:,1] % self.nTheta

                loop_segs = np.where( \
                    np.logical_and(pl_segments[ind0:ind1,0,1] == self.nTheta-1,\
                                   pl_segments[ind0:ind1,1,1] == 0))
                pl_segments[ind0+loop_segs[0],1,1] = self.nTheta

            else:
                pl_segments[ind0:ind1,0,0] = \
                    2*i*self.nPhi - np.floor(self.segments[:,0]/self.nTheta)
                pl_segments[ind0:ind1,0,1] = \
                    self.nTheta - (self.segments[:,0] % self.nTheta)
                pl_segments[ind0:ind1,1,0] = \
                    2*i*self.nPhi - np.floor(self.segments[:,1]/self.nTheta)
                pl_segments[ind0:ind1,1,1] = \
                    self.nTheta - (self.segments[:,1] % self.nTheta)

                loop_segs = np.where( \
                    np.logical_and(pl_segments[ind0:ind1,0,1] == 1, \
                                   pl_segments[ind0:ind1,1,1] == self.nTheta))
                pl_segments[ind0+loop_segs[0],1,1] = 0

            if quantity=='currents':
                pl_quantity[ind0:ind1] = self.currents[:]*1e-6
            elif quantity=='constrained segments':
                pl_quantity[ind0:ind1][self.constrained_segments( \
                                                       include='explicit')] = 1
                pl_quantity[ind0:ind1][self.constrained_segments( \
                                                       include='implicit')] = -1
            else:
                raise ValueError('Unrecognized quantity for plotting')

        lc = LineCollection(pl_segments, **kwargs)
        lc.set_array(pl_quantity)
        if quantity=='currents':
            lc.set_clim(np.max(np.abs(self.currents*1e-6))*np.array([-1, 1]))
        elif quantity=='constrained segments':
            lc.set_clim([-1, 1])
        lc.set_cmap('coolwarm')

        if ax is None:
            fig = pl.figure()
            ax = fig.add_subplot()

        ax.set_xlim((-1, nHalfPeriods*self.nPhi + 1))
        ax.set_ylim((-1, self.nTheta + 1))

        ax.set_xlabel('Toroidal index')
        ax.set_ylabel('Poloidal index')
        cb = pl.colorbar(lc)
        if quantity=='currents':
            cb.set_label('Current (MA)')
        elif quantity=='constrained segments':
            cb.set_label('1 = constrained; 0 = free')


        ax.add_collection(lc)

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

        vertices = np.zeros((self.nTheta*self.nPhi, 4, 2))
        #vertices[:,:,0] = \
        #    np.floor(self.segments[self.cell_key[:,:],0]/self.nTheta)
        #vertices[:,:,1] = \
        #    self.segments[self.cell_key[:,:],0] % self.nTheta
        vertices[:,0,0] = \
            np.floor(self.segments[self.cell_key[:,0],0]/self.nTheta)
        vertices[:,1,0] = \
            np.floor(self.segments[self.cell_key[:,0],1]/self.nTheta)
        vertices[:,2,0] = \
            np.floor(self.segments[self.cell_key[:,2],1]/self.nTheta)
        vertices[:,3,0] = \
            np.floor(self.segments[self.cell_key[:,2],0]/self.nTheta)
        #vertices[:,4,0] = \
        #    np.floor(self.segments[self.cell_key[:,3],0]/self.nTheta)
        vertices[:,0,1] = \
            self.segments[self.cell_key[:,0],0] % self.nTheta
        vertices[:,1,1] = \
            self.segments[self.cell_key[:,0],1] % self.nTheta
        vertices[:,2,1] = \
            self.segments[self.cell_key[:,2],1] % self.nTheta
        vertices[:,3,1] = \
            self.segments[self.cell_key[:,2],0] % self.nTheta
        #vertices[:,4,1] = \
        #    self.segments[self.cell_key[:,3],0] % self.nTheta

        #halfNTheta = int(self.nTheta/2)
        #vertices[halfNTheta:self.nTheta,3,1] = \
        #    self.nTheta + 1 - vertices[halfNTheta:self.nTheta,3,1]
        #vertices[-halfNTheta:,1,1] = \
        #    self.nTheta - 1 - vertices[-halfNTheta:,1,1]

        loop_segs = np.where(\
            np.logical_and(vertices[:,0,1] == self.nTheta-1, \
                           vertices[:,2,1] == 0))
        vertices[loop_segs,2:,1] = self.nTheta
        
        free_cell_ids = self.get_free_cells()
        all_values = np.zeros((self.cell_key.shape[0]))

        if len(cell_values) == self.cell_key.shape[0]:
            all_values[:] = np.reshape(cell_values, (-1))[:]
        elif len(cell_values) == len(free_cell_ids):
            all_values[free_cell_ids] = np.reshape(cell_values, (-1))[:]
        else:
            raise ValueError('Input cell_values doesn''t have the correct ' \
                             'number of elements')

        pc = PolyCollection(vertices)
        pc.set_array(all_values)
        pc.set_edgecolor((0,0,0))
        pc.set_clim(np.max(np.abs(all_values))*np.array([-1, 1]))
        pc.set_cmap('coolwarm')

        if ax is None:
            fig = pl.figure()
            ax = fig.add_subplot()

        ax.set_xlim((-1, self.nPhi + 1))
        ax.set_ylim((-1, self.nTheta + 1))

        ax.set_xlabel('Toroidal index')
        ax.set_ylabel('Poloidal index')
        cb = pl.colorbar(pc)
 
        if value_label:
            cb.set_label(value_label)

        ax.add_collection(pc)

