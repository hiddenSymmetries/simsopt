import unittest
from pathlib import Path
import numpy as np
from simsopt.geo import SurfaceRZFourier, ToroidalWireframe, CircularPort, \
                        windowpane_wireframe

try:
    import pyevtk
except ImportError:
    pyevtk = None

TEST_DIR = (Path(__file__).parent / ".." / "test_files").resolve()

def surf_torus(nfp, rmaj, rmin):
    surf = SurfaceRZFourier(nfp=nfp, mpol=1, ntor=0)
    surf.set_rc(0, 0, rmaj)
    surf.set_rc(1, 0, rmin)
    surf.set_zs(1, 0, rmin)
    return surf

def subtended_angle(x, y):
    """
    Finds the angle subtended by a set of points within 2D space (assumed to
    be less than pi).

    Parameters
    ----------
        x, y: 1d double arrays
            x- and y-coordinates of each point. Must have at least 2 elements.

    Returns
    -------
        angle: double
            Subtended angle (in radians)
    """

    if len(x) < 2 or len(y) < 2:
        raise ValueError('x and y must have at least two elements each')

    if len(x) != len(y):
        raise ValueError('x and y must have the same number of elements')

    n = len(x)

    # Distances of each point from the origin, i.e. lengths of each vector (x,y)
    dists = np.sqrt(x**2 + y**2)

    # Formula for the subtended angle between two points
    def subtended(i,j):
        ratio = (x[i]*x[j] + y[i]*y[j])/(dists[i]*dists[j])
        if np.abs(ratio) > 1:
            return np.arccos(np.sign(ratio))
        else:
            return np.arccos(ratio)

    # Initialize the angle to that subtended by the first two points
    ind0 = 0
    ind1 = 1
    angle = subtended(ind0, ind1)

    for i in range(2,n):

        # Angles subtended between test point and reference points
        angle_0 = subtended(i,ind0)
        angle_1 = subtended(i,ind1)

        # If test point subtends a larger angle, replace a reference point
        if angle_0 > angle or angle_1 > angle:

            if angle_0 > angle_1:
                angle = angle_0
                ind1 = i
            else:
                angle = angle_1
                ind0 = i
    
    return angle

class Testing(unittest.TestCase):

    def test_toroidal_wireframe_constructor(self):
        """
        Runs a few consistency checks for errors in construction of a 
        ToroidalWireframe class instance
        """

        nfp = 3
        rmaj = 2
        rmin = 1
        surf_wf = surf_torus(nfp, rmaj, rmin)

        nPhi = 4
        nTheta = 4

        wf = ToroidalWireframe(surf_wf, nPhi, nTheta)

        #-----------------------------------------------------------------------
        # Check basic class instance quantities
        #-----------------------------------------------------------------------

        self.assertEqual(wf.nfp, nfp)
        self.assertEqual(wf.nSegments, len(wf.segments))
        self.assertEqual(wf.nSegments, 2*nPhi*nTheta)
        self.assertEqual(wf.nTorSegments, wf.nPolSegments)
        self.assertEqual(wf.nSegments, wf.nTorSegments + wf.nPolSegments)
        self.assertEqual(len(wf.nodes), 2*nfp)

        # Verify that nodes for each half-period are toroidally localized
        for i in range(nfp*2):
            self.assertEqual(wf.nNodes, np.shape(wf.nodes[i])[0])
            angle = subtended_angle(wf.nodes[i][:,0], wf.nodes[i][:,1])
            self.assertAlmostEqual(angle, np.pi/nfp)

        # Verify no more than 4 segments connected to each node
        node_count = np.zeros(wf.nNodes, dtype=np.int64)
        for i in range(wf.nSegments):
            node_count[wf.segments[i,0]] += 1
            node_count[wf.segments[i,1]] += 1
        self.assertTrue(np.max(node_count) == 4)

        #-----------------------------------------------------------------------
        # Segment array
        #-----------------------------------------------------------------------

        # Correctness of node connections for interior segments
        self.assertFalse(np.any(wf.segments[ 7,:] - np.array([ 7,11])))
        self.assertFalse(np.any(wf.segments[25,:] - np.array([11, 8])))

        # Correctness of node connections for segments on symmetry plane
        self.assertFalse(np.any(wf.segments[14,:] - np.array([14,18])))
        self.assertFalse(np.any(wf.segments[31,:] - np.array([17,18])))

        #-----------------------------------------------------------------------
        # Segment connection matrix
        #-----------------------------------------------------------------------

        # Correctness of connections for interior nodes
        self.assertFalse(np.any(wf.connected_segments[4,:] \
                                - np.array([0,21,4,18])))
        self.assertFalse(np.any(wf.connected_segments[9,:] \
                                - np.array([5,22,9,23])))

        # Correctness of connections for a node on a symmetry plane 
        self.assertFalse(np.any(wf.connected_segments[1,:] \
                                - np.array([3,16,1,17])))

        # Correctness of connections for a node on a symmetry plane at z=0
        self.assertFalse(np.any(wf.connected_segments[2,:] \
                                - np.array([2,17,2,17])))

        # Verify that no node is connected to more than 4 segments
        for i in range(wf.nNodes):
            count_i = np.sum(wf.segments[wf.connected_segments[i,:],0] == i) + \
                      np.sum(wf.segments[wf.connected_segments[i,:],1] == i)
            if i > 0 and i < nTheta/2 \
                or i > wf.nNodes-nTheta and i < wf.nNodes-nTheta/2:
                self.assertEqual(count_i, 3)
            elif i > nTheta/2 and i < nTheta \
                or i > wf.nNodes-nTheta/2 and i < wf.nNodes:
                self.assertEqual(count_i, 1)
            else:
                self.assertEqual(count_i, 4)

        #-----------------------------------------------------------------------
        # Cell key
        #-----------------------------------------------------------------------

        nCells = wf.cell_key.shape[0]

        # Verify that adjacent segments around cells share nodes
        for i in range(nCells):

            if i >= np.round(wf.nTheta/2) and i < wf.nTheta:
                # Index offset for mirrored segments on symmetry plane
                self.assertEqual(wf.segments[wf.cell_key[i,0],0], \
                    (wf.nTheta - wf.segments[wf.cell_key[i,3],1]) % wf.nTheta)
                self.assertEqual(wf.segments[wf.cell_key[i,2],0], \
                    (wf.nTheta - wf.segments[wf.cell_key[i,3],0]) % wf.nTheta)

            else:
                # Non-mirrored or interior cells (no index offset needed)
                self.assertEqual(wf.segments[wf.cell_key[i,0],0], \
                                 wf.segments[wf.cell_key[i,3],0])
                self.assertEqual(wf.segments[wf.cell_key[i,2],0], \
                                 wf.segments[wf.cell_key[i,3],1])

            if i >= nCells - np.round(wf.nTheta/2):
                # Correction for mirrored segments on symmetry plane
                node_symmPlane = wf.nNodes - wf.nTheta
                symmPlane_node_0 = wf.segments[wf.cell_key[i,1],0] % wf.nTheta
                symmPlane_node_1 = wf.segments[wf.cell_key[i,1],1] % wf.nTheta
                offs_0 = node_symmPlane \
                             + (wf.nTheta - symmPlane_node_0) % wf.nTheta
                offs_1 = node_symmPlane \
                             + (wf.nTheta - symmPlane_node_1) % wf.nTheta
                self.assertEqual(wf.segments[wf.cell_key[i,0],1], offs_1)
                self.assertEqual(offs_0, wf.segments[wf.cell_key[i,2],1])

            else:
                # Non-mirrored or interior cells (no index offset needed)
                self.assertEqual(wf.segments[wf.cell_key[i,0],1], \
                                 wf.segments[wf.cell_key[i,1],0])
                self.assertEqual(wf.segments[wf.cell_key[i,1],1], \
                                 wf.segments[wf.cell_key[i,2],1])

        #-----------------------------------------------------------------------
        # Cell neighbors
        #-----------------------------------------------------------------------

        # Verify that neighboring cells share segments in common
        for i in range(nCells):

            # Indices of neighboring cells (rows of cell_key)
            [nbr_npol, nbr_ptor, nbr_ppol, nbr_ntor] = wf.cell_neighbors[i,:]

            # Columns of shared segments in cell_key, accounting for mirroring
            # on symmetry planes
            iShareNpol = 2
            iSharePpol = 0
            iSharePtor = 1 if i >= (nCells - wf.nTheta) else 3
            iShareNtor = 3 if i < wf.nTheta else 1

            self.assertEqual(wf.cell_key[i,0], wf.cell_key[nbr_npol,iShareNpol])
            self.assertEqual(wf.cell_key[i,1], wf.cell_key[nbr_ptor,iSharePtor])
            self.assertEqual(wf.cell_key[i,2], wf.cell_key[nbr_ppol,iSharePpol])
            self.assertEqual(wf.cell_key[i,3], wf.cell_key[nbr_ntor,iShareNtor])

    def test_toroidal_wireframe_constraints(self):
        """
        Consistency checks for the constraint handling functionality in the 
        ToroidalWireframe class
        """

        nfp = 3
        rmaj = 2
        rmin = 1
        surf_wf = surf_torus(nfp, rmaj, rmin)

        nPhi = 4
        nTheta = 4

        wf = ToroidalWireframe(surf_wf, nPhi, nTheta)

        # Check number of continuity constraints (added on construction)
        self.assertEqual(len(wf.constraints.keys()), nPhi*nTheta - 2)

        # An isolated current-carrying segment violates continuity
        test_cur = 1e6
        wf.currents[0] = test_cur
        self.assertFalse(wf.check_constraints(constraint_tol=test_cur*1e-12))
        wf.currents[0] = 0
        self.assertTrue(wf.check_constraints)

        # A segment constraint can't be double-added or double-removed
        wf.add_segment_constraints(1)
        self.assertRaises(ValueError, wf.add_segment_constraints, 1)
        wf.remove_segment_constraints(1)
        self.assertRaises(ValueError, wf.remove_segment_constraints, 1)

        # Verify that an implicit constraint is added automatically
        wf.set_segments_constrained([0, 4, 18])
        csegs = wf.constrained_segments()
        self.assertEqual(len(csegs), 4)
        for i in [0, 4, 18, 21]:
            self.assertTrue(i in csegs)
            if i == 21:
                self.assertTrue('implicit_segment_%d' % (i) in wf.constraints)
            else:
                self.assertTrue('segment_%d' % (i) in wf.constraints)

        # Removing explicit constraint should remove associated implicit constr.
        wf.set_segments_free([4])
        csegs = wf.constrained_segments()
        self.assertEqual(len(csegs), 2)
        for i in [0, 18]:
            self.assertTrue(i in csegs)
            self.assertTrue('segment_%d' % (i) in wf.constraints)

        # Current loop through constrained segments should violate constraints
        wf.currents[18:22] = test_cur
        self.assertFalse(wf.check_constraints())
        wf.free_all_segments()
        self.assertTrue(wf.check_constraints())
        self.assertEqual(len(wf.constrained_segments()), 0)
        wf.currents[:] = 0

        # Consistency checks for the poloidal current constraint
        wf.add_poloidal_current_constraint(test_cur)
        self.assertFalse(wf.check_constraints())
        ntf = int(nPhi/2)
        with self.assertRaises(ValueError):
            wf.add_tfcoil_currents(ntf, 2*test_cur*(1/(2*ntf*wf.nfp)))
        wf.add_tfcoil_currents(ntf, test_cur*(1/(2*ntf*wf.nfp)))
        self.assertTrue(wf.check_constraints())
        wf.remove_poloidal_current_constraint()
        wf.currents[:] = 0
        self.assertTrue(wf.check_constraints())

        # Consistency checks for the toroidal current constraint
        wf.add_toroidal_current_constraint(test_cur)
        wf.currents[1:wf.nTorSegments:wf.nTheta] = test_cur
        self.assertFalse(wf.check_constraints()) # violates continuity when 
                                                 # enforcing stell symm
        wf.currents[1:wf.nTorSegments:wf.nTheta] = test_cur
        wf.currents[3:wf.nTorSegments:wf.nTheta] = test_cur
        self.assertFalse(wf.check_constraints()) # wrong total current
        wf.currents[1:wf.nTorSegments:wf.nTheta] = 0.5*test_cur
        wf.currents[3:wf.nTorSegments:wf.nTheta] = 0.5*test_cur
        self.assertTrue(wf.check_constraints())
        wf.remove_toroidal_current_constraint()
        wf.currents[:] = 0
        self.assertTrue(wf.check_constraints())

        #-----------------------------------------------------------------------
        # Check correctness of the toroidal break feature
        #-----------------------------------------------------------------------

        with self.assertRaises(ValueError):
            wf.set_toroidal_breaks(2, 1)
        with self.assertRaises(ValueError):
            wf.set_toroidal_breaks(1, 3)

        wf.set_toroidal_breaks(1, 1)
        csegs = wf.constrained_segments()
        self.assertEqual(len(csegs), 4)
        for i in [4, 5, 6, 7]:
            self.assertTrue(i in csegs)
        wf.free_all_segments()

        wf.set_toroidal_breaks(1,2)
        csegs = wf.constrained_segments()
        self.assertEqual(len(csegs), 12)
        for i in [4, 5, 6, 7, 8, 9, 10, 11, 22, 23, 24, 25]:
            self.assertTrue(i in csegs)
        wf.free_all_segments()

        wf.set_toroidal_breaks(1, 2, allow_pol_current=True)
        csegs = wf.constrained_segments()
        self.assertEqual(len(csegs), 8)
        for i in [4, 5, 6, 7, 8, 9, 10, 11]:
            self.assertTrue(i in csegs)
        wf.free_all_segments()

        wf.set_toroidal_current(test_cur)
        with self.assertRaises(ValueError):
           wf.set_toroidal_breaks(1, 2)
        wf.remove_toroidal_current_constraint()

        #-----------------------------------------------------------------------
        # Check that poloidal currents play nicely with toroidal breaks
        #-----------------------------------------------------------------------

        wf.add_tfcoil_currents(1, test_cur)
        wf.set_toroidal_breaks(1, 1, allow_pol_current=False)
        self.assertTrue(wf.check_constraints())
        wf.free_all_segments()

        wf.set_toroidal_breaks(1, 2, allow_pol_current=False)
        self.assertFalse(wf.check_constraints())
        wf.free_all_segments()

        wf.set_toroidal_breaks(1, 2, allow_pol_current=True)
        self.assertTrue(wf.check_constraints())
        wf.currents[:] = 0
        wf.free_all_segments()

        #-----------------------------------------------------------------------
        # Check correctness of the get_free_cells method
        #-----------------------------------------------------------------------

        wf.set_toroidal_breaks(1, 2)
        freeCells = wf.get_free_cells(form='indices')
        freeCellsLogical = wf.get_free_cells()
        self.assertEqual(len(freeCells), 8)
        for i in range(16):
            if i in [0, 1, 2, 3, 12, 13, 14, 15]:
                self.assertTrue(i in freeCells)
                self.assertTrue(freeCellsLogical[i])
            else:
                self.assertFalse(i in freeCells)
                self.assertFalse(freeCellsLogical[i])
        wf.free_all_segments()

        wf.set_segments_constrained(np.arange(wf.nTorSegments))
        self.assertFalse(np.any(wf.get_free_cells()))
        wf.free_all_segments()

    def test_toroidal_wireframe_constraint_matrices(self):
        """
        Consistency checks for the construction of constraint matrices by the
        ToroidalWireframe class
        """

        nfp = 3
        rmaj = 2
        rmin = 1
        surf_wf = surf_torus(nfp, rmaj, rmin)

        nPhi = 4
        nTheta = 4

        wf = ToroidalWireframe(surf_wf, nPhi, nTheta)

        #-----------------------------------------------------------------------
        # Baseline case: with no added constraints (only continuity constraints)
        #-----------------------------------------------------------------------

        # Basic properties of matrices with only continuity constraints
        B0, d0 = wf.constraint_matrices()
        self.assertAlmostEqual(np.max(np.abs(d0)), 0)
        self.assertEqual(B0.shape[0], nPhi*nTheta-2)
        self.assertEqual(B0.shape[1], 2*nPhi*nTheta)
        self.assertEqual(d0.shape[0], B0.shape[0])
        self.assertEqual(d0.shape[1], 1)
        self.assertEqual(np.linalg.matrix_rank(B0), nPhi*nTheta-2)

        # No constraints should be redundant
        B1, d1 = wf.constraint_matrices(remove_redundancies=False)
        self.assertEqual(B0.shape, B1.shape)
        self.assertEqual(d0.shape, d1.shape)
        self.assertFalse(np.any(B0-B1))
        self.assertFalse(np.any(d0-d1))

        # There should be no constrained segments to remove
        B2, d2 = wf.constraint_matrices(remove_redundancies=True)
        self.assertEqual(B0.shape, B2.shape)
        self.assertEqual(d0.shape, d2.shape)
        self.assertFalse(np.any(B0-B2))
        self.assertFalse(np.any(d0-d2))

        # Attempting to assume no crossings should produce an error
        with self.assertRaises(RuntimeError):
            B3, d3 = wf.constraint_matrices(assume_no_crossings=True)

        #-----------------------------------------------------------------------
        # Case with enough constrained segments to have no crossings
        #-----------------------------------------------------------------------

        wf.set_segments_constrained(np.arange(wf.nTorSegments))
        wf.set_segments_constrained([16, 17, 22, 23, 24, 25, 30, 31])
        nConst = len(wf.constrained_segments())

        # With no crossings assumed
        B0, d0 = wf.constraint_matrices(assume_no_crossings=True)
        self.assertEqual(np.max(np.abs(d0)), 0)
        self.assertEqual(B0.shape[0], nConst + 6)
        self.assertEqual(B0.shape[1], 2*nPhi*nTheta)
        self.assertEqual(d0.shape[0], B0.shape[0])
        self.assertEqual(d0.shape[1], 1)

        # No crossings + constrained segments removed
        B1, d1 = wf.constraint_matrices(assume_no_crossings=True, \
                                        remove_constrained_segments=True)
        self.assertEqual(B1.shape[0], 6)
        self.assertEqual(B1.shape[1], 8)
        self.assertEqual(d1.shape[0], 6)

        # Default behavior (all segments included; no closed loop assumption)
        B2, d2 = wf.constraint_matrices()
        self.assertEqual(np.max(np.abs(d2)), 0)
        self.assertEqual(B2.shape[0], nConst + 8)
        self.assertEqual(B2.shape[1], 2*nPhi*nTheta)

        # Including all constraints, including the redundant ones
        B3, d3 = wf.constraint_matrices(remove_redundancies=False)
        self.assertEqual(np.max(np.abs(d3)), 0)
        self.assertEqual(B3.shape[0], nPhi*nTheta-2 + nConst)
        self.assertEqual(B3.shape[1], 2*nPhi*nTheta)
        self.assertEqual(d3.shape[0], B3.shape[0])
        self.assertEqual(d3.shape[1], 1)

    def test_toroidal_wireframe_collision_checking(self):
        """
        Tests method that imparts segment constraints where collisions would
        occur with other objects
        """

        nfp = 3
        rmaj = 2
        rmin = 1
        surf_wf = surf_torus(nfp, rmaj, rmin)

        nPhi = 4
        nTheta = 4

        wf = ToroidalWireframe(surf_wf, nPhi, nTheta)

        # A port whose end lies near (but does not cover) a node
        p1 = CircularPort(ox=2, oy=0, oz=0, ax=0, ay=0, az=1, ir=0.001, \
                          thick=0.001, l0=0, l1=0.999)
        wf.constrain_colliding_segments(p1.collides)
        self.assertEqual(len(wf.constrained_segments()), 0)

        # The node is blocked if a sufficient gap spacing is added
        wf.constrain_colliding_segments(p1.collides, gap=0.01) 
        csegs = wf.constrained_segments()
        self.assertEqual(len(csegs), 4)
        for i in [1, 3, 16, 17]:
            self.assertTrue(i in csegs)
        wf.free_all_segments()

        # A port that encloses all segments
        p2 = CircularPort(ox=0, oy=0, oz=0, ax=0, ay=0, az=1, ir=3.5, thick=0, \
                          l0=-1.5, l1=1.5)
        wf.constrain_colliding_segments(p2.collides)
        self.assertEqual(len(wf.unconstrained_segments()), 0)
        wf.free_all_segments()

    def test_toroidal_wireframe_windowpane(self):
        """
        Tests the construction of a wireframe with windowpane constraints
        """

        nfp = 3
        rmaj = 2
        rmin = 1
        surf_wf = surf_torus(nfp, rmaj, rmin)

        nCoils_tor = 4
        nCoils_pol = 6
        size_tor = 2
        size_pol = 2
        gap_tor = 2
        gap_pol = 2
        wf = windowpane_wireframe(surf_wf, nCoils_tor, nCoils_pol, size_tor, \
                                  size_pol, gap_tor, gap_pol)

        usegs = wf.unconstrained_segments()
        n_usegs = 2*(size_tor + size_pol)*nCoils_tor*nCoils_pol
        self.assertEqual(len(usegs), n_usegs)

    @unittest.skipIf(pyevtk is None, "pyevtk not found")
    def test_toroidal_wireframe_to_vtk(self):
        """
        Tests export of a wireframe to a VTK file
        """

        import os
 
        surf_wf = surf_torus(3, 2, 0.5)
        wf = windowpane_wireframe(surf_wf, 5, 10, 4, 4, 2, 2)

        filename = 'temp_wframe_vtk_test_file'
        filepath = os.path.join(os.path.dirname(__file__), filename)
        wf.to_vtk(filepath)
        os.remove(filepath + '.vtu')

