"""
wireframe.py

Definitions for the ToroidalWireframe class
"""

import numpy as np
import warnings
from simsopt.geo.surfacerzfourier import SurfaceRZFourier

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

        if nTheta % 2:
            nTheta = nTheta + 1
            warnings.warn('nTheta must be even. Increasing to %d' % (nTheta))

        if nPhi % 2:
            nPhi = nPhi + 1
            warnings.warn('nPhi must be even. Increasing to %d' % (nPhi))

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
        node_inds = np.arange(self.nNodes).reshape(nodes_surf.shape[:2])

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
        segments_tor[:,0] = node_inds[:-1, :].reshape((self.nTorSegments))
        segments_tor[:,1] = node_inds[1:,  :].reshape((self.nTorSegments))

        # Map nodes to index in the segment array of segment originating 
        # from the respective node
        self.torSegmentKey = -np.ones(nodes_surf.shape[:2]).astype(np.int64)
        self.torSegmentKey[:-1,:] = \
            np.arange(self.nTorSegments).reshape((nPhi, nTheta))

        # Poloidal segments (on symmetry planes, only include segments for z>0)
        segments_pol = np.zeros((self.nPolSegments, 2))
        self.polSegmentKey = -np.ones(nodes_surf.shape[:2]).astype(np.int64)
        HalfNTheta = int(nTheta/2)

        segments_pol[:HalfNTheta, 0] = node_inds[0, :HalfNTheta]
        segments_pol[:HalfNTheta, 1] = node_inds[0, 1:HalfNTheta+1]
        self.polSegmentKey[0, :HalfNTheta] = np.arange(HalfNTheta) + self.nTorSegments
        for i in range(1, nPhi):
            polInd0 = HalfNTheta + (i-1)*nTheta
            polInd1 = polInd0 + nTheta
            segments_pol[polInd0:polInd1, 0] = node_inds[i, :]
            segments_pol[polInd0:polInd1-1, 1] = node_inds[i, 1:]
            segments_pol[polInd1-1, 1] = node_inds[i, 0]
            self.polSegmentKey[i, :] = np.arange(polInd0, polInd1) + self.nTorSegments

        segments_pol[-HalfNTheta:, 0] = node_inds[-1, :HalfNTheta]
        segments_pol[-HalfNTheta:, 1] = node_inds[-1, 1:HalfNTheta+1]
        self.polSegmentKey[-1, :HalfNTheta] = \
            np.arange(self.nPolSegments-HalfNTheta, self.nPolSegments) + self.nTorSegments

        # Join the toroidal and poloidal segments into a single array
        self.segments = \
            np.ascontiguousarray(np.zeros((self.nSegments, 2)).astype(np.int64))
        self.segments[:self.nTorSegments, :] = segments_tor[:, :]
        self.segments[self.nTorSegments:, :] = segments_pol[:, :]

        # Initialize currents to zero
        self.currents = np.ascontiguousarray(np.zeros((self.nSegments)))

        # Define constraints to enforce current continuity
        # Constraint equations have the form B*x = d, where:
        #   x is the array of currents in each segment
        #   B is a matrix of coefficients of the currents in each equation
        #   d is an array of constant terms on the RHS of each equation

        self.nConstraints = self.nTorSegments - 2
        self.constraints_B = \
            np.ascontiguousarray(np.zeros((self.nConstraints, self.nSegments)))
        self.constraints_d = np.ascontiguousarray(np.zeros((self.nConstraints)))

        # Populate B matrix with coefficients to enforce continuity at each node
        count = 0
        for i in range(nPhi+1):
            for j in range(nTheta):

                if i == 0:
                    if j == 0 or j >= nTheta/2:
                        continue
                    ind_tor_in  = self.torSegmentKey[i, nTheta-j]
                    ind_tor_out = self.torSegmentKey[i, j]
                    ind_pol_in  = self.polSegmentKey[i, j-1]
                    ind_pol_out = self.polSegmentKey[i, j]

                elif i > 0 and i < nPhi:
                    ind_tor_in  = self.torSegmentKey[i-1, j]
                    ind_tor_out = self.torSegmentKey[i, j]
                    if j == 0:
                        ind_pol_in  = self.polSegmentKey[i, nTheta-1]
                    else:
                        ind_pol_in  = self.polSegmentKey[i, j-1]
                    ind_pol_out = self.polSegmentKey[i, j]

                else:
                    if j == 0 or j >= nTheta/2:
                        continue
                    ind_tor_in  = self.torSegmentKey[i-1, j]
                    ind_tor_out = self.torSegmentKey[i-1, nTheta-j]
                    ind_pol_in  = self.polSegmentKey[i, j-1]
                    ind_pol_out = self.polSegmentKey[i, j]

                self.constraints_B[count, [ind_tor_in,  ind_pol_in ]] = -1
                self.constraints_B[count, [ind_tor_out, ind_pol_out]] = 1
                count = count + 1

    def make_plot(self, ax=None):
        """
        Make a plot of the wireframe grid, including nodes and segments.
        """

        import matplotlib.pylab as pl
        from mpl_toolkits.mplot3d.art3d import Line3DCollection

        pl_segments = np.zeros((2*self.nfp*self.nSegments, 2, 3))
        pl_currents = np.zeros((2*self.nfp*self.nSegments))
   
        for i in range(2*self.nfp):
            ind0 = i*self.nSegments
            ind1 = (i+1)*self.nSegments
            pl_segments[ind0:ind1,:,:] = self.nodes[i][:,:][self.segments[:,:]]
            pl_currents[ind0:ind1] = self.currents[:]*1e-6

        lc = Line3DCollection(pl_segments)
        lc.set_array(pl_currents)

        if ax is None:
            fig = pl.figure()
            ax = fig.add_subplot(projection='3d')

            ax.set_xlim([np.min(pl_segments[:,:,0], axis=(0,1)),
                         np.max(pl_segments[:,:,0], axis=(0,1))])
            ax.set_ylim([np.min(pl_segments[:,:,1], axis=(0,1)),
                         np.max(pl_segments[:,:,1], axis=(0,1))])
            ax.set_zlim([np.min(pl_segments[:,:,2], axis=(0,1)),
                         np.max(pl_segments[:,:,2], axis=(0,1))])

            ax.set_aspect('equal')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            cb = pl.colorbar(lc)
            cb.set_label('Current (MA)')

        ax.add_collection(lc)

        return(ax)

