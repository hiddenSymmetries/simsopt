from scipy.spatial import Delaunay
import numpy as np
import matplotlib.pyplot as plt

__all__ = ['hull2D']

class hull2D():
    """Class to generate 2D non-convex hulls using the Chi-algorithm, described in 
    
    Matt Duckham et al. “Efficient Generation of Simple Polygons for Characterizing the Shape of a Set of Points in the Plane”. In: Pattern Recognition 41.10 (Oct. 1, 2008), pp. 3224–3236. issn: 0031-3203. doi: 10.1016/j.patcog.2008.03.023.

    Args:
        - xflat: Points in 2D space. Should be a np.array of shape (npts, 2)
        - X3D: Points in 3D space. This is only used to extract the upper and lower envelop of the hull. If not needed, use X3D=None
    """

    def __init__(self, XFLAT, X3D=None):
        self.x3d = X3D
        x = XFLAT[:,0]
        y = XFLAT[:,1]
        self.points = np.array([self.point(xx, yy, ii) for ii, (xx, yy) in enumerate(zip(XFLAT[:,0],XFLAT[:,1]))])
        delaunay = Delaunay(np.stack([x,y],axis=1), furthest_site=False, incremental=True )

        self.triangles = dict()
        for s in delaunay.simplices:
            ptind = tuple(np.sort([s[0],s[1],s[2]]))
            self.triangles[ptind] = self.triangle(self.points[s[0]], self.points[s[1]], self.points[s[2]], key=ptind)

        
        self.segments = dict()
        for pt in self.points:
            tmp = []
            ind = np.where(delaunay.simplices==pt.index)[0]
            for s in delaunay.simplices[ind,:]:
                tmp.append(s)
            tmp = np.unique(tmp)
            ind = np.where(tmp==pt.index)

            for new_ind in np.delete(tmp, ind): 
                if new_ind>pt.index:
                    key = tuple(np.sort([pt.index,new_ind]))
                    self.segments[key] =  self.segment(pt, self.points[new_ind], key)

        self.edges = self.find_edge()

        test, tkey = self.is_regular()         
        while not test:
            self.pop_singular_triangle(self.triangles[tkey])
            test, tkey = self.is_regular()         



    class point():
        """Point in 2D space, to which we associate an index
        
        Args:
            - x: x-coordinate
            - y: y-coordinate
            - index: index of the point
        """
        def __init__(self, x, y, index):
            self.x = x
            self.y = y
            self.index = index

    class segment():
        """Segment in 2D space, defined by two points, and to which we associate a key. Here we assume we work in cartesian coordinates

        Args:
            - x0: First point, instance of the class hull2D.point
            - x1: Second point, instance of the class hull2D.point
        """
        def __init__(self, x0, x1, key):
            self.points = np.array([x0,x1])
            self.key = key

        def plot(self, ax, color='k'):
            """Plot the segment
            
            Args:
                - ax: instance of matplotlib.axes._axes.Axes
                - color: define the color of the segment
            """
            xx = [p.x for p in self.points]
            yy = [p.y for p in self.points]
            ax.plot(xx, yy, color=color)

        def length(self):
            """Returns the length of the segment
            """
            x0 = self.points[0]
            x1 = self.points[1]
            return np.sqrt( (x0.x-x1.x)**2 + (x0.y-x1.y)**2 )


    class triangle():
        """Represent a triangle in 2D space, defined by three points, and to which we associate a key.
        
        Args:
            - p1, p2, p3: three instances of hull2D.point
            - key: key associated to this triangle.
        """
        def __init__(self, p1, p2, p3, key):
            self.points = np.array([p1, p2, p3])
            self.key = key


        def plot(self, ax=None):
            """Plot the triangle
            
            Args:
                - ax: instance of matplotlib.axes._axes.Axes
            """

            if ax is None:
                _, ax = plt.subplots()

            xx = [pt.x for pt in self.points]
            yy = [pt.y for pt in self.points]
            ax.fill(xx, yy)
            
        
    def find_triangle_with_segment(self, s):
        """Given a segment, find triangles bounded by this segment.
        
        Args:
            - s: instance of hull2D.segment
        Output:
            - t1, t2: Two bounding triangles. t2 is None if segment is only shared by one triangle.
        """
        # Get segment's points indices
        i0 = s.points[0].index
        i1 = s.points[1].index

        # Find all triangle keys that have both i0 and i1
        keys = np.array(list(self.triangles.keys()))
        ind = np.intersect1d(np.where(keys==i0)[0], np.where(keys==i1)[0])

        # Return triangles
        if ind.size==2:
            key1 = tuple(keys[ind[0]])
            key2 = tuple(keys[ind[1]])
            return self.triangles[key1], self.triangles[key2]
        elif ind.size==1:
            key1 = tuple(keys[ind[0]])
            return self.triangles[key1], None
        else:
            return ValueError('Segment is not part of a triangle')

    def is_segment_at_edge(self, s):
        """Check if segment is forming the edge of the hull. This is easily identified - if the segment is only appearing in one triangle, then it is an edge segment.
        
        Args:
            - s: instance of hull2D.segment
        Output:
            - boolean, True if edge segment, False otherwise.
        """
        _, t2 = self.find_triangle_with_segment( s )
        if t2 is None:
            return True
        else:
            return False

    def pop_segment(self, segment):
        """Remove a segment from the hull. Can only remove a segment if it is an edge segment (see hull2D.is_segment_at_edge). The associated triangle is also removed. The hole in the hull is then replaced by the two other segments forming the popped triangle.
        
        Args:
            segment: instance of hull2D.segment
        
        Exceptions:
            ValueError: if the segment is not an edge segment
        """
        key = segment.key
        if not self.is_segment_at_edge(segment):
            raise ValueError('Trying to pop a segment not at the edge')

        # Pop segment
        self.segments.pop(key)

        # Find triangle and its segment - replace hole in edge
        t, _ = self.find_triangle_with_segment(segment)
        skey1 = (t.key[0],t.key[1])
        skey2 = (t.key[0],t.key[2])
        skey3 = (t.key[1],t.key[2])

        # Re-evaluate edges
        ind = self.edges.index( key )
        self.edges.pop( ind )
        for skey in [skey1, skey2, skey3]:
            if skey!=key:
                self.edges.insert(ind, skey)
                               
        # Pop triangle
        self.triangles.pop(t.key)

    def pop_singular_triangle(self, triangle):
        # Find point indices
        pindex = np.sort([p.index for p in triangle.points])
        if self.is_segment_at_edge( self.segments[(pindex[0],pindex[1])] ):
            key = (pindex[0],pindex[1])
            self.segments.pop( key )

            ind = self.edges.index( key )
            self.edges.pop( ind )
        else:
            new_edge = (pindex[0],pindex[1])

        if self.is_segment_at_edge( self.segments[(pindex[1],pindex[2])] ):
            key = (pindex[1],pindex[2])
            self.segments.pop( key )

            ind = self.edges.index( key )
            self.edges.pop( ind )
        else:
            new_edge = (pindex[1],pindex[2])

        if self.is_segment_at_edge( self.segments[(pindex[0],pindex[2])] ):
            key = (pindex[0],pindex[2])
            self.segments.pop( key )

            ind = self.edges.index( key )
            self.edges.pop( ind )
        else:
            new_edge = (pindex[0],pindex[2])

        self.edges.insert(ind, new_edge)
        self.triangles.pop(triangle.key)



    def pop_triangle(self, triangle):
        """Pop triangle
        
        Args:
            - triangle: instance of hull2D.triangle
        
        Exceptions:
            - ValueError if the triangle is not an edge triangle.
        """
        # Find point indices
        pindex = np.sort([p.index for p in triangle.points])
        
        # Find which segment is an edge segment. If none are at the edge, raise ValueError
        if self.is_segment_at_edge( self.segments[(pindex[0],pindex[1])] ):
            self.pop_segment( self.segments[(pindex[0],pindex[1])] )
        elif self.is_segment_at_edge( self.segments[(pindex[1],pindex[2])] ):
            self.pop_segment( self.segments[(pindex[1],pindex[2])] )
        elif self.is_segment_at_edge( self.segments[(pindex[0],pindex[2])] ):
            self.pop_segment( self.segments[(pindex[0],pindex[2])] )
        else:
            raise ValueError('Triangle is not at the edge')
        
    
    def find_edge(self):
        """Find a closed path of edge segments. If the domain has no "holes", this forms the non-convex hull.

        Output:
            - edge_keys: instance of numpy.array, containing the keys of the edge segments.
        """

        # First scan through segments until finding one edge segment
        edge_keys = []
        for key, s in self.segments.items():
            if self.is_segment_at_edge( s ):
                edge_keys.append(key)
                break

        # Now hop from one point to the next one, only taking edge segments
        last_key = edge_keys[-1] # This is the last segment key we traveled from
        next_segment = self.segments[last_key] # This is the segment
        first_point = next_segment.points[0] # One of the last segment point
        next_point = next_segment.points[1] # The other last segment point

        keys = np.array(list(self.segments.keys()))
        # We loop until the next point is the same as the very first point of the path
        while next_point.index!=first_point.index: 
            # Find all segments that start from the next point
            segments = np.array([self.segments[tuple(k)] for k in keys[np.where(keys==next_point.index)[0]]])

            # Remove last segment we just traveled on
            ind = np.where( [s is next_segment for s in segments] )[0]
            segments = np.delete(segments, ind)

            # Find next one - there should be only one other segment starting from next_point and that is an edge segment
            is_edge = [self.is_segment_at_edge(s) for s in segments]
            next_segment = segments[ np.where( is_edge ) ][0]
            edge_keys.append(next_segment.key)

            # Identify new point
            if next_segment.points[0] is next_point:
                next_point = next_segment.points[1]
            else:
                next_point = next_segment.points[0]

        # Return the keys
        return edge_keys
    

    def sort_edge_keys(self, edges):
        """ Sort the keys from the edge segments

        In general, edge_keys element are (i0,i1), with i0<i1. This function reorders all edge segments to form a chain, as
        (i0,i1), (i1,i2), (i2,i3), ... , (in,i0)

        Args:
            - edges: obtained from hull2D.find_edge()

        Output:
            - sorted_edges
        """
        tmp = np.unique(self.edges,axis=0)
        edges = [tuple(e) for e in tmp]
        sorted_edges = [edges[0]]
        last_pt = sorted_edges[-1][1]
        edges.pop(0)

        while sorted_edges[-1][1]!=sorted_edges[0][0]:
            arr_edges = np.array(edges)
            ind = np.where(arr_edges==last_pt)
            
            ii = ind[0][0]
            jj = ind[1][0]
            if jj==0:
                last_pt = edges[ii][1]
                sorted_edges.append(edges[ii])
            else:
                last_pt = edges[ii][0]
                sorted_edges.append((edges[ii][1],edges[ii][0]))
                
            edges.pop(ii)
        return sorted_edges



    def envelop(self, dmax, upper_or_lower):
        """ Get upper and lower envelop. Maybe should be moved somewhere else - not part of hull2D

        Args:
            dmax: Max length of edge segment
            upper_or_lower: 'upper' or 'lower'

        Output:
            ind: Indices of hull2D.x and hull2D.y corresponding to the upper or lower envelop

        Exception:
            ValueError: If incorrect 'upper_or_lower' input
        """

        if upper_or_lower=='upper':
            f = np.argmax
        elif upper_or_lower=='lower':
            f = np.argmin
        else:
            raise ValueError("upper_or_lower should be 'upper' or 'lower'!")
        
        # First we evaluate the non-convex hull and get the points indices
        out=1
        while out==1:
            out = self.remove_largest_edge_segment(dmax)
        edge_index = self.sort_edge_keys( self.edges )
        node_index = [e[0] for e in edge_index]

        # Recover y-ccordinates
        y = np.array([p.y for p in self.points])

        # Use information from the x3d set of points to split the hull at relevant points
        N, M, _ = self.x3d.shape
        left_indices = np.arange(0,M)
        ind = np.intersect1d(node_index, left_indices)
        ind0 = ind[f(np.array(y[ind]))]
        
        right_indices = np.arange((N-1)*M,N*M)
        ind = np.intersect1d(node_index, right_indices)
        ind1 = ind[f(np.array(y[ind]))]

        # Re-organize edge points so that i0 is smaller than i1.
        i0 = np.where(node_index==ind0)[0][0]
        i1 = np.where(node_index==ind1)[0][0]
        if i1<i0:
            start=i1
            end=i0
        else:
            start=i0
            end=i1
            
        # Need to find point in between both limits of the envelop
        ii = int((i1+i0)/2) # One candidate between i0 and i1
        ll = len(node_index)
        jj = int(np.mod( i1+ ((ll-i1) + i0)/2, ll-1 )) # Another candidate, going the other way around the hull
        ind = f(np.array([y[node_index[ii]], y[node_index[jj]]]))
        if ind == 0:
            i2 = ii
        else:
            i2 = jj
        
        if i2<start or i2>end:
            ind = node_index[end+1:] + node_index[:start]
        else:
            ind = node_index[start:end+1]

        return ind
    

    def remove_largest_edge_segment(self, dmax):
        """Find largest edge segment and pop it

        Args:
            - dmax: Max distance to pop. If all segments are smaller than dmax, no segment is popped and this method returns 0
        
        Output:
            - flag: 1 if segment has been popped, 0 if all segments are smaller than dmax, or if updated triangularisation is singular
        """
        edge_keys = self.edges
        lengths = [self.segments[k].length() for k in edge_keys]

        index_to_pop = np.argmax(lengths)
        segment_to_pop = self.segments[edge_keys[ index_to_pop ]]
        if lengths[index_to_pop]<=dmax:
            return 0 

        # Check if triagularization is still regular after the pop:
        t1, _ = self.find_triangle_with_segment( segment_to_pop )
        tt, _ = self.is_regular(triangle_mask = t1.key)
        if tt:
            self.pop_segment( segment_to_pop )
            #print(f'Popped segment {edge_keys[index_to_pop]}')
            return 1
        else:
            return 0

    def is_regular(self, triangle_mask=None):
        """Check if the triangularisation is regular
        
        A triagularisation is irregular if any triangle has two segments that are edges. Equivalently, if any edge node only belongs to one triangle, the triangularisation is irregular

        Args:
            - triangle_mask: Mask some triangles from the calculation

        Output:
            - True: triangularisation is irregular
            - False: triangularisation is regular
        """
        # Find edges
        edge_indices = self.edges
        # Find edge point indices
        edge_vertices_index = np.unique([self.segments[k].points[0].index for k in edge_indices] + [self.segments[k].points[1].index for k in edge_indices])

        # check to how many simplices each edge node belongs
        triangle_keys = np.array(list(self.triangles.keys()))
        # apply mask
        if triangle_mask is not None:
            ind = np.where(np.all(triangle_keys==triangle_mask,axis=1))[0][0]
            triangle_keys = np.append(triangle_keys[:ind], triangle_keys[ind+1:], axis=0)
        for v in edge_vertices_index:
            ind = np.where(triangle_keys==v)[0]
            if ind.size==1:
                return False, tuple(triangle_keys[ind[0]])
        return True, 0
        
    def plot_edge(self, ax=None, color='k'):
        """ Plot hull
        
        Args:
            - ax: instance of matplotlib.axes._axes.Axes. If not provided, a new figure is generated
            - color: set edge color
        """
        # Open new figure if no ax is provided
        if ax is None:
            _, ax = plt.subplots()
            
        # Plot segments one by one
        edge_keys = self.edges
        for key in edge_keys:
            self.segments[key].plot(ax=ax, color=color)
        
    
    def plot(self, ax=None):
        """ Plot all points and triangles
        
        Args:
            - ax: instance of matplotlib.axes._axes.Axes. If not provided, a new figure is generated
        """
        # Open new figure if no ax is provided
        if ax is None:
            _, ax = plt.subplots()

        # Plot all points
        xplot = [x.x for x in self.points]
        yplot = [x.y for x in self.points]
        ax.scatter(xplot, yplot, s=10)

        # Plot all segments
        for _, s in self.segments.items():
            s.plot(ax=ax, color='r')