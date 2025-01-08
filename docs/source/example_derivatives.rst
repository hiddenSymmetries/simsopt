Using the Derivative class
============================

In this tutorial we will review how to use the :obj:`~simsopt._core.derivative.Derivative` class
to define the derivatives of Optimizable objects. First, lets import our some modules.

.. code-block::

    import numpy as np
    from simsopt._core import Optimizable
    from simsopt.field import Current, coils_via_symmetries, BiotSavart
    from simsopt.geo import create_equally_spaced_curves, SurfaceRZFourier
    from scipy.optimize import approx_fprime
    from simsopt._core.derivative import Derivative, derivative_dec


Lets start by defining some degrees of freedom. Our objective function
will depend on a ``BiotSavart`` object and a ``Surface`` object.

.. code-block::

    # define surface
    nfp = 2
    surf = SurfaceRZFourier(nfp=nfp, stellsym=True, mpol=1, ntor=1)
    surf.unfix_all()
    surf.get('rc(0,0)')
    surf.set('rc(0,0)', 1.0)
    surf.set('rc(1,0)', 0.2)
    surf.set('zs(1,0)', 0.2)

    # coils
    ncoils = 4
    base_curves = create_equally_spaced_curves(ncoils, nfp, stellsym=True, R0=1.0,
                                            R1=0.5, order=3)
    base_currents = [Current(1.0) * 1e5 for i in range(ncoils)]
    coils = coils_via_symmetries(base_curves, base_currents, nfp, stellsym=True)
    biot_savart = BiotSavart(coils)


Notice how the BiotSavart object does not have any DOFs of its own, it simply 
holds curves and currents. Instead, all of the DOFs of a BiotSavart object are 
held by the Curve and Current object themselves.

On the other hand, a surface has its own DOFs.

We can plot the coils and surface to make sure we have defined reasonable values.

.. code-block::

    # plot for sanity
    from simsopt.geo import plot as sms_plot
    sms_plot(coils+[surf])


Now lets define an Optimizable objective function and walk through the process
of computing derivatives. The Optimizable will have an ``Error`` method which
is the objective function, 
    :math:`J = \frac{1}{2 n_\theta n_\phi} \sum_{ij} B_{ij}^2`
where the sum is taken over quadrature nodes on the surface and ``B`` is the 
field strenth of the BiotSavart field.

The ``dError`` method is where we will define the
partial derivatives of the objective with respect to the surface and BiotSavart objects. The
``dError_decorated`` function uses the ``derivative_dec`` to convert the partial derivatives
from ``dError`` into a full gradient of the objective with respect to all free degrees of
freedom (DOFS).

Read the documentation of the functions in the object to learn the
specifics of constructing a derivative.

.. code-block::

    class Objective(Optimizable):

        def __init__(self, bs, surf):
            self.bs = bs
            self.surf = surf
            Optimizable.__init__(self, depends_on=[bs, surf])

        def Error(self):
            """
            The objective function.
            """
            xyz = self.surf.gamma()
            nphi, ntheta, _ = np.shape(xyz)
            xyz = xyz.reshape((-1,3)) # (nphi * ntheta, 3)
            self.bs.set_points(xyz)
            B = self.bs.B()
            J = 0.5 * np.sum(B**2) / (nphi * ntheta)
            return J

        def dError(self):
            """
            Compute the partial derivatives of the Error function.
            
            Calling a function with no derivative_dec decorator, like dErrror(), will return 
            a Derivative object. We can then get the partial derivatives by doing, for instance,
                `partials = dError()`
                `dError_by_dsurf = partials(surf)`
                `dError_by_dbs = partials(bs)`

            This function will return a Derivative object created from a dictionary.
            The keys for the dictionary are each Optimizable object that is used in the 
            Error function which 'owns' a DOF. For example, the surf object owns the Fourier 
            coefficients which describe the surface. So we will add
                `derivs = {}`
                `derivs[surf] = array of surface derivatives`
            On the other hand, the BiotSavart object does not 'own' any DOFs, so, in general, we would not add 
            anything to the dictionary for the BiotSavart object. All of the DOFs in in the BiotSavart 
            object are owned by the curves and currents which make up the coils. So for each
            coil in the BiotSavart object we do,
                `derivs[coil.current] = array of current derivatives`
                `derivs[coil.curve] = array of curve derivatives`
            The values of the dictionary should be the actual derivative arrays associated to that
            Optimizable object.
            
            Conveniently, for a BiotSavart object ALL derivatives can be accessed through the
                B_vjp(...)
            method. The B_vjp() method returns a Derivative object containing the vector jacobian 
            product of the derivatives with another vector. So for our case, we will directly use B_vjp
            to compute the derivatives we need.

            Finally, we return a Derivative object, created from the dictionary.
                `Derivative(derivs)`
            """
            xyz = self.surf.gamma()
            nphi, ntheta, _ = np.shape(xyz)
            xyz = xyz.reshape((-1,3)) # (nphi * ntheta, 3)
            self.bs.set_points(xyz)
            B = self.bs.B() # (ntheta * nphi, 3)

            # make a dictionary of derivatives
            derivs = {}

            """
            derivative with respect to surface dofs 
                dError/dcoeff = 1/(ntheta * nphi) sum_i (dX/dcoeff)^T(dB/dX)^T B
            """
            dB_by_dX = self.bs.dB_by_dX() # (ntheta * nphi, 3, 3)
            dgamma_by_dsurf = self.surf.dgamma_by_dcoeff() # (ntheta, nphi, 3, n_surf_dof)
            dgamma_by_dsurf = dgamma_by_dsurf.reshape((nphi * ntheta, 3, -1))
            n_surf_dofs = np.shape((dgamma_by_dsurf))[-1]
            dError_by_dsurf = np.zeros(n_surf_dofs)
            for ii in range(len(B)):
                dError_by_dsurf += dgamma_by_dsurf[ii].T @ (dB_by_dX[ii].T @ B[ii]) / (ntheta * nphi)
            derivs[self.surf] = dError_by_dsurf
            dError_by_dsurf = Derivative(derivs)

            """ derivative with respect to ALL curve/current dofs """
            dError_by_dbs = self.bs.B_vjp(B / (nphi * ntheta)) # Derivative object

            """ Derivative objects are summable. In the case that an Optimizable object, such
            as a surface, exists in both Derivative objects, then the derivatives of the Optimizable
            will be summed. If it only exists in one Derivative object, then the derivative of the sum
            will be just the sole derivative value. For example,

                # if deriv1 and deriv2 are Derivative objects with an Optimizable surface then,
                (deriv1 + deriv2)[surface] = deriv1[surface] + deriv2[surface]

                # if the surface only exists in deriv1 and not deriv2
                (deriv1 + deriv2)[surface] = deriv1[surface] 

            Derivatives can also be multiplied. However, the multiplication rules differ slightly from
            the addition rules.
            """        
            dError_by_all = dError_by_dsurf + dError_by_dbs
            return dError_by_all
        
        """
        We did not use the derivative decorator when constructing the dError function.
        The derivative decorator is optional, but has a key impact on functionality.
        Calling a decorated function like dErrror_decorated() will return a gradient
        with respect to all dofs, as opposed to a Derivative object. This is useful for numerical
        optimization purposes where the full gradient is used, rather than partial derivatives.

        ex:
            gradient = self.dError_decorated()
        """
        @derivative_dec
        def dError_decorated(self):
            return self.dError()


Evaluating the objective and gradient is easy. As described above, we have two methods for 
compute derivative: ``dError_decorated`` and ``dError``. ``dError_decorated`` computes
the gradient with respect to all free DOFS, while ``dError`` computes partial derivatives
with respect to an parent Optimizable objects within the ``Objective`` class.

.. code-block::

    obj = Objective(bs=biot_savart, surf=surf)

    # objective value
    print(obj.Error())

    # gradient w.r.t all free dofs
    print(obj.dError_decorated())

    # partials
    partials = obj.dError()
    dError_by_dsurf = partials(surf)
    dError_by_dbs = partials(biot_savart)
    print(dError_by_dsurf) # partial
    print(dError_by_dbs) # partial

We can check that the derivatives are correct using finite differences.

.. code-block::

    # check derivative w.r.t. surface dofs w/ finite difference
    obj.unfix_all()
    biot_savart.fix_all()
    x = obj.x
    def fun(x):
        surf.x = x
        return obj.Error()
    dError_by_dsurf_fd = approx_fprime(x, fun, epsilon=1e-7)
    print('surf dof finite difference error', np.max(np.abs(dError_by_dsurf_fd - dError_by_dsurf)))

    # check derivative w.r.t. coil dofs w/ finite difference
    obj.unfix_all()
    surf.fix_all()
    x = obj.x
    def fun(x):
        biot_savart.x = x
        return obj.Error()
    dError_by_dbs_fd = approx_fprime(x, fun, epsilon=1e-6)
    print('coil dof finite difference error', np.max(np.abs(dError_by_dbs_fd - dError_by_dbs)))