.. _example_vmec:

Running VMEC 
============

VMEC solves the ideal MHD equations in three-dimensional toroidal geometries, by minimizing an energy functional, 

.. math::
  W = \int \left( \frac{B^2}{2\mu_0} + \frac{p}{\gamma -1}\right) \, d^3 x 

where :math:`B` is the magnetic field strength, :math:`p` is the plasma pressure, and :math:`\gamma` is the adiabatic index. The functional is minimized subject to the constraint of a divergence-free magnetic field, :math:`\nabla \cdot B = 0`, continuously nested magnetic surfaces labeled by :math:`\psi`, fixed rotational transform profile, :math:`\iota(\psi)`, and either a fixed boundary shape or specified magnetic field in the vacuum region outside the plasma.
The divergence-free constraint is enforced through the expression,

.. math::
  B = \nabla \psi \times \nabla \alpha. 

Here :math:`\psi` is assumed to be the toroidal flux function, and :math:`\alpha` labels field lines on magnetic surfaces. 
Given a background coordinate system with poloidal angle :math:`\theta` and cylindrical toroidal angle :math:`\phi` along with the normalized toroidal flux :math:`s`, the field-line label is determined through the expression :math:`\alpha = \theta - \iota \phi + \lambda(s,\theta,\phi)` for some periodic function :math:`\lambda(\theta,\phi)`.

The energy functional is then minimized with respect to :math:`R(\theta,\phi)` and :math:`Z(\theta,\phi)`, which describe the shape of the flux surfaces in cylindrical coordinates, and :math:`\lambda(s,\theta,\phi)`, which determines the field-line label. 

See :ref:`mhd` for more general information about MHD equilibria in simsopt. 

In this tutorial, we describe how to run the VMEC equilibrium code. Other resources are the `STELLOPT wiki <https://princetonuniversity.github.io/STELLOPT/VMEC>`_ and the `VMEC++ documentation <https://arxiv.org/pdf/2502.04374>`_.

Input parameters
^^^^^^^^^^^^^^^^

You can initialize this class either from a VMEC ``input.<extension>`` file or from a ``wout_<extension>.nc`` output file. 
See, for example, ``tests/test_files/input.LandremanPaul2021_QA_lowres`` and ``tests/test_files/wout_LandremanPaul2021_QA_lowres.nc``.  
If neither is provided, a default input file is used. When this class is initialized from an input file, it is possible to modify the input parameters and run the VMEC code. When this class is initialized from a ``wout`` file, all the data from the
``wout`` file are available in memory but the VMEC code cannot be
re-run, since some of the input data (e.g. radial multigrid
parameters) are not available in the wout file.

The primary inputs to VMEC are geometry information (the shape of the last closed flux surface and/or the coil shapes and locations) and two profile functions (pressure and one of current or rotational transform), described in the two subsequent sections.
A more comprehensive list can be found in the `VMEC++ documentation <https://arxiv.org/pdf/2502.04374>`_. 
All quantities are in SI units. 

These parameters can be directly modified in the ``input.extension`` Fortran namelist.
The input parameters to VMEC are also accessible as attributes of
the ``indata`` attribute. For example, if ``vmec`` is an instance
of :obj:`~simsopt.mhd.vmec.Vmec`, then you can read or write the input resolution
parameters using ``vmec.indata.mpol``, ``vmec.indata.ntor``,
``vmec.indata.ns_array``, etc. 

Geometry parameters 
-------------------

The parameters in ``indata`` defining the boundary surface,
``rbc``, ``rbs``, ``zbc``, and ``zbs`` from the
``indata`` attribute are always ignored, and these arrays are
instead taken from the simsopt surface object associated to the
``boundary`` attribute. If ``boundary`` is a surface based on some
other representation than VMEC's Fourier representation, the
surface will automatically be converted to VMEC's representation
(:obj:`~simsopt.geo.surfacerzfourier.SurfaceRZFourier`) before
each run of VMEC. You can replace ``boundary`` with a new surface
object, of any type that implements the conversion function
``to_RZFourier()``. If this is a fixed-boundary calculation, ``boundary`` describes the shape of the fixed outer boundary. 
In the case of a free-boundary calculation, it describes the shape of the initial guess for the boundary. 

If one runs VMEC directly outside of the ``simsopt`` interface, the boundary surface parameters are interpreted as follows: 
``rbc(n,m)``, ``zbs(n,m)``, ``rbs(n,m)``, ``zbc(n,m)`` are the Fourier coefficients describing the boundary surface, 
where :math:`n` is a toroidal mode number and :math:`m` is a poloidal mode number. 
These correspond with the ``rc(m,n)``, ``zs(m,n)``, ``rs(m,n)``, ``zc(m,n)`` coefficients in the :obj:`~simsopt.geo.surfacerzfourier.SurfaceRZFourier` representation. 

Other input parameters related to geometry are described below:

- ``lasym``: A Boolean determining whether stellarator asymmetric modes should be included in the equilibrium calculation. If ``false``, stellarator symmetry is assumed.
- ``nfp``: The number of field periods. 
- ``mpol`` and ``ntor``: The number of poloidal and toroidal mode numbers used to compute the equilibrium solution with a Fourier representation. These are to be intepreted in the same way as the attributes of the same name in :obj:`~simsopt.geo.surfacerzfourier.SurfaceRZFourier`. However, they need not have the same value as the corresponding attributes in :obj:`~simsopt.geo.surfacerzfourier.SurfaceRZFourier`.
- ``phiedge``: The value of the toroidal flux through the boundary magnetic surface. This sets the overall scale of the equilibrium magnetic field.
- ``raxis`` and ``zaxis``: Fourier modes describing the initial guess for the magnetic axis curve as a function of the cylindrical toroidal angle, :math:`R_a(\phi)`, :math:`Z_a(\phi)`, with :math:`R_a(\phi) = \sum_n \mathrm{raxis}(n) \cos(-n \mathrm{nfp}\phi)` and :math:`Z_a(\phi) = \sum_n \mathrm{zaxis}(n) \sin(-n \mathrm{nfp}\phi)`.

Profile parameters 
------------------

To run VMEC, two input profiles must be specified: pressure and
either iota or toroidal current.  Each of these profiles can be
specified in several ways. One way is to specify the profile in
the input file used to initialize the :obj:`~simsopt.mhd.vmec.Vmec` object. For
instance, the pressure profile is determined by the variables
``pmass_type``, ``am``, ``am_aux_s``, and ``am_aux_f``, described below. You can
also modify these variables from python via the ``indata``
attribute, e.g. ``vmec.indata.am = [1.0e5, -1.0e5]``. 

Another option is to assign a :obj:`simsopt.mhd.profiles.Profile` object
to the attributes ``pressure_profile``, ``current_profile``, or
``iota_profile``. This approach allows for the profiles to be
optimized, and it allows you to use profile shapes defined in
python that are not available in the fortran VMEC code. To explain
this approach we focus here on the pressure profile; the iota and
current profiles are analogous. If the ``pressure_profile``
attribute of a :obj:`~simsopt.mhd.vmec.Vmec` object is ``None`` (the default), then a
simsopt :obj:`~simsopt.mhd.profiles.Profile` object is not used,
and instead the settings from ``Vmec.indata`` (initialized from
the input file) are used. If a
:obj:`~simsopt.mhd.profiles.Profile` object is assigned to the
``pressure_profile`` attribute, then an :ref:`edge in the dependency graph <dependecies>` is introduced, so the :obj:`~simsopt.mhd.vmec.Vmec`
object then depends on the dofs of the
:obj:`~simsopt.mhd.profiles.Profile` object. Whenever VMEC is run,
the simsopt :obj:`~simsopt.mhd.profiles.Profile` is converted to
either a polynomial (power series) or cubic spline in the
normalized toroidal flux :math:`s`, depending on whether
``indata.pmass_type`` is ``"power_series"`` or
``"cubic_spline"``. (The current profile is different in that
either ``"cubic_spline_ip"`` or ``"cubic_spline_i"`` is specified
instead of ``"cubic_spline"``, where ``cubic_spline_ip`` sets :math:`I'(s)` while ``cubic_spline_i`` sets :math:`I(s)`.) The number of terms in the power
series or number of spline nodes is determined by the attributes
``n_pressure``, ``n_current``, and ``n_iota``.  If a cubic spline
is used, the spline nodes are uniformly spaced from :math:`s=0` to
1. Note that the choice of whether a polynomial or spline is used
for the VMEC calculation is independent of the subclass of
:obj:`~simsopt.mhd.profiles.Profile` used. Also, whether the iota
or current profile is used is always determined by the
``indata.ncurr`` attribute: 0 for iota, 1 for current. Example::

    from sismopt.mhd.profiles import ProfilePolynomial, ProfileSpline, ProfilePressure, ProfileScaled
    from simsopt.util.constants import ELEMENTARY_CHARGE
    import numpy as np
    from simsopt.mhd import Vmec

    ne = ProfilePolynomial(1.0e20 * np.array([1, 0, 0, 0, -0.9]))
    Te = ProfilePolynomial(8.0e3 * np.array([1, -0.9]))
    Ti = ProfileSpline([0, 0.5, 0.8, 1], 7.0e3 * np.array([1, 0.9, 0.8, 0.1]))
    ni = ne
    pressure = ProfilePressure(ne, Te, ni, Ti)  # p = ne * Te + ni * Ti
    pressure_Pa = ProfileScaled(pressure, ELEMENTARY_CHARGE)  # Te and Ti profiles were in eV, so convert to SI here.
    vmec = Vmec(filename)
    vmec.pressure_profile = pressure_Pa
    vmec.indata.pmass_type = "cubic_spline"
    vmec.n_pressure = 8  # Use 8 spline nodes

When a current profile is used, ``VMEC``  automatically updates ``curtor`` so that the total toroidal current :math:`I(s=1)` matches that of the specified profile.

VMEC input parameters related to the pressure and current profiles are described below:

- ``ncurr``: An integer determining whether the equilibrium calculation is performed at fixed rotational transform profile (``ncurr=0``) or at fixed toroidal current profile (``ncurr=1``). The rotational transform and current are specified using the ``ai_*`` and ``ac_*`` input parameters, respectively. 
- ``pcurr_type``: A string specifying the type of current profile. The most commonly used options are ``power_series``, ``power_series_i``, ``akima_spline_i``, ``akima_spline_ip``, ``cubic_spline_i``, ``cubic_spline_ip``, ``line_segment_i``, and ``line_segment_ip``. Other options are described in the `VMEC++ documentation <https://arxiv.org/pdf/2502.04374>`_. Options ending in ``_ip`` specify the derivative of the toroidal current profile with respect to the normalized toroidal flux, :math:`s`, the "I-prime" profile, while options ending in ``_i`` specify the toroidal current profile. Power series options are specified by the ``ac`` input parameters, while the others are described by the ``ac_aux_s`` and ``ac_aux_f`` input parameters. 
- ``ac``: A polynomial description of the integrated toroidal current profile with respect to the normalized toroidal flux, :math:`s`, :math:`I_T(s) = \sum_{i=0}^{N} \mathrm{ac}(i) s^{i-1}` if ``pcurr_type`` is ``power_series_i``, or :math:`I_T'(s) = \sum_{i=0}^{N} \mathrm{ac}(i) s^{i-1}` if ``pcurr_type`` is ``power_series_ip``. If ``ncurr`` is 0 or if another ``pcurr_type`` is specified, this input is ignored. 
- ``ac_aux_s`` and ``ac_aux_f``: These inputs are used to specify the current profile as a spline or line segment. The ``ac_aux_s`` specifies values of the normalized toroidal flux, :math:`s`, while ``ac_aux_f`` specifies the corresponding values of the current, :math:`I_T(s)`, or derivative of the current, :math:`I_T'(s)`, depending on ``pcurr_type``. The length of these two input arrays should be the same. 
- ``piota_type``: A string specifying the type of rotational transform profile. The most commonly used options are ``power_series``, ``akima_spline``, ``cubic_spline``, and ``line_segment``. Other options are described in the `VMEC++ documentation <https://arxiv.org/pdf/2502.04374>`_. With ``power_series``, the profile is specified by the ``ai`` input parameters, while the others are described by the ``ai_aux_s`` and ``ai_aux_f`` input arrays.
- ``ai``: A polynomial description of the rotational transform profile with respect to the normalized toroidal flux, :math:`s`, :math:`\iota(s) = \sum_{i=0}^{N} \mathrm{ai}(i) s^{i-1}`. This input is only used if ``ncurr`` is 0 and ``piota_type`` is ``power_series``. 
- ``ai_aux_s`` and ``ai_aux_f``: These inputs are used to specify the rotational transform profile as a spline or line segment. The ``ai_aux_s`` specifies values of the normalized toroidal flux, :math:`s`, while ``ai_aux_f`` specifies the corresponding values of the rotational transform, :math:`\iota(s)`. The length of these two input arrays should be the same.
- ``gamma``: The adiabatic index (ratio of specific heats). If 0 (default), the ``am_*`` input parameters specify the pressure profile. Otherwise, these specify the mass profile. We recommend using the default value of 0.
- ``pmass_type``: A string specifying the type of pressure profile. The most commonly used options are ``power_series``, ``akima_spline``, ``cubic_spline``, and ``line_segment``. Other options are described in the `VMEC++ documentation <https://arxiv.org/pdf/2502.04374>`_. With ``power_series``, the profile is specified by the ``am`` input parameters, while the others are described by the ``am_aux_s`` and ``am_aux_f`` input arrays.
- ``am``: A polynomial description of the pressure with respect to :math:`s`, :math:`p(s) = \sum_{i} \mathrm{am}(i) s^{i-1}`. This input is only used if ``pmass_type`` is ``power_series``.
- ``am_aux_s`` and ``am_aux_f``: These inputs are used to specify the pressure profile as a spline or line segment. The ``am_aux_s`` specifies values of the normalized toroidal flux, :math:`s`, while ``am_aux_f`` specifies the corresponding values of the pressure, :math:`p(s)`. The length of these two input arrays should be the same.
- ``pres_scale``: A scale factor applied to the pressure profile :math:`p(s)` to modify the amplitude of the pressure profile. 

Resolution parameters 
---------------------

The VMEC solution is computed with a multistage method, beginning with a small number of surfaces and increasing until the desired resolution is achieved. The stages are described by ``ns_array``, ``ftol_array``, and ``niter_array``. 
As an example, here are the parameters used in ``tests/test_files/input.LandremanPaul2021_QA_lowres``::  

  NS_ARRAY    =      16       50 75
  NITER_ARRAY =     600     3000 3000
  FTOL_ARRAY  = 1.0E-16  1.0E-11 1.0E-13

Here, the equilibrium solve proceeds in three stages: 

- First with 16 surfaces until 600 iteractions or a total force residual of :math:`10^{-16}` is achieved,
- Second with 50 surfaces until 3000 iterations or a total force residual of :math:`10^{-11}` is achieved, and
- Finally with 75 surfaces until 3000 iterations or a total force residual of :math:`10^{-13}` is achieved. If this desired force residual is not achieved, an ``ObjectiveError`` will be raised, as discussed in :ref:`Interpreting VMEC errors <interp>`.  

The other resolution parameters specified in ``tests/test_files/input.LandremanPaul2021_QA_lowres`` include::

  NSTEP =  200
  DELT =   9.00E-01

Here the input resolution parameters are described in further detail: 

- ``ns_array``: An array of the number of radial gridpoints to use during each iteration of the calculation. Each element defines the number of magnetic surfaces to include in the calculation at each stage. In order to achieve convergence, it is typically necessary to begin with a small number of surfaces (10-20) and increase to your desired resolution (typically 75-150 is sufficient) in increments of 20-40.
- ``ftol_array``: An array defining the tolerances in the force residual used at each grid level. This should have the same number of elements as ``ns_array``. Typically the finest grid should have a value of :math:`10^{-11}-10^{-15}`. The coarse grids can have larger tolerances. The VMEC calculation is performed by minimizing an energy functional until this normalized tolerance in the force residual is achieved.
- ``niter_array``: The maximum number of iterations to use at each iteration of the calculation. This array should be of the same size as ``ftol_array`` and ``ns_array``. If the number of iterations exceeds ``niter`` during the finest grid evaluation, the code will exit with an error. If it exceeds ``niter`` during the coarser grid evaluations, the calculation will proceed to the next grid size defined by the next element of ``ns_array``. Typical values at the finest grid are 3000-5000, while the coarser grids can sometimes have smaller values (e.g., 500-1000). 
- ``nstep``: The number of iterations between output of the force residual as the energy is minimized.
- ``delt``: This parameter controls the step size in the minimization of the energy functional. Typical values are the range 0.2-0.9. This control parameter should not be changed unless one is having difficulty obtaining convergence. 
- ``ntheta``: The number of poloidal grid points for evaluation in real space. This defaults to :math:`2 \times \mathrm{mpol} + 6`. 
- ``nzeta``: The number of toroidal grid points for evaluation in real space. This defaults to :math:`2 \times \mathrm{ntor} + 4`. In the context of a free-boundary calculation, the ``mgrid`` resolution parameter ``nphi`` should be an integer multiple of ``nzeta``. 

Free-boundary parameters
------------------------

- ``lfreeb``: A Boolean determining whether the calculation is performed in free-boundary mode. If ``true``, the VMEC calculation will be performed with a free boundary. If ``false``, the VMEC calculation will be performed with a fixed boundary.
- ``mgrid_file``: The name of the MGRID netcdf file. This can be produced with :obj:`~simsopt.field.mgrid.MGrid` or the ``xgrid`` executable in `STELLOPT <https://github.com/PrincetonUniversity/STELLOPT/tree/develop/MAKEGRID>`_ executable. 
- ``extcur``: An array of coil currents used to specify the external magnetic field. This scales up the magnetic field described in the MGRID file. If the MGRID file is produced with simsopt, then this should be set to 1.0. If the MGRID file is produced with STELLOPT, typically this should be set to the entries in the ``extcur.<extension>`` produced by ``xgrid``. 
- ``nvacskip``: The number of iterations to skip without iteration with the vacuum field. Defaults to 1. Sometimes increasing this number can help with convergence. Typical values are between 1 and 10. 

As an example, the free-boundary parameters used in ``examples/2_Intermediate/free_boundary_vmec.py`` are::

  LFREEB = T
  MGRID_FILE = 'mgrid.w7x.nc'
  EXT_CUR = 1.0

Running VMEC 
^^^^^^^^^^^^

VMEC is run either when the :meth:`~simsopt.mhd.vmec.Vmec.run()` function is called, or when any of the output functions like :meth:`~simsopt.mhd.vmec.Vmec.aspect()` or :meth:`~simsopt.mhd.vmec.Vmec.iota_axis()` are called.

When VMEC is run multiple times, the default behavior is that all
``wout`` output files will be deleted except for the first and
most recent iteration on worker group 0. If you wish to keep all
the ``wout`` files, you can set ``keep_all_files = True``. If you
want to save the ``wout`` file for a certain intermediate
iteration, you can set the ``files_to_delete`` attribute to ``[]``
after that run of VMEC.

A caching mechanism is implemented, using the attribute
``need_to_run_code``. Whenever VMEC is run, or if the class is
initialized from a ``wout`` file, this attribute is set to
``False``. Subsequent calls to :meth:`run()` or output functions
like :meth:`aspect()` will not actually run VMEC again, until
``need_to_run_code`` is changed to ``True``. The attribute
``need_to_run_code`` is automatically set to ``True`` whenever the
state vector ``.x`` is changed, and when dofs of the ``boundary``
are changed. However, ``need_to_run_code`` is not automatically
set to ``True`` when entries of ``indata`` are modified.

Once VMEC has run at least once, or if the class is initialized
from a ``wout`` file, all of the quantities in the ``wout`` output
file are available as attributes of the ``wout`` attribute.  For
example, if ``vmec`` is an instance of :obj:`~simsopt.mhd.vmec.Vmec`, then the flux
surface shapes can be obtained from ``vmec.wout.rmnc`` and
``vmec.wout.zmns``.

.. warning::
    Since the underlying fortran implementation of VMEC uses global
    module variables, it is not possible to have more than one python
    :obj:`~simsopt.mhd.vmec.Vmec` object with different parameters; changing the parameters of
    one would change the parameters of the other.

VMEC outputs are saved on the so-called half and full grids. The full grid is linearly spaced from :math:`s=0` to 1 (including both endpoints), while the points on the half grid are located halfway between adjacent points on the full grid. Using the ``ncdump`` command, one can see which outputs are saved on the full and half grid::

  ncdump -h wout_<extension>.nc

  ...

  double rmnc(radius, mn_mode) ;
    rmnc:long_name = "cosmn component of cylindrical R, full mesh" ;
    rmnc:units = "m" ;
  double zmns(radius, mn_mode) ;
    zmns:long_name = "sinmn component of cylindrical Z, full mesh" ;
    zmns:units = "m" ;
  double lmns(radius, mn_mode) ;
    lmns:long_name = "sinmn component of lambda, half mesh" ;
  double gmnc(radius, mn_mode_nyq) ;
    gmnc:long_name = "cosmn component of jacobian, half mesh" ;
  double bmnc(radius, mn_mode_nyq) ;
    bmnc:long_name = "cosmn component of mod-B, half mesh" ;
  double bsubumnc(radius, mn_mode_nyq) ;
    bsubumnc:long_name = "cosmn covariant u-component of B, half mesh" ;
  double bsubvmnc(radius, mn_mode_nyq) ;
    bsubvmnc:long_name = "cosmn covariant v-component of B, half mesh" ;
  double bsubsmns(radius, mn_mode_nyq) ;
    bsubsmns:long_name = "sinmn covariant s-component of B, half mesh" ;
  double currumnc(radius, mn_mode_nyq) ;
    currumnc:long_name = "cosmn covariant u-component of J, full mesh" ;
  double currvmnc(radius, mn_mode_nyq) ;
    currvmnc:long_name = "cosmn covariant v-component of J, full mesh" ;

.. _interp:

Interpreting VMEC errors 
------------------------

VMEC can produce a variety of errors. Sometimes they can be circumvented by modifying resolution parameters, and sometimes they are triggered from the input geometry. Here we discuss some interpretation of typical VMEC errors. 

The standard output (visible if ``vmec.verbose=True``) can give insight into how the convergence is progressing. You will see blocks of text for each value of ``ns`` in ``ns_array``. Here is an example the force residual is seen to be decreasing with the number of iterations::
    
    NS =  100 NO. FOURIER MODES =   13 FTOLV =  1.000E-12 NITER =   4000
    PROCESSOR COUNT - RADIAL:    1  VACUUM:    1

    ITER    FSQR      FSQZ      FSQL    RAX(v=0)    DELT        WMHD      DEL-BSQ

    1  5.32E+00  1.63E+00  3.61E-07  3.333E-01  9.00E-01  5.2233E-03  7.055E-02
    200  4.25E-08  2.99E-08  1.76E-11  3.332E-01  9.00E-01  5.2231E-03  6.967E-02
    400  9.96E-10  9.57E-10  5.67E-13  3.331E-01  9.00E-01  5.2231E-03  6.983E-02
    600  1.83E-11  2.05E-11  5.26E-14  3.330E-01  9.00E-01  5.2231E-03  6.994E-02
    766  9.80E-13  6.57E-13  7.64E-15  3.330E-01  9.00E-01  5.2231E-03  6.998E-02

    EXECUTION TERMINATED NORMALLY

The columns of the output data are interpreted as follows:

- Columns 2-4 provide the force residuals corresponding to :math:`R`, :math:`Z`, and :math:`\lambda`. When each residual falls below ``ftol``, the executation terminates successfully.
- Column 5 provides the major radius of the magnetic axis at :math:`\phi=0`. 
- Column 6 provides the current value of the step size. 
- Column 7 dispays the current value of the energy functional, :math:`W`. 
- In the case of a free-boundary calculation, Column 8 provides the normalized jump in the total pressure across the plasma boundary. 

On the other hand, the following output indicates that the VMEC calculation has not converged::
  
  NS =   25 NO. FOURIER MODES =  116 FTOLV =  1.000E-10 NITER =   2000
  PROCESSOR COUNT - RADIAL:    1

  ITER    FSQR      FSQZ      FSQL    RAX(v=0)    DELT       WMHD

    1  1.10E-01  3.33E-02  4.22E-04  2.775E-01  9.00E-01  1.3020E-02
  200  1.55E-02  3.09E-05  1.68E-04  2.774E-01  9.00E-01  1.3019E-02
  400  1.84E-02  3.19E-05  1.46E-04  2.773E-01  9.00E-01  1.3019E-02
  600  8.98E-03  1.64E-05  9.26E-05  2.773E-01  9.00E-01  1.3019E-02
  800  9.16E-03  1.64E-05  8.39E-05  2.773E-01  9.00E-01  1.3019E-02
 1000  7.44E-03  1.37E-05  7.26E-05  2.773E-01  9.00E-01  1.3019E-02
 1200  6.41E-03  1.22E-05  6.86E-05  2.773E-01  9.00E-01  1.3019E-02
 1400  6.03E-03  1.20E-05  7.04E-05  2.773E-01  9.00E-01  1.3019E-02
 1600  8.37E-03  1.70E-05  9.62E-05  2.772E-01  9.00E-01  1.3019E-02
 1800  9.33E-03  1.88E-05  1.05E-04  2.772E-01  9.00E-01  1.3019E-02
 2000  9.96E-03  2.01E-05  1.10E-04  2.772E-01  9.00E-01  1.3019E-02

  NS =  100 NO. FOURIER MODES =  116 FTOLV =  1.000E-12 NITER =   4000
  PROCESSOR COUNT - RADIAL:    1

  ITER    FSQR      FSQZ      FSQL    RAX(v=0)    DELT       WMHD

    1  3.93E-01  1.50E-01  2.53E-04  2.772E-01  9.00E-01  1.3018E-02
  200  9.79E-07  5.70E-08  9.84E-10  2.774E-01  6.87E-01  1.3018E-02
  400  3.58E-08  2.76E-09  4.35E-11  2.772E-01  6.87E-01  1.3018E-02
  600  5.12E-09  4.91E-10  3.09E-12  2.772E-01  6.87E-01  1.3018E-02
  800  1.27E-09  1.36E-10  4.77E-13  2.772E-01  6.87E-01  1.3018E-02
 1000  4.85E-10  2.98E-11  2.01E-13  2.772E-01  6.87E-01  1.3018E-02
 1200  3.55E-10  1.25E-11  8.46E-14  2.772E-01  6.87E-01  1.3018E-02
 1400  3.50E-10  1.01E-11  7.16E-14  2.772E-01  6.87E-01  1.3018E-02
 1600  3.28E-10  8.89E-12  6.47E-14  2.772E-01  6.87E-01  1.3018E-02
 1800  2.93E-10  7.27E-12  5.59E-14  2.772E-01  6.87E-01  1.3018E-02
 2000  2.67E-10  6.34E-12  5.10E-14  2.771E-01  6.87E-01  1.3018E-02
 2200  2.52E-10  5.93E-12  4.96E-14  2.771E-01  6.87E-01  1.3018E-02
 2400  2.45E-10  5.73E-12  4.77E-14  2.771E-01  6.87E-01  1.3018E-02
 2600  2.45E-10  5.67E-12  4.76E-14  2.771E-01  6.87E-01  1.3018E-02
 2800  2.50E-10  5.72E-12  4.85E-14  2.771E-01  6.87E-01  1.3018E-02
 3000  2.59E-10  5.88E-12  5.01E-14  2.771E-01  6.87E-01  1.3018E-02
 3200  2.74E-10  6.17E-12  5.25E-14  2.771E-01  6.87E-01  1.3018E-02
 3400  2.95E-10  6.57E-12  5.58E-14  2.771E-01  6.87E-01  1.3018E-02
 3600  3.21E-10  7.09E-12  6.01E-14  2.771E-01  6.87E-01  1.3018E-02
 3800  3.54E-10  7.73E-12  6.52E-14  2.771E-01  6.87E-01  1.3018E-02
 4000  3.93E-10  8.49E-12  7.12E-14  2.770E-01  6.87E-01  1.3018E-02

    simsopt._core.util.ObjectiveFailure: VMEC did not converge. ierr=2

An ``ObjectiveError`` is raised if VMEC fails, and the ``ierr`` code indicates the type of error encountered on the Fortran side.
The interpretation of the error code is listed below:

- `norm_term_flag=0`: Normal termination. 
- `bad_jacobian_flag=1`: An initial guess for the toroidal coordinates is constructed from the magnetic axis guess and boundary shape, resulting in an ill-defined Jacobian. 
- `more_iter_flag=2`: More iterations are required. 
- `jac75_flag=4`: An ill-defined Jacobian was detected 75 times (decrease ``delt``). 
- `input_error_flag=5`: Input file is not properly formatted.  
- `phiedge_error_flag=7`: ``phiedge`` has the wrong sign in vacuum subroutine (only for free boundary). 
- `ns_error_flag=8`: ``ns_array`` must not all be zeros.  
- `misc_error_flag=9`: Error reading ``mgrid`` file.  
- `successful_term_flag=11`: Normal termination. 
- `bsub_bad_js1_flag=12`: bsubu or bsubv js=1 component (on-axis covariant magnetic field components) are non-zero. 
- `r01_bad_value_flag=13`: rmnc(0,1) is zero. Check that this component is non-zero in the input file. 
- `arz_bad_value_flag=14`: arnorm (average of :math:`|dr/d\theta|^2`) or aznorm (average of :math:`|dr/d\phi|^2`) equals zero. 

In the above example, we can see the force minimization is not successful, as the maximum number of iterations, 4000, was exceeded before ``ftol`` could be achieved. 
There are a few ways to proceed. ``niter`` can be increased to allow the force minimization to converge. We can also adjust the staging parameters.
In this case, the staging parameters are specified as follows::

    NS_ARRAY    = 5      12        25     100
    NITER_ARRAY = 2000     2000    2000   4000  
    FTOL_ARRAY  = 1e-8 1.00E-08  1.00E-10 1e-12 

Since the force residual did not get very small in the ``ns=25`` stage, in this case VMEC will converge by adding an intermediate stage with ``ns=50``::

    NS_ARRAY    = 5      12        25     50     100
    NITER_ARRAY = 2000     2000    2000   2000   4000  
    FTOL_ARRAY  = 1e-8 1.00E-08  1.00E-10 1e-10 1e-12

Sometimes adjusting ``delt`` (in either direction) can also help convergence. 

Often if ``mpol`` and ``ntor`` get too large and in very strongly shaped geometries, it becomes challenging to converge. 