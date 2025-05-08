Running VMEC 
============

In this tutorial, we describe how to run the VMEC equilibrium code. Other resources are the `STELLOPT wiki <https://princetonuniversity.github.io/STELLOPT/VMEC>`_ and the `VMEC++ documentation <https://arxiv.org/pdf/2502.04374>`_.

Input parameters
^^^^^^^^^^^^^^^^

Typical input parameters are described below. A more comprehensive list can be found in the `VMEC++ documentation <https://arxiv.org/pdf/2502.04374>`_. These parameters can be directly modified in the ``input.*`` Fortran namelist, or by modifying the corresponding attribute of a :obj:`~simsopt.mhd.vmec.Vmec` object, e.g.::
   
    vmec.indata.nfp = 5

All quantities are in SI units. 

Geometry parameters 
-------------------

- ``LASYM``: a Boolean determining whether stellarator asymmetric modes should be included in the equilibrium calculation. If ``false``, stellarator symmetry is assumed.
- ``RBC(n,m)`` and ``ZBS(n,m)``: the Fourier coefficients describing the boundary surface, :math:`R_{m,n}^c` and :math:`Z_{m,n}^s`. Here :math:`n` is a toroidal mode number and :math:`m` is a poloidal mode number. These correspond with the ``rc(m,n)`` and ``zs(m,n)`` coefficients in the :obj:`~simsopt.geo.surfacerzfourier.SurfaceRZFourier` representation. If this is a fixed-boundary calculation, these describe the shape of the fixed outer boundary. In the case of a free-boundary calculation, these describe the shape of the initial guess for the boundary. 
- ``NFP``: the number of field periods. This should be consistent with the representation in :obj:`~simsopt.geo.surfacerzfourier.SurfaceRZFourier`.
- ``MPOL`` and ``NTOR``: the number of poloidal and toroidal mode numbers used to compute the equilibrium solution with a Fourier discretization. These correspond with the attributes of the same name in :obj:`~simsopt.geo.surfacerzfourier.SurfaceRZFourier`.
- ``PHIEDGE``: the value of the toroidal flux in SI units through the boundary magnetic surface. This sets the overall scale of the equilibrium magnetic field.
- ``RAXIS`` and ``ZAXIS``: Fourier modes describing the initial guess for the magnetic axis curve as a function of the cylindrical toroidal angle, :math:`R_a(\phi)`, :math:`Z_a(\phi)`, :math:`R_a(\phi) = \sum_n \mathrm{RAXIS}(n) \cos(-n \mathrm{NFP}\phi)` and :math:`Z_a(\phi) = \sum_n \mathrm{ZAXIS}(n) \sin(-n \mathrm{NFP}\phi)`.

Profile parameters 
------------------

- ``NCURR``: an integer determining whether the equilibrium calculation is performed at fixed rotational transform profile (``NCURR=0``) or at fixed toroidal current profile (``NCURR=1``). The rotational transform and current are specified using the ``AI_*`` and ``AC_*`` input parameters, respectively. 
- ``PCURR_TYPE``: a string specifying the type of current profile. The most commonly used options are ``power_series``, ``power_series_i``, ``akima_spline_i``, ``akima_spline_ip``, ``cubic_spline_i``, ``cubic_spline_ip``, ``line_segment_i``, and ``line_segment_ip``. Other options are described in the `VMEC++ documentation <https://arxiv.org/pdf/2502.04374>`_. Options ending in ``_ip`` specify the derivative of the toroidal current profile with respect to the normalized toroidal flux, :math:`s`, the "I-prime" profile, while options ending in ``_i`` specify the toroidal current profile. Power series options are specified by the ``AC`` input parameters, while the others are described by the ``AC_AUX_S`` and ``AC_AUX_F`` input parameters. 
- ``AC``: a polynomial description of the integrated toroidal current profile with respect to the normalized toroidal flux, :math:`s`, :math:`I_T(s) = \sum_{i=0}^{N} \mathrm{AC}(i) s^{i-1}` if ``PCURR_TYPE`` is ``power_series_i``, or :math:`I_T'(s) = \sum_{i=0}^{N} \mathrm{AC}(i) s^{i-1}` if ``PCURR_TYPE`` is ``power_series_ip``. If ``NCURR`` is 0 or if another ``PCURR_TYPE`` is specified, this input is ignored. 
- ``AC_AUX_S`` and ``AC_AUX_F``: these inputs are used to specify the current profile as a spline or line segment. The ``AC_AUX_S``specifies values of the normalized toroidal flux, :math:`s`, while ``AC_AUX_F`` specifies the corresponding values of the current, :math:`I_T(s)`, or derivative of the current, :math:`I_T'(s)`, depending on ``PCURR_TYPE``. The length of these two input arrays should be the same. 
- ``PIOTA_TYPE``: a string specifying the type of rotational transform profile. The most commonly used options are ``power_series``, ``akima_spline``, ``cubic_spline``, and ``line_segment``. Other options are described in the `VMEC++ documentation <https://arxiv.org/pdf/2502.04374>`_. With ``power_series``, the profile is specified by the ``AI`` input parameters, while the others are described by the ``AI_AUX_S`` and ``AI_AUX_F`` input arrays.
- ``AI``: a polynomial description of the rotational transform profile with respect to the normalized toroidal flux, :math:`s`, :math:`\iota(s) = \sum_{i=0}^{N} \mathrm{AI}(i) s^{i-1}`. This input is only used if ``NCURR`` is 0 and ``PIOTA_TYPE`` is ``power_series``. 
- ``AI_AUX_S`` and ``AI_AUX_F``: these inputs are used to specify the rotational transform profile as a spline or line segment. The ``AI_AUX_S``specifies values of the normalized toroidal flux, :math:`s`, while ``AI_AUX_F`` specifies the corresponding values of the rotational transform, :math:`\iota(s)`. The length of these two input arrays should be the same.
- ``PMASS_TYPE``: a string specifying the type of pressure profile. The most commonly used options are ``power_series``, ``akima_spline``, ``cubic_spline``, and ``line_segment``. Other options are described in the `VMEC++ documentation <https://arxiv.org/pdf/2502.04374>`_. With ``power_series``, the profile is specified by the ``AM`` input parameters, while the others are described by the ``AM_AUX_S`` and ``AM_AUX_F`` input arrays.
- ``AM``: a polynomial description of the pressure with respect to :math:`s`, :math:`p(s) = \sum_{i} \mathrm{AM}(i) s^{i-1}`. This input is only used if ``PMASS_TYPE`` is ``power_series``.
- ``AM_AUX_S`` and ``AM_AUX_F``: these inputs are used to specify the pressure profile as a spline or line segment. The ``AM_AUX_S``specifies values of the normalized toroidal flux, :math:`s`, while ``AM_AUX_F`` specifies the corresponding values of the pressure, :math:`p(s)`. The length of these two input arrays should be the same.
- ``PRES_SCALE``: A scale factor applied to the pressure profile :math:`p(s)` to modify the amplitude of the pressure profile. 
- ``GAMMA``: the adiabatic index (ratio of specific heats). If 0 (default), the ``AM_`` input parameters specify the pressure profile. Otherwise, these specify the mass profile. We recommend using the default value of 0.

Resolution parameters 
---------------------

The VMEC solution is computed with a multistage method, beginning with a small number of surfaces and increasing until the desired resolution is achieved. The stages are described by ``NS_ARRAY``, ``FTOL_ARRAY``, and ``NITER_ARRAY``.

- ``NS_ARRAY``: an array of the number of radial gridpoints to use during each iteration of the calculation. Each element defines the number of magnetic surfaces to include in the calculation at each stage. In order to achieve convergence, it is typically necessary to begin with a small number of surfaces (10-20) and increase to your desired resolution (typically 75-150 is sufficient) in increments of 20-40.
- ``FTOL_ARRAY``: an array defining the tolerances in the force residual used at each grid level. This should have the same number of elements as ``NS_ARRAY``. Typically the finest grid should have a value of :math:`10^{-11}-10^{-15}`. The coarse grids can have larger tolerances. The VMEC calculation is performed by minimizing an energy functional until this normalized tolerance in the force residual is achieved.
- ``NITER_ARRAY``: The maximum number of iterations to use at each iteration of the calculation. This array should be of the same size as ``FTOL_ARRAY`` and ``NS_ARRAY``. If the number of iterations exceeds ``NITER`` during the finest grid evaluation, the code will exit with an error. If it exceeds ``NITER`` during the coarser grid evaluations, the calculation will proceed to the next grid size defined by the next element of ``NS_ARRAY``. Typical values at the finest grid are 3000-5000, while the coarser grids can sometimes have smaller values (e.g., 500-1000). 
- ``NSTEP``: the number of iterations between output of the force residual as the energy is minimized.
- ``DELT``: this parameter controls the step size in the minimization of the energy functional. Typical values are the range 0.2-0.9. This control parameter should not be changed unless one is having difficulty obtaining convergence. 
- ``NTHETA``: the number of poloidal grid points for evaluation in real space. This defaults to :math:`2 \times \mathrm{mpol} + 6`. 
- ``NZETA``: the number of toroidal grid points for evaluation in real space. This defaults to :math:`2 \times \mathrm{ntor} + 4`. In the context of a free-boundary calculation, the ``MGRID`` resolution parameter ``nphi`` should be an integer multiple of ``NZETA``. 

Free-boundary parameters
------------------------

- ``LFREEB``: a Boolean determining whether the calculation is performed in free-boundary mode. If ``true``, the VMEC calculation will be performed with a free boundary. If ``false``, the VMEC calculation will be performed with a fixed boundary.
- ``MGRID_FILE``: the name of the MGRID netcdf file. This can be produced with :obj:`~simsopt.field.mgrid.MGrid` or the ``xgrid`` executable in `STELLOPT <https://github.com/PrincetonUniversity/STELLOPT/tree/develop/MAKEGRID>`_ executable. 
- ``EXTCUR``: an array of coil currents used to specify the external magnetic field. This scales up the magnetic field described in the MGRID file. If the MGRID file is produced with simsopt, then this should be set to 1.0. If the MGRID file is produced with STELLOPT, typically this should be set to the entries in the ``extcur.*`` produced by ``xgrid``. 
- ``NVACSKIP``: the number of iterations to skip without iteration with the vacuum field. Defaults to 1. Sometimes increasing this number can help with convergence. Typical values are between 1 and 10. 

Interpreting VMEC errors 
^^^^^^^^^^^^^^^^^^^^^^^^

VMEC can produce a variety of errors. Sometimes they can be circumvented by modifying resolution parameters, and sometimes they are triggered from the input geometry. Here we discuss some interpretation of typical VMEC errors. 

The standard output can give insight into how the convergence is progressing. You will see blocks of text for each value of ``NS`` in ``NS_ARRAY``. Here is an example the force residual is seen to be decreasing with the number of iterations::
    
    NS =  100 NO. FOURIER MODES =   13 FTOLV =  1.000E-12 NITER =   4000
    PROCESSOR COUNT - RADIAL:    1  VACUUM:    1

    ITER    FSQR      FSQZ      FSQL    RAX(v=0)    DELT        WMHD      DEL-BSQ

    1  5.32E+00  1.63E+00  3.61E-07  3.333E-01  9.00E-01  5.2233E-03  7.055E-02
    200  4.25E-08  2.99E-08  1.76E-11  3.332E-01  9.00E-01  5.2231E-03  6.967E-02
    400  9.96E-10  9.57E-10  5.67E-13  3.331E-01  9.00E-01  5.2231E-03  6.983E-02
    600  1.83E-11  2.05E-11  5.26E-14  3.330E-01  9.00E-01  5.2231E-03  6.994E-02
    766  9.80E-13  6.57E-13  7.64E-15  3.330E-01  9.00E-01  5.2231E-03  6.998E-02

    EXECUTION TERMINATED NORMALLY

Columns 2-4 provide the force residuals corresponding to :math:`R`, :math:`Z`, and :math:`\lambda`. When the residuals fall below ``FTOL``, the executation terminates successfully::
  
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

Here we can see the force minimization is not successful, as the maximum number of iterations, 4000, was exceeded before ``FTOL`` could be achieved. There are a few ways to proceed. ``NITER`` can be increased to allow the force minimization to converge. In this case, the iteration parameters are specified as follows::

    NS_ARRAY    = 5      12        25     100
    NITER_ARRAY = 2000     2000    2000   4000  
    FTOL_ARRAY  = 1e-8 1.00E-08  1.00E-10 1e-12 

Since the force residual did not get very small in the ``NS=25`` stage, this error can be eliminated by adding an intermediate stage::

    NS_ARRAY    = 5      12        25     50     100
    NITER_ARRAY = 2000     2000    2000   2000   4000  
    FTOL_ARRAY  = 1e-8 1.00E-08  1.00E-10 1e-10 1e-12

Sometimes adjusting ``DELT`` (in either direction) can help convergence. 

Often if the mode number resolution gets too large, it becomes more challenging to converge. 
