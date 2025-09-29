import numpy as np
import warnings
from simsopt.geo.curverzfourier import CurveRZFourier
from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt.field.coil import Current
from simsopt.field.biotsavart import BiotSavart
from simsopt.field.coil import Coil 
from simsopt.field import coils_via_symmetries
from simsopt.geo.curve import RotatedCurve
from simsopt.geo.curvexyzfouriersymmetries import CurveXYZFourierSymmetries

from pathlib import Path
THIS_DIR = (Path(__file__).parent).resolve()

__all__ = [
    "get_data", 
    "configurations", 
    "get_ncsx_data",
    "get_hsx_data",
    "get_giuliani_data",
    "get_w7x_data",
]

configurations = ["ncsx", "hsx", "giuliani", "w7x", "lhd_like"]

def get_data(name, **kwargs):
    """
    Load one of several pre-defined stellarator coil configurations plus its
    Biot–Savart operator in a single call.

    Parameters
    ----------
    name : str
        Which configuration to load. Available values are:
        ``"ncsx"``, ``"hsx"``, ``"giuliani"``, ``"w7x"``, ``"lhd_like"``.
    kwargs : dict
        Configuration-specific parameters. See the sections below for details.
        
        .. note::
        
            The following keys are supported in ``kwargs`` based on the value
            of ``name``:

            **ncsx**

            -  ``coil_order`` *(int, default=25)*  
               Order of the curves representing the coils.
            -  ``points_per_period`` *(int, default=10)*  
               Points-per-period for quadrature.
            -  ``magnetic_axis_order`` *(int, default=10)*  
               Order of the curve representing the magnetic axis.

            **hsx**

            -  ``coil_order`` *(int, default=16)*
            -  ``points_per_period`` *(int, default=10)*
            -  ``magnetic_axis_order`` *(int, default=10)*

            **giuliani**

            -  ``coil_order`` *(int, default=16)*
            -  ``magnetic_axis_order`` *(int, default=10)*
            -  ``points_per_period`` *(int, default=10)*
            -  ``length`` *(int, default=18)*  
               Total length for the nine-stage optimization.
            -  ``nsurfaces`` *(int, default=5)*  
               Number of surfaces for the optimization.

            **w7x**

            -  ``coil_order`` *(int, default=48)*
            -  ``points_per_period`` *(int, default=2)*
            -  ``magnetic_axis_order`` *(int, default=10)*

            **lhd_like**

            -  ``numquadpoints_circular`` *(int, default=400)*  
               Number of quadrature points for the six circular coils.
            -  ``numquadpoints_helical`` *(int, default=1000)*  
               Number of quadrature points for each helical coil.
            -  ``numquadpoints_axis`` *(int, default=30)*  
               Number of quadrature points for the magnetic axis.


    Returns
    -------
    tuple
        *5-element tuple* ``(base_curves, base_currents, ma, nfp, bs)``, where:

        base_curves : list of :class:`CurveXYZFourier` 
            The curves representing the unique coils of the configuration (excluding symmetry copies). Fidelity and number of degrees-of-freedom are determined by ``coil_order`` and ``points_per_period``.
        base_currents : list of :class:`Current`
            Corresponding coil currents.
        ma : :class:`CurveRZFourier`
            The magnetic axis, of order ``magnetic_axis_order``.
        nfp : int
            Number of field periods.
        bs : :class:`BiotSavart`
            The Biot–Savart operator assembled from the complete physical coil set.
            
                .. note::
                    - For all configurations **except** ``"lhd_like"``, the coils are expanded via ``coils_via_symmetries(curves, currents, nfp, True)`` before building ``bs``.                    
                    - For ``"lhd_like"``, each ``CurveXYZFourier`` and its ``Current`` are passed directly into ``Coil(curve, current)`` (no symmetry expansion).

    Notes
    -----
    
    **Available configurations:**

    ``name="ncsx"``
        **Get a configuration that corresponds to the modular coils of the NCSX
        experiment (circular coils are not included).**

    ``name="hsx"``
        **Get a configuration that corresponds to the modular coils of the HSX
        experiment.**

    ``name="giuliani"``
       **This example simply loads the coils after the nine stage optimization runs discussed in**

            *A. Giuliani, F. Wechsung, M. Landreman, G. Stadler, A. Cerfon,
            “Direct computation of magnetic surfaces in Boozer coordinates
            and coil optimization for quasi-symmetry,” Journal of Plasma Phys.*

    ``name="w7x"`` 
        **Get the W7-X coils and magnetic axis.**

        Note that this function returns 7 coils: the 5 unique non-planar
        modular coils, and the 2 planar (A and B) coils. The coil currents
        correspond to the "Standard configuration", in which the planar A and 
        B coils carry no current. The shapes come from Fourier-transforming 
        the `xgrid` file `coils.w7x_v001`, provided by Joachim Geiger (Sept 27, 2022) in an
        email to Florian Wechsung and others,the Fourier modes here reproduce the Cartesian
        coordinate data from coils.w7x_v001 to **~ 1e-13 meters**. 
        
        **Some description from Joachim**:

            *"I have attached two coils-files which contain the filaments
            suitable for generating the mgrid-file for VMEC with xgrid. They
            contain the non-planar and the planar coils in a one-filament
            approximation as used here for equilibrium calculations.  The two
            coil-sets are slightly different with respect to the planar coils
            (the non-planar coil geometry is the same), the w7x_v001 being the
            CAD-coil-set while the w7x-set has slightly older planar coils
            which I had used in a large number of calculations. The difference
            is small, but, depending on what accuracy is needed, noticeable.
            In case you want only one coil-set, I would suggest to use the
            CAD-coils, i.e. w7x_v001, although the other coil-set had been
            used in the PPCF-paper for citation.  If there are any further
            questions, do not hesitate to contact me."*
        
    ``name="lhd_like"`` 
        **Get the coils and axis for an LHD-like configuration.**

        This coil set is a single-filament approximation of the coils in LHD, the
        Large Helical Device in Japan. Each filament corresponds to the center of
        the winding pack of the real finite-thickness LHD coils. The coil currents
        correspond to the configuration in which the major radius of the magnetic
        axis is 3.6 m.

        This configuration has 6 circular coils and 2 helical coils. In the
        lists of curves and currents, the order is OVU, OVL, ISU, ISL, IVU, IVL,
        helical1, helical2. Here, U and L indicate upper and lower, OV and IV indicate
        the outer and inner vertical field coils, and IS indicates the inner shaping
        coils.

        These coils were generated from data generously provided by Yasuhiro Suzuki.
        They produce a configuration similar to that used in Suzuki, Y., K. Y.
        Watanabe, and S. Sakakibara. "Theoretical studies of equilibrium beta limit
        in LHD plasmas." *Physics of Plasmas 27*, 10 (2020).
        
        **Special Note:** For ``"lhd_like"``, the returned ``nfp`` (5) reflects the coil periodicity,
        while the magnetic axis uses ``nfp=10``.

    """
    
    def add_default_args(kw_old, **kw_new):
        for k, v in kw_new.items():
            if k not in kw_old:
                kw_old[k] = v

    ma = None
    nfp = None # will be assigned in every branch
    cfg = name.lower()

    if cfg == "ncsx":
        """Get a configuration that corresponds to the modular coils of the NCSX experiment (circular coils are not included)."""
        add_default_args(kwargs, coil_order=25, magnetic_axis_order=10, points_per_period=10)
        coil_order = kwargs.pop("coil_order")
        magnetic_axis_order = kwargs.pop("magnetic_axis_order")
        points_per_period = kwargs.pop("points_per_period")
        filename = THIS_DIR / "NCSX.dat"
        base_curves = CurveXYZFourier.load_curves_from_file(filename, order=coil_order, ppp=points_per_period)
        nfp = 3
        
        base_currents = [Current(c) for c in [6.52271941985300E+05, 6.51868569367400E+05, 5.37743588647300E+05]]
        
        cR = [
            1.471415400740515, 0.1205306261840785, 0.008016125223436036, -0.000508473952304439,
            -0.0003025251710853062, -0.0001587936004797397, 3.223984137937924e-06, 3.524618949869718e-05,
            2.539719080181871e-06, -9.172247073731266e-06, -5.9091166854661e-06, -2.161311017656597e-06,
            -5.160802127332585e-07, -4.640848016990162e-08, 2.649427979914062e-08, 1.501510332041489e-08,
            3.537451979994735e-09, 3.086168230692632e-10, 2.188407398004411e-11, 5.175282424829675e-11,
            1.280947310028369e-11, -1.726293760717645e-11, -1.696747733634374e-11, -7.139212832019126e-12,
            -1.057727690156884e-12, 5.253991686160475e-13]
        sZ = [
            0.06191774986623827, 0.003997436991295509, -0.0001973128955021696, -0.0001892615088404824,
            -2.754694372995494e-05, -1.106933185883972e-05, 9.313743937823742e-06, 9.402864564707521e-06,
            2.353424962024579e-06, -1.910411249403388e-07, -3.699572817752344e-07, -1.691375323357308e-07,
            -5.082041581362814e-08, -8.14564855367364e-09, 1.410153957667715e-09, 1.23357552926813e-09,
            2.484591855376312e-10, -3.803223187770488e-11, -2.909708414424068e-11, -2.009192074867161e-12,
            1.775324360447656e-12, -7.152058893039603e-13, -1.311461207101523e-12, -6.141224681566193e-13,
            -6.897549209312209e-14]

    elif cfg == "hsx":
        """Get a configuration that corresponds to the modular coils of the HSX experiment."""
        add_default_args(kwargs, coil_order=16, magnetic_axis_order=10, points_per_period=10)
        coil_order = kwargs.pop("coil_order")
        magnetic_axis_order = kwargs.pop("magnetic_axis_order")
        points_per_period  = kwargs.pop("points_per_period")
        filename = THIS_DIR / "HSX.dat"
        base_curves = CurveXYZFourier.load_curves_from_file(filename, order=coil_order, ppp=points_per_period)
        nfp = 4
        base_currents = [Current(c) for c in [-1.500725500000000e+05] * 6]
        cR = [1.221168734647426701e+00, 2.069298947130969735e-01, 1.819037041932574511e-02, 4.787659822787012774e-05,
            -3.394778038757981920e-05, 4.051690884402789139e-05, 1.066865447680375597e-05, -1.418831703321225589e-05,
            2.041664078576817539e-05, 2.407340923216046553e-05, -1.281275289727263035e-05, -2.712941403326357315e-05,
            1.828622086757983125e-06, 1.945955315401206440e-05, 1.409134021563425399e-05, 4.572199318143535127e-06,
            3.136573559452139703e-07, -3.918158977823491866e-07, -2.204187636324686728e-07, -4.532041599796651056e-08,
            2.878479243210971143e-08, 2.102768080992785704e-08, -1.267816940685333911e-08, -2.268541399245120326e-08,
            -8.015316098897114283e-09, 6.401201778979550964e-09]
        sZ = [1.670393448410154857e-01, 1.638250511845155272e-02, 1.656424673977177490e-04, -1.506417857585283353e-04,
            8.367238367133577161e-05, -1.386982370447437845e-05, -7.536154112897463947e-06, -1.533108076767641072e-05,
            -9.966838351213697000e-06, 2.561158318745738406e-05, -1.212668371257951164e-06, -1.476513099369021112e-05,
            -3.716380502156798402e-06, 3.381104573944970371e-06, 2.605458694352088474e-06, 5.701177408478323677e-07,
            -1.056254779440627595e-07, -1.112799365280694501e-07, -5.381768314066269919e-08, -1.484193645281248712e-08,
            1.160936870766209295e-08, 1.466392841646290274e-08, 1.531935984912975004e-09, -6.857347022910395347e-09,
            -4.082678667917087128e-09]

    elif cfg == "giuliani":
        """This example simply loads the coils after the nine‐stage optimization runs discussed in Giuliani et al., J. Plasma Phys."""
        add_default_args(kwargs, coil_order=16, magnetic_axis_order=10, points_per_period=10, length=18, nsurfaces=5)
        coil_order, magnetic_axis_order, points_per_period = kwargs.pop("coil_order"), kwargs.pop("magnetic_axis_order"), kwargs.pop("points_per_period")
        length, nsurfaces = kwargs.pop("length"), kwargs.pop("nsurfaces")
        filename = THIS_DIR / f"GIULIANI_length{length}_nsurfaces{nsurfaces}"
        base_curves   = CurveXYZFourier.load_curves_from_file(filename.with_suffix(".curves"),
                                                         order=coil_order, ppp=points_per_period)
        base_currents = [Current(c) for c in np.loadtxt(filename.with_suffix(".currents"))]
        dofs = np.loadtxt(filename.with_suffix(".ma"))
        cR = dofs[:26]
        sZ = dofs[26:]
        nfp = 2

    elif cfg == "w7x":
        """Get the W7-X coils and magnetic axis."""
        add_default_args(kwargs, coil_order=48, magnetic_axis_order=10, points_per_period=2)
        coil_order, magnetic_axis_order, points_per_period = kwargs.pop("coil_order"), kwargs.pop("magnetic_axis_order"), kwargs.pop("points_per_period")
        filename = THIS_DIR / "W7-X.dat"
        base_curves = CurveXYZFourier.load_curves_from_file(filename, order=coil_order, ppp=points_per_period)
        nfp = 5
        turns = 108
        base_currents = [Current(15000.0 * turns)] * 5 + [Current(0.0), Current(0.0)]
        cR = [
            5.56069066955626, 0.370739830964738, 0.0161526928867275,
            0.0011820724983052, 3.43773868380292e-06, -4.71423775536881e-05,
            -6.23133271265022e-05, 2.06622580616597e-06, -0.000113256675159501,
            5.0296895894932e-07, 7.02308687052046e-05, 5.13479338167885e-05,
            1.90731007856885e-05
        ]
        sZ = [
            +0.308156954586225,    +0.0186374002410851,
            +0.000261743895528833, -5.78207516751575e-05, +0.000129121205314107,
            -3.08849630026052e-05, -1.95450172782866e-05, +8.32136392792337e-05,
            -6.19785500011441e-05, -1.36521157246782e-05, +1.46281683516623e-05,
            +1.5142136543872e-06
        ]

    elif cfg == "lhd_like":
        """Return the coils and axis for an LHD‐like configuration."""
        add_default_args(kwargs,
            numquadpoints_circular=400,
            numquadpoints_helical=1000,
            numquadpoints_axis=30,
        )
        nq_circ = kwargs.pop("numquadpoints_circular")
        nq_hel  = kwargs.pop("numquadpoints_helical")
        nq_ax   = kwargs.pop("numquadpoints_axis")

        # LHD‐like parameters
        nfp = 5  # Even though LHD has nfp=10 overall, each helical coil by itself has nfp=5.
        order_circ = 1
        order_hel = 6
        stellsym = True
        ntor = 1  # Number of toroidal turns for each helical coil to bite its tail
        base_curves = [
            CurveXYZFourier(nq_circ, order_circ),
            CurveXYZFourier(nq_circ, order_circ),
            CurveXYZFourier(nq_circ, order_circ),
            CurveXYZFourier(nq_circ, order_circ),
            CurveXYZFourier(nq_circ, order_circ),
            CurveXYZFourier(nq_circ, order_circ),
            CurveXYZFourierSymmetries(nq_hel, order_hel, nfp, stellsym, ntor),
        ]
        base_curves.append(RotatedCurve(base_curves[-1], phi=np.pi/5, flip=False))

        # Set the shape of the first pair of circular coils, OVU and OVL:
        R = 5.55
        Z = 1.55
        base_curves[0].x = [0, 0, R, 0, R, 0, +Z, 0, 0]
        base_curves[1].x = [0, 0, R, 0, R, 0, -Z, 0, 0]

        # Set the shape of the second pair of circular coils, ISU and ISL:
        R = 2.82
        Z = 2.0
        base_curves[2].x = [0, 0, R, 0, R, 0, +Z, 0, 0]
        base_curves[3].x = [0, 0, R, 0, R, 0, -Z, 0, 0]
        
        
        # Set the shape of the third pair of circular coils, IVU and IVL:
        R = 1.8
        Z = 0.8
        base_curves[4].x = [0, 0, R, 0, R, 0, +Z, 0, 0]
        base_curves[5].x = [0, 0, R, 0, R, 0, -Z, 0, 0]

        # Set the shape of the helical coils:
        base_curves[6].x = [3.850062473963758, 0.9987505207248398, 0.049916705720487310, 0.0012492189452854780, 1.0408856336378722e-05, 0, 0, 0, 0, 0, -1.0408856336392461e-05, 0, 0, -0.9962526034072403, -0.049958346351996670, -0.0012486983723145407, -2.082291883655196e-05, 0, 0]

        base_currents = [
           Current(2824400.0), Current(2824400.0),
           Current(682200.0),  Current(682200.0),
           Current(-2940000.0), Current(-2940000.0),
           Current(-5400000.0), Current(-5400000.0)]

        # magnetic axis
        ma = CurveRZFourier(quadpoints=nq_ax, order=6, nfp=10, stellsym=True)
        ma.x = [
            3.591808210975107,
            0.03794646194915659,
            0.00016372996351568552,
            -3.8324273652135154e-07,
            -7.090559083982798e-09,
            -7.966967131848883e-11,
            -5.308175230062491e-13,
            -0.03663986968740222,
            -0.00016230047363370836,
            4.326127544845136e-07,
            1.1123540323857856e-08,
            6.833905523642707e-11,
            4.612346787214785e-13,
        ]

    else:
        raise ValueError(f"Unrecognized configuration name {name!r}; "
                         f"choose from {configurations!r}")

    if cfg != "lhd_like":   # ie. not lhd_like
        # building the magnetic axis
        nump = magnetic_axis_order * points_per_period
        if nump % 2 == 0:   # ensure an odd node count
            nump += 1
        ma = CurveRZFourier(nump, magnetic_axis_order, nfp, True)
        ma.rc[:] = cR[:magnetic_axis_order + 1]
        ma.zs[:] = sZ[:magnetic_axis_order]
        ma.x = ma.get_dofs()

        coils = coils_via_symmetries(base_curves, base_currents, nfp, True)
        bs = BiotSavart(coils)
    else: 
        # ma already defined above in the lhd_like elif case 
        
        coils = [Coil(curve, current) for curve, current in zip(base_curves, base_currents)]
        bs = BiotSavart(coils)
        

    return base_curves, base_currents, ma, nfp, bs



def get_ncsx_data(Nt_coils=25, Nt_ma=10, ppp=10):
    """
    Get a configuration that corresponds to the modular coils of the NCSX experiment (circular coils are not included).

    Args:
        Nt_coils: order of the curves representing the coils.
        Nt_ma: order of the curve representing the magnetic axis.
        ppp: point-per-period: number of quadrature points per period

    Returns: 3 element tuple containing the coils, currents, and the magnetic axis.
    
    .. deprecated:: 1.11.0
       Use :func:`get_data` instead:
       ``get_data('ncsx', coil_order=..., magnetic_axis_order=..., points_per_period=...)``.
    """
    warnings.warn(
        "get_ncsx_data is deprecated and will be removed in the next major release; "
        "please use get_data('ncsx', coil_order=..., magnetic_axis_order=..., points_per_period=...) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    base_curves, base_currents, ma, *_ = get_data(
        "ncsx",
        coil_order=Nt_coils,
        magnetic_axis_order=Nt_ma,
        points_per_period=ppp,
    )
    return base_curves, base_currents, ma


def get_hsx_data(Nt_coils=16, Nt_ma=10, ppp=10):
    """
    Get a configuration that corresponds to the modular coils of the HSX experiment.

    Args:
        Nt_coils: order of the curves representing the coils.
        Nt_ma: order of the curve representing the magnetic axis.
        ppp: point-per-period: number of quadrature points per period

    Returns: 3 element tuple containing the coils, currents, and the magnetic axis.
    
    .. deprecated:: 1.11.0
       Use :func:`get_data` instead:
       ``get_data('hsx', coil_order=..., magnetic_axis_order=..., points_per_period=...)``.
    """
    warnings.warn(
        "get_hsx_data is deprecated and will be removed in the next major release; "
        "please use get_data('hsx', coil_order=..., magnetic_axis_order=..., points_per_period=...) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    base_curves, base_currents, ma, *_ = get_data(
        "hsx",
        coil_order=Nt_coils,
        magnetic_axis_order=Nt_ma,
        points_per_period=ppp,
    )
    return base_curves, base_currents, ma


def get_giuliani_data(Nt_coils=16, Nt_ma=10, ppp=10, length=18, nsurfaces=5):
    """
    This example simply loads the coils after the nine stage optimization runs discussed in

       A. Giuliani, F. Wechsung, M. Landreman, G. Stadler, A. Cerfon, Direct computation of magnetic surfaces in Boozer coordinates and coil optimization for quasi-symmetry. Journal of Plasma Physics.

    Args:
        Nt_coils: order of the curves representing the coils.
        Nt_ma: order of the curve representing the magnetic axis.
        ppp: point-per-period: number of quadrature points per period

    Returns: 3 element tuple containing the coils, currents, and the magnetic axis.
        
    .. deprecated:: 1.11.0
       Use :func:`get_data` instead:
       ``get_data('giuliani', coil_order=..., magnetic_axis_order=..., points_per_period=..., length=..., nsurfaces=...)``.
    """
    warnings.warn(
        "get_giuliani_data is deprecated and will be removed in the next major release; "
        "please use get_data('giuliani', coil_order=..., magnetic_axis_order=..., "
        "points_per_period=..., length=..., nsurfaces=...) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    base_curves, base_currents, ma, *_ = get_data(
        "giuliani",
        coil_order=Nt_coils,
        magnetic_axis_order=Nt_ma,
        points_per_period=ppp,
        length=length,
        nsurfaces=nsurfaces,
    )
    return base_curves, base_currents, ma


def get_w7x_data(Nt_coils=48, Nt_ma=10, ppp=2):
    """
    Get the W7-X coils and magnetic axis.

    Note that this function returns 7 coils: the 5 unique nonplanar
    modular coils, and the 2 planar (A and B) coils. The coil currents
    returned by this function correspond to the "Standard
    configuration", in which the planar A and B coils carry no current.
    
    Args:
        Nt_coils: order of the curves representing the coils.
        Nt_ma: order of the curve representing the magnetic axis.
        ppp: point-per-period: number of quadrature points per period.

    Returns: 3 element tuple containing the coils, currents, and the magnetic axis.
    
    .. deprecated:: 1.11.0
       Use :func:`get_data` instead:
       ``get_data('w7x', coil_order=..., magnetic_axis_order=..., points_per_period=...)``.
    """
    warnings.warn(
        "get_w7x_data is deprecated and will be removed in the next major release; "
        "please use get_data('w7x', coil_order=..., magnetic_axis_order=..., points_per_period=...) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    base_curves, base_currents, ma, *_ = get_data(
        "w7x",
        coil_order=Nt_coils,
        magnetic_axis_order=Nt_ma,
        points_per_period=ppp,
    )
    return base_curves, base_currents, ma