import numpy as np
from simsopt.geo.curverzfourier import CurveRZFourier
from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt.field.coil import Current
from simsopt.field.biotsavart import BiotSavart
from simsopt.field.coil import Coil 

from pathlib import Path
THIS_DIR = (Path(__file__).parent).resolve()

__all__ = ["get_data", "configurations"]
configurations = ["ncsx", "hsx", "giuliani", "w7x"]

def get_data(name, **kwargs):
    """
    Load one of several pre-defined stellarator coil configurations plus its
    Biot–Savart operator in a single call.

    Parameters
    ----------
    name : str
        Which configuration to load. Available values are:
        ``"ncsx"``, ``"hsx"``, ``"giuliani"``, ``"w7x"``.
    Nt_coils : int, optional
        Order of the curves representing the coils.
    Nt_ma : int, optional
        Order of the curve representing the magnetic axis.
    ppp : int, optional
        Points-per-period: number of quadrature points per period.
    length : int, optional
        (*Giuliani only*) Total length for the nine‐stage optimization.
    nsurfaces : int, optional
        (*Giuliani only*) Number of surfaces used in the optimization.

    Returns
    -------
    4-element tuple (`curves`, `currents`, `ma`,  `bs`)
    - `curves` : list of CurveXYZFourier  
      The coil curves, with length determined by `Nt_coils` and `ppp`.  
    - `currents` : list of Current  
      Corresponding coil currents.  
    - `ma` : CurveRZFourier  
      The magnetic axis, of order `Nt_ma`, with `nfp` field periods.  
    - `bs` : BiotSavart  
      The Biot–Savart operator assembled from `curves` and `currents`.


    Notes
    -----
    All of the above keyword arguments may be overridden via `kwargs`.
    
    **Configurations**

    *NCSX*  
    Get a configuration that corresponds to the modular coils of the NCSX
    experiment (circular coils are not included).

    *HSX*  
    Get a configuration that corresponds to the modular coils of the HSX
    experiment.

    *Giuliani*  
    This example simply loads the coils after the nine stage optimization
    runs discussed in

      A. Giuliani, F. Wechsung, M. Landreman, G. Stadler, A. Cerfon,
      “Direct computation of magnetic surfaces in Boozer coordinates
      and coil optimization for quasi-symmetry,” *Journal of Plasma Phys.*

    *W7X*  
    Get the W7-X coils and magnetic axis.

    Note that this function returns 7 coils: the 5 unique non-planar
    modular coils, and the 2 planar (A and B) coils. The coil currents
    correspond to the "Standard configuration", in which the planar A and 
    B coils carry no current. The shapes come from Fourier-transforming 
    the `xgrid` file `coils.w7x_v001`, provided by Joachim Geiger (Sept 27, 2022) in an
    email to Florian Wechsung and others,the Fourier modes here reproduce the Cartesian
    coordinate data from coils.w7x_v001 to ~ 1e-13 meters. Some
    description from Joachim:

    > "I have attached two coils-files which contain the filaments
    > suitable for generating the mgrid-file for VMEC with xgrid. They
    > contain the non-planar and the planar coils in a one-filament
    > approximation as used here for equilibrium calculations.  The two
    > coil-sets are slightly different with respect to the planar coils
    > (the non-planar coil geometry is the same), the w7x_v001 being the
    > CAD-coil-set while the w7x-set has slightly older planar coils
    > which I had used in a large number of calculations. The difference
    > is small, but, depending on what accuracy is needed, noticeable.
    > In case you want only one coil-set, I would suggest to use the
    > CAD-coils, i.e. w7x_v001, although the other coil-set had been
    > used in the PPCF-paper for citation.  If there are any further
    > questions, do not hesitate to contact me."
    
    """
    
    def add_default_args(kw_old, **kw_new):
        for k, v in kw_new.items():
            if k not in kw_old:
                kw_old[k] = v

    cfg = name.lower()

    if cfg == "ncsx":
        """Get a configuration that corresponds to the modular coils of the NCSX experiment (circular coils are not included)."""
        add_default_args(kwargs, Nt_coils=25, Nt_ma=10, ppp=10)
        Nt_coils = kwargs.pop("Nt_coils")
        Nt_ma = kwargs.pop("Nt_ma")
        ppp = kwargs.pop("ppp")
        filename = THIS_DIR / "NCSX.dat"
        curves = CurveXYZFourier.load_curves_from_file(filename, order=Nt_coils, ppp=ppp)
        nfp = 3
        
        currents = [Current(c) for c in [6.52271941985300E+05, 6.51868569367400E+05, 5.37743588647300E+05]]
        
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
        add_default_args(kwargs, Nt_coils=16, Nt_ma=10, ppp=10)
        Nt_coils = kwargs.pop("Nt_coils")
        Nt_ma = kwargs.pop("Nt_ma")
        ppp  = kwargs.pop("ppp")
        filename = THIS_DIR / "HSX.dat"
        curves = CurveXYZFourier.load_curves_from_file(filename, order=Nt_coils, ppp=ppp)
        nfp = 4
        currents = [Current(c) for c in [-1.500725500000000e+05] * 6]
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
        add_default_args(kwargs, Nt_coils=16, Nt_ma=10, ppp=10, length=18, nsurfaces=5)
        Nt_coils, Nt_ma, ppp = kwargs.pop("Nt_coils"), kwargs.pop("Nt_ma"), kwargs.pop("ppp")
        length, nsurfaces = kwargs.pop("length"), kwargs.pop("nsurfaces")
        filename = THIS_DIR / f"GIULIANI_length{length}_nsurfaces{nsurfaces}"
        curves   = CurveXYZFourier.load_curves_from_file(filename.with_suffix(".curves"),
                                                         order=Nt_coils, ppp=ppp)
        currents = [Current(c) for c in np.loadtxt(filename.with_suffix(".currents"))]
        dofs = np.loadtxt(filename.with_suffix(".ma"))
        cR = dofs[:26]
        sZ = dofs[26:]
        nfp = 2

    elif cfg == "w7x":
        """Get the W7-X coils and magnetic axis."""
        add_default_args(kwargs, Nt_coils=48, Nt_ma=10, ppp=2)
        Nt_coils, Nt_ma, ppp = kwargs.pop("Nt_coils"), kwargs.pop("Nt_ma"), kwargs.pop("ppp")
        filename = THIS_DIR / "W7-X.dat"
        curves = CurveXYZFourier.load_curves_from_file(filename, order=Nt_coils, ppp=ppp)
        nfp = 5
        turns = 108
        currents = [Current(15000.0 * turns)] * 5 + [Current(0.0), Current(0.0)]
        cR = [
            5.56069066955626, 0.370739830964738, 0.0161526928867275,
            0.0011820724983052, 3.43773868380292e-06, -4.71423775536881e-05,
            -6.23133271265022e-05, 2.06622580616597e-06, -0.000113256675159501,
            5.0296895894932e-07, 7.02308687052046e-05, 5.13479338167885e-05,
            1.90731007856885e-05
        ]
        sZ = [
            0, -0.308156954586225, -0.0186374002410851,
            -0.000261743895528833, 5.78207516751575e-05, -0.000129121205314107,
            3.08849630026052e-05, 1.95450172782866e-05, -8.32136392792337e-05,
            6.19785500011441e-05, 1.36521157246782e-05, -1.46281683516623e-05,
            -1.5142136543872e-06
        ]

    else:
        raise ValueError(f"Unrecognized configuration name {name!r}; "
                         f"choose from {configurations!r}")

    # building the magnetic axis
    nump = Nt_ma * ppp
    nump = nump + 1 if nump % 2 == 0 else nump
    ma   = CurveRZFourier(nump, Nt_ma, nfp, True)
    ma.rc[:] = cR[: (Nt_ma + 1)]
    ma.zs[:] = sZ[:Nt_ma]
    ma.x    = ma.get_dofs()

    # assemble Biot–Savart
    coils_for_bs = [Coil(curve, current) for curve, current in zip(curves, currents)]
    bs = BiotSavart(coils_for_bs)

    return curves, currents, ma, bs
