"""Physical constants, taken from :mod:`scipy.constants` (current CODATA release)."""
from scipy import constants as _constants

PROTON_MASS = _constants.m_p  # kg
NEUTRON_MASS = _constants.m_n  # kg
ELEMENTARY_CHARGE = _constants.e  # C
ONE_EV = _constants.eV  # J
ALPHA_PARTICLE_MASS = _constants.physical_constants["alpha particle mass"][0]  # kg
ALPHA_PARTICLE_CHARGE = 2*ELEMENTARY_CHARGE
FUSION_ALPHA_PARTICLE_ENERGY = 3.52e6 * ONE_EV
