from .vmec import vmec_found, Vmec
from .spec import py_spec_found, pyoculus_found, Spec, Residue
from .boozer import booz_xform_found, Boozer, Quasisymmetry

#try:
#    import vmec
#except BaseException as err:
#    print('Unable to load VMEC module, so some functionality will not be available.')
#    print('Reason VMEC module was not loaded:')
#    print(err)

#try:
#    from .vmec import Vmec
#    vmec_found = True
#except ImportError as err:
#    vmec_found = False
#    print('Unable to load VMEC module, so some functionality will not be available.')
#    print('Reason VMEC module was not loaded:')
#    print(err)

#if vmec_found:
#    from .vmec import Vmec
