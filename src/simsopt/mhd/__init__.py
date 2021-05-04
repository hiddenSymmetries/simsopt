from .vmec import Vmec  # , vmec_found
from .spec import Spec, Residue  # , spec_found
from .boozer import Boozer, Quasisymmetry  # , boozer_found

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
