# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the LGPL License

try:
    from .vmec import *
except BaseException as err:
    print('Unable to load VMEC module, so some functionality will not be available.')
    print('Reason VMEC module was not loaded:')
    print(err)
