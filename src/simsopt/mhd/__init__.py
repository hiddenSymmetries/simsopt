# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the MIT License

from importlib import import_module

__all__ = []


def _export_module(module_name, optional=False):
    try:
        module = import_module(f".{module_name}", __name__)
    except ModuleNotFoundError:
        if optional:
            return None
        raise

    names = getattr(module, "__all__", [])
    globals().update({name: getattr(module, name) for name in names})
    __all__.extend(names)
    return module


profiles = _export_module("profiles")
bootstrap = _export_module("bootstrap")
spec = _export_module("spec")
virtual_casing = _export_module("virtual_casing")
vmec = _export_module("vmec", optional=True)
vmec_diagnostics = _export_module("vmec_diagnostics", optional=True)
boozer = _export_module("boozer", optional=True)
