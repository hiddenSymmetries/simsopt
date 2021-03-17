from jax import jit as jaxjit
from .config import parameters


def jit(fun):
    if parameters['jit']:
        return jaxjit(fun)
    else:
        return fun
