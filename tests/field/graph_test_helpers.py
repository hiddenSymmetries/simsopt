"""Helpers for tests that check what simsopt's dependency graph holds."""
import gc


def coil_graph(coils):
    """Return the coils and every Optimizable they depend on, each once."""
    return list({id(o): o for c in coils for o in [c] + c.ancestors}.values())


def call_without_cyclic_gc(graph, fn):
    """
    Run ``fn`` with the cyclic garbage collector disabled, so that a collection cannot free (and so hide) a
    temporary it leaves registered. Return its value and the live child counts of the objects in ``graph``
    before and after.
    """
    def live_children():
        return [sum(ref() is not None for ref in o._children) for o in graph]

    gc.collect()
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        before = live_children()
        value = fn()
        after = live_children()
    finally:
        if was_enabled:
            gc.enable()
    return value, before, after
