"""The B, B-VJP and dB/dX kernels keep their per-coil reductions fast.

Two kernel forms made them several times or about twice as slow
(the reasons are in ``biotsavart.py``, at the integrands and the contraction):

* a per-coil reduce whose operand keeps a length-3 component axis minor
  (openxla/xla#52437), checked in the compiled HLO;
* a stack, a block sum or the 1/Q scale on a per-coil quadrature sum before its
  contraction with the currents, which XLA fuses with the reduction (A100: the
  reduce moves into the stack's concatenate fusion; XLA:CPU: the op becomes its
  YNN epilogue), checked in the traced program.
"""

from __future__ import annotations

from unittest_jax_support import JaxTestCase, optional_jax_imports

with optional_jax_imports():
    import simsopt_jax  # noqa: F401
    import jax
    import jax.numpy as jnp
    from jax.extend.core import jaxprs_in_params
    from simsopt_jax.core import biotsavart as biotsavart_core


import re
from dataclasses import dataclass

# Sizes are (coils, nodes, points). Five coils is the only length-5 axis: 24
# nodes (padded to 32 by the pairwise sum), 12-node blocks (padded to 16),
# 7-point chunks with a 6-point tail of 20 points, and 3 Cartesian components.
_COIL_COUNT = 5
_SIZES = (_COIL_COUNT, 24, 20)
_POINT_CHUNK_SIZE = 7
# (coil_chunk_size, quadrature_block_size, point_chunk_size): unblocked, and
# two quadrature blocks as in the fast-lane tuning (128 nodes in 64-node blocks).
_TUNINGS = ((0, 0, _POINT_CHUNK_SIZE), (0, 12, _POINT_CHUNK_SIZE))
# The fast-lane tiling at which XLA formed the A100 concatenate fusions: one
# 1024-point chunk and 128 nodes in two 64-node blocks; 31 coils is the only
# length-31 axis.
_FAST_LANE_COIL_COUNT = 31
_FAST_LANE_SIZES = (_FAST_LANE_COIL_COUNT, 128, 1024)
_FAST_LANE_TUNING = (0, 64, 1024)

# Forward kernel name -> ``_DiffMode`` member name.
_FORWARD_DIFF_MODES = {"B": "VALUE", "dB_by_dX": "JACOBIAN"}

_COMPUTATION = re.compile(r"^(?:ENTRY\s+)?%([\w.\-]+)\s.*\{\s*$")
_INSTRUCTION = re.compile(
    r"^\s*(?:ROOT\s+)?%(?P<name>[\w.\-]+)\s*=\s*"
    r"(?P<shape>\([^=]*?\)|\S+)\s+(?P<opcode>[\w\-]+)\((?P<rest>.*)$"
)
_ARRAY_SHAPE = re.compile(r"^[a-z]\w*\[(?P<dims>[\d,]*)\](?:\{(?P<layout>[\d,]*))?")
_REDUCED_DIMENSIONS = re.compile(r"dimensions=\{([\d,]*)\}")


@dataclass(frozen=True)
class _Instruction:
    name: str
    shape: str
    opcode: str
    operands: tuple[str, ...]
    attributes: str


def _parse_hlo(text):
    """Map each computation of an HLO module's text to its instructions."""
    computations: dict[str, list[_Instruction]] = {}
    current: list[_Instruction] = []
    for line in text.splitlines():
        computation = _COMPUTATION.match(line)
        if computation:
            current = computations.setdefault(computation.group(1), [])
            continue
        instruction = _INSTRUCTION.match(line)
        if instruction is None:
            continue
        rest = instruction.group("rest")
        operand_text, _, attributes = rest.partition(")")
        current.append(
            _Instruction(
                name=instruction.group("name"),
                shape=instruction.group("shape"),
                opcode=instruction.group("opcode"),
                operands=tuple(re.findall(r"%([\w.\-]+)", operand_text)),
                attributes=attributes,
            )
        )
    return computations


def _per_coil_reduces(computations, coil_count):
    """Map each per-coil reduce of the module to its operand's shape and layout.

    A per-coil reduce keeps the coil axis: it sums a per-coil-node array over
    quadrature nodes, or over the point chunk in the VJP. Reductions over the
    coil axis (the current contraction) are not per-coil. HLO instruction names
    are unique within a module. Values: (operand shape, dims, minor axis,
    reduced axes).
    """
    instructions = [inst for body in computations.values() for inst in body]
    shapes = {instruction.name: instruction.shape for instruction in instructions}
    found = {}
    for instruction in instructions:
        if instruction.opcode != "reduce":
            continue
        shape = shapes[instruction.operands[0]]
        ((dims_text, layout),) = _ARRAY_SHAPE.findall(shape)
        dims = tuple(int(size) for size in dims_text.split(",") if size)
        minor = int(layout.split(",")[0]) if layout else len(dims) - 1
        (axes,) = _REDUCED_DIMENSIONS.findall(instruction.attributes)
        reduced = {int(axis) for axis in axes.split(",") if axis}
        if any(
            size == coil_count and axis not in reduced for axis, size in enumerate(dims)
        ):
            found[instruction.name] = (shape, dims, minor, reduced)
    return found


def _kernel_and_arguments(kernel_name, tuning, sizes):
    """Return one kernel and float64 shape structs of ``sizes`` for its inputs."""
    coils, nodes, point_count = sizes

    def spec(*shape):
        return jax.ShapeDtypeStruct(shape, jnp.float64)

    points, coil_data, currents = (
        spec(point_count, 3),
        spec(coils, nodes, 3),
        spec(coils),
    )
    if kernel_name == "B_vjp":
        kernel = biotsavart_core._make_B_vjp_kernel(*tuning)
        return kernel, (points, points, coil_data, coil_data, currents)
    kernel = biotsavart_core._make_kernel(
        biotsavart_core._Integrand.B,
        biotsavart_core._DiffMode[_FORWARD_DIFF_MODES[kernel_name]],
        *tuning,
    )
    return kernel, (points, coil_data, coil_data, currents)


def _compiled_hlo(kernel_name, tuning, sizes):
    """Compile one kernel for JAX's default device and return its HLO."""
    kernel, arguments = _kernel_and_arguments(kernel_name, tuning, sizes)
    return _parse_hlo(kernel.lower(*arguments).compile().as_text())


def _per_coil_quadrature_sums(jaxpr, coil_count):
    """Yield ``(sum, reads)`` for each per-coil quadrature sum in ``jaxpr`` and
    the jaxprs nested in it.

    ``pairwise_sum_axis`` ends in a ``squeeze`` of a ``reduce_sum``; its result
    is a per-coil sum when it keeps the coil axis. ``reads`` holds the
    equations that read the sum, and None if its jaxpr returns it.
    """
    producers = {var: eqn for eqn in jaxpr.eqns for var in eqn.outvars}
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == "squeeze" and coil_count in eqn.outvars[0].aval.shape:
            source = producers.get(eqn.invars[0])
            if source is not None and source.primitive.name == "reduce_sum":
                (total,) = eqn.outvars
                reads = [
                    reader
                    for reader in jaxpr.eqns
                    if any(operand is total for operand in reader.invars)
                ]
                returned = any(result is total for result in jaxpr.outvars)
                yield total, reads + [None] * returned
        for sub_jaxpr in jaxprs_in_params(eqn.params):
            yield from _per_coil_quadrature_sums(sub_jaxpr, coil_count)


def _is_contraction(eqn):
    """Whether ``eqn`` contracts a per-coil sum with a vector: the currents, or
    in the VJP the cotangent of that contraction."""
    if eqn is None or eqn.primitive.name != "dot_general":
        return False
    return sorted(len(var.aval.shape) for var in eqn.invars)[0] == 1


class TestBiotSavartReduceLayout(JaxTestCase):
    def test_per_coil_reduces_keep_no_minor_component_axis(self):
        """No per-coil reduce of the compiled B-VJP or dB/dX kernel keeps a
        length-3 component axis minor."""
        for kernel_name in ("B_vjp", "dB_by_dX"):
            for tuning in _TUNINGS:
                with self.subTest(kernel=kernel_name, tuning=tuning), self.case():
                    self._case_per_coil_reduces_keep_no_minor_component_axis(
                        kernel_name, tuning
                    )

    def _case_per_coil_reduces_keep_no_minor_component_axis(self, kernel_name, tuning):
        """Run one row with case-local objects released before runtime cleanup."""
        reduces = _per_coil_reduces(
            _compiled_hlo(kernel_name, tuning, _SIZES), _COIL_COUNT
        )
        # The guard is vacuous if the parser finds no per-coil reduce at all.
        self.assertTrue(reduces, "no per-coil reduce found in the optimized HLO")
        offending = [
            f"{shape} dimensions={sorted(reduced)}"
            for shape, dims, minor, reduced in reduces.values()
            if dims[minor] == 3 and minor not in reduced
        ]
        self.assertEqual(offending, [], "per-coil reduces with a minor 3-axis")

    def test_per_coil_quadrature_sums_feed_only_their_contraction(self):
        """In the traced B, dB/dX and B-VJP kernels each per-coil quadrature sum
        goes straight into its contraction, at this module's sizes and at the
        fast-lane tiling."""
        cases = (
            (_SIZES, _TUNINGS[0]),
            (_SIZES, _TUNINGS[1]),
            (_FAST_LANE_SIZES, _FAST_LANE_TUNING),
        )
        for kernel_name in ("B", "dB_by_dX", "B_vjp"):
            for sizes, tuning in cases:
                with self.subTest(kernel=kernel_name, sizes=sizes, tuning=tuning):
                    with self.case():
                        self._case_per_coil_quadrature_sums_feed_only_their_contraction(
                            kernel_name, sizes, tuning
                        )

    def _case_per_coil_quadrature_sums_feed_only_their_contraction(
        self, kernel_name, sizes, tuning
    ):
        """Run one row with case-local objects released before runtime cleanup."""
        coils, nodes, _ = sizes
        kernel, arguments = _kernel_and_arguments(kernel_name, tuning, sizes)
        # disable_jit inlines jit calls and unrolls fori_loops, so each sum and
        # every equation that reads it, in the VJP too, are in one jaxpr.
        with jax.disable_jit():
            traced = jax.make_jaxpr(kernel)(*arguments)
        sums = list(_per_coil_quadrature_sums(traced.jaxpr, coils))
        # One sum per component and quadrature block for each traced point
        # chunk and derivative pass; none found would make the guard vacuous.
        block_size = tuning[1]
        blocks = 1 if block_size <= 0 else (nodes + block_size - 1) // block_size
        self.assertTrue(
            sums and len(sums) % (3 * blocks) == 0,
            f"{len(sums)} per-coil quadrature sums for 3 components x {blocks} blocks",
        )
        offending = [
            f"{total.aval.shape} -> "
            + ", ".join(
                "result" if eqn is None else eqn.primitive.name for eqn in reads
            )
            for total, reads in sums
            if not reads or not all(map(_is_contraction, reads))
        ]
        self.assertEqual(
            offending, [], "per-coil quadrature sums used before their contraction"
        )
