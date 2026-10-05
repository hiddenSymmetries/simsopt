"""Behaviour of the JAX host/device boundary owners.

``simsopt_jax.runtime.host_boundary`` owns the strict transfer guard, its one
permit (``allow_host_transfers``) and the host readers;
``simsopt_jax.backend.dtypes`` owns placement. Code under
``disallow_host_transfers()`` moves data only through these owners, so each
must work under the strict guard, and the permit must lift the guard for its
own block and no further.

The owner tests exercise the owners only, so a census also scans
``src/simsopt_jax`` and ``src/simsopt_jax_adapters`` for direct calls of the
explicit JAX transfer and readiness primitives (which the strict guard does not
refuse) and admits them only inside the owner functions listed in
``_TRANSFER_OWNER_FUNCTIONS``. Owners are keyed by file and qualified function
name, never by line, so edits inside an owner do not disturb the census.
"""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import ast
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from simsopt_jax.backend.dtypes import explicit_device_array, runtime_device_put_tree
from simsopt_jax.runtime.host_boundary import (
    allow_host_transfers,
    block_until_ready,
    disallow_host_transfers,
    host_array,
    host_tree_after_ready,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
_SCANNED_ROOTS = ("src/simsopt_jax", "src/simsopt_jax_adapters")
_TRANSFER_PRIMITIVES = frozenset(
    {
        "block_until_ready",
        "device_get",
        "device_put",
        "transfer_guard",
        "transfer_guard_device_to_device",
        "transfer_guard_device_to_host",
        "transfer_guard_host_to_device",
    }
)
_JAX_MODULE = "jax"
_OTHER_VALUE = ""

# Every function that calls a transfer primitive directly, as
# ``path::Qualified.function``. Adding an entry admits a new boundary owner; an
# entry whose function no longer calls a primitive fails as stale.
_TRANSFER_OWNER_FUNCTIONS = frozenset(
    {
        "src/simsopt_jax/backend/dtypes.py::_device_put",
        "src/simsopt_jax/backend/dtypes.py::_unplaced_device_put",
        "src/simsopt_jax/backend/dtypes.py::runtime_device_put_tree",
        "src/simsopt_jax/runtime/host_boundary.py::allow_host_transfers",
        "src/simsopt_jax/runtime/host_boundary.py::block_until_ready",
        "src/simsopt_jax/runtime/host_boundary.py::disallow_host_transfers",
        "src/simsopt_jax/runtime/host_boundary.py::host_value",
    }
)


class _TransferCall(NamedTuple):
    owner: str
    primitive: str


class _TransferCallScanner(ast.NodeVisitor):
    """Collects direct ``jax`` transfer-primitive calls with their enclosing function.

    A call counts when its callee resolves lexically to ``jax.<primitive>``:
    through ``import jax`` (any alias, at any scope), ``from jax import
    <primitive>`` or an assignment alias of either. Method calls such as
    ``array.block_until_ready()`` do not count.
    """

    def __init__(self, relative_path: str) -> None:
        self._relative_path = relative_path
        self._scope_names: list[str] = []
        self._scope_is_class: list[bool] = [False]
        self._bindings: list[dict[str, str]] = [{}]
        self.calls: set[_TransferCall] = set()

    def _lookup(self, name: str) -> str:
        innermost = len(self._bindings) - 1
        for index in range(innermost, -1, -1):
            if index != innermost and self._scope_is_class[index]:
                continue
            if name in self._bindings[index]:
                return self._bindings[index][name]
        return _OTHER_VALUE

    def _resolve(self, expression: ast.expr) -> str:
        if isinstance(expression, ast.Name):
            return self._lookup(expression.id)
        if (
            isinstance(expression, ast.Attribute)
            and expression.attr in _TRANSFER_PRIMITIVES
            and isinstance(expression.value, ast.Name)
            and self._lookup(expression.value.id) == _JAX_MODULE
        ):
            return expression.attr
        return _OTHER_VALUE

    def _bind(self, target: ast.expr, value: str) -> None:
        if isinstance(target, ast.Name):
            self._bindings[-1][target.id] = value

    def _visit_scope(
        self, name: str, is_class: bool, bindings: dict[str, str], body: list[ast.stmt]
    ) -> None:
        self._bindings[-1][name] = _OTHER_VALUE
        self._scope_names.append(name)
        self._scope_is_class.append(is_class)
        self._bindings.append(bindings)
        for statement in body:
            self.visit(statement)
        self._bindings.pop()
        self._scope_is_class.pop()
        self._scope_names.pop()

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            is_jax = alias.name == "jax" or (
                alias.name.startswith("jax.") and alias.asname is None
            )
            bound = alias.asname or alias.name.split(".", maxsplit=1)[0]
            self._bindings[-1][bound] = _JAX_MODULE if is_jax else _OTHER_VALUE

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            is_primitive = node.module == "jax" and alias.name in _TRANSFER_PRIMITIVES
            self._bindings[-1][alias.asname or alias.name] = (
                alias.name if is_primitive else _OTHER_VALUE
            )

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        value = self._resolve(node.value)
        for target in node.targets:
            self.visit(target)
            self._bind(target, value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self.visit(node.value)
            self._bind(node.target, self._resolve(node.value))

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for expression in (*node.decorator_list, *node.bases):
            self.visit(expression)
        for keyword in node.keywords:
            self.visit(keyword.value)
        self._visit_scope(node.name, True, {}, node.body)

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        self.visit(node.args)
        parameters = (
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
            *(argument for argument in (node.args.vararg, node.args.kwarg) if argument),
        )
        self._visit_scope(
            node.name,
            False,
            {parameter.arg: _OTHER_VALUE for parameter in parameters},
            node.body,
        )

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node: ast.Call) -> None:
        primitive = self._resolve(node.func)
        if primitive in _TRANSFER_PRIMITIVES:
            scope = ".".join(self._scope_names) or "<module>"
            self.calls.add(_TransferCall(f"{self._relative_path}::{scope}", primitive))
        self.generic_visit(node)


def _transfer_calls(source: str, relative_path: str) -> frozenset[_TransferCall]:
    scanner = _TransferCallScanner(relative_path)
    scanner.visit(ast.parse(source, filename=relative_path))
    return frozenset(scanner.calls)


def _tree_transfer_calls(repo_root: Path) -> frozenset[_TransferCall]:
    return frozenset(
        call
        for root in _SCANNED_ROOTS
        for path in sorted((repo_root / root).rglob("*.py"))
        for call in _transfer_calls(
            path.read_text(encoding="utf-8"), path.relative_to(repo_root).as_posix()
        )
    )


def _census_violations(
    calls: frozenset[_TransferCall], owners: frozenset[str]
) -> tuple[list[str], list[str]]:
    """Return (calls outside an owner, owners that call no primitive)."""
    unadmitted = sorted(
        f"{call.owner} ({call.primitive})" for call in calls if call.owner not in owners
    )
    stale = sorted(owners - {call.owner for call in calls})
    return unadmitted, stale


def test_only_owner_functions_call_jax_transfer_primitives() -> None:
    unadmitted, stale = _census_violations(
        _tree_transfer_calls(REPO_ROOT), _TRANSFER_OWNER_FUNCTIONS
    )

    assert not unadmitted, "Direct JAX transfer calls outside an owner:\n  " + (
        "\n  ".join(unadmitted)
    )
    assert not stale, "Owners that no longer call a primitive:\n  " + "\n  ".join(stale)


def test_census_rejects_a_stray_transfer_outside_an_owner() -> None:
    calls = _transfer_calls(
        "import jax\n\n"
        "def owner(value):\n"
        "    return jax.device_put(value)\n\n"
        "def stray(value):\n"
        "    return jax.device_put(value)\n",
        "src/simsopt_jax/module.py",
    )

    unadmitted, stale = _census_violations(
        calls, frozenset({"src/simsopt_jax/module.py::owner"})
    )

    assert unadmitted == ["src/simsopt_jax/module.py::stray (device_put)"]
    assert stale == []


def test_census_reports_an_owner_that_no_longer_calls_a_primitive() -> None:
    calls = _transfer_calls(
        "import jax\n\ndef owner(value):\n    return value\n",
        "src/simsopt_jax/module.py",
    )

    assert _census_violations(
        calls, frozenset({"src/simsopt_jax/module.py::owner"})
    ) == ([], ["src/simsopt_jax/module.py::owner"])


def test_census_resolves_jax_aliases_and_qualifies_nested_scopes() -> None:
    calls = _transfer_calls(
        "import jax as jx\n"
        "from jax import device_get as fetch\n"
        "import jax.numpy\n"
        "put = jax.device_put\n\n"
        "class Holder:\n"
        "    def method(self, value):\n"
        "        def inner():\n"
        "            from jax import transfer_guard\n"
        "            return transfer_guard('allow')\n"
        "        return jx.block_until_ready(fetch(put(value))), inner\n\n"
        "def local(value):\n"
        "    import jax\n"
        "    return jax.device_get(value)\n\n"
        "def not_counted(value, device_put):\n"
        "    value.block_until_ready()\n"
        "    return device_put(value)\n",
        "src/simsopt_jax/module.py",
    )

    assert calls == {
        _TransferCall("src/simsopt_jax/module.py::Holder.method", "block_until_ready"),
        _TransferCall("src/simsopt_jax/module.py::Holder.method", "device_get"),
        _TransferCall("src/simsopt_jax/module.py::Holder.method", "device_put"),
        _TransferCall(
            "src/simsopt_jax/module.py::Holder.method.inner", "transfer_guard"
        ),
        _TransferCall("src/simsopt_jax/module.py::local", "device_get"),
    }


def test_tree_census_scans_both_roots(tmp_path: Path) -> None:
    stray = "import jax\n\ndef stray(value):\n    return jax.device_get(value)\n"
    for root in _SCANNED_ROOTS:
        (tmp_path / root / "sub").mkdir(parents=True)
        (tmp_path / root / "sub" / "stray.py").write_text(stray, encoding="utf-8")
    (tmp_path / "src/simsopt").mkdir()
    (tmp_path / "src/simsopt/unscanned.py").write_text(stray, encoding="utf-8")

    assert _tree_transfer_calls(tmp_path) == {
        _TransferCall(f"{root}/sub/stray.py::stray", "device_get")
        for root in _SCANNED_ROOTS
    }


def test_allow_host_transfers_lifts_the_strict_guard_for_its_block_only() -> None:
    device_array = jax.device_put(np.asarray([1.0, 2.0], dtype=np.float64))
    host_values = np.asarray([3.0, 4.0], dtype=np.float64)

    with disallow_host_transfers():
        with pytest.raises(jax.errors.JaxRuntimeError):
            _ = device_array + host_values
        with allow_host_transfers():
            permitted = device_array + host_values
        with pytest.raises(jax.errors.JaxRuntimeError):
            _ = device_array + host_values

    np.testing.assert_array_equal(jax.device_get(permitted), [4.0, 6.0])


def test_placement_owners_place_host_values_under_the_strict_guard() -> None:
    tree = {
        "float": np.asarray([1.0, 2.0], dtype=np.float32),
        "integer": (np.asarray(3, dtype=np.int16),),
    }
    reference = jax.device_put(np.zeros(2, dtype=np.float64))

    with disallow_host_transfers():
        placed_tree = runtime_device_put_tree(tree)
        placed_array = explicit_device_array(
            [5.0, 6.0], dtype=jnp.float64, reference=reference
        )

    assert isinstance(placed_tree["float"], jax.Array)
    np.testing.assert_array_equal(jax.device_get(placed_tree["float"]), tree["float"])
    assert jax.device_get(placed_tree["integer"][0]).item() == 3
    assert placed_array.dtype == jnp.float64
    assert placed_array.sharding == reference.sharding
    np.testing.assert_array_equal(jax.device_get(placed_array), [5.0, 6.0])


def test_host_readers_return_writeable_host_copies_under_the_strict_guard() -> None:
    value = {
        "vector": jnp.asarray([1.0, 2.0], dtype=jnp.float64),
        "scalar": (jnp.asarray(3, dtype=jnp.int32),),
    }

    with disallow_host_transfers():
        array = host_array(value["vector"], dtype=np.float32)
        tree = host_tree_after_ready(value)

    assert isinstance(array, np.ndarray)
    assert array.dtype == np.float32
    assert array.flags.writeable
    np.testing.assert_array_equal(array, [1.0, 2.0])
    assert isinstance(tree["vector"], np.ndarray)
    assert tree["vector"].flags.writeable
    assert tree["scalar"][0].item() == 3


def test_runtime_device_put_tree_preserves_structure_and_exact_leaf_dtypes() -> None:
    value = {
        "float": np.asarray([1.0, 2.0], dtype=np.float32),
        "integer": (np.asarray(3, dtype=np.int16),),
    }

    placed = runtime_device_put_tree(value)

    assert placed.keys() == value.keys()
    assert placed["float"].dtype == jnp.float32
    assert placed["integer"][0].dtype == jnp.int16


def test_readiness_and_host_tree_preserve_pytree_structure() -> None:
    value = {
        "vector": jnp.asarray([1.0, 2.0], dtype=jnp.float64),
        "scalar": (jnp.asarray(3, dtype=jnp.int32),),
    }

    ready = block_until_ready(value)
    host = host_tree_after_ready(ready)

    assert host.keys() == value.keys()
    np.testing.assert_array_equal(host["vector"], np.asarray([1.0, 2.0]))
    assert host["scalar"][0].item() == 3
    assert host["vector"].flags.writeable
