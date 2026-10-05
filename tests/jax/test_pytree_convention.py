"""Keep JAX registration and immutable record conventions centralized."""

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import ast
from collections.abc import Callable
from collections import Counter
from dataclasses import FrozenInstanceError, dataclass, field
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import pytest

from simsopt_jax.pytree import pytree_dataclass, pytree_node, registered_pytree_classes

ROOT = Path(__file__).resolve().parents[2]
TREES = (
    "src/simsopt_jax",
    "src/simsopt_jax_adapters",
)
REGISTRATION_NAMES = frozenset(
    (
        "register_dataclass",
        "register_pytree_node",
        "register_pytree_node_class",
        "register_pytree_with_keys",
        "register_pytree_with_keys_class",
        "register_static",
    )
)
NAMEDTUPLE_NAMES = frozenset(("NamedTuple", "namedtuple"))
ALIAS_NAMES = REGISTRATION_NAMES | NAMEDTUPLE_NAMES | {"pytree_dataclass", "pytree_node"}
Violation = tuple[str, str, str, str]

# Any exemption must match the exact occurrence multiset below.
PENDING: tuple[Violation, ...] = ()


class ConventionVisitor(ast.NodeVisitor):
    """Count prohibited references without interpreting aliases or scope bindings."""

    def __init__(self, path: str):
        self.path = path
        self.classes: list[str] = []
        self.assignment_owners: tuple[str, ...] = ()
        self.call_owners: list[str | None] = []
        self.function_depth = 0
        self.violations: list[Violation] = []

    def record(self, kind: str, name: str):
        if (
            kind in {"registration", "import", "star-import"}
            and (name in REGISTRATION_NAMES or kind == "star-import")
            and self.path == "src/simsopt_jax/pytree.py"
        ):
            return
        call_owner = self.call_owners[-1] if self.call_owners else None
        if kind == "registration" and call_owner:
            owners = (call_owner,)
        elif self.classes:
            owners = (self.classes[-1],)
        elif call_owner:
            owners = (call_owner,)
        else:
            owners = self.assignment_owners or ("<module>",)
        self.violations.extend((self.path, owner, kind, name) for owner in owners)

    def reference(self, name: str, *, imported: bool = False):
        if name in REGISTRATION_NAMES:
            self.record("import" if imported else "registration", name)
        if name in NAMEDTUPLE_NAMES:
            self.record("import" if imported else "namedtuple", name)

    def visit_Name(self, node: ast.Name):
        self.reference(node.id)

    def visit_Attribute(self, node: ast.Attribute):
        self.reference(node.attr)
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant):
        if isinstance(node.value, str):
            self.reference(node.value)

    def check_import_alias(self, alias: ast.alias, module: str | None = None):
        name = alias.name.rsplit(".", 1)[-1]
        if alias.asname and (
            name in ALIAS_NAMES
            or alias.name == "dataclasses.field"
            or module == "dataclasses" and name == "field"
        ):
            self.record("alias", name)

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            self.check_import_alias(alias)
            for name in alias.name.split("."):
                self.reference(name, imported=True)
            if alias.asname:
                self.reference(alias.asname, imported=True)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module:
            for name in node.module.split("."):
                self.reference(name, imported=True)
        for alias in node.names:
            self.check_import_alias(alias, node.module)
            if alias.name == "*" and node.module in {"jax", "jax.tree_util"}:
                self.record("star-import", "*")
            self.reference(alias.name, imported=True)
            if alias.asname:
                self.reference(alias.asname, imported=True)

    def visit_Dict(self, node: ast.Dict):
        for key in node.keys:
            if isinstance(key, ast.Constant) and key.value == "static":
                self.record("static-metadata", "static")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        target = node.args[0] if node.args else None
        self.call_owners.append(target.id if isinstance(target, ast.Name) else None)
        if (
            isinstance(node.func, ast.Name) and node.func.id == "dict"
            or isinstance(node.func, ast.Attribute) and node.func.attr == "dict"
        ):
            for keyword in node.keywords:
                if keyword.arg == "static":
                    self.record("static-metadata", "static")
        self.generic_visit(node)
        self.call_owners.pop()

    def visit_ClassDef(self, node: ast.ClassDef):
        self.classes.append(node.name)
        self.generic_visit(node)
        self.classes.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef):
        self.function_depth += 1
        self.generic_visit(node)
        self.function_depth -= 1

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Assign(self, node: ast.Assign | ast.AnnAssign):
        previous = self.assignment_owners
        if not self.classes and not self.function_depth:
            targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
            self.assignment_owners = tuple(
                child.id
                for target in targets
                for child in ast.walk(target)
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store)
            )
        self.generic_visit(node)
        self.assignment_owners = previous

    visit_AnnAssign = visit_Assign


def inspect_source(path: str, source: str) -> ConventionVisitor:
    visitor = ConventionVisitor(path)
    visitor.visit(ast.parse(source, filename=path))
    return visitor


def violations_in_source(path: str, source: str) -> list[Violation]:
    return inspect_source(path, source).violations


@pytest.fixture(scope="module")
def repository_inspections() -> list[ConventionVisitor]:
    return [
        inspect_source(path.relative_to(ROOT).as_posix(), path.read_text())
        for tree in TREES
        for path in sorted((ROOT / tree).rglob("*.py"))
    ]


def test_registration_and_record_convention(repository_inspections):
    actual = Counter(
        violation
        for visitor in repository_inspections
        for violation in visitor.violations
    )
    expected = Counter(PENDING)
    assert actual == expected, (
        f"Unexpected occurrences: {dict(actual - expected)}; "
        f"stale occurrences: {dict(expected - actual)}"
    )


def test_pending_has_no_stale_entries(repository_inspections):
    actual = Counter(
        violation
        for visitor in repository_inspections
        for violation in visitor.violations
    )
    assert not Counter(PENDING) - actual
    assert all(
        kind in {"import", "registration", "namedtuple", "star-import"}
        for _, _, kind, _ in PENDING
    )
    assert all(name != "register_dataclass" for _, _, _, name in PENDING)


@pytest.mark.parametrize("identifier", sorted(REGISTRATION_NAMES | NAMEDTUPLE_NAMES))
@pytest.mark.parametrize("form", ("name", "attribute", "import", "getattr"))
def test_guard_rejects_every_prohibited_reference(identifier, form):
    sources = {
        "name": f"{identifier}(Payload)",
        "attribute": f"trees.{identifier}(Payload)",
        "import": f"from library import {identifier}",
        "getattr": f"getattr(trees, '{identifier}')",
    }
    kind = "registration" if identifier in REGISTRATION_NAMES else "namedtuple"
    owner = "Payload" if form in {"name", "attribute"} else "trees"
    if form == "import":
        kind, owner = "import", "<module>"
    assert violations_in_source("example.py", sources[form]) == [
        ("example.py", owner, kind, identifier)
    ]


@pytest.mark.parametrize(
    "source, expected",
    (
        (
            "from functools import partial\n@partial(trees.register_dataclass, data_fields=['x'], meta_fields=[])\nclass Payload: pass",
            (("Payload", "registration", "register_dataclass"),),
        ),
        (
            "register = partial(trees.register_static)\nregister(Payload)",
            (("register", "registration", "register_static"),),
        ),
        (
            "register: object = trees.register_dataclass\nregister(Payload)",
            (("register", "registration", "register_dataclass"),),
        ),
        (
            "from jax import tree_util as trees\ndef unrelated():\n    trees = other\ntrees.register_dataclass(Payload)",
            (("Payload", "registration", "register_dataclass"),),
        ),
        (
            "from jax.tree_util import register_static as register\ndef unrelated():\n    from functools import partial as register\nregister(Payload)",
            (("<module>", "alias", "register_static"), ("<module>", "import", "register_static")),
        ),
        ("from jax import *", (("<module>", "star-import", "*"),)),
        ("from jax.tree_util import *", (("<module>", "star-import", "*"),)),
        (
            "Payload = typing.NamedTuple('Payload', [('x', int)])",
            (("Payload", "namedtuple", "NamedTuple"),),
        ),
        (
            "Payload = collections.namedtuple('Payload', 'x')",
            (("Payload", "namedtuple", "namedtuple"),),
        ),
        (
            "class Payload(collections.namedtuple('Base', 'x')): pass",
            (("Payload", "namedtuple", "namedtuple"),),
        ),
        (
            "from typing import NamedTuple\nclass Payload(NamedTuple): pass",
            (
                ("<module>", "import", "NamedTuple"),
                ("Payload", "namedtuple", "NamedTuple"),
            ),
        ),
        (
            "from collections import namedtuple as record\nPayload = record('Payload', 'x')",
            (("<module>", "alias", "namedtuple"), ("<module>", "import", "namedtuple")),
        ),
        (
            "class Payload:\n    register = trees.register_static\n    def method(self):\n        return getattr(trees, 'register_dataclass')",
            (
                ("Payload", "registration", "register_static"),
                ("trees", "registration", "register_dataclass"),
            ),
        ),
        (
            "def function():\n    register = trees.register_static\n    return register(Payload)",
            (("<module>", "registration", "register_static"),),
        ),
        (
            "Payload = getattr(typing, 'NamedTuple')('Payload', [('x', int)])",
            (("typing", "namedtuple", "NamedTuple"),),
        ),
        (
            "Payload = getattr(collections, 'namedtuple')('Payload', 'x')",
            (("collections", "namedtuple", "namedtuple"),),
        ),
        (
            "Payload = trees.register_static(Target)",
            (("Target", "registration", "register_static"),),
        ),
    ),
    ids=(
        "partial_decorator",
        "partial_call",
        "annotated_alias",
        "function_local_assignment_shadowing",
        "function_local_import_shadowing",
        "jax_star_import",
        "tree_util_star_import",
        "typing_factory",
        "collections_factory",
        "collections_factory_base",
        "namedtuple_import_and_base",
        "namedtuple_factory_alias_import",
        "class_body_and_method",
        "function_body_assignment",
        "typing_getattr",
        "collections_getattr",
        "call_target_over_assignment",
    ),
)
def test_guard_attributes_prohibited_references(source, expected):
    assert violations_in_source("example.py", source) == [
        ("example.py", owner, kind, name) for owner, kind, name in expected
    ]


def test_guard_does_not_infer_aliases_for_shadowed_function_parameters():
    assert not violations_in_source(
        "example.py", "def unrelated(register): register(Payload)"
    )


@pytest.mark.parametrize("registration_name", sorted(REGISTRATION_NAMES))
def test_guard_permits_registration_only_in_helper(registration_name):
    source = f"import jax\njax.tree_util.{registration_name}(Payload)"
    assert not violations_in_source("src/simsopt_jax/pytree.py", source)
    source = "import typing\nclass Payload(typing.NamedTuple): pass"
    assert violations_in_source("src/simsopt_jax/pytree.py", source) == [
        ("src/simsopt_jax/pytree.py", "Payload", "namedtuple", "NamedTuple")
    ]
    source = "from typing import NamedTuple as Record"
    assert violations_in_source("src/simsopt_jax/pytree.py", source) == [
        ("src/simsopt_jax/pytree.py", "<module>", "alias", "NamedTuple"),
        ("src/simsopt_jax/pytree.py", "<module>", "import", "NamedTuple")
    ]


def test_pending_counts_missing_occurrences_as_stale():
    original = "class Payload(NamedTuple):\n    alias = NamedTuple"
    expected = Counter(violations_in_source("example.py", original))
    missing = Counter(violations_in_source("example.py", "class Payload(NamedTuple): pass"))
    assert expected[("example.py", "Payload", "namedtuple", "NamedTuple")] == 2
    assert missing != expected
    assert expected - missing == Counter({("example.py", "Payload", "namedtuple", "NamedTuple"): 1})
    assert not missing - expected


def test_pending_cannot_authorize_a_different_registration_api():
    path = "src/simsopt_jax/core/field.py"
    source = "@jax.tree_util.register_pytree_node_class\nclass Payload: pass"
    expected = Counter(violations_in_source(path, source))
    replacement = source.replace("register_pytree_node_class", "register_dataclass", 1)
    actual = Counter(violations_in_source(path, replacement))
    assert actual != expected
    assert any(name == "register_dataclass" for _, _, _, name in actual - expected)
    assert any(
        name == "register_pytree_node_class" for _, _, _, name in expected - actual
    )


@pytest.mark.parametrize(
    "source, name",
    (
        ("from typing import NamedTuple as Record", "NamedTuple"),
        ("from collections import namedtuple as record", "namedtuple"),
        ("from dataclasses import field as f", "field"),
        ("import dataclasses.field as f", "field"),
        ("from simsopt_jax.pytree import pytree_dataclass as record", "pytree_dataclass"),
        ("from simsopt_jax.pytree import pytree_node as record", "pytree_node"),
        ("import typing.NamedTuple as Record", "NamedTuple"),
        *((f"from jax.tree_util import {name} as register", name) for name in sorted(REGISTRATION_NAMES)),
    ),
)
def test_guard_rejects_prohibited_import_aliases(source, name):
    assert ("example.py", "<module>", "alias", name) in violations_in_source("example.py", source)
    assert ("src/simsopt_jax/pytree.py", "<module>", "alias", name) in violations_in_source("src/simsopt_jax/pytree.py", source)


@pytest.mark.parametrize("source", ("import typing as t", "import dataclasses as dc", "from jax import tree_util as trees", "from other import field as f"))
def test_guard_permits_module_and_unrelated_import_aliases(source):
    assert not violations_in_source("example.py", source)


def test_pending_cannot_hide_replaced_namedtuple_import_alias():
    path = "src/simsopt_jax/core/_device_scalars.py"
    source = (
        "from typing import NamedTuple\n"
        "class _FieldCacheStatus(NamedTuple):\n"
        "    terminal: bool\n"
    )
    expected = Counter(violations_in_source(path, source))
    replacement = source.replace("from typing import NamedTuple", "import typing\nfrom typing import NamedTuple as Record")
    replacement = replacement.replace("class _FieldCacheStatus(NamedTuple):", "class _FieldCacheStatus(typing.NamedTuple):")
    replacement += "\nclass Surprise(Record):\n    value: int\n"
    actual = Counter(violations_in_source(path, replacement))
    assert actual - expected == Counter({(path, "<module>", "alias", "NamedTuple"): 1})
    assert not expected - actual


def test_pending_cannot_hide_registration_target_inside_exempt_class():
    path = "src/simsopt_jax/core/field.py"
    target = "_FieldPullbackInputs"
    marker = f"@jax.tree_util.register_pytree_node_class\n@dataclass(frozen=True, slots=True)\nclass {target}:"
    source = marker + "\n    pass\n"
    expected = Counter(violations_in_source(path, source))
    surprise = "class Surprise:\n    def tree_flatten(self): return (), None\n    @classmethod\n    def tree_unflatten(cls, aux, children): return cls()\n\n"
    replacement = source.replace(marker, surprise + f"@dataclass(frozen=True, slots=True)\nclass {target}:\n    jax.tree_util.register_pytree_node_class(Surprise)")
    actual = Counter(violations_in_source(path, replacement))
    assert actual - expected == Counter({(path, "Surprise", "registration", "register_pytree_node_class"): 1})
    assert expected - actual == Counter({(path, target, "registration", "register_pytree_node_class"): 1})


@pytest.mark.parametrize(
    "source",
    (
        "from simsopt_jax.pytree import pytree_dataclass as record\n@record(data=('value',))\nclass Payload:\n    value: int = field(metadata={'static': True})",
        "from dataclasses import field as f\n@pytree_dataclass(data=('value',))\nclass Payload:\n    value: int = f(metadata={'static': True})",
        "META = {'static': True}\n@pytree_dataclass(data=('value',))\nclass Payload:\n    value: int = field(metadata=META)",
        "@pytree_dataclass(data=('value',))\nclass Payload:\n    value: int = field(metadata=dict(static=True))",
        "@dataclass(frozen=True)\nclass Base:\n    value: int = field(metadata={'static': True})\n@pytree_dataclass(data=('value',))\nclass Payload(Base): pass",
        "def function():\n    return {'static': False}",
        "META = builtins.dict(static=False)",
        "@pytree_dataclass(data=('value',))\nclass Payload:\n    def method(self):\n        return {'static': True}",
    ),
    ids=("aliased_helper", "aliased_field", "named_mapping", "dict_constructor", "inherited_declaration", "function_literal", "qualified_dict_constructor", "method_literal"),
)
def test_guard_rejects_static_metadata_without_matching_declarations(source):
    violations = violations_in_source("example.py", source)
    assert any(kind == "static-metadata" and name == "static" for _, _, kind, name in violations)


@pytest.mark.parametrize("source", ("META = {'description': 'value'}", "META = dict(description='value')", "field(metadata=META)"))
def test_guard_permits_metadata_without_static_keys(source):
    assert not violations_in_source("example.py", source)


def test_plain_class_is_frozen_and_round_trips_in_data_order():
    @pytree_dataclass(data=("second", "first"), meta=("mode",))
    class Payload:
        first: int
        second: int
        mode: str

    payload = Payload(first=1, second=2, mode="test")
    with pytest.raises(FrozenInstanceError):
        setattr(payload, "first", 3)
    leaves, treedef = jax.tree_util.tree_flatten(payload)
    assert leaves == [2, 1]
    assert jax.tree_util.tree_unflatten(treedef, leaves) == payload
    assert jax.tree_util.tree_unflatten(treedef, [20, 10]) == Payload(10, 20, "test")


def test_mutable_dataclass_is_rejected():
    @dataclass
    class Mutable:
        value: int

    with pytest.raises(TypeError, match="Mutable must be a frozen dataclass"):
        pytree_dataclass(data=("value",))(Mutable)


@pytest.mark.parametrize(
    "data, meta, message",
    (
        (("value",), ("value", "mode"), "data and meta overlap"),
        (("value",), (), "missing=\\['mode'\\]"),
        (("value", "extra"), ("mode",), "unknown=\\['extra'\\]"),
    ),
)
def test_invalid_partition_is_rejected(data, meta, message):
    class Payload:
        value: int
        mode: str

    before = registered_pytree_classes()
    with pytest.raises(ValueError, match=message):
        pytree_dataclass(data=data, meta=meta)(Payload)
    assert registered_pytree_classes() == before


def test_existing_dataclass_options_and_constructor_are_preserved():
    @dataclass(frozen=True, init=False, eq=False)
    class Payload:
        value: int
        derived: int = field(init=False)

        def __init__(self, value):
            object.__setattr__(self, "value", value)
            object.__setattr__(self, "derived", 2 * value)

    assert pytree_dataclass(data=("value",))(Payload) is Payload
    assert Payload(3).derived == 6
    assert Payload(3) != Payload(3)
    leaves, treedef = jax.tree_util.tree_flatten(Payload(3))
    assert jax.tree_util.tree_unflatten(treedef, leaves).derived == 6


def test_registry_returns_immutable_snapshot():
    before = registered_pytree_classes()

    @pytree_dataclass(data=("value",))
    class Payload:
        value: int

    assert isinstance(before, tuple)
    assert registered_pytree_classes() == (*before, Payload)
    assert Payload not in before


@pytest.mark.parametrize(
    "data, meta, partition",
    (
        (("value", "value"), ("mode",), "data"),
        (("value",), ("mode", "mode"), "meta"),
    ),
)
def test_duplicate_partition_names_are_rejected_without_registering(data, meta, partition):
    class Payload:
        value: int
        mode: str

    before = registered_pytree_classes()
    with pytest.raises(ValueError, match=f"duplicate names in {partition}"):
        pytree_dataclass(data=data, meta=meta)(Payload)
    assert registered_pytree_classes() == before
    payload = cast(Callable[..., Payload], Payload)(value=1, mode="test")
    assert jax.tree_util.tree_leaves(payload) == [payload]


def test_undecorated_subclass_is_frozen_and_all_fields_round_trip():
    @dataclass(frozen=True)
    class Base:
        value: int

    @pytree_dataclass(data=("extra", "value"))
    class Child(Base):
        extra: int = 2

    child = Child(value=1, extra=3)
    with pytest.raises(FrozenInstanceError):
        setattr(child, "extra", 9)
    with pytest.raises(FrozenInstanceError):
        setattr(child, "unregistered", 9)
    leaves, treedef = jax.tree_util.tree_flatten(child)
    assert leaves == [3, 1]
    assert jax.tree_util.tree_unflatten(treedef, [5, 4]) == Child(value=4, extra=5)
    assert Child in registered_pytree_classes()


def test_undecorated_subclass_cannot_omit_its_new_fields():
    @dataclass(frozen=True)
    class Base:
        value: int

    class Child(Base):
        extra: int = 2

    before = registered_pytree_classes()
    with pytest.raises(ValueError, match="missing=\\['extra'\\]"):
        pytree_dataclass(data=("value",))(Child)
    assert registered_pytree_classes() == before


def test_custom_node_preserves_sealed_constructor_and_ordered_reconstruction():
    before = registered_pytree_classes()

    @dataclass(frozen=True, slots=True, init=False)
    class Payload:
        first: object
        second: object
        mode: str

        def __init__(self, *args, **kwargs):
            raise RuntimeError("producer only")

        def tree_flatten(self):
            return (self.second, self.first), self.mode

        @classmethod
        def tree_unflatten(cls, mode, children):
            second, first = children
            instance = object.__new__(cls)
            object.__setattr__(instance, "first", first)
            object.__setattr__(instance, "second", second)
            object.__setattr__(instance, "mode", mode)
            return instance

    assert pytree_node(Payload) is Payload
    assert registered_pytree_classes() == (*before, Payload)
    with pytest.raises(RuntimeError, match="producer only"):
        Payload(1, 2, "sum")
    payload = Payload.tree_unflatten("sum", (jnp.asarray(2.0), jnp.asarray(1.0)))
    leaves, treedef = jax.tree_util.tree_flatten(payload)
    assert [float(value) for value in leaves] == [2.0, 1.0]
    restored = jax.tree_util.tree_unflatten(treedef, leaves)
    assert restored.mode == "sum"
    with pytest.raises(FrozenInstanceError):
        restored.first = 3
    assert not hasattr(restored, "__dict__")

    def weighted_sum(value):
        return value.first + 2 * value.second

    assert float(jax.jit(weighted_sum)(restored)) == 5.0
    gradient = jax.grad(weighted_sum)(restored)
    assert gradient.mode == "sum"
    assert [float(value) for value in jax.tree_util.tree_leaves(gradient)] == [2.0, 1.0]


def test_custom_node_rejects_mutable_dataclass_without_registering():
    @dataclass
    class Mutable:
        value: int

        def tree_flatten(self):
            return (self.value,), None

        @classmethod
        def tree_unflatten(cls, _meta, children):
            return cls(*children)

    before = registered_pytree_classes()
    with pytest.raises(TypeError, match="Mutable must be a frozen dataclass"):
        pytree_node(Mutable)
    assert registered_pytree_classes() == before
    payload = Mutable(1)
    assert jax.tree_util.tree_leaves(payload) == [payload]


def test_custom_node_freezes_undecorated_subclass_and_keeps_custom_leaf_order():
    @dataclass(frozen=True)
    class Base:
        first: int

    @pytree_node
    class Payload(Base):
        second: int

        def tree_flatten(self):
            return (self.second, self.first), None

        @classmethod
        def tree_unflatten(cls, _meta, children):
            second, first = children
            return cls(first=first, second=second)

    payload = Payload(first=1, second=2)
    with pytest.raises(FrozenInstanceError):
        setattr(payload, "second", 3)
    leaves, treedef = jax.tree_util.tree_flatten(payload)
    assert leaves == [2, 1]
    assert jax.tree_util.tree_unflatten(treedef, [4, 3]) == Payload(first=3, second=4)


def test_dataclass_reconstruction_calls_post_init():
    constructed = []

    @pytree_dataclass(data=("value",))
    class Payload:
        value: int

        def __post_init__(self):
            constructed.append(self.value)

    leaves, treedef = jax.tree_util.tree_flatten(Payload(1))
    assert leaves == [1]
    assert jax.tree_util.tree_unflatten(treedef, [2]).value == 2
    assert constructed == [1, 2]
