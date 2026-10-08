from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

from simsopt_jax.core.state_tokens import make_state_token_factory


def test_state_token_factories_are_independent_monotonic_sequences():
    first = make_state_token_factory()
    second = make_state_token_factory()

    assert [first(), first(), first()] == [0, 1, 2]
    assert [second(), second()] == [0, 1]
    assert first() == 3
