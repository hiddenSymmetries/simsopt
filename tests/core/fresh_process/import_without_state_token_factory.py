"""Fresh-process child: import a JAX test module whose import is broken.

``make_state_token_factory`` is removed before ``core.test_state_tokens``
imports it, so that import must raise ``ImportError`` even though the optional
JAX runtime is supported. The test module import deliberately follows the
removal at module level.
"""

import simsopt_jax.core.state_tokens as state_tokens

del state_tokens.make_state_token_factory

import core.test_state_tokens  # noqa: F401
