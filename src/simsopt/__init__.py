# ===================ATTENTION=================================================
# Don't abuse this file by importing all variables from all modules to top-level.
# Import only the important classes that should be at top-level.
# Follow the same logic in the sub-packages.
# ===================END ATTENTION=============================================
  
try:
    from importlib import metadata
except ImportError:
    # Running on pre-3.8 Python; use importlib-metadata package
    import importlib_metadata as metadata

__version__ = metadata.version('simsopt')

#__all__ = ['LeastSquaresProblem', 'LeastSquaresTerm', 'Surface', 'Target']
