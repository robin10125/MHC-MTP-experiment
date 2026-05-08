"""Importable experiment API.

The original combined experiment implementation is kept in
`combined-mhc-mtp.py`.  This module exposes its public classes/functions under
an import-safe Python module name so training code can use:

    from experiment import ExperimentConfig, build_model
"""

from experiment_core import *  # noqa: F401,F403
