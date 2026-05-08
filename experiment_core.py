"""Loader for the combined mHC + MTP experiment module.

`combined-mhc-mtp.py` has a hyphen in its filename, so it cannot be imported
with a normal Python import statement.  This shim loads it once and re-exports
the symbols used by the training and control code.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


_MODULE_PATH = Path(__file__).with_name("combined-mhc-mtp.py")
_SPEC = importlib.util.spec_from_file_location("_combined_mhc_mtp", _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Could not load {_MODULE_PATH}")

_module = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_module)


ExperimentConfig = _module.ExperimentConfig
SumReduction = _module.SumReduction
SinkhornMixReduction = _module.SinkhornMixReduction
mHCTrunk = _module.mHCTrunk
mHCWithMTP = _module.mHCWithMTP
build_model = _module.build_model
make_batch = _module.make_batch
train_step = _module.train_step
run_condition = _module.run_condition
compare_conditions = _module.compare_conditions
