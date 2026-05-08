#!/usr/bin/env python3
"""Canonical training entry point.

The implementation lives in control_train.py, which supports the main
mHC+MTP experiment and the control models.  This wrapper keeps the documented
`python train.py ...` command working.
"""

from control_train import main


if __name__ == "__main__":
    main()
