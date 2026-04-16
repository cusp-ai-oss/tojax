# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0
import traceback


def get_caller() -> str:
    """Returns file, line number, and code of the caller's caller."""
    stack = traceback.extract_stack()
    # stack[-1] = get_caller, stack[-2] = direct caller, stack[-3] = caller's caller
    frame = stack[-3]
    return f"{frame.filename}:{frame.lineno} {frame.line}"
