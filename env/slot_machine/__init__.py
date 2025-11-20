"""Minimal `slot_machine` package to provide MultiArmedBandit for local runs.

This is a lightweight implementation intended to satisfy imports for
development/testing. It mimics the expected interface used by the
environment classes in `env/`.
"""

from .MultiArmedBandit import MultiArmedBandit

__all__ = ["MultiArmedBandit"]
