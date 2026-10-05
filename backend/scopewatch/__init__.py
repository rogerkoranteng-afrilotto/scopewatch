"""Scopewatch - measurement on the laparoscopic operating field.

Scopewatch watches the camera that is already in the room. It measures the blood
on the field with an uncertainty range instead of leaving it to the eye, it
timestamps when the rate of change crosses a threshold, it infers the operative
phase from the instruments and the scene, and when the phase says the irreversible
step is near without a recorded safety view it holds a checkpoint that a person
must confirm or dismiss with a reason.

It is decision support and a retrospective measurement instrument. It is not a
medical device, it has never been used in a field trial, and it takes no clinical
action of any kind. See docs/report.md, "Responsible use".
"""

from __future__ import annotations

from visioncore import assert_opencv5

assert_opencv5()

__version__ = "1.0.0"

__all__ = ["__version__"]
