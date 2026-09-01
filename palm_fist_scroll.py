"""Scroll by hand shape: open palm scrolls up, fist scrolls down.

Chosen over pointing gestures because shape is what the tracker separates most
reliably. Every earlier failure came from reading a DIRECTION off the hand -
which way the thumb points, how far a fingertip sits from the wrist - and those
readings break when the hand rotates out of the image plane and the fingers
foreshorten. An open palm and a closed fist are opposite extremes of the same
single measurement, so there is no direction to misread and nothing in between
that could be mistaken for either.

Neither shape resembles the two-finger volume pose, which needs exactly two
fingers out and two curled - the midpoint between these two.

The cost is that up/down is arbitrary and has to be learned. Palm is up on the
reasoning that an open hand is the "more" gesture.
"""

import numpy as np

from phase3_thumb_scroll import ThumbScroller
from trace_pose import margins

# Asymmetric on purpose. Extension is the reliable reading - an extended finger
# is unambiguous - while "curled" is what a foreshortened finger also looks
# like, so the fist demands a clearly negative margin rather than merely
# not-extended.
PALM_EXT = 0.20         # all four clearly extended
FIST_CURL = -0.10       # all four clearly curled


def palm_fist_state(pts):
    """+1 open palm (scroll up), -1 fist (scroll down), 0 otherwise.

    Second value is the weakest finger's margin, i.e. how close the pose came
    to not counting - shown in the overlay so a rejected pose is diagnosable.
    """
    m = margins(pts)
    vals = [m[f] for f in ("index", "middle", "ring", "pinky")]

    if all(v > PALM_EXT for v in vals):
        return 1, float(min(vals))
    if all(v < FIST_CURL for v in vals):
        return -1, float(max(vals))
    return 0, float(np.mean(vals))


class PalmFistScroller(ThumbScroller):
    detect = staticmethod(palm_fist_state)
