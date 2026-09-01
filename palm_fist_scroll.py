"""Scroll by hand shape: open palm scrolls up, fist scrolls down.

Chosen over pointing gestures because shape is what the tracker separates most
reliably. Earlier designs read a DIRECTION off the hand - which way the thumb
points, how far a fingertip sits from the wrist - and those readings break when
the hand rotates out of the image plane and the fingers foreshorten. An open
palm and a closed fist are opposite extremes of the same single measurement, so
there is no direction to misread and nothing in between could be mistaken for
either.

Neither shape resembles the two-finger volume pose, which needs exactly two
fingers out and two curled - the midpoint between these two.

Direction comes from WHICH pose you make, not from how the hand moves, so the
hand holds still while scrolling. That matters: an earlier drag-to-scroll
design lost the hand in 40.9% of scrolling frames to motion blur, against 5.7%
for a static pose.

What a static pose gives up is analog speed, so speed comes from how long you
hold it instead: slow at first for precise nudges, accelerating to a fast
travel speed, the way holding an arrow key behaves.

The cost is that up/down is arbitrary and has to be learned. Palm is up, on the
reasoning that an open hand is the "more" gesture.
"""

from collections import deque

import numpy as np

from phase2_pose_detect import FRAMES_TO_ENGAGE, MIN_HAND_SIZE_PX
from phase3_scroll import Wheel
from trace_pose import margins

# Releasing takes longer than engaging: a still hand still drops out of
# tracking occasionally, and a dropped frame should not chop the scroll.
FRAMES_TO_RELEASE = 10

# Asymmetric on purpose. Extension is the reliable reading - an extended finger
# is unambiguous - while "curled" is what a foreshortened finger also looks
# like, so the fist demands a clearly negative margin rather than merely
# not-extended.
PALM_EXT = 0.20         # all four clearly extended
FIST_CURL = -0.10       # all four clearly curled

BASE_RATE = 260.0       # wheel units/sec at the start of a hold
MAX_RATE = 1400.0       # after ramping up
RAMP_TIME = 1.6         # seconds to reach full speed
ONSET_DELAY = 0.12      # ignore the first moment, while the pose settles


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


class PalmFistScroller:
    # Swappable so an alternative gesture can reuse the ramp and hysteresis.
    detect = staticmethod(palm_fist_state)

    def __init__(self):
        self.wheel = Wheel()
        self.direction = 0          # current engaged direction
        self.candidate = 0
        self.true_run = 0
        self.false_run = 0
        self.held_since = 0.0
        self.armed = False
        self.ramp = True
        self.base = BASE_RATE
        self.last_emit = 0.0
        self.total = 0.0
        self.holds = 0
        self.reach = 0.0

    def rate(self, now):
        held = now - self.held_since - ONSET_DELAY
        if held <= 0:
            return 0.0
        if not self.ramp:
            return self.base
        t = min(1.0, held / RAMP_TIME)
        # ease-in: gentle at the start where precision matters, fast later
        return self.base + (MAX_RATE - self.base) * (t * t)

    def update(self, state, dt, now):
        d, reach = (0, 0.0)
        if state is not None and state.size >= MIN_HAND_SIZE_PX:
            d, reach = self.detect(state.pts)
        self.reach = reach

        if d != 0 and (self.direction == 0 or d == self.direction):
            self.true_run += 1
            self.false_run = 0
            self.candidate = d
        else:
            self.false_run += 1
            self.true_run = 0

        if self.direction == 0 and self.true_run >= FRAMES_TO_ENGAGE:
            self.direction = self.candidate
            self.held_since = now
            self.holds += 1
        elif self.direction != 0 and self.false_run >= FRAMES_TO_RELEASE:
            self.direction = 0

        emitted = 0.0
        if self.direction:
            # Image y grows downward, but this is a pose not a movement: an
            # open palm means up, which is a POSITIVE wheel delta.
            emitted = self.direction * self.rate(now) * dt
            if self.armed:
                self.wheel.scroll(emitted)
            self.total += abs(emitted)

        self.last_emit = emitted
        return emitted
