"""Volume as a dial: hold two fingers up and rotate your hand.

The gesture BMW uses is a rotating finger, and rotation is the right shape for
volume - it is continuous, has no natural end stop, and maps to a knob everyone
has already used.

Why this pose and this measurement:

  The two-finger pose (index + middle up, ring + pinky curled) was measured at
  100% detection across the whole enrollment set, and it is not used by the
  scroll gestures, so the two cannot be confused.

  Rotation is read from the wrist -> middle-knuckle angle, which enrollment
  showed is stable and spans a wide range naturally (-51 to +66 degrees just
  from ordinary hand tilt). It needs no extra tracking machinery.

  Volume is set as a continuous scalar through Core Audio rather than by
  sending VK_VOLUME_UP keys, which move in fixed 2% steps and would make a
  smooth rotation feel like a ratchet.

Rotate clockwise to raise, anticlockwise to lower.
"""

import numpy as np

from phase2_pose_detect import FRAMES_TO_ENGAGE, MIN_HAND_SIZE_PX, pose_ok
from trace_pose import margins

# Rotating the hand foreshortens the extended fingers, so the pose test
# briefly fails mid-gesture. A short release window turns that into a
# dropped dial - and worse, the same frames look like a thumbs pose.
FRAMES_TO_RELEASE = 22
DEADZONE_DEG = 0.6          # below this it is landmark noise, not intent
VOL_PER_DEGREE = 0.006      # ~85 degrees of rotation covers half the range
ONSET_FRAMES = 3            # ignore rotation while the pose settles
MAX_STEP_DEG = 25.0         # reject implausible jumps (tracking glitches)


class SystemVolume:
    """Core Audio master volume, with the COM object opened lazily."""

    def __init__(self):
        self._vol = None
        self.available = True

    def _iface(self):
        if self._vol is None:
            from pycaw.utils import AudioUtilities
            self._vol = AudioUtilities.GetSpeakers().EndpointVolume
        return self._vol

    def get(self):
        try:
            return float(self._iface().GetMasterVolumeLevelScalar())
        except Exception:
            self.available = False
            return 0.0

    def set(self, value):
        value = float(np.clip(value, 0.0, 1.0))
        try:
            self._iface().SetMasterVolumeLevelScalar(value, None)
            return value
        except Exception:
            self.available = False
            return value


def angle_delta(new, old):
    """Shortest signed rotation from old to new, in degrees.

    Straight subtraction breaks when the angle wraps past +-180: a hand moving
    a few degrees would read as a 350 degree spin and slam the volume.
    """
    return (new - old + 180.0) % 360.0 - 180.0


class VolumeDial:
    def __init__(self, volume=None):
        self.vol = volume or SystemVolume()
        self.engaged = False
        self.true_run = 0
        self.false_run = 0
        self.frames = 0
        self.last_angle = None
        self.level = None           # cached so we do not read COM every frame
        self.rotated = 0.0          # cumulative degrees this gesture
        self.gestures = 0
        self.enabled = True
        self.per_degree = VOL_PER_DEGREE

    def update(self, state):
        """Feed one frame. Returns the current level, or None if not engaged."""
        ok = (state is not None and state.size >= MIN_HAND_SIZE_PX
              and pose_ok(margins(state.pts), strict=not self.engaged))

        if ok:
            self.true_run += 1
            self.false_run = 0
        else:
            self.false_run += 1
            self.true_run = 0

        if not self.engaged and self.true_run >= FRAMES_TO_ENGAGE:
            self._engage(state)
        elif self.engaged and self.false_run >= FRAMES_TO_RELEASE:
            self.engaged = False
            self.last_angle = None
            return None

        if not self.engaged or state is None:
            return self.level if self.engaged else None

        self.frames += 1
        delta = angle_delta(state.angle, self.last_angle)
        self.last_angle = state.angle

        # Same reason the scroll gate latches an origin: the frames while the
        # hand is rising into the pose carry rotation that was never intended.
        if self.frames <= ONSET_FRAMES:
            return self.level
        if abs(delta) < DEADZONE_DEG or abs(delta) > MAX_STEP_DEG:
            return self.level

        self.rotated += delta
        if self.enabled:
            self.level = self.vol.set(self.level + delta * self.per_degree)
        else:
            self.level = float(np.clip(self.level + delta * self.per_degree, 0.0, 1.0))
        return self.level

    def _engage(self, state):
        self.engaged = True
        self.gestures += 1
        self.frames = 0
        self.rotated = 0.0
        self.last_angle = state.angle
        self.level = self.vol.get()
