# kinesics

*kinesics* - the study of communication through body and hand movement.

Webcam hand-gesture control for Windows — scroll and volume, no extra hardware.
Inspired by BMW's in-car gesture control, which uses a time-of-flight sensor;
this does it with the laptop camera you already have.

Runs in the background from a tray icon with a small always-on-top preview.

## Gestures

| Gesture | Action |
|---|---|
| Open palm, held | scroll up |
| Fist, held | scroll down |
| Open palm, tapped | page up |
| Fist, tapped | page down |
| Two fingers up, rotate your hand like a dial | volume |
| Horns - index and pinky up, middle and ring down | toggle mute |

Scroll speed ramps the longer you hold — gentle at first for nudging a line or
two, accelerating to a fast travel speed. Volume is continuous, roughly 85° of
rotation per half the range.

## Requirements

Windows, a webcam, and Python 3.11–3.12 (MediaPipe has no wheels for 3.13+).

```
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The hand-landmark model (~7.8 MB) downloads from Google's MediaPipe model host
on first run.

## Running

```
.venv\Scripts\pythonw.exe gesture_scroll.py            background, tray icon
.venv\Scripts\python.exe  gesture_scroll.py --console  with a log console
.venv\Scripts\python.exe  gesture_scroll.py --install-startup
```

`ctrl+alt+G` arms/disarms, `ctrl+alt+Q` quits. It starts armed by default.

The camera is opened only while armed — disarmed means the webcam LED is off
and no frames are read.

## Settings

`settings.json`, read at startup:

| Key | Meaning |
|---|---|
| `target` | `cursor` sends real wheel events to the window under the pointer (works everywhere). `foreground` posts to the focused window instead, so the pointer does not matter — but Chrome and Electron apps may ignore posted messages. |
| `armed_on_start` | open the camera at launch |
| `overlay` | show the always-on-top preview |
| `base_rate` / `max_rate` / `ramp` | scroll speed and acceleration |
| `volume` / `volume_per_degree` | volume dial, and its sensitivity |
| `mute` | horns mute toggle |
| `tap_to_page` | a brief palm/fist pages instead of scrolling |
| `hotkey` | e.g. `ctrl+alt+G`, `ctrl+shift+F9`, `alt+HOME` |

## How it works

```
webcam -> MediaPipe hand landmarks -> geometric pose rule -> hysteresis gate -> SendInput / Core Audio
```

Pose recognition is **pure geometry, no machine learning**. Each finger is
extended if its tip is farther from the wrist than its PIP joint, normalised by
hand size. Measured against 359 enrolled frames spanning a 3.6× size range and
−51° to +66° of hand tilt, this rule classified every frame correctly, and beat
three alternatives (PIP joint angle, tip-to-MCP distance, straightness ratio) —
the nearest had a separation of d=2.3 versus d=8.8.

Paging uses duration rather than another hand shape. The finger-count space
is full, and a hand moving between any two shapes sweeps through the ones
between - the cause of every gesture collision in this project. Duration is
an independent axis that cannot be passed through by accident. Release
within ~0.4 s and it sends Page Up/Down as a real key press, which follows
keyboard focus and so needs no window targeting at all.

Mute is edge-triggered: it fires once per gesture and re-arms only after
the hand has clearly let go, so a long hold cannot flip it repeatedly.

The mute pose is horns rather than a single raised finger. One finger was
tried first and fired constantly while scrolling, because a hand opening or
closing passes through it - the index is usually first to extend and last to
curl, so it sits alone longer than any reasonable hold threshold. Horns
requires non-adjacent fingers in opposite states, which a hand in transit
between palm and fist never produces.

Volume goes through Core Audio as a continuous scalar rather than
`VK_VOLUME_UP` key presses, which move in fixed 2% steps and would make a
smooth rotation feel like a ratchet.

## Things measured the hard way

Each of these changed the design:

- **A still hand beats a moving one.** The first design was drag-to-scroll.
  Downward strokes ran ~1.5× faster than upward ones, motion blurred the frame,
  and tracking dropped mid-stroke. Switching to a static pose cut hand-loss
  during scrolling from **40.9% to 5.7%**.
- **Fingers you must hold curled get tired and drift open.** A two-finger pose
  broke into 0.3 s fragments as the ring and pinky crept open (ring margin
  −0.255 while held, +0.100 when it broke). A Schmitt trigger — strict to
  engage, loose to stay engaged — took median hold from 0.37 s to 1.93 s.
- **2D finger extension breaks under rotation.** Rotating the hand
  foreshortens extended fingers so they read as curled, which made the volume
  gesture briefly impersonate the old thumbs pose. Hence palm/fist, which are
  opposite extremes of one measurement with no direction to misread, plus a
  lockout while the dial is engaged.
- **Finger spread is too noisy to build on** — 2.4σ separation, σ=0.26.

## Files

| File | |
|---|---|
| `gesture_scroll.py` | the daemon: tray icon, hotkeys, overlay, scroll targeting |
| `palm_fist_scroll.py` | the scroll gesture, its speed ramp and hysteresis |
| `volume.py` | the volume dial |
| `phase1_landmark_view.py` | camera thread, landmarker setup, diagnostic viewer |
| `phase2_pose_detect.py` | the pose gate and its hysteresis |
| `enroll.py` | guided enrollment — walks you through hand variations, records them |
| `trace_pose.py` | per-frame diagnosis of why a pose is or is not recognised |
| `measure.py`, `analyze.py`, `metric_compare.py`, `replay_gate.py` | the measurement tools behind the numbers above |

`enroll.py` is worth running once: it prompts you through tilts, distances and
both hands like a phone's face-unlock setup, and confirms tracking works by
requiring you to reach each position on command.

## Known limitations

- Windows only (Core Audio and `SendInput`).
- `target: foreground` may not work in Chrome or Electron apps; use `cursor`.
- False-positive rate in ordinary use has not been measured. Every gesture
  logged so far was deliberate.
