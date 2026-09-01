"""Gesture scroll, running in the background.

No window. A tray icon shows state, a global hotkey arms and disarms it from
anywhere, and the camera is only opened while armed - so when it is off, the
webcam LED is off, the CPU is idle, and nothing is watching.

    pythonw gesture_scroll.py              run in the background (no console)
    python  gesture_scroll.py --console    same, but keep a console for logs
    python  gesture_scroll.py --install-startup
    python  gesture_scroll.py --remove-startup

Hotkeys (work from any application):
    ctrl+alt+G   arm / disarm
    ctrl+alt+Q   quit

Gestures:
    open palm                                     scroll up
    fist                                          scroll down
    two fingers up, rotate your hand like a dial  volume
    horns (index + pinky up, middle + ring down)  toggle mute

Scrolling goes to the window under the mouse cursor, or to the focused window
if "target" is set to "foreground" in settings.json.
"""

import ctypes
import json
import os
import sys
import threading
import time
from ctypes import wintypes
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from palm_fist_scroll import FIST_CURL, PALM_EXT
from phase1_landmark_view import CONNECTIONS, FINGERTIPS

# --------------------------------------------------------------------------
# global hotkeys
# --------------------------------------------------------------------------

MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_NOREPEAT = 0x4000
WM_HOTKEY = 0x0312

HOTKEY_TOGGLE = 1
HOTKEY_QUIT = 2


class HotkeyListener(threading.Thread):
    """RegisterHotKey needs its own message pump, and the pump must live in the
    same thread that registered the keys - hence a dedicated thread."""

    def __init__(self, on_toggle, on_quit, binding):
        super().__init__(daemon=True)
        self.binding = binding
        self.on_toggle = on_toggle
        self.on_quit = on_quit
        self.ready = threading.Event()
        self.ok = False
        self._thread_id = None

    def run(self):
        user32 = ctypes.windll.user32
        self._thread_id = ctypes.windll.kernel32.GetCurrentThreadId()
        mods, vk = self.binding
        a = user32.RegisterHotKey(None, HOTKEY_TOGGLE, mods, vk)
        b = user32.RegisterHotKey(None, HOTKEY_QUIT, MOD_CONTROL | MOD_ALT | MOD_NOREPEAT, 0x51)
        self.ok = bool(a and b)
        self.ready.set()
        if not self.ok:
            return

        msg = wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            if msg.message == WM_HOTKEY:
                if msg.wParam == HOTKEY_TOGGLE:
                    self.on_toggle()
                elif msg.wParam == HOTKEY_QUIT:
                    self.on_quit()
                    break
        user32.UnregisterHotKey(None, HOTKEY_TOGGLE)
        user32.UnregisterHotKey(None, HOTKEY_QUIT)

    def stop(self):
        if self._thread_id:
            ctypes.windll.user32.PostThreadMessageW(self._thread_id, 0x0012, 0, 0)  # WM_QUIT


# --------------------------------------------------------------------------
# tray icon
# --------------------------------------------------------------------------

COLORS = {
    "off": (110, 110, 120),
    "armed": (70, 160, 240),
    "up": (90, 210, 110),
    "down": (90, 210, 110),
}


def make_icon(state):
    """Small hand-ish glyph: a bar with an arrow showing what is happening."""
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    c = COLORS[state]
    d.ellipse((4, 4, 60, 60), fill=(*c, 60), outline=(*c, 255), width=3)
    if state == "up":
        d.polygon([(32, 16), (46, 38), (18, 38)], fill=(*c, 255))
        d.rectangle((28, 38, 36, 50), fill=(*c, 255))
    elif state == "down":
        d.polygon([(32, 48), (46, 26), (18, 26)], fill=(*c, 255))
        d.rectangle((28, 14, 36, 26), fill=(*c, 255))
    elif state == "armed":
        d.ellipse((26, 26, 38, 38), fill=(*c, 255))
    else:
        d.line((22, 32, 42, 32), fill=(*c, 255), width=5)
    return img


# --------------------------------------------------------------------------
# where the scroll goes
# --------------------------------------------------------------------------

WM_MOUSEWHEEL = 0x020A
WHEEL_DELTA = 120


class ForegroundWheel:
    """Scroll the focused window instead of the one under the pointer.

    Injected wheel events (the default) always go where the mouse pointer is,
    like a real wheel - which is surprising when your hand is nowhere near the
    mouse. This posts WM_MOUSEWHEEL straight to the foreground window instead.

    The tradeoff is real: posted messages are synthetic, and applications that
    do their own input handling - Chrome and most Electron apps - may ignore
    them entirely. Whole notches only, since apps receiving a posted message do
    their own integer division by WHEEL_DELTA and would floor a fraction to 0.
    """

    def __init__(self, log=None):
        self.residual = 0.0
        self.log = log
        self.last_other = None
        self.pid = ctypes.windll.kernel32.GetCurrentProcessId()

    def _is_ours(self, hwnd):
        pid = ctypes.c_ulong()
        ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        return pid.value == self.pid

    def note_foreground(self):
        """Remember the last foreground window that is not one of ours.

        Called every frame. Without it the overlay - which is topmost and can
        take focus - becomes the target, and the scroll lands in our own
        preview instead of the user's document.
        """
        hwnd = ctypes.windll.user32.GetForegroundWindow()
        if hwnd and not self._is_ours(hwnd):
            self.last_other = hwnd

    def target_window(self):
        hwnd = ctypes.windll.user32.GetForegroundWindow()
        if hwnd and not self._is_ours(hwnd):
            return hwnd
        if self.last_other and ctypes.windll.user32.IsWindow(self.last_other):
            return self.last_other
        # Cold start: our own overlay was foreground and nothing else has been
        # seen yet, so there is no remembered window. Without this the first
        # gestures of a session are silently dropped.
        return self._topmost_other()

    def _topmost_other(self):
        """Highest window in z-order that is visible, titled, and not ours."""
        user32 = ctypes.windll.user32
        found = []

        @ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
        def visit(hwnd, _):
            if (user32.IsWindowVisible(hwnd) and not user32.IsIconic(hwnd)
                    and not self._is_ours(hwnd) and user32.GetWindowTextLengthW(hwnd) > 0):
                found.append(hwnd)
                return False           # z-order means the first hit is the top
            return True

        user32.EnumWindows(visit, 0)
        return found[0] if found else None

    def scroll(self, units):
        self.residual += units
        notches = int(self.residual / WHEEL_DELTA)
        if notches == 0:
            return
        self.residual -= notches * WHEEL_DELTA

        user32 = ctypes.windll.user32
        hwnd = self.target_window()
        if not hwnd or not user32.IsWindow(hwnd):
            return

        rect = wintypes.RECT()
        user32.GetWindowRect(hwnd, ctypes.byref(rect))
        x = (rect.left + rect.right) // 2
        y = (rect.top + rect.bottom) // 2

        # Deliver to the child at that point - posting to the outer frame often
        # goes nowhere. RealChildWindowFromPoint searches only inside hwnd, so
        # unlike WindowFromPoint it cannot return our own topmost overlay when
        # the overlay happens to sit over the target's centre.
        client = wintypes.POINT((rect.right - rect.left) // 2,
                                (rect.bottom - rect.top) // 2)
        target = user32.RealChildWindowFromPoint(hwnd, client) or hwnd

        delta = notches * WHEEL_DELTA
        wparam = (delta & 0xFFFF) << 16
        lparam = (y << 16) | (x & 0xFFFF)
        user32.PostMessageW(target, WM_MOUSEWHEEL, wparam, lparam)


def window_title(hwnd):
    user32 = ctypes.windll.user32
    buf = ctypes.create_unicode_buffer(120)
    user32.GetWindowTextW(user32.GetAncestor(hwnd, 2) or hwnd, buf, 120)
    return buf.value[:60] or "<no title>"


def describe_target(settings, wheel=None):
    """Name the window that just received the scroll, for the log."""
    user32 = ctypes.windll.user32
    if str(settings.get("target", "cursor")).lower() == "foreground":
        hwnd = wheel.target_window() if wheel else user32.GetForegroundWindow()
        if not hwnd:
            return "foreground window: <none found>"
        return f"foreground window: {window_title(hwnd)}"
    pt = wintypes.POINT()
    user32.GetCursorPos(ctypes.byref(pt))
    hwnd = user32.WindowFromPoint(pt)
    return f"window under cursor ({pt.x},{pt.y}): {window_title(hwnd)}"


def make_wheel(settings, log):
    from phase3_scroll import Wheel
    if str(settings.get("target", "cursor")).lower() == "foreground":
        log("scroll target: foreground window (posted messages; some apps ignore these)")
        return ForegroundWheel(log)
    log("scroll target: window under the mouse pointer")
    return Wheel()


# --------------------------------------------------------------------------
# on-screen overlay
# --------------------------------------------------------------------------

OVERLAY = "gesture scroll"

GWL_EXSTYLE = -20
WS_EX_NOACTIVATE = 0x08000000
WS_EX_TOOLWINDOW = 0x00000080


def make_non_activating(title):
    """Stop the overlay from ever taking focus.

    It is topmost, so without this it becomes the foreground window and the
    'scroll the focused window' path targets our own preview. TOOLWINDOW also
    keeps it out of alt-tab, where it has no business being.
    """
    user32 = ctypes.windll.user32
    hwnd = user32.FindWindowW(None, title)
    if not hwnd:
        return False
    style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
    user32.SetWindowLongW(hwnd, GWL_EXSTYLE,
                          style | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW)
    return True


def draw_overlay(frame, sc, state, dial=None, muter=None):
    """Small live view: what the camera sees, what the rule thinks, what it did.

    Every stage is shown separately, because a failure at any one of them looks
    identical from the outside - nothing scrolls.
    """
    import cv2

    view = cv2.resize(frame, (360, 202))
    h, w = view.shape[:2]

    if state is not None:
        # Full skeleton. The gestures are read from all four fingers, so the
        # overlay has to show all four - a partial drawing makes a misread
        # finger invisible, which is the one thing this view exists to catch.
        s = w / frame.shape[1]
        px = (state.pts * s).astype(int)
        for a, b in CONNECTIONS:
            cv2.line(view, tuple(px[a]), tuple(px[b]), (90, 200, 120), 2, cv2.LINE_AA)
        for i in FINGERTIPS:
            cv2.circle(view, tuple(px[i]), 4, (60, 90, 255), -1, cv2.LINE_AA)

    if muter is not None and muter.progress > 0:
        label = f"MUTE... {muter.progress * 100:3.0f}%"
        color = (120, 160, 255)
    elif muter is not None and muter.muted:
        label, color = "MUTED", (120, 160, 255)
    elif dial is not None and dial.engaged:
        label = f"VOLUME {(dial.level or 0) * 100:3.0f}%"
        color = (240, 190, 90)
    elif sc.direction > 0:
        label, color = "SCROLLING UP", (110, 235, 110)
    elif sc.direction < 0:
        label, color = "SCROLLING DOWN", (110, 235, 110)
    elif state is None:
        label, color = "no hand seen", (120, 120, 130)
    else:
        label, color = "hand seen - palm / fist / horns", (0, 190, 240)

    cv2.rectangle(view, (0, 0), (w, 26), (0, 0, 0), -1)
    cv2.putText(view, label, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)
    if sc.direction or (dial is not None and dial.engaged):
        cv2.rectangle(view, (0, 0), (w - 1, h - 1), color, 4)

    cv2.rectangle(view, (0, h - 22), (w, h), (0, 0, 0), -1)
    cv2.putText(view, f"pose {sc.reach:+.2f} (palm >{PALM_EXT} / fist <{FIST_CURL})"
                f"   holds {sc.holds}",
                (8, h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (185, 185, 185), 1, cv2.LINE_AA)
    return view


# --------------------------------------------------------------------------
# the worker
# --------------------------------------------------------------------------

class Engine:
    """Owns the camera and the model, and only while armed.

    Releasing them on disarm is the point: no webcam LED, no CPU cost, and no
    frames being read when the user has not asked for it.
    """

    def __init__(self, log, settings):
        self.log = log
        self.settings = settings
        self.armed = False
        self.state = "off"
        self.running = True
        self.thread = None
        self.lock = threading.Lock()
        self.on_state = lambda s: None
        self.holds = 0
        self.units = 0.0
        self.saw_hand = False
        self.hold_start_units = 0.0
        self.dial = None
        self.muter = None

    def toggle(self):
        with self.lock:
            self.armed = not self.armed
            if self.armed:
                self.thread = threading.Thread(target=self._loop, daemon=True)
                self.thread.start()
                self.log("armed")
            else:
                self.log("disarmed")
        self._set("armed" if self.armed else "off")

    def shutdown(self):
        self.running = False
        self.armed = False

    def _set(self, s):
        if s != self.state:
            self.state = s
            self.on_state(s)

    def _loop(self):
        # Imported lazily so startup is instant and nothing heavy loads until
        # the user actually arms it.
        from enroll import analyse
        from phase1_landmark_view import CameraThread, build_landmarker, ensure_model
        from phase2_pose_detect import MIN_HAND_SIZE_PX
        from palm_fist_scroll import PalmFistScroller
        import cv2
        import mediapipe as mp

        ensure_model()
        try:
            cam = CameraThread()
        except RuntimeError as e:
            self.log(f"camera unavailable: {e}")
            self.armed = False
            self._set("off")
            return

        landmarker = build_landmarker()
        sc = PalmFistScroller()
        sc.armed = True
        sc.base = float(self.settings['base_rate'])
        sc.ramp = bool(self.settings['ramp'])
        sc.wheel = make_wheel(self.settings, self.log)

        # Volume wins while it is engaged, and for a moment after. A rotating
        # hand transiently reads as a fist (foreshortened fingers look
        # curled), so without a lockout every volume gesture ends in a
        # spurious scroll down.
        volume_until = 0.0
        VOLUME_COOLDOWN = 1.2
        # And the mirror of it: a hand that just finished scrolling is on its
        # way through the two-finger shape, not asking for the dial.
        scroll_until = 0.0
        SCROLL_COOLDOWN = 0.7

        dial = None
        if self.settings.get('volume', True):
            from volume import VolumeDial
            dial = VolumeDial()
            dial.per_degree = float(self.settings.get('volume_per_degree', 0.006))
            self.log('volume dial on: two fingers up, rotate to change volume')

        muter = None
        if self.settings.get('mute', True):
            from volume import MuteToggle
            muter = MuteToggle()
            self.log('mute on: hold horns (index + pinky up) to toggle')
        last_stamp, idx = 0.0, 0

        # A small always-on-top preview. Without it the daemon is a black box:
        # a gesture can be detected, the scroll computed, and nothing visibly
        # happen, with no way to tell which stage failed.
        overlay = bool(self.settings.get("overlay", True))
        if overlay:
            cv2.namedWindow(OVERLAY, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(OVERLAY, 360, 202)
            cv2.setWindowProperty(OVERLAY, cv2.WND_PROP_TOPMOST, 1)
            cv2.moveWindow(OVERLAY, 40, 40)
            cv2.waitKey(1)
            make_non_activating(OVERLAY)

        try:
            while self.running and self.armed:
                frame, stamp = cam.read()
                if frame is None or stamp == last_stamp:
                    time.sleep(0.003)
                    continue
                dt = min(stamp - last_stamp, 0.2) if last_stamp else 1 / 30
                last_stamp = stamp

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = landmarker.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                    int(idx * 1000 / 30))
                idx += 1

                h, w = frame.shape[:2]
                st = None
                if res.hand_landmarks:
                    pts = np.array([[p.x * w, p.y * h] for p in res.hand_landmarks[0]])
                    st = analyse(pts, res.handedness[0][0].category_name)

                if hasattr(sc.wheel, "note_foreground"):
                    sc.wheel.note_foreground()

                if sc.direction:
                    scroll_until = time.perf_counter() + SCROLL_COOLDOWN

                if dial is not None:
                    was = dial.engaged
                    dial.update(None if time.perf_counter() < scroll_until
                                and not dial.engaged else st)
                    if dial.engaged:
                        volume_until = time.perf_counter() + VOLUME_COOLDOWN
                    if dial.engaged and not was:
                        self.log(f'VOLUME dial engaged at {(dial.level or 0)*100:.0f}%')
                    elif was and not dial.engaged:
                        self.log(f'  volume {(dial.level or 0)*100:.0f}% '
                                 f'(rotated {dial.rotated:+.0f} deg)')
                    self.dial = dial

                if muter is not None:
                    toggled = muter.update(st)
                    if toggled is not None:
                        self.log(f"MUTE {'on' if toggled else 'off'}")
                    if muter.held:
                        volume_until = time.perf_counter() + 0.4
                    self.muter = muter

                # Hide the hand from the scroller while volume owns it.
                st_for_scroll = None if time.perf_counter() < volume_until else st

                before_dir = sc.direction
                before_units = sc.total
                sc.update(st_for_scroll, dt, time.perf_counter())
                self.holds = sc.holds
                self.units = sc.total

                # Log every stage transition. "Nothing scrolls" looks identical
                # whether the hand is unseen, the pose is rejected, or the
                # events go to the wrong window - these lines separate them.
                if sc.direction != before_dir:
                    if sc.direction:
                        self.log(f"{'PALM -> up' if sc.direction > 0 else 'FIST -> down'} "
                                 f"(margin {sc.reach:+.2f})")
                    else:
                        self.log(f"  sent {sc.total - self.hold_start_units:.0f} units "
                                 f"-> {describe_target(self.settings, sc.wheel)}")
                    self.hold_start_units = sc.total

                seen_hand = st is not None
                if seen_hand != self.saw_hand:
                    self.saw_hand = seen_hand
                    self.log("hand in view" if seen_hand else "hand left view")

                self._set("up" if sc.direction > 0 else
                          "down" if sc.direction < 0 else "armed")

                if overlay:
                    cv2.imshow(OVERLAY, draw_overlay(frame, sc, st, dial, muter))
                    if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                        self.armed = False
        finally:
            if overlay:
                cv2.destroyWindow(OVERLAY)
            cam.release()
            landmarker.close()
            self._set("off")
            self.log(f"camera released ({self.holds} holds, {self.units / 120:.0f} notches)")


# --------------------------------------------------------------------------
# autostart
# --------------------------------------------------------------------------

def startup_path():
    return (Path(os.environ["APPDATA"]) / "Microsoft" / "Windows" / "Start Menu"
            / "Programs" / "Startup" / "gesture-scroll.bat")


def install_startup():
    here = Path(__file__).resolve().parent
    pyw = here / ".venv" / "Scripts" / "pythonw.exe"
    if not pyw.exists():
        print(f"pythonw not found at {pyw}")
        return
    p = startup_path()
    p.write_text(f'@echo off\r\nstart "" "{pyw}" "{here / "gesture_scroll.py"}"\r\n')
    print(f"installed {p}")
    print("It will start disarmed on next login; ctrl+alt+G to arm.")


def remove_startup():
    p = startup_path()
    if p.exists():
        p.unlink()
        print(f"removed {p}")
    else:
        print("not installed")


# --------------------------------------------------------------------------

DEFAULTS = {
    # Start armed. Arming was the single point of failure in testing: the
    # gesture was recognised and the scroll computed, then discarded because
    # the toggle had never fired. Requiring a hotkey that may be swallowed by
    # another app, before anything can work at all, is a bad default.
    "armed_on_start": True,
    "base_rate": 260.0,
    "max_rate": 1400.0,
    "ramp": True,
    "hotkey": "ctrl+alt+G",
    "overlay": True,
    "target": "cursor",   # "cursor" or "foreground"
    "volume": True,       # two fingers up, rotate like a dial
    "mute": True,         # one finger up, held, toggles mute
    "volume_per_degree": 0.006,
}

SETTINGS_PATH = Path(__file__).parent / "settings.json"

VK = {**{c: 0x41 + i for i, c in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ")},
      **{str(d): 0x30 + d for d in range(10)},
      "F1": 0x70, "F2": 0x71, "F3": 0x72, "F4": 0x73, "F5": 0x74, "F6": 0x75,
      "F7": 0x76, "F8": 0x77, "F9": 0x78, "F10": 0x79, "F11": 0x7A, "F12": 0x7B,
      "SPACE": 0x20, "INSERT": 0x2D, "HOME": 0x24, "END": 0x23}
MOD_SHIFT = 0x0004
MOD_WIN = 0x0008


def load_settings():
    s = dict(DEFAULTS)
    if SETTINGS_PATH.exists():
        try:
            s.update(json.loads(SETTINGS_PATH.read_text()))
        except (ValueError, OSError):
            pass                      # a broken file must not stop it starting
    return s


def parse_hotkey(spec):
    """'ctrl+alt+G' -> (modifier bits, virtual key code), or None if unparseable."""
    parts = [p.strip().lower() for p in spec.split("+") if p.strip()]
    if not parts:
        return None
    mods = 0
    for p in parts[:-1]:
        mods |= {"ctrl": MOD_CONTROL, "control": MOD_CONTROL, "alt": MOD_ALT,
                 "shift": MOD_SHIFT, "win": MOD_WIN}.get(p, 0)
    key = VK.get(parts[-1].upper())
    return (mods | MOD_NOREPEAT, key) if key else None


ERROR_ALREADY_EXISTS = 183


def claim_single_instance():
    """RegisterHotKey is exclusive, so a second copy silently loses the
    hotkeys and leaves two cameras fighting over the device. Refuse to start
    instead. The handle is deliberately leaked - it lives as long as we do."""
    ctypes.windll.kernel32.CreateMutexW(None, False, "Local\\gesture-scroll-single")
    return ctypes.windll.kernel32.GetLastError() != ERROR_ALREADY_EXISTS


def main():
    if "--install-startup" in sys.argv:
        return install_startup()
    if "--remove-startup" in sys.argv:
        return remove_startup()

    console = "--console" in sys.argv
    settings = load_settings()
    if "--armed" in sys.argv:
        settings["armed_on_start"] = True
    if "--disarmed" in sys.argv:
        settings["armed_on_start"] = False
    if not claim_single_instance():
        msg = "kinesics is already running (check the tray icon)"
        if console:
            print(msg)
        else:
            ctypes.windll.user32.MessageBoxW(None, msg, "kinesics", 0x40)
        return
    log_file = Path(__file__).parent / "logs" / "daemon.log"
    log_file.parent.mkdir(exist_ok=True)

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        if console:
            print(line, flush=True)
        with log_file.open("a") as f:
            f.write(line + "\n")

    import pystray

    engine = Engine(log, settings)
    icon = pystray.Icon("kinesics", make_icon("off"), "kinesics (disarmed)")

    def refresh(state):
        icon.icon = make_icon(state)
        icon.title = {
            "off": "kinesics - disarmed",
            "armed": "kinesics - armed: palm up, fist down, rotate two fingers for volume",
            "up": "kinesics - scrolling up",
            "down": "kinesics - scrolling down",
        }[state]

    engine.on_state = refresh

    def quit_all(*_):
        engine.shutdown()
        hotkeys.stop()
        icon.stop()

    icon.menu = pystray.Menu(
        pystray.MenuItem(lambda _: "Disarm" if engine.armed else "Arm",
                         lambda *_: engine.toggle()),
        pystray.MenuItem("Quit  (ctrl+alt+Q)", quit_all),
    )

    spec = str(settings["hotkey"])
    binding = parse_hotkey(spec) or parse_hotkey(DEFAULTS["hotkey"])
    hotkeys = HotkeyListener(engine.toggle, quit_all, binding)
    hotkeys.start()
    hotkeys.ready.wait(timeout=3)
    if not hotkeys.ok:
        log(f"WARNING: could not register {spec} / ctrl+alt+Q. Another app may be "
            f"swallowing them - use the tray menu, or set a different "
            f'"hotkey" in {SETTINGS_PATH.name}.')

    if settings["armed_on_start"]:
        engine.toggle()
        log(f"started ARMED - open palm scrolls up, fist scrolls down "
            f"({spec} to disarm)")
    else:
        log(f"started, disarmed - {spec} to arm")

    icon.run()
    log("exited")


if __name__ == "__main__":
    main()
