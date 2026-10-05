"""External, foreground-only macOS capture and input.

This module never reads Roblox memory, network traffic, or game internals. Imports
of macOS bindings are lazy so configuration checks and tests work on other OSes.
Mouse deltas in action files are fractions of the captured window's dimensions.
"""

from __future__ import annotations

import json
import math
import platform
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

ROBLOX_BUNDLE_ID = "com.roblox.RobloxPlayer"
KEY_CODES = {
    "a": 0, "s": 1, "d": 2, "f": 3, "h": 4, "g": 5, "z": 6,
    "x": 7, "c": 8, "v": 9, "b": 11, "q": 12, "w": 13,
    "e": 14, "r": 15, "y": 16, "t": 17, "1": 18, "2": 19,
    "3": 20, "4": 21, "6": 22, "5": 23, "9": 25, "7": 26,
    "8": 28, "0": 29, "o": 31, "u": 32, "i": 34, "p": 35,
    "return": 36, "l": 37, "j": 38, "k": 40, "n": 45, "m": 46,
    "tab": 48, "space": 49, "shift": 56, "alt": 58, "control": 59,
    "left": 123, "right": 124, "down": 125, "up": 126,
}
ESCAPE_CODE = 53
F8_CODE = 100
BUTTONS = {"left": 0, "right": 1, "middle": 2}


class DesktopError(RuntimeError):
    """Desktop is unavailable or a safety condition interrupted execution."""


class FocusLost(DesktopError):
    """Roblox is not the foreground application."""


class EmergencyStop(DesktopError):
    """The physical Escape key requested an immediate stop."""


@dataclass(frozen=True)
class Action:
    name: str
    keys: tuple[str, ...] = ()
    dx: float = 0.0
    dy: float = 0.0
    button: str | None = None
    duration: float | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "name": self.name,
            "keys": list(self.keys),
            "mouse": {"dx": self.dx, "dy": self.dy, "button": self.button},
        }
        if self.duration is not None:
            result["duration"] = self.duration
        return result


def parse_actions(value: Any) -> list[Action]:
    """Validate the shared training/recording/runtime action vocabulary."""
    if isinstance(value, dict):
        value = value.get("actions")
    if not isinstance(value, list) or not value:
        raise ValueError("Actions must be a nonempty list (or {'actions': [...]})")
    actions: list[Action] = []
    names: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"Action {index} must be an object")
        if set(item) - {"name", "keys", "mouse", "duration"}:
            raise ValueError(f"Action {index} contains an unknown field")
        name = item.get("name")
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError(f"Action {index} needs a unique nonempty name")
        names.add(name)
        keys = item.get("keys", [])
        if not isinstance(keys, list) or any(not isinstance(k, str) or k not in KEY_CODES for k in keys):
            raise ValueError(f"{name}: invalid keys; Escape and F8 are reserved")
        if len(set(keys)) != len(keys):
            raise ValueError(f"{name}: duplicate keys")
        mouse = item.get("mouse", {})
        if not isinstance(mouse, dict):
            raise ValueError(f"{name}: mouse must be an object")
        if set(mouse) - {"dx", "dy", "button"}:
            raise ValueError(f"{name}: unknown mouse field")
        deltas = []
        for axis in ("dx", "dy"):
            v = mouse.get(axis, 0.0)
            if isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or abs(v) > 1:
                raise ValueError(f"{name}: mouse {axis} must be finite in [-1, 1]")
            deltas.append(float(v))
        button = mouse.get("button")
        if button is not None and (not isinstance(button, str) or button not in BUTTONS):
            raise ValueError(f"{name}: button must be left, right, middle, or null")
        duration = item.get("duration")
        if duration is not None and (isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or not 0.01 <= duration <= 2):
            raise ValueError(f"{name}: duration must be between 0.01 and 2 seconds")
        actions.append(Action(name, tuple(keys), *deltas, button, duration))
    return actions


def load_actions(path: str | Path) -> list[Action]:
    return parse_actions(json.loads(Path(path).read_text()))


@dataclass(frozen=True)
class Window:
    window_id: int
    pid: int
    x: float
    y: float
    width: int
    height: int


@dataclass(frozen=True)
class RawInput:
    keys: tuple[str, ...] = ()
    buttons: tuple[str, ...] = ()
    x: float = 0.0
    y: float = 0.0
    escape: bool = False
    toggle: bool = False
    mouse_total_dx: float | None = None
    mouse_total_dy: float | None = None


class DesktopBackend(Protocol):
    """Small injectable boundary; fake implementations need no GUI permissions."""

    def is_focused(self) -> bool: ...
    def window(self) -> Window: ...
    def capture(self) -> Any: ...
    def input_state(self) -> RawInput: ...
    def key(self, name: str, down: bool) -> None: ...
    def mouse_button(self, name: str, down: bool) -> None: ...
    def mouse_move(self, dx: float, dy: float) -> None: ...
    def diagnostics(self) -> dict[str, Any]: ...


class MacDesktop:
    """Capture a Roblox window; deliver standard OS keyboard/mouse events.

    The caller must already have granted Screen Recording and Accessibility.
    Construction and diagnostics never invoke a permission prompt or activate an
    app. An external user switching apps ends play rather than stealing focus.
    """

    def __init__(self) -> None:
        if platform.system() != "Darwin":
            raise DesktopError("Live desktop control requires macOS; use --dry-run elsewhere")
        try:
            import Quartz
            from AppKit import NSWorkspace
        except ImportError as exc:
            raise DesktopError("Install the mac extras: pip install -e '.[mac]'") from exc
        self.q = Quartz
        self.workspace = NSWorkspace.sharedWorkspace()
        self._capture_mode = "quartz"
        self._held_mouse_button: str | None = None
        self._mouse_totals = [0.0, 0.0]
        self._mouse_lock = threading.Lock()
        self._monitor_stop = threading.Event()
        self._monitor_ready = threading.Event()
        self._monitor_thread: threading.Thread | None = None
        self._mouse_monitor_available = False
        self._tap_callback = None

    def start_mouse_monitor(self) -> bool:
        """Listen to real movement deltas, including when Roblox locks the cursor.

        This uses a listen-only event tap. Missing Input Monitoring permission
        returns False without opening a macOS permission dialog.
        """
        if self._monitor_thread and self._monitor_thread.is_alive():
            return self._mouse_monitor_available
        preflight = getattr(self.q, "CGPreflightListenEventAccess", None)
        if preflight is None or not preflight():
            return False
        self._monitor_stop.clear()
        self._monitor_ready.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_mouse, name="roblox-mouse-observer", daemon=True)
        self._monitor_thread.start()
        self._monitor_ready.wait(timeout=1)
        return self._mouse_monitor_available

    def _monitor_mouse(self) -> None:
        q = self.q
        tap = source = loop = None
        try:
            def callback(proxy: Any, event_type: int, event: Any, refcon: Any) -> Any:
                if event_type in (q.kCGEventTapDisabledByTimeout, q.kCGEventTapDisabledByUserInput):
                    if tap is not None and not self._monitor_stop.is_set():
                        q.CGEventTapEnable(tap, True)
                    return event
                with self._mouse_lock:
                    self._mouse_totals[0] += q.CGEventGetIntegerValueField(event, q.kCGMouseEventDeltaX)
                    self._mouse_totals[1] += q.CGEventGetIntegerValueField(event, q.kCGMouseEventDeltaY)
                return event

            self._tap_callback = callback  # Retain Python callback while C owns it.
            mask = 0
            for event_type in (q.kCGEventMouseMoved, q.kCGEventLeftMouseDragged, q.kCGEventRightMouseDragged, q.kCGEventOtherMouseDragged):
                mask |= 1 << event_type
            tap = q.CGEventTapCreate(q.kCGSessionEventTap, q.kCGHeadInsertEventTap,
                q.kCGEventTapOptionListenOnly, mask, callback, None)
            if tap is None:
                return
            source = q.CFMachPortCreateRunLoopSource(None, tap, 0)
            loop = q.CFRunLoopGetCurrent()
            q.CFRunLoopAddSource(loop, source, q.kCFRunLoopCommonModes)
            q.CGEventTapEnable(tap, True)
            self._mouse_monitor_available = True
            self._monitor_ready.set()
            while not self._monitor_stop.is_set():
                q.CFRunLoopRunInMode(q.kCFRunLoopDefaultMode, .05, True)
        except Exception:
            self._mouse_monitor_available = False
        finally:
            self._monitor_ready.set()
            self._mouse_monitor_available = False
            if tap is not None:
                q.CGEventTapEnable(tap, False)
            if source is not None and loop is not None:
                q.CFRunLoopRemoveSource(loop, source, q.kCFRunLoopCommonModes)

    def stop_mouse_monitor(self) -> None:
        self._monitor_stop.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1)

    def is_focused(self) -> bool:
        app = self.workspace.frontmostApplication()
        return app is not None and str(app.bundleIdentifier()) == ROBLOX_BUNDLE_ID

    def require_focus(self) -> None:
        if not self.is_focused():
            raise FocusLost("Roblox lost foreground focus; input stopped")

    def window(self) -> Window:
        # Fullscreen transitions briefly remove a foreground window from the
        # on-screen list. No input is sent while waiting for it to reappear.
        deadline = time.monotonic() + 1.5
        while True:
            try:
                return self._window_once()
            except DesktopError as exc:
                if not str(exc).startswith('No visible Roblox game window') or time.monotonic() >= deadline:
                    raise
                time.sleep(.05)

    def _window_once(self) -> Window:
        self.require_focus()
        app = self.workspace.frontmostApplication()
        pid = int(app.processIdentifier())
        options = self.q.kCGWindowListOptionOnScreenOnly | self.q.kCGWindowListExcludeDesktopElements
        candidates = []
        for item in self.q.CGWindowListCopyWindowInfo(options, self.q.kCGNullWindowID) or []:
            if int(item.get(self.q.kCGWindowOwnerPID, -1)) != pid or int(item.get(self.q.kCGWindowLayer, -1)) != 0:
                continue
            b = item[self.q.kCGWindowBounds]
            if b["Width"] >= 200 and b["Height"] >= 150:
                candidates.append(Window(int(item[self.q.kCGWindowNumber]), pid, float(b["X"]), float(b["Y"]), int(b["Width"]), int(b["Height"])))
        if not candidates:
            raise DesktopError(f"No visible Roblox game window (foreground pid={pid}); launch a game before recording or play")
        return max(candidates, key=lambda w: w.width * w.height)

    def capture(self) -> Any:
        from PIL import Image

        preflight = getattr(self.q, "CGPreflightScreenCaptureAccess", None)
        if preflight is not None and not preflight():
            raise DesktopError("Screen Recording permission is not granted for this Python process")
        window = self.window()
        if self._capture_mode == "quartz":
            try:
                image = self.q.CGWindowListCreateImage(
                    self.q.CGRectNull, self.q.kCGWindowListOptionIncludingWindow,
                    window.window_id, self.q.kCGWindowImageBoundsIgnoreFraming,
                )
                if image is None:
                    raise DesktopError("Quartz returned an empty screenshot")
                width, height = self.q.CGImageGetWidth(image), self.q.CGImageGetHeight(image)
                # Explicitly render to RGBA instead of assuming native pixel layout.
                space = self.q.CGColorSpaceCreateDeviceRGB()
                data = bytearray(width * height * 4)
                context = self.q.CGBitmapContextCreate(data, width, height, 8, width * 4, space,
                    self.q.kCGImageAlphaPremultipliedLast | self.q.kCGBitmapByteOrder32Big)
                if context is None:
                    raise DesktopError("Cannot create screenshot pixel buffer")
                self.q.CGContextDrawImage(context, self.q.CGRectMake(0, 0, width, height), image)
                frame = Image.frombytes("RGBA", (width, height), bytes(data)).convert("RGB")
                self.require_focus()
                return frame
            except FocusLost:
                raise
            except Exception:
                # macOS releases that remove CGWindowListCreateImage retain the
                # public screencapture utility. This path may have lower FPS.
                self._capture_mode = "screencapture"
        with tempfile.TemporaryDirectory(prefix="roblox-capture-") as directory:
            path = Path(directory) / "frame.png"
            proc = subprocess.run(["/usr/sbin/screencapture", "-x", "-o", "-l", str(window.window_id), "-t", "png", str(path)], capture_output=True, timeout=3)
            if proc.returncode or not path.exists():
                detail = proc.stderr.decode("utf-8", errors="replace").strip()
                raise DesktopError(f"Screen capture failed; verify Screen Recording permission. {detail}")
            with Image.open(path) as source:
                frame = source.convert("RGB")
        self.require_focus()
        return frame

    def input_state(self) -> RawInput:
        q = self.q
        source = q.kCGEventSourceStateCombinedSessionState
        keys = tuple(k for k, code in KEY_CODES.items() if q.CGEventSourceKeyState(source, code))
        buttons = tuple(k for k, code in BUTTONS.items() if q.CGEventSourceButtonState(source, code))
        event = q.CGEventCreate(None)
        point = q.CGEventGetLocation(event)
        with self._mouse_lock:
            totals = tuple(self._mouse_totals) if self._mouse_monitor_available else (None, None)
        return RawInput(keys, buttons, float(point.x), float(point.y),
            bool(q.CGEventSourceKeyState(source, ESCAPE_CODE)),
            bool(q.CGEventSourceKeyState(source, F8_CODE)), *totals)

    def escape_pressed(self) -> bool:
        """Check the stop key without polling the entire demonstration state."""
        return bool(self.q.CGEventSourceKeyState(
            self.q.kCGEventSourceStateCombinedSessionState, ESCAPE_CODE))

    def key(self, name: str, down: bool) -> None:
        if down:
            self.require_focus()
            preflight = getattr(self.q, "CGPreflightPostEventAccess", None)
            if preflight is not None and not preflight():
                raise DesktopError("Accessibility permission is not granted for OS input events")
        event = self.q.CGEventCreateKeyboardEvent(None, KEY_CODES[name], down)
        self.q.CGEventPost(self.q.kCGHIDEventTap, event)

    def mouse_button(self, name: str, down: bool) -> None:
        if down:
            self.require_focus()
            preflight = getattr(self.q, "CGPreflightPostEventAccess", None)
            if preflight is not None and not preflight():
                raise DesktopError("Accessibility permission is not granted for OS input events")
        q = self.q
        point = q.CGEventGetLocation(q.CGEventCreate(None))
        event_type = {
            "left": (q.kCGEventLeftMouseUp, q.kCGEventLeftMouseDown),
            "right": (q.kCGEventRightMouseUp, q.kCGEventRightMouseDown),
            "middle": (q.kCGEventOtherMouseUp, q.kCGEventOtherMouseDown),
        }[name][int(down)]
        event = q.CGEventCreateMouseEvent(None, event_type, point, BUTTONS[name])
        q.CGEventPost(q.kCGHIDEventTap, event)
        if down:
            self._held_mouse_button = name
        elif self._held_mouse_button == name:
            self._held_mouse_button = None

    def mouse_move(self, dx: float, dy: float) -> None:
        self.require_focus()
        if dx == 0 and dy == 0:
            return
        preflight = getattr(self.q, "CGPreflightPostEventAccess", None)
        if preflight is not None and not preflight():
            raise DesktopError("Accessibility permission is not granted for OS input events")
        q = self.q
        w = self.window()
        state = self.input_state()
        x = min(w.x + w.width - 1, max(w.x + 1, state.x + dx * w.width))
        y = min(w.y + w.height - 1, max(w.y + 1, state.y + dy * w.height))
        event_type, button = q.kCGEventMouseMoved, 0
        for name, event in (("left", q.kCGEventLeftMouseDragged), ("right", q.kCGEventRightMouseDragged), ("middle", q.kCGEventOtherMouseDragged)):
            # CGEventPost is asynchronous; our issued button-down is a more
            # reliable drag indicator than querying the OS immediately after it.
            if name == self._held_mouse_button or (self._held_mouse_button is None and name in state.buttons):
                event_type, button = event, BUTTONS[name]
                break
        event = q.CGEventCreateMouseEvent(None, event_type, (x, y), button)
        q.CGEventSetIntegerValueField(event, q.kCGMouseEventDeltaX, round(dx * w.width))
        q.CGEventSetIntegerValueField(event, q.kCGMouseEventDeltaY, round(dy * w.height))
        self.require_focus()
        q.CGEventPost(q.kCGHIDEventTap, event)

    def diagnostics(self) -> dict[str, Any]:
        q = self.q
        app = self.workspace.frontmostApplication()
        preflight = getattr(q, "CGPreflightScreenCaptureAccess", None)
        access = getattr(q, "CGPreflightPostEventAccess", None) or getattr(q, "AXIsProcessTrusted", None)
        return {
            "platform": platform.platform(), "target_bundle": ROBLOX_BUNDLE_ID,
            "foreground_bundle": str(app.bundleIdentifier()) if app else None,
            "roblox_focused": self.is_focused(),
            "screen_recording_granted": bool(preflight()) if preflight else None,
            "accessibility_granted": bool(access()) if access else None,
            "input_monitoring_granted": bool(q.CGPreflightListenEventAccess()) if hasattr(q, "CGPreflightListenEventAccess") else None,
            "raw_mouse_deltas_available": self._mouse_monitor_available,
            "capture_backend": self._capture_mode,
        }


class InputController:
    """Own injected input and release it on exceptions, stops, or focus changes."""

    def __init__(self, backend: DesktopBackend) -> None:
        self.backend = backend
        self.held_keys: set[str] = set()
        self.held_button: str | None = None
        self._lock = threading.RLock()

    def check(self) -> None:
        escape_check = getattr(self.backend, 'escape_pressed', None)
        escape = escape_check() if escape_check is not None else self.backend.input_state().escape
        if escape:
            self.release_all()
            raise EmergencyStop("Escape pressed")
        if not self.backend.is_focused():
            self.release_all()
            raise FocusLost("Roblox lost foreground focus; input stopped")

    def apply(self, action: Action) -> None:
        with self._lock:
            self.check()
            target_keys = set(action.keys)
            for key in sorted(self.held_keys - target_keys):
                self.backend.key(key, False)
                self.held_keys.discard(key)
            if self.held_button is not None and self.held_button != action.button:
                self.backend.mouse_button(self.held_button, False)
                self.held_button = None
            for key in sorted(target_keys - self.held_keys):
                self.check()
                self.backend.key(key, True)
                self.held_keys.add(key)
            if action.button is not None and action.button != self.held_button:
                self.check()
                self.backend.mouse_button(action.button, True)
                self.held_button = action.button
            self.check()
            self.backend.mouse_move(action.dx, action.dy)

    def release_all(self) -> None:
        """Send only cleanup key-up/button-up events, including after focus loss."""
        errors: list[Exception] = []
        with self._lock:
            for key in tuple(self.held_keys):
                try:
                    self.backend.key(key, False)
                except Exception as exc:
                    errors.append(exc)
                else:
                    self.held_keys.discard(key)
            if self.held_button is not None:
                try:
                    self.backend.mouse_button(self.held_button, False)
                except Exception as exc:
                    errors.append(exc)
                else:
                    self.held_button = None
        if errors:
            raise DesktopError(f"Failed to release {len(errors)} injected inputs: {errors[0]}")

    def __enter__(self) -> "InputController":
        return self

    def __exit__(self, *_: Any) -> None:
        self.release_all()


class Watchdog:
    """Independent focus/Escape/heartbeat check while model inference executes."""

    def __init__(self, controller: InputController, timeout: float = 2.0,
                 memory_limit_mb: float = 5500, rss: Callable[[], float] | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        if timeout <= 0 or memory_limit_mb <= 0:
            raise ValueError("Watchdog timeout and memory limit must be positive")
        self.controller = controller
        self.timeout = timeout
        self.memory_limit_mb = memory_limit_mb
        self.rss = rss or process_rss_mb
        self.clock = clock
        self.last_heartbeat = clock()
        self.stop_event = threading.Event()
        self.reason: str | None = None
        self._thread: threading.Thread | None = None

    def heartbeat(self) -> None:
        self.last_heartbeat = self.clock()

    def check_once(self) -> None:
        try:
            self.controller.check()
            if self.clock() - self.last_heartbeat > self.timeout:
                raise DesktopError(f"Inference/capture watchdog exceeded {self.timeout:.1f}s")
            if self.rss() > self.memory_limit_mb:
                raise DesktopError(f"Resident memory exceeded {self.memory_limit_mb:.0f} MB")
        except Exception as exc:
            self.reason = str(exc)
            self.stop_event.set()
            try:
                self.controller.release_all()
            except Exception as release_exc:
                self.reason += f"; cleanup: {release_exc}"

    def raise_if_stopped(self) -> None:
        if self.stop_event.is_set():
            raise DesktopError(self.reason or "Watchdog stopped")

    def __enter__(self) -> "Watchdog":
        self.heartbeat()
        self._thread = threading.Thread(target=self._run, name="roblox-input-watchdog", daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self.stop_event.wait(0.025):
            self.check_once()

    def __exit__(self, *_: Any) -> None:
        self.stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        self.controller.release_all()


def process_rss_mb() -> float:
    """Current resident MB, using psutil when available (not peak allocation)."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1_000_000
    except ImportError:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / (1_000_000 if platform.system() == "Darwin" else 1000)


def portable_diagnostics() -> dict[str, Any]:
    import importlib.util
    return {"platform": platform.platform(), "target_bundle": ROBLOX_BUNDLE_ID,
        "python": platform.python_version(), "rss_mb": round(process_rss_mb(), 2),
        "quartz_available": importlib.util.find_spec("Quartz") is not None,
        "torch_available": importlib.util.find_spec("torch") is not None,
        "live_input": False}


def countdown(seconds: float) -> None:
    """Visible focus countdown; never selects, activates, or opens an app."""
    if not math.isfinite(seconds) or seconds < 0 or seconds > 60:
        raise ValueError("Countdown must be between 0 and 60 seconds")
    if seconds:
        print(f"Switch to Roblox. Starting in {seconds:g}s; Escape stops. Do not type sensitive text.", flush=True)
        time.sleep(seconds)
