"""External keyboard/mouse workbench for developing and testing the potato bot.

Only foreground Roblox pixels and ordinary OS inputs are used. This module does
not train or claim to be a neural policy. Command files are local operator input.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import signal
import time
from dataclasses import asdict

from .desktop import Action, InputController, MacDesktop, Watchdog, ROBLOX_BUNDLE_ID, KEY_CODES


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def activate_roblox() -> None:
    """One explicit activation when starting a session, never during play."""
    from AppKit import NSRunningApplication, NSApplicationActivateIgnoringOtherApps
    apps = NSRunningApplication.runningApplicationsWithBundleIdentifier_(ROBLOX_BUNDLE_ID)
    if not apps:
        raise RuntimeError('Roblox is not running')
    apps[0].activateWithOptions_(NSApplicationActivateIgnoringOtherApps)


def serve(directory: Path, seconds: float = 900, activate: bool = False) -> int:
    directory.mkdir(parents=True, exist_ok=True)
    inbox, status = directory / 'command.json', directory / 'status.json'
    backend = MacDesktop()
    if activate:
        activate_roblox()
        time.sleep(.5)
    start, last_id = time.monotonic(), None
    if inbox.exists():
        try:
            last_id = json.loads(inbox.read_text()).get('id')
        except (OSError, ValueError):
            pass
    controller = InputController(backend)
    count = 0
    def publish(phase: str, **extra):
        atomic_json(status, {'phase': phase, 'completed': count, 'command_id': last_id,
                            'elapsed': time.monotonic() - start, **extra})
    def preview():
        frame = backend.capture()
        temp = directory / 'current.tmp.png'
        frame.save(temp)
        temp.replace(directory / 'current.png')
        return frame
    def move_to(x, y):
        if not all(isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 1 for v in (x, y)):
            raise ValueError('Coordinates must be normalized within the Roblox window')
        w, raw = backend.window(), backend.input_state()
        backend.mouse_move((w.x + x*w.width - raw.x)/w.width, (w.y + y*w.height - raw.y)/w.height)
    try:
        with controller, Watchdog(controller, timeout=3, memory_limit_mb=1000) as watchdog:
            def wait(duration):
                end = min(start + seconds, time.monotonic()+duration)
                while time.monotonic() < end:
                    controller.check()
                    watchdog.raise_if_stopped()
                    watchdog.heartbeat()
                    time.sleep(min(.01, max(0, end-time.monotonic())))
            controller.check()
            preview()
            publish('ready', diagnostics=backend.diagnostics(), window=asdict(backend.window()), pointer=asdict(backend.input_state()))
            last_preview = time.monotonic()
            while time.monotonic()-start < seconds:
                controller.check()
                watchdog.raise_if_stopped()
                watchdog.heartbeat()
                command = None
                if inbox.exists():
                    try:
                        command = json.loads(inbox.read_text())
                    except (OSError, ValueError):
                        pass
                if command and isinstance(command.get('id'), str) and command['id'] != last_id:
                    last_id = command['id']
                    op = command.get('op')
                    duration = float(command.get('seconds', .15))
                    if not math.isfinite(duration) or not .01 <= duration <= 10:
                        raise ValueError('Command duration must be 0.01..10 seconds')
                    publish('running', op=op)
                    try:
                        if op == 'stop':
                            break
                        if op == 'record':
                            import subprocess, sys
                            path=Path(command['path']).resolve()
                            if path.exists():
                                raise ValueError('Recording destination already exists')
                            subprocess.Popen([sys.executable,'-m','roblox_learner.game_video',str(path)], start_new_session=True)
                            wait(.2)
                        elif op == 'keys':
                            keys = command.get('keys', [])
                            if not isinstance(keys, list) or any(k not in KEY_CODES for k in keys):
                                raise ValueError('Unsupported keyboard control')
                            controller.apply(Action('keys', tuple(keys)))
                            wait(duration)
                        elif op in ('click', 'button'):
                            if op == 'click':
                                move_to(command['x'], command['y'])
                                wait(.05)
                            controller.apply(Action('click', button=command.get('button','left')))
                            wait(duration)
                        elif op == 'aim':
                            dx, dy = float(command.get('dx',0)), float(command.get('dy',0))
                            if not all(math.isfinite(v) and abs(v)<=1000 for v in (dx,dy)):
                                raise ValueError('Aim delta out of range')
                            steps = max(1, int(duration*30))
                            q=backend.q
                            w=backend.window()
                            center=(w.x+w.width/2,w.y+w.height/2)
                            q.CGWarpMouseCursorPosition(center)
                            wait(.05)
                            for i in range(steps):
                                controller.check()
                                raw=backend.input_state()
                                event=q.CGEventCreateMouseEvent(None,q.kCGEventMouseMoved,center,0)
                                q.CGEventSetIntegerValueField(event,q.kCGMouseEventDeltaX,round((i+1)*dx/steps)-round(i*dx/steps))
                                q.CGEventSetIntegerValueField(event,q.kCGMouseEventDeltaY,round((i+1)*dy/steps)-round(i*dy/steps))
                                q.CGEventPost(q.kCGHIDEventTap,event)
                                wait(duration/steps)
                        elif op == 'look':
                            controller.apply(Action('look', button='right'))
                            wait(.05)
                            dx, dy = float(command.get('dx',0)), float(command.get('dy',0))
                            if not all(math.isfinite(v) and abs(v)<=1 for v in (dx,dy)):
                                raise ValueError('Look delta out of range')
                            steps = max(1, int(duration*30))
                            for _ in range(steps):
                                backend.mouse_move(dx/steps,dy/steps)
                                wait(duration/steps)
                        elif op == 'drag':
                            points=command['points']
                            if not isinstance(points,list) or not 2<=len(points)<=200:
                                raise ValueError('Drag requires 2..200 points')
                            move_to(*points[0]); wait(.05)
                            controller.apply(Action('drag',button=command.get('button','left')))
                            for point in points[1:]:
                                move_to(*point)
                                wait(duration/(len(points)-1))
                        elif op == 'scroll':
                            delta=int(command.get('delta',0))
                            if abs(delta)>50:
                                raise ValueError('Scroll out of range')
                            q=backend.q
                            q.CGEventPost(q.kCGHIDEventTap,q.CGEventCreateScrollWheelEvent(None,q.kCGScrollEventUnitLine,1,delta))
                            wait(duration)
                        elif op == 'wait':
                            wait(duration)
                        else:
                            raise ValueError('Unknown operation')
                    finally:
                        controller.release_all()
                    wait(.15)
                    preview()
                    count+=1
                    publish('complete', op=op)
                    with (directory/'commands.jsonl').open('a') as log:
                        log.write(json.dumps({'command':command,'time':time.time()})+'\n')
                    last_preview=time.monotonic()
                elif time.monotonic()-last_preview>1:
                    preview(); last_preview=time.monotonic()
                wait(.05)
    except (Exception, KeyboardInterrupt) as exc:
        controller.release_all()
        publish('stopped', error=str(exc))
        return 1
    publish('stopped', reason='duration_or_operator_stop')
    return 0


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control-dir', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=900)
    parser.add_argument('--activate', action='store_true')
    args=parser.parse_args(argv)
    if not 1 <= args.seconds <= 7200:
        parser.error('seconds must be 1..7200')
    def stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    return serve(args.control_dir,args.seconds,args.activate)


if __name__ == '__main__':
    raise SystemExit(main())
