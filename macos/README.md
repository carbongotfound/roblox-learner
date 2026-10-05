# Local macOS launcher

`Launcher.swift` builds a small, visible AppKit application with bundle ID
`org.robloxlearner.desktop`. It stays alive while launching the project's Python
commands so macOS can associate their permissions with **Roblox Learner**.
Permission inheritance must be verified using **Diagnose Python child** on the
actual installation; the native app's granted status alone is insufficient.

This is a locally signed application shim. Python, the isolated environment, and
the repository are **not** bundled into it. It is not Apple notarized or an
official Roblox application. Building it changes no system permissions.

## Build

From the repository, with dependencies already installed in an isolated venv:

```sh
python scripts/build_macos_app.py \
  --python /absolute/path/to/venv/bin/python \
  --control-dir /absolute/path/to/agent-control \
  --output '/absolute/path/to/Roblox Learner.app'
open '/absolute/path/to/Roblox Learner.app'
```

The script requires macOS and Xcode Command Line Tools. It compiles Swift, writes
the explicit runtime paths into the app, ad-hoc signs the bundle, and verifies its
signature. Open the **app bundle** through Launch Services, not its internal
executable. Keep that same built app while granting and testing permissions.
Rebuilding or moving runtime paths may require rebuilding and granting access
again. The build refuses an existing output unless `--replace` is supplied.

## Permissions and diagnosis

The three visible grant buttons invoke Apple's supported permission APIs:

- **Accessibility** permits ordinary OS keyboard/mouse input.
- **Input Monitoring** reads demonstration controls, raw mouse motion, and Escape.
- **Screen Recording** captures the Roblox window.

macOS presents its own approval UI. The launcher never changes the TCC database,
disables system security, or approves a permission for the user. Screen Recording
changes may require quitting and reopening the app. Read both the app status and
the Python child's diagnosis afterward; both must report the expected grants.
The app does not automatically open Roblox or move focus during a running episode.

The public APIs are documented by Apple:
[AXIsProcessTrustedWithOptions](https://developer.apple.com/documentation/applicationservices/1459186-axisprocesstrustedwithoptions),
[CGRequestListenEventAccess](https://developer.apple.com/documentation/coregraphics/cgrequestlisteneventaccess()),
and [CGRequestScreenCaptureAccess](https://developer.apple.com/documentation/coregraphics/cgrequestscreencaptureaccess()).

## Request protocol

The launcher polls `<control-dir>/agent-request.json`. Write a complete JSON file
atomically, using a new alphanumeric/hyphen/underscore `id` for every request:

```json
{"id":"diagnose-001","module":"roblox_learner.play","args":["--diagnose"]}
```

Only the project's `play`, `record`, `collect`, `train`, `evaluate`, and `benchmark`
modules are allowed. Arguments are passed directly to Python as an array, never
through a shell. For example, a finite recording session uses:

```json
{"id":"record-001","module":"roblox_learner.record","args":["--config","configs/obby.json","--output","demos/obby-001","--seconds","120","--countdown","8"]}
```

`collect` supports the teacher-directed collector's own CLI options. `play`
requires a trained checkpoint; the launcher never substitutes scripted inputs or
untrained weights for a trained model.

Explicit native requests are also available:

```json
{"id":"permission-001","command":"request_accessibility"}
```

Other native commands are `request_input_monitoring`,
`request_screen_recording`, `native_diagnose`, and `stop`. The app remembers the
last consumed request ID across restarts so restarting it cannot silently replay
the same recording/play request. Only one Python child runs at a time.

Status is written atomically to `agent-status.json`. Each child gets
`agent-runs/<id>/stdout.log` and `stderr.log`; their current tails also appear in
the app. A `stop` request or **Stop agent** sends SIGINT so Python can release
controls in its cleanup handlers. Quitting the app waits for that child cleanup.
Escape and foreground-focus checks remain enforced inside the Python runtime.
