#!/usr/bin/env python3
"""Build a locally ad-hoc-signed AppKit launcher with an explicit Python runtime.

This is an app shim, not a bundled Python distribution. Use Launch Services
(`open 'Roblox Learner.app'`) so macOS sees the app's bundle identity. Rebuilding
an ad-hoc-signed binary can require granting permissions again; retain the built
app while granting/testing permissions. No permissions are changed by this build.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import plistlib
import shutil
import subprocess
import tempfile

BUNDLE_ID = "org.robloxlearner.desktop"


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", required=True, type=Path, help="Python executable inside the installed isolated environment")
    parser.add_argument("--repository", type=Path, default=repo)
    parser.add_argument("--control-dir", required=True, type=Path, help="Directory for agent-request.json, status, and logs")
    parser.add_argument("--output", type=Path, default=repo.parent / "Roblox Learner.app")
    parser.add_argument("--build-cache", type=Path, help="Writable Swift module cache; defaults to control-dir/swift-cache")
    parser.add_argument("--replace", action="store_true", help="Replace exactly the existing output app, possibly invalidating prior TCC approval")
    args = parser.parse_args()
    if platform.system() != "Darwin":
        parser.error("The native launcher must be built on macOS with Xcode Command Line Tools")
    # Keep the venv symlink path: resolving it would erase the environment identity.
    python = args.python.absolute()
    repository = args.repository.resolve()
    control = args.control_dir.resolve()
    output = args.output.absolute()
    if not python.is_file() or not os.access(python, os.X_OK):
        parser.error(f"Python executable is unavailable: {python}")
    if output.suffix != ".app":
        parser.error("--output must name a .app bundle")
    if output.exists() and not args.replace:
        parser.error("Output app already exists; keep its signing identity or use --replace explicitly")
    source = repository / "macos" / "Launcher.swift"
    if not source.is_file():
        parser.error(f"Launcher source does not exist: {source}")
    control.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    cache = args.build_cache.resolve() if args.build_cache else control / "swift-cache"
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="roblox-launcher-build-", dir=control) as build_directory:
        app = Path(build_directory) / output.name
        contents = app / "Contents"
        executable = contents / "MacOS" / "RobloxLearner"
        resources = contents / "Resources"
        executable.parent.mkdir(parents=True)
        resources.mkdir()
        info = {
            "CFBundleIdentifier": BUNDLE_ID,
            "CFBundleName": "Roblox Learner",
            "CFBundleDisplayName": "Roblox Learner",
            "CFBundleExecutable": "RobloxLearner",
            "CFBundlePackageType": "APPL",
            "CFBundleShortVersionString": "0.1.0",
            "CFBundleVersion": "1",
            "LSMinimumSystemVersion": "13.0",
            "NSHighResolutionCapable": True,
            "NSPrincipalClass": "NSApplication",
            "NSHumanReadableCopyright": "MIT licensed; locally built, not Apple notarized.",
            "NSScreenCaptureUsageDescription": "Capture only the foreground Roblox window to train and run a local visual game policy.",
            "NSInputMonitoringUsageDescription": "Record demonstration controls and detect the Escape emergency-stop key.",
        }
        with (contents / "Info.plist").open("wb") as stream:
            plistlib.dump(info, stream)
        (resources / "launcher.json").write_text(json.dumps({"python": str(python), "repository": str(repository), "controlDirectory": str(control)}, indent=2) + "\n")
        arch = platform.machine()
        subprocess.run(["xcrun", "swiftc", str(source), "-o", str(executable), "-O", "-swift-version", "5",
            "-target", f"{arch}-apple-macos13.0", "-module-cache-path", str(cache),
            "-framework", "AppKit", "-framework", "ApplicationServices", "-framework", "CoreGraphics"], check=True)
        subprocess.run(["/usr/bin/codesign", "--force", "--sign", "-", "--identifier", BUNDLE_ID, "--timestamp=none", str(app)], check=True)
        subprocess.run(["/usr/bin/codesign", "--verify", "--deep", "--strict", str(app)], check=True)
        if output.exists():
            shutil.rmtree(output)
        shutil.move(str(app), output)
    print(json.dumps({"app": str(output), "bundle_id": BUNDLE_ID, "python": str(python),
        "repository": str(repository), "request_file": str(control / "agent-request.json"),
        "status_file": str(control / "agent-status.json"), "signing": "local ad-hoc; not notarized",
        "launch": f"open {output}", "permissions_changed": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
