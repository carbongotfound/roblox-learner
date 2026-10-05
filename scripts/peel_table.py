"""Screen-checked mouse peeling for an already-entered Peel THE Potato table.

Requires the authorized potato workbench to be running. Navigation, purchases,
key collection, and escape are not automated by this narrow controller.
"""
import argparse
import json
from pathlib import Path
import shutil
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control-dir', type=Path, required=True)
    parser.add_argument('--evidence-dir', type=Path, required=True)
    parser.add_argument('--batches', type=int, default=5)
    parser.add_argument('--stroke-seconds', type=float, default=9)
    args = parser.parse_args()
    if not 1 <= args.batches <= 200:
        parser.error('--batches must be 1..200')
    if not .1 <= args.stroke_seconds <= 10:
        parser.error('--stroke-seconds must be 0.1..10')
    import Foundation
    import Vision
    root = args.control_dir.resolve()
    args.evidence_dir.mkdir(parents=True, exist_ok=True)
    points = [[.46,.4],[.46,.5],[.46,.61],[.50,.61],[.50,.5],
              [.50,.4],[.54,.4],[.54,.5],[.54,.61]]
    for batch in range(args.batches):
        status = json.loads((root/'status.json').read_text())
        if status['phase'] not in ('ready', 'complete'):
            raise SystemExit('Workbench is stopped or busy; refusing concurrent input')
        frame = root/'current.png'
        if time.time() - frame.stat().st_mtime > 5:
            raise SystemExit('Screenshot is stale')
        request = Vision.VNRecognizeTextRequest.alloc().init()
        request.setRecognitionLevel_(1)
        handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(
            Foundation.NSURL.fileURLWithPath_(str(frame)), {})
        handler.performRequests_error_([request], None)
        text = ' '.join(str(item.topCandidates_(1)[0].string())
                        for item in request.results()).upper()
        if 'AVE' not in text or 'AUTO PEEL' not in text or 'BUY ROBUX' in text:
            path = args.evidence_dir/f'screen-change-{time.time_ns()}.png'
            shutil.copyfile(frame, path)
            raise SystemExit(f'Screen changed. Inspect {path}\n{text}')
        command = {'id': str(time.time_ns()), 'op': 'drag', 'seconds': args.stroke_seconds,
                   'points': sum([points if i % 2 == 0 else points[::-1]
                                  for i in range(14)], [])}
        temporary = root/'command.tmp'
        temporary.write_text(json.dumps(command))
        temporary.replace(root/'command.json')
        deadline = time.monotonic() + 40
        while time.monotonic() < deadline:
            status = json.loads((root/'status.json').read_text())
            if status['phase'] == 'stopped':
                raise SystemExit(str(status))
            if status.get('command_id') == command['id'] and status['phase'] == 'complete':
                break
            time.sleep(.1)
        else:
            raise SystemExit('Workbench did not complete the command; no retry sent')
        print(f'Completed peeling batch {batch + 1}/{args.batches}', flush=True)


if __name__ == '__main__':
    main()
