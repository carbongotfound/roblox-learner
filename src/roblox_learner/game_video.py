"""Record only the foreground Roblox window, with timestamps and explicit gaps."""
from pathlib import Path
import signal, subprocess, time, json
from PIL import Image, ImageDraw, ImageOps
from .desktop import MacDesktop

def record(path, seconds=7200):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.with_suffix('.pid').write_text(str(__import__('os').getpid()))
    backend=MacDesktop(); running=True
    def stop(*_):
        nonlocal running
        running=False
    signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
    enc=subprocess.Popen(['/opt/homebrew/bin/ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pixel_format','rgb24','-video_size','1280x720','-framerate','5','-i','pipe:0','-an','-c:v','h264_videotoolbox','-b:v','2500k',str(path)],stdin=subprocess.PIPE)
    start=time.monotonic(); index=0; gaps=0
    try:
        while running and time.monotonic()-start<seconds:
            try:
                frame=ImageOps.pad(backend.capture(),(1280,720))
                label='ROBLOX WINDOW'
            except Exception:
                frame=Image.new('RGB',(1280,720),'black'); label='PAUSED: Roblox window unavailable or not foreground'; gaps+=1
            ImageDraw.Draw(frame).text((10,700),time.strftime('%Y-%m-%d %H:%M:%S')+' | '+label,fill='white',stroke_width=1,stroke_fill='black')
            enc.stdin.write(frame.tobytes()); index+=1
            time.sleep(max(0,start+index/5-time.monotonic()))
    finally:
        enc.stdin.close(); enc.wait(timeout=20)
        path.with_suffix('.json').write_text(json.dumps({'frames':index,'fps':5,'gap_frames':gaps,'elapsed_seconds':time.monotonic()-start},indent=2))
if __name__=='__main__':
    import sys
    record(sys.argv[1])
