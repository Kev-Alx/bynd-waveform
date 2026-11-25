"""Static audio comparison server with byte ranges and a persistent heartbeat."""
import functools
import json
import os
import re
import signal
import threading
import time
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path('/workspace/improvement')
DIST = ROOT / 'audio_browser/dist'
STATUS = ROOT / 'audio_browser_server_status.json'
STATE = {'status': 'starting', 'pid': os.getpid(), 'bind': '0.0.0.0', 'port': 8888,
         'url': 'http://localhost:8888/', 'directory': str(DIST), 'requests': 0,
         'started_at_utc': datetime.now(timezone.utc).isoformat()}
LOCK = threading.Lock()


def update(**values):
    with LOCK:
        STATE.update(updated_at_utc=datetime.now(timezone.utc).isoformat(), **values)
        temp = STATUS.with_suffix('.tmp')
        temp.write_text(json.dumps(STATE, indent=2))
        temp.replace(STATUS)


class Handler(SimpleHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def end_headers(self):
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Cache-Control', 'public, max-age=86400' if urlsplit(self.path).path.endswith('.wav') else 'no-cache')
        super().end_headers()

    def send_head(self):
        self.remaining = None
        with LOCK:
            STATE['requests'] += 1
        path = urlsplit(self.path).path
        if path.startswith('/_data/'):
            self.path = '/index.html'
        filename = Path(self.translate_path(self.path))
        header = self.headers.get('Range')
        if not header or not filename.is_file():
            return super().send_head()
        size = filename.stat().st_size
        match = re.fullmatch(r'bytes=(\d*)-(\d*)', header.strip())
        if match and (match[1] or match[2]):
            start = int(match[1]) if match[1] else max(0, size-int(match[2]))
            end = min(int(match[2]), size-1) if match[1] and match[2] else size-1
        else:
            start, end = size, 0
        if start >= size or start > end:
            self.send_response(416)
            self.send_header('Content-Range', f'bytes */{size}')
            self.send_header('Content-Length', '0')
            self.end_headers()
            return None
        stream = filename.open('rb')
        stream.seek(start)
        self.remaining = end-start+1
        self.send_response(206)
        self.send_header('Content-Type', self.guess_type(str(filename)))
        self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.send_header('Content-Length', str(self.remaining))
        self.send_header('Last-Modified', self.date_time_string(filename.stat().st_mtime))
        self.end_headers()
        return stream

    def copyfile(self, source, outputfile):
        try:
            if self.remaining is None:
                super().copyfile(source, outputfile)
            else:
                remaining = self.remaining
                while remaining:
                    chunk = source.read(min(remaining, 262144))
                    if not chunk:
                        break
                    outputfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass  # Browser stopped or changed the selected audio.


def main():
    if not (DIST/'index.html').is_file():
        raise FileNotFoundError('Build the audio comparison app before serving it')
    server = ThreadingHTTPServer(('0.0.0.0', 8888), functools.partial(Handler, directory=str(DIST)))
    server.daemon_threads = True
    done = threading.Event()
    def heartbeat():
        while not done.wait(30):
            update(status='running')
    def stop(signum, frame):
        update(status='stopping', signal=signum)
        threading.Thread(target=server.shutdown, daemon=True).start()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    update(status='running')
    threading.Thread(target=heartbeat, daemon=True).start()
    print('Audio comparison ready: http://0.0.0.0:8888/ (Docker port 8888)', flush=True)
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        done.set();server.server_close();update(status='stopped')


if __name__ == '__main__':
    try:
        main()
    except BaseException as e:
        update(status='failed', error=repr(e));raise
