#!/usr/bin/env python3
"""Post nvidia-smi to a Discord webhook once or every two hours (Python stdlib).

Configure locally with a hidden prompt; run `--help` for commands.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import fcntl
import getpass
import hashlib
import http.client
import json
import logging
from logging.handlers import RotatingFileHandler
import math
import os
from pathlib import Path
import re
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
from urllib import error, parse, request
import uuid

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / '.discord-gpu.json'
DEFAULT_RUNTIME = ROOT / 'reproduction-runs' / 'discord-gpu'
TAIPEI = timezone(timedelta(hours=8), 'Asia/Taipei')
LOGGER = logging.getLogger('discord-gpu')


class NotificationError(Exception):
    """Safe-to-display error; never contains the webhook URL/token."""


def timestamp():
    return datetime.now(TAIPEI).isoformat(timespec='seconds')


def webhook_url(value):
    """Accept Discord incoming webhooks only; always request send confirmation."""
    try:
        url = parse.urlsplit(value.strip())
        allowed = {'discord.com', 'discordapp.com', 'canary.discord.com', 'ptb.discord.com'}
        valid = (url.scheme == 'https' and url.hostname in allowed
                 and url.port in (None, 443) and not url.username and not url.password
                 and not url.fragment
                 and re.fullmatch(r'/api(?:/v\d+)?/webhooks/[0-9]+/[A-Za-z0-9_.-]+', url.path))
        query = parse.parse_qs(url.query, strict_parsing=True) if url.query else {}
        if not valid or set(query) - {'wait', 'thread_id'}:
            raise ValueError
        if 'thread_id' in query and (len(query['thread_id']) != 1 or not query['thread_id'][0].isdigit()):
            raise ValueError
        params = {'wait': 'true'}
        if 'thread_id' in query:
            params['thread_id'] = query['thread_id'][0]
        return parse.urlunsplit(('https', url.hostname, url.path, parse.urlencode(params), ''))
    except (ValueError, TypeError, AttributeError):
        raise NotificationError('Use a valid HTTPS Discord incoming-webhook URL.') from None


def load_config(path):
    try:
        info = path.stat()
        if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077:
            raise NotificationError(f'Config must be a private file: chmod 600 {path}')
        config = json.loads(path.read_text())
        if not isinstance(config, dict):
            raise ValueError
        url = webhook_url(config['webhook_url'])
        label = str(config.get('label') or socket.gethostname()).replace('\n', ' ')[:80]
        return {'webhook_url': url, 'label': label}
    except FileNotFoundError:
        raise NotificationError('Not configured. Run this script with the configure command first.') from None
    except (KeyError, ValueError, TypeError, OSError):
        raise NotificationError('Cannot read configuration; run configure again or check file permissions.') from None


def configure(path):
    url = webhook_url(getpass.getpass('Discord webhook URL (hidden): '))
    label = input(f'Machine label [{socket.gethostname()}]: ').strip() or socket.gethostname()
    # No URL in command history, logs, process arguments, or terminal output.
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'w') as stream:
        os.fchmod(stream.fileno(), 0o600)
        json.dump({'webhook_url': url, 'label': label[:80]}, stream, indent=2)
        stream.write('\n')
    print(f'Saved private configuration to {path}. No message sent.')


def collect_gpu(binary='/usr/bin/nvidia-smi'):
    try:
        result = subprocess.run([binary], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding='utf-8', errors='replace', timeout=30, check=False)
        return result.stdout or '(nvidia-smi produced no output)\n', result.returncode
    except FileNotFoundError:
        return 'nvidia-smi was not found. Check NVIDIA drivers and container GPU access.\n', 127
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or b''
        if isinstance(output, bytes):
            output = output.decode('utf-8', errors='replace')
        return output + '\nnvidia-smi timed out after 30 seconds.\n', 124
    except OSError:
        return 'nvidia-smi could not be executed. Check executable permissions.\n', 126


def make_payload(output, label, gpu_exit, at):
    header = f'GPU status: {label}\n{at} (Asia/Taipei) | nvidia-smi exit: {gpu_exit}'
    safe = output.replace('```', "''' ")
    content = f'{header}\n```text\n{safe.rstrip()}\n```'
    # Conservative UTF-16 budget; full original output is attached if needed.
    attach = len(content.encode('utf-16-le')) // 2 > 1900 or '```' in output
    if attach:
        content = header + '\nFull nvidia-smi output is attached as nvidia-smi.txt.'
    payload = {'content': content, 'allowed_mentions': {'parse': []}}
    if not attach:
        return json.dumps(payload).encode(), 'application/json'
    payload['attachments'] = [{'id': 0, 'filename': 'nvidia-smi.txt'}]
    boundary = 'gpu-' + uuid.uuid4().hex
    data = (f'--{boundary}\r\nContent-Disposition: form-data; name="payload_json"\r\n'
            'Content-Type: application/json\r\n\r\n').encode() + json.dumps(payload).encode()
    data += (f'\r\n--{boundary}\r\nContent-Disposition: form-data; name="files[0]"; '
             'filename="nvidia-smi.txt"\r\nContent-Type: text/plain; charset=utf-8\r\n\r\n').encode()
    data += output.encode('utf-8') + f'\r\n--{boundary}--\r\n'.encode()
    return data, 'multipart/form-data; boundary=' + boundary


class NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise NotificationError('Discord returned an unexpected redirect; webhook was not forwarded.')


def post_report(url, body, content_type):
    opener = request.build_opener(NoRedirect())
    for attempt in range(3):
        delay = 2 ** attempt
        req = request.Request(url, data=body, method='POST', headers={
            'Content-Type': content_type, 'User-Agent': 'GPUStatusReporter/1.0'})
        try:
            with opener.open(req, timeout=20) as response:
                result = response.read()
                if not 200 <= response.status < 300:
                    raise NotificationError('Discord returned an unexpected HTTP status.')
                try:
                    return str(json.loads(result).get('id', 'accepted'))
                except (ValueError, AttributeError):
                    return 'accepted'
        except error.HTTPError as exc:
            status = exc.code
            if status == 429:
                try:
                    delay = float(json.loads(exc.read()).get('retry_after', delay))
                except (ValueError, TypeError, AttributeError):
                    pass
                if not math.isfinite(delay) or delay > 30:
                    raise NotificationError('Discord rate limit exceeds the short retry window; retry later.') from None
            elif status in (401, 403, 404):
                raise NotificationError(f'Discord HTTP {status}: webhook invalid/deleted or channel access denied. Reconfigure it.') from None
            elif not 500 <= status < 600:
                raise NotificationError(f'Discord rejected the report (HTTP {status}).') from None
            if attempt == 2:
                raise NotificationError(f'Discord HTTP {status}: delivery failed after 3 attempts.') from None
        except (error.URLError, OSError, http.client.HTTPException):
            if attempt == 2:
                raise NotificationError('Cannot reach Discord after 3 attempts. Check outbound HTTPS, DNS and certificates.') from None
        time.sleep(max(0, delay))
    raise NotificationError('Discord delivery failed.')


def read_state(runtime):
    try:
        return json.loads((runtime / 'status.json').read_text())
    except (OSError, ValueError):
        return {}


def write_state(runtime, **updates):
    runtime.mkdir(parents=True, exist_ok=True)
    state = {**read_state(runtime), **updates}
    with tempfile.NamedTemporaryFile(mode='w', dir=runtime, delete=False) as stream:
        json.dump(state, stream, indent=2)
        stream.write('\n')
        name = stream.name
    os.replace(name, runtime / 'status.json')


@contextmanager
def daemon_lock(runtime):
    runtime.mkdir(parents=True, exist_ok=True)
    with (runtime / 'daemon.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise NotificationError('The GPU notifier is already running; no duplicate was started.') from None
        lock.seek(0)
        lock.truncate()
        lock.write(str(os.getpid()))
        lock.flush()
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def running_pid(runtime):
    runtime.mkdir(parents=True, exist_ok=True)
    with (runtime / 'daemon.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return None
        except BlockingIOError:
            lock.seek(0)
            try:
                pid = int(lock.read().strip())
            except ValueError:
                return None
    try:
        args = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        if str(Path(__file__).resolve()).encode() in args and b'run' in args:
            return pid
    except OSError:
        pass
    return None


def send_once(config_path, runtime, binary):
    config = load_config(config_path)
    at = timestamp()
    output, gpu_exit = collect_gpu(binary)
    write_state(runtime, last_attempt=at, gpu_exit=gpu_exit, label=config['label'])
    body, content_type = make_payload(output, config['label'], gpu_exit, at)
    try:
        message = post_report(config['webhook_url'], body, content_type)
    except NotificationError as exc:
        write_state(runtime, last_error=str(exc))
        raise
    write_state(runtime, last_sent=at, last_error=None, discord_message_id=message)
    LOGGER.info('Report delivered; nvidia-smi exit=%s', gpu_exit)
    return gpu_exit


def send_job(config_path, status_path, name=None):
    """Notify once per finished job attempt, using the existing private webhook.

    Command arguments and environment variables are deliberately omitted.
    A receipt distinguishes delivery failure from the actual compute result.
    """
    status_path = Path(status_path)
    try:
        raw = status_path.read_bytes()
        job = json.loads(raw)
        if job['status'] not in {'complete', 'failed', 'interrupted'} or 'exit_code' not in job:
            raise ValueError
    except (OSError, ValueError, KeyError, TypeError):
        raise NotificationError('A finished job JSON with an exit code is required.') from None
    event = hashlib.sha256(raw).hexdigest()
    receipt = status_path.with_name(status_path.stem + '.discord.json')
    try:
        previous = json.loads(receipt.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        previous = {}
    if previous.get('event_sha256') == event and previous.get('status') == 'sent':
        return previous['discord_message_id']
    state = {'event_sha256': event, 'job_status': job['status'], 'exit_code': job['exit_code'],
             'attempted_at': timestamp(), 'status': 'pending'}

    def save():
        temporary = receipt.with_suffix(receipt.suffix + '.tmp')
        temporary.write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
        temporary.replace(receipt)

    save()
    try:
        config = load_config(config_path)
        label = str(name or status_path.parent.name).replace('\n', ' ')[:120]
        content = (f"Job {job['status']}: {label}\n{config['label']} | {timestamp()} (Asia/Taipei)\n"
                   f"Exit code: {job['exit_code']}\n"
                   f"Status: {str(status_path.resolve())[:350]}\n"
                   f"Log: {str(job.get('log', 'see status file'))[:350]}")
        payload = {'content': content, 'allowed_mentions': {'parse': []}}
        message = post_report(config['webhook_url'], json.dumps(payload).encode('utf-8'), 'application/json')
    except NotificationError as exc:
        state.update(status='failed', error=str(exc))
        save()
        raise
    state.update(status='sent', sent_at=timestamp(), discord_message_id=message)
    save()
    return message


def run_loop(config_path, runtime, binary, interval):
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    with daemon_lock(runtime):
        write_state(runtime, pid=os.getpid(), interval_seconds=interval, started_at=timestamp())
        LOGGER.info('Started; sending immediately, then every %s seconds', interval)
        try:
            while not stop.is_set():
                started = time.monotonic()
                try:
                    # Reload configuration each time so a rotated webhook is picked up.
                    send_once(config_path, runtime, binary)
                except NotificationError as exc:
                    write_state(runtime, last_error=str(exc))
                    LOGGER.error('%s', exc)
                except Exception as exc:
                    # Unknown exceptions may contain URLs; log their type only.
                    reason = f'Unexpected report error: {type(exc).__name__}'
                    write_state(runtime, last_error=reason)
                    LOGGER.error('%s', reason)
                elapsed = time.monotonic() - started
                stop.wait(max(0, interval - elapsed))
        finally:
            write_state(runtime, pid=None, stopped_at=timestamp())
            LOGGER.info('Stopped')


def setup_logging(runtime):
    runtime.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(runtime / 'notifier.log', maxBytes=1_000_000, backupCount=3)
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG, help='Private local JSON; default: /workspace/.discord-gpu.json')
    parser.add_argument('--runtime', type=Path, default=DEFAULT_RUNTIME, help='Status, lock and rotated logs directory')
    parser.add_argument('--nvidia-smi', default='/usr/bin/nvidia-smi', help='Executable path; never interpreted by a shell')
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ['configure', 'dry-run', 'once', 'stop', 'status']:
        commands.add_parser(name)
    job = commands.add_parser('job', help='Send one completion/failure notification for a job JSON')
    job.add_argument('--status', type=Path, required=True)
    job.add_argument('--name')
    for name in ['start', 'run']:
        sub = commands.add_parser(name)
        sub.add_argument('--interval-seconds', type=int, default=7200)
    args = parser.parse_args(argv)
    config_path, runtime = args.config.absolute(), args.runtime.resolve()
    try:
        if args.command == 'configure':
            configure(config_path)
        elif args.command == 'job':
            send_job(config_path, args.status, args.name)
            print('Discord accepted the job notification.')
        elif args.command == 'dry-run':
            output, gpu_exit = collect_gpu(args.nvidia_smi)
            print(f'GPU status preview | {timestamp()} (Asia/Taipei) | nvidia-smi exit: {gpu_exit}\n{output}')
            print('Dry run only: no Discord request sent.')
        elif args.command == 'status':
            print(json.dumps({**read_state(runtime), 'running': bool(pid := running_pid(runtime)), 'pid': pid}, indent=2))
        elif args.command == 'stop':
            pid = running_pid(runtime)
            if pid:
                os.kill(pid, signal.SIGTERM)
                print(f'Stop requested for notifier PID {pid}. Check status to confirm shutdown.')
            else:
                print('Notifier is not running.')
        elif args.command == 'start':
            load_config(config_path)
            if args.interval_seconds < 60:
                raise NotificationError('Interval must be at least 60 seconds; default is 7200 (two hours).')
            if pid := running_pid(runtime):
                print(f'Already running as PID {pid}; no duplicate started.')
                return 0
            runtime.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, str(Path(__file__).resolve()), '--config', str(config_path), '--runtime', str(runtime),
                       '--nvidia-smi', args.nvidia_smi, 'run', '--interval-seconds', str(args.interval_seconds)]
            with (runtime / 'launcher.log').open('a') as log:
                child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                         start_new_session=True, close_fds=True)
            for _ in range(50):
                if pid := running_pid(runtime):
                    print(f'Started notifier PID {pid}: immediately, then every {args.interval_seconds // 60} minutes. Logs: {runtime / "notifier.log"}')
                    return 0
                if child.poll() is not None:
                    raise NotificationError(f'Notifier exited during startup. Check {runtime / "launcher.log"}.')
                time.sleep(0.1)
            raise NotificationError('Startup is taking longer than expected; check status before trying start again.')
        else:
            setup_logging(runtime)
            load_config(config_path)
            if args.command == 'once':
                gpu_exit = send_once(config_path, runtime, args.nvidia_smi)
                print(f'Discord accepted the report (nvidia-smi exit: {gpu_exit}).')
            else:
                if args.interval_seconds < 60:
                    raise NotificationError('Interval must be at least 60 seconds.')
                run_loop(config_path, runtime, args.nvidia_smi, args.interval_seconds)
        return 0
    except NotificationError as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1
    except (OSError, EOFError, KeyboardInterrupt):
        print('Operation interrupted or a local file/process could not be accessed. No credential is printed.', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
