"""Offline notifier tests: all Discord requests are mocked."""
from email.parser import BytesParser
from email.policy import default
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from urllib import error

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'discord_gpu_report.py'
spec = importlib.util.spec_from_file_location('discord_gpu_report', SCRIPT)
gpu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpu)
FAKE_URL = 'https://discord.com/api/webhooks/123456789/TOKEN_FOR_OFFLINE_TESTS_ONLY'


class ReporterTests(unittest.TestCase):
    def test_rejects_non_discord_webhooks_without_echoing_secret(self):
        for url in ['http://discord.com/api/webhooks/123/private',
                    'https://discord.com.attacker.example/api/webhooks/123/private',
                    'https://discord.com@attacker.example/api/webhooks/123/private',
                    'https://discord.com/api/webhooks/123/private#fragment']:
            with self.subTest(url=url), self.assertRaises(gpu.NotificationError) as caught:
                gpu.webhook_url(url)
            self.assertNotIn('private', str(caught.exception))

    def test_requires_confirmation_and_preserves_thread_destination(self):
        result = gpu.webhook_url(FAKE_URL + '?thread_id=444&wait=false')
        self.assertIn('wait=true', result)
        self.assertIn('thread_id=444', result)

    def test_short_output_is_complete_and_cannot_ping_everyone(self):
        body, kind = gpu.make_payload('GPU 0: 81%\n', '@everyone', 0, '2026-10-17T18:00:00+08:00')
        payload = json.loads(body)
        self.assertEqual(kind, 'application/json')
        self.assertIn('GPU 0: 81%', payload['content'])
        self.assertEqual(payload['allowed_mentions'], {'parse': []})

    def test_large_output_attaches_every_byte_in_one_message(self):
        output = ('GPU 0 memory/utilization 🟩\n' * 200) + 'LAST LINE\n'
        body, kind = gpu.make_payload(output, 'test-gpu', 0, '2026-10-17T18:00:00+08:00')
        message = BytesParser(policy=default).parsebytes(
            f'Content-Type: {kind}\r\nMIME-Version: 1.0\r\n\r\n'.encode() + body)
        fields = {p.get_param('name', header='content-disposition'): p for p in message.iter_parts()}
        payload = json.loads(fields['payload_json'].get_payload(decode=True))
        self.assertLessEqual(len(payload['content'].encode('utf-16-le')) // 2, 2000)
        self.assertEqual(fields['files[0]'].get_payload(decode=True), output.encode())
        self.assertEqual(payload['allowed_mentions'], {'parse': []})

    def test_missing_binary_becomes_an_error_report(self):
        with patch.object(gpu.subprocess, 'run', side_effect=FileNotFoundError):
            output, code = gpu.collect_gpu('/missing/nvidia-smi')
        self.assertEqual(code, 127)
        self.assertIn('not found', output)

    def test_timeout_keeps_partial_output(self):
        failure = subprocess.TimeoutExpired(['nvidia-smi'], 30, output=b'partial GPU status\n')
        with patch.object(gpu.subprocess, 'run', side_effect=failure):
            output, code = gpu.collect_gpu()
        self.assertEqual(code, 124)
        self.assertIn('partial GPU status', output)
        self.assertIn('timed out', output)

    def test_rate_limit_is_retried_and_confirmation_returned(self):
        failure = error.HTTPError(FAKE_URL, 429, 'rate limit', {}, io.BytesIO(b'{"retry_after": 0.1}'))
        response = MagicMock()
        response.status = 200
        response.read.return_value = b'{"id": "confirmed-message"}'
        opener = MagicMock()
        opener.open.side_effect = [failure, response]
        response.__enter__.return_value = response
        with patch.object(gpu.request, 'build_opener', return_value=opener), patch.object(gpu.time, 'sleep') as sleep:
            result = gpu.post_report(FAKE_URL, b'{}', 'application/json')
        self.assertEqual(result, 'confirmed-message')
        self.assertEqual(opener.open.call_count, 2)
        sleep.assert_called_once_with(0.1)

    def test_deleted_webhook_is_not_retried_and_token_is_not_logged(self):
        failure = error.HTTPError(FAKE_URL, 404, FAKE_URL, {}, io.BytesIO())
        opener = MagicMock()
        opener.open.side_effect = failure
        with patch.object(gpu.request, 'build_opener', return_value=opener), patch.object(gpu.time, 'sleep') as sleep:
            with self.assertRaises(gpu.NotificationError) as caught:
                gpu.post_report(FAKE_URL, b'{}', 'application/json')
        self.assertNotIn('TOKEN_FOR_OFFLINE_TESTS_ONLY', str(caught.exception))
        self.assertEqual(opener.open.call_count, 1)
        sleep.assert_not_called()

    def test_redirect_does_not_forward_webhook(self):
        with self.assertRaises(gpu.NotificationError):
            gpu.NoRedirect().redirect_request(None, None, 302, '', {}, 'https://attacker.example/')

    def test_private_configuration_and_hidden_configure(self):
        with tempfile.TemporaryDirectory() as folder:
            config = Path(folder) / 'secret.json'
            with patch.object(gpu.getpass, 'getpass', return_value=FAKE_URL), patch('builtins.input', return_value='my-gpu'), patch('sys.stdout', new_callable=io.StringIO) as stdout:
                gpu.configure(config)
            self.assertEqual(config.stat().st_mode & 0o777, 0o600)
            self.assertEqual(gpu.load_config(config)['label'], 'my-gpu')
            self.assertNotIn('TOKEN_FOR_OFFLINE_TESTS_ONLY', stdout.getvalue())
            config.chmod(0o644)
            with self.assertRaises(gpu.NotificationError):
                gpu.load_config(config)

    def test_second_daemon_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            runtime = Path(folder)
            with gpu.daemon_lock(runtime):
                with self.assertRaises(gpu.NotificationError):
                    with gpu.daemon_lock(runtime):
                        self.fail('A duplicate daemon acquired the lock')
            # A stale PID file with no held lock must not count as an active daemon.
            self.assertIsNone(gpu.running_pid(runtime))

    def test_dry_run_never_reads_credentials_or_sends(self):
        with patch.object(gpu, 'collect_gpu', return_value=('GPU status\n', 0)), patch.object(gpu, 'load_config') as config, patch.object(gpu, 'post_report') as send, patch('sys.stdout', new_callable=io.StringIO):
            result = gpu.main(['dry-run'])
        self.assertEqual(result, 0)
        config.assert_not_called()
        send.assert_not_called()

    def test_failed_delivery_does_not_kill_two_hour_schedule(self):
        with tempfile.TemporaryDirectory() as folder:
            stop = MagicMock()
            stop.is_set.side_effect = [False, False, True]
            with patch.object(gpu.threading, 'Event', return_value=stop), patch.object(gpu.signal, 'signal'), patch.object(gpu.time, 'monotonic', side_effect=[0, 2, 7200, 7203]), patch.object(gpu, 'send_once', side_effect=[gpu.NotificationError('offline'), 0]) as send, patch.object(gpu.LOGGER, 'error'):
                gpu.run_loop(Path(folder)/'unused.json', Path(folder), '/usr/bin/nvidia-smi', 7200)
            self.assertEqual(send.call_count, 2)
            self.assertEqual([c.args[0] for c in stop.wait.call_args_list], [7198, 7197])
            self.assertIsNone(gpu.read_state(Path(folder))['pid'])

    def test_job_notification_is_deduplicated_and_omits_command_secrets(self):
        with tempfile.TemporaryDirectory() as folder:
            status = Path(folder) / 'job.json'
            status.write_text(json.dumps({'status': 'complete', 'exit_code': 0,
                'command': ['tool', 'PRIVATE_COMMAND_TOKEN'], 'log': '/workspace/job.log'}))
            with patch.object(gpu, 'load_config', return_value={'webhook_url': FAKE_URL, 'label': 'gpu'}), \
                 patch.object(gpu, 'post_report', return_value='message-1') as send:
                self.assertEqual(gpu.send_job(Path(folder)/'private.json', status, 'defense'), 'message-1')
                self.assertEqual(gpu.send_job(Path(folder)/'private.json', status, 'defense'), 'message-1')
            send.assert_called_once()
            payload = json.loads(send.call_args.args[1])
            self.assertIn('Job complete: defense', payload['content'])
            self.assertNotIn('PRIVATE_COMMAND_TOKEN', payload['content'])
            self.assertEqual(payload['allowed_mentions'], {'parse': []})
            self.assertEqual(json.loads(status.with_name('job.discord.json').read_text())['status'], 'sent')

    def test_job_delivery_failure_preserves_result_and_can_be_retried(self):
        with tempfile.TemporaryDirectory() as folder:
            status = Path(folder) / 'job.json'
            original = json.dumps({'status': 'failed', 'exit_code': 7})
            status.write_text(original)
            with patch.object(gpu, 'load_config', return_value={'webhook_url': FAKE_URL, 'label': 'gpu'}), \
                 patch.object(gpu, 'post_report', side_effect=[gpu.NotificationError('offline'), 'message-2']):
                with self.assertRaises(gpu.NotificationError):
                    gpu.send_job(Path(folder)/'private.json', status)
                self.assertEqual(status.read_text(), original)
                self.assertEqual(json.loads(status.with_name('job.discord.json').read_text())['status'], 'failed')
                self.assertEqual(gpu.send_job(Path(folder)/'private.json', status), 'message-2')

    def test_running_job_is_not_announced_as_finished(self):
        with tempfile.TemporaryDirectory() as folder:
            status = Path(folder) / 'job.json'
            status.write_text(json.dumps({'status': 'running'}))
            with patch.object(gpu, 'load_config') as config, patch.object(gpu, 'post_report') as send:
                with self.assertRaises(gpu.NotificationError):
                    gpu.send_job(Path(folder)/'private.json', status)
                config.assert_not_called()
                send.assert_not_called()


if __name__ == '__main__':
    unittest.main()
