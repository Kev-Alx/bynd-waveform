# Send nvidia-smi to Discord every two hours

## Job and script completion notifications

The user requested Discord notifications for every job/script completion on 17 October 2026. `run_logged.py --notify-discord` sends a completion, failure, or interruption message through this reporter after saving the compute result. Use `--job-name NAME` for a readable label. Alternatively, set `BEYONDWAVEFORM_NOTIFY_DISCORD=1`; `Supervisor.phase` then wraps each stage so it sends its own completion notice as well as the overall job.

```bash
/opt/venv/bin/python scripts/run_logged.py --notify-discord --job-name my-job --log /workspace/reproduction-runs/my-job.log --status /workspace/reproduction-runs/my-job/job.json -- /opt/venv/bin/python scripts/example.py
/usr/bin/python3 scripts/discord_gpu_report.py job --status /workspace/reproduction-runs/my-job/job.json --name my-job
```

Delivery receipts are stored beside each job as `job.discord.json`. A successful receipt suppresses duplicate sends for the same finished attempt; a failed receipt can be retried with the `job` command. Notification failure preserves the actual job exit code. Messages omit command arguments/environment variables and disable automatic mentions. A notification sender needs read access to the private config; a root-owned mode-600 file cannot be read by an unprivileged job. Keep mode 600 and give the job-running user ownership when appropriate. Existing root GPU monitoring continues to work after that ownership change.

The script is [discord_gpu_report.py](/workspace/scripts/discord_gpu_report.py). It uses Python 3's standard library and runs on the GPU machine/container. No Discord bot account, Python package installation, cron or systemd is required.

**Current state:** implemented and tested locally. Delivery is not active until you create a webhook, configure it, and run `start` below. No Discord message was sent during implementation.

## 1. Create a Discord webhook

Use an ordinary **text channel** in a server you own or can manage, for example `#gpu-status`. This setup posts to that channel, rather than to a direct-message conversation.

1. Open the server's **Server Settings → Integrations → Webhooks**.
2. Choose **Create Webhook** (or **New Webhook**).
3. Name it **GPU Status** and select `#gpu-status` as its destination.
4. Save changes if prompted, then click **Copy Webhook URL**.

You need the **Manage Webhooks** permission, or help from the server administrator. Keep the URL private: it contains the token that permits posting to that channel. Enter it only in the hidden local prompt below; do not paste it into chat.

Official references: [Discord webhook setup](https://support.discord.com/hc/en-us/articles/228383668-Intro-to-Webhooks), [webhook permissions/API](https://docs.discord.com/developers/resources/webhook).

## 2. Configure it on the GPU machine

Run this in the Linux terminal **inside the current GPU container**, where `/workspace` and `nvidia-smi` exist:

```bash
/usr/bin/python3 /workspace/scripts/discord_gpu_report.py configure
```

At `Discord webhook URL (hidden):`, paste the copied URL and press Enter. Nothing will appear as you paste; that is expected. Next, enter a recognizable machine label, such as `TITAN-RTX`, or press Enter for the hostname.

The script saves `/workspace/.discord-gpu.json` with mode `600` (only the owner can read/write it). That filename has been added to `.gitignore`. The URL is not placed in command-line arguments, terminal output or notifier logs.

Configuration alone does not send anything or start a schedule.

## 3. Preview and send a test

Preview the GPU output without contacting Discord:

```bash
/usr/bin/python3 /workspace/scripts/discord_gpu_report.py dry-run
```

Send one actual message to the configured channel:

```bash
/usr/bin/python3 /workspace/scripts/discord_gpu_report.py once
```

Check that a message appears in Discord. A normal `nvidia-smi` table may exceed Discord's message-length limit; in that case, open the attached **nvidia-smi.txt** file. It contains the full output, including the process table, without truncation.

A successful `once` command means Discord accepted the report. The message separately shows `nvidia-smi`'s exit code: `0` is success. If the GPU command fails or times out, its error is reported to Discord too.

## 4. Start the two-hour schedule

```bash
/usr/bin/python3 /workspace/scripts/discord_gpu_report.py start
```

This sends immediately, then runs every **7,200 seconds (two hours)**. For example, starting at 17:30 Asia/Taipei gives reports around 17:30, 19:30, 21:30, and so on. The manual `once` test and `start` each send a report, so two messages during setup are expected.

The process runs in the background and survives closing the terminal/SSH session. It uses a lock to prevent two background notifier copies from running simultaneously. A second `start` reports the existing process rather than starting another sender.

The Discord message header shows **Asia/Taipei (UTC+8)**. The timestamp inside the original `nvidia-smi` output retains the GPU machine's own timezone.

## 5. Check or stop it

Check whether it is running and inspect the last attempt, last delivery and any error:

```bash
/usr/bin/python3 /workspace/scripts/discord_gpu_report.py status
```

Inspect recent logs:

```bash
tail -n 30 /workspace/reproduction-runs/discord-gpu/notifier.log
```

Stop the background notifier:

```bash
/usr/bin/python3 /workspace/scripts/discord_gpu_report.py stop
```

Use `status` afterward to confirm it has stopped. A send already in progress may finish before shutdown. Logs rotate automatically at approximately 1 MB with three backups.

If you replace/delete the webhook, rerun `configure`. A running notifier reloads its configuration before each report, so it will use the replacement URL on the next scheduled send. `once` can test the replacement immediately.

## Keeping it running

- The GPU machine/container must remain powered on, with outbound HTTPS access to Discord.
- A terminal disconnect is fine. A **container stop/restart or host reboot stops the notifier**; run `start` again afterward, or invoke that same command from your container's startup mechanism.
- This implementation does not change the Docker image or entrypoint and does not enable automatic startup after reboot. Keep `/workspace` on its persistent mount to retain the script and private configuration when recreating the container.
- Enable the appropriate channel/app notifications on your Discord client if you also want alerts on your phone. Messages do not intentionally ping anyone.

## Optional: use an existing Linux cron service instead

The current container has no cron daemon or `crontab` command. The background loop above is the ready-to-use option here.

If you already have a functioning cron service **in the same GPU environment with the same paths and user access**, you can use one-shot mode instead. Stop the background loop first and prepare the log directory:

```bash
/usr/bin/python3 /workspace/scripts/discord_gpu_report.py stop
mkdir -p /workspace/reproduction-runs/discord-gpu
crontab -e
```

Add this single line to that user's crontab:

```cron
0 */2 * * * /usr/bin/python3 /workspace/scripts/discord_gpu_report.py once >> /workspace/reproduction-runs/discord-gpu/cron.log 2>&1
```

This runs at minute zero of every even-numbered hour in the cron service's timezone. The cron user must be able to read the private config and access the GPU. A cron job on the Docker host cannot use the container's `/workspace` path automatically; it would need an appropriate `docker exec` command for the actual container name. Use one scheduler, either this cron entry or the background loop.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| `Not configured` | Run `configure` in the same environment/user account that will run the notifier. |
| Config permission error | Run `chmod 600 /workspace/.discord-gpu.json`. |
| Discord HTTP 401/403/404 | Recreate or copy the correct webhook, verify the destination channel, then rerun `configure`. |
| Discord HTTP 429 or network error | Check logs and connectivity. The script makes up to three short attempts; a failed send does not stop future two-hour reports. |
| `nvidia-smi` error in the report | Check NVIDIA driver/container GPU access. The notifier sends the error rather than silently skipping the report. |
| No messages after reboot | Run `start` again; the detached process does not survive a container/host restart. |

## Checks performed

- Ran `nvidia-smi` successfully against the visible NVIDIA TITAN RTX.
- Ran the script's real `dry-run`; no network request was sent.
- Passed 13 offline unit tests covering complete long-output attachments, no automatic mentions, hidden/private configuration, rate limits, invalid webhooks, credential-safe errors, redirect refusal, GPU failures, duplicate-process locking and continued scheduling after a delivery failure.
- Confirmed `status` reports `running: false` before user configuration/activation.

To rerun the offline checks:

```bash
/usr/bin/python3 -m unittest discover -s /workspace/tests -p test_discord_gpu_report.py -v
```
