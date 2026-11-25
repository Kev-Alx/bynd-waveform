# Speech experiment audio comparison

Local container interface: **http://localhost:8888/**, or the Docker host's address on the mapped port. The server binds `0.0.0.0:8888` and serves only this app's compiled `dist` directory. This delivery uses the user's existing container port; no cloud deployment is required.

## What is included

- 67 scored conditions across 15 studies/control groups; 2,664 scored recordings over 48 distinct development sentences.
- 23 condition rows use the historical 24-recording cohort; 44 use the 48-recording cohort. Repeated controls remain labeled per study.
- One, two or three run panels, original WAV playback/download, common-scale waveform previews, reference and recognized transcripts, per-recording WER/quality, and full-run WER/quality status.
- Sentence search, explicit unavailable-cohort states, a shared-sentence filter, linked playback positions and playback-speed controls.
- Complete cohort/study/quality filtering, sorting and CSV export; plain-language changes and future steps.

Audio is copied byte-for-byte from the saved experiments. Only PCM-identical recordings share an asset. No volume normalization, resampling, trimming, lossy compression or model inference is applied by this interface.

## Rebuild

The complete build command also stages the audio assets after the Data app compiler clears `dist`:

```bash
/opt/venv/bin/python /workspace/improvement/build_audio_browser.py
```

To regenerate reviewed data and verify the source receipts first:

```bash
/opt/venv/bin/python /workspace/improvement/build_audio_browser_data.py
```

Data regeneration sets `buildStatus` to `creating`. After completing changes/checks, set it to `complete` in `src/data.json` and run the complete build command. Preserve the app ID and existing source receipts.

## Server operation

The current server runs in a detached session. Check `/workspace/improvement/audio_browser_server_status.json` and `/workspace/improvement/audio_browser_server_job.json` before launching another instance. Do not kill an unrelated process to obtain the port.

The command to serve an available port 8888 is:

```bash
/opt/venv/bin/python -u /workspace/improvement/serve_audio_browser.py
```

Long-lived launches should use `/workspace/scripts/run_logged.py`, `stdin=DEVNULL` and `start_new_session=True`, as recorded in this task. The wrapper captures an explicit exit receipt; the server updates a heartbeat every 30 seconds. The server supports byte-range audio requests for seeking.

Logs/status are beside the project under `/workspace/improvement`: `audio_browser_server.log`, `audio_browser_server_status.json`, and `audio_browser_server_job.json`. The research run's separate `JOB_STATUS.*` files remain unchanged.

## Verification

`audio_browser_data_receipt.json` reconciles every saved per-condition count, pooled S/D/I/word denominator, mean quality score and audio hash. `audio_browser_build_receipt.json` checks all playback assets are staged. `audio_browser_browser_receipt.json` records actual Chromium playback, position switching, seek/speed/panel controls, filters, CSV export, mobile geometry and runtime errors. Browser dependencies live entirely in the sibling `audio_browser_qa` directory.

The original implementation and frozen experimental code were not edited by this interface task. No new GPU research run was launched. See `../16_all_results.md` and `../17_summary_and_next_steps.md` for the scientific outcome and proposed next steps.
