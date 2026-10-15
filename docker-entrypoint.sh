#!/bin/bash
set -euo pipefail

# Generate SSH host keys if they don't already exist.
ssh-keygen -A

# Install the public key supplied by the Docker run command.
if [[ -f /run/secrets/codex_authorized_keys ]]; then
    install -o codex -g codex -m 600 \
        /run/secrets/codex_authorized_keys \
        /home/codex/.ssh/authorized_keys
else
    echo "ERROR: /run/secrets/codex_authorized_keys is missing."
    echo "Mount your Windows public key into the container."
    exit 1
fi

mkdir -p /run/sshd
chmod 755 /run/sshd

# Validate SSH configuration before starting.
sshd -t

echo "Starting SSH server..."
exec /usr/sbin/sshd -D -e