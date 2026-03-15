#!/bin/bash
set -euo pipefail

INSTALL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Create unprivileged service user (with video group for GPU access)
if ! id cost-tracker &>/dev/null; then
    useradd --system --no-create-home --shell /usr/sbin/nologin cost-tracker
fi
usermod -aG video cost-tracker

# Python venv
if [ ! -d "$INSTALL_DIR/venv" ]; then
    python3 -m venv "$INSTALL_DIR/venv"
fi
"$INSTALL_DIR/venv/bin/pip" install -q -r "$INSTALL_DIR/requirements.txt"

# Environment file with ENTSO-E token
if [ ! -f /etc/default/cost-tracker ]; then
    echo 'ENTSOE_TOKEN=your-entsoe-token-here' > /etc/default/cost-tracker
    chmod 600 /etc/default/cost-tracker
    echo "WARNING: Set your ENTSO-E token in /etc/default/cost-tracker"
fi

# Set ownership (readable by cost-tracker, owned by root)
chown -R root:cost-tracker "$INSTALL_DIR"
chmod -R g+r "$INSTALL_DIR"

# Install systemd service (patched with actual install dir)
sed "s|/opt/cost-function|$INSTALL_DIR|g" "$INSTALL_DIR/deploy/cost-tracker.service" \
    > /etc/systemd/system/cost-tracker.service
systemctl daemon-reload
systemctl enable cost-tracker
systemctl restart cost-tracker
echo "Service started. Check: systemctl status cost-tracker"
