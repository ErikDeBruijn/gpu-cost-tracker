#!/bin/bash
set -euo pipefail

# Create unprivileged service user (with video group for GPU access)
if ! id cost-tracker &>/dev/null; then
    useradd --system --no-create-home --shell /usr/sbin/nologin cost-tracker
fi
# Add to video group for nvidia-smi/NVML access
usermod -aG video cost-tracker

# Deploy code
mkdir -p /opt/cost-function
cp cost_service.py power_monitor.py epex_prices.py requirements.txt /opt/cost-function/

# Python venv
if [ ! -d /opt/cost-function/venv ]; then
    python3 -m venv /opt/cost-function/venv
fi
/opt/cost-function/venv/bin/pip install -q -r /opt/cost-function/requirements.txt

# Environment file with ENTSO-E token
if [ ! -f /etc/default/cost-tracker ]; then
    echo 'ENTSOE_TOKEN=your-entsoe-token-here' > /etc/default/cost-tracker
    chmod 600 /etc/default/cost-tracker
    echo "WARNING: Set your ENTSO-E token in /etc/default/cost-tracker"
fi

# Set ownership
chown -R cost-tracker:cost-tracker /opt/cost-function

# Install and start service
cp deploy/cost-tracker.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable cost-tracker
systemctl restart cost-tracker
echo "Service started. Check: systemctl status cost-tracker"
