#!/bin/bash

# ---- CONFIGURATION ----
SSH_HOST="xf09id1-det-ioc2.nsls2.bnl.gov"
WINDOWS_IP="xf09id1-merlin1.nsls2.bnl.gov"

# Optional: RDP credentials (leave empty to prompt or use GUI client)
RDP_USER="administrator"
#RDP_PASS="your_windows_password"

echo "Enter your username (for tunnel ssh):"

read SSH_USER

# ---- SSH Tunnel ----
echo "Cleaning up any old tunnels..."
pkill -f "ssh -f -N -L ${LOCAL_PORT}:${WINDOWS_IP}:3389"
echo "Starting SSH tunnel to $WINDOWS_IP via $SSH_HOST..."
ssh -f -N -L ${LOCAL_PORT}:${WINDOWS_IP}:3389 ${SSH_USER}@${SSH_HOST}

if [ $? -ne 0 ]; then
    echo "Failed to create SSH tunnel."
    exit 1
fi

echo "SSH tunnel established. Connecting to RDP at localhost:${LOCAL_PORT}..."

# ---- Launch RDP Client ----
# Using xfreerdp
#xfreerdp /v:localhost:${LOCAL_PORT} /u:${RDP_USER} /p:${RDP_PASS} +clipboard /cert:ignore
echo "Enter the Merlin Windows credential..."
xfreerdp /v:localhost:${LOCAL_PORT} /u:${RDP_USER} /cert:ignore /smart-sizing:1440x1080

# Optional: clean up tunnel (use pkill if needed)
echo "Clean up the tunnel"
pkill -f "ssh -f -N -L ${LOCAL_PORT}:${WINDOWS_IP}:3389"
