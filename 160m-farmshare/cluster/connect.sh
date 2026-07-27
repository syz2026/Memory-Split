#!/usr/bin/env bash
# One-time interactive auth from the LOCAL machine; keeps a multiplexed SSH
# session alive so subsequent ssh/rsync (and this repo's sync scripts) run
# non-interactively. Adapted from LoopRFA's cluster/local/connect.sh.
#
# Usage: bash cluster/connect.sh SUNETID [HOST]
#   HOST defaults to rice-04.farmshare.stanford.edu (a specific login node);
#   pass login.farmshare.stanford.edu for the load-balanced pool, or
#   dtn.farmshare.stanford.edu to warm the transfer node that
#   sync_push.sh / sync_pull.sh use (each host needs its own warm-up —
#   ControlMaster sockets are per-host).
#
# The socket path matches ssh_config.example and the sync scripts
# (~/.ssh/cm-%r@%h:%p), so anything using those rides this session.
set -euo pipefail
SUNET=${1:?usage: connect.sh SUNETID [HOST]}
# SUNetIDs are lowercase; a stray capital or @stanford.edu suffix breaks auth.
SUNET=$(printf '%s' "$SUNET" | tr '[:upper:]' '[:lower:]' | sed 's/@.*//')
HOST=${2:-${FARMSHARE_HOST:-rice-04.farmshare.stanford.edu}}
echo "Connecting to $HOST as $SUNET ..."
SOCK_PATTERN="$HOME/.ssh/cm-%r@%h:%p"
SOCK_LITERAL="$HOME/.ssh/cm-$SUNET@$HOST:22"
mkdir -p "$HOME/.ssh"
# FarmShare is password+Duo only. Disable pubkey auth so ssh-agent key offers
# can't exhaust the server's MaxAuthTries before the password prompt appears
# ("Too many authentication failures").
ssh -o ControlMaster=auto -o ControlPath="$SOCK_PATTERN" -o ControlPersist=8h \
    -o ServerAliveInterval=30 -o ServerAliveCountMax=10 -o TCPKeepAlive=yes \
    -o PubkeyAuthentication=no \
    -o PreferredAuthentications=keyboard-interactive,password \
    "$SUNET@$HOST" \
    'echo "FarmShare session live on $(hostname) as $(whoami)"'
cat <<EOF
Session established (persists 8h). Non-interactive reuse:
  ssh -o ControlPath="$SOCK_PATTERN" $SUNET@$HOST '<command>'
  rsync -e "ssh -o ControlPath=\"$SOCK_PATTERN\"" ...
This repo's sync_push.sh / sync_pull.sh reuse it automatically for the DTN
host (warm that separately: bash cluster/connect.sh $SUNET dtn.farmshare.stanford.edu).
Close with: ssh -O exit -o ControlPath="$SOCK_LITERAL" "$SUNET@$HOST"
EOF
