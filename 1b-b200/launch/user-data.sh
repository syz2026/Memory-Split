#!/bin/bash
# Bootstrap for the MemorySplit 135m B200 capacity-block node.
# Goal: have fast local scratch ready before the operator gets a shell.
# Must never block SSM agent startup, so failures are logged, not fatal.

exec > >(tee -a /var/log/ms-bootstrap.log) 2>&1
set -x
echo "ms-bootstrap start $(date -u +%FT%TZ)"

STATUS=/var/log/ms-bootstrap.status
echo "RUNNING" > "$STATUS"

# Shell access comes only from SSM here (no key pair is attached), so getting
# the agent healthy is the single most important thing this script does and it
# runs before anything that could fail.
if ! systemctl is-active --quiet amazon-ssm-agent && ! systemctl is-active --quiet snap.amazon-ssm-agent.amazon-ssm-agent; then
  echo "ssm agent not active - installing"
  snap install amazon-ssm-agent --classic 2>/dev/null \
    || (curl -fsSL -o /tmp/ssm.deb https://s3.amazonaws.com/ec2-downloads-windows/SSMAgent/latest/debian_amd64/amazon-ssm-agent.deb \
        && dpkg -i /tmp/ssm.deb)
  systemctl enable --now snap.amazon-ssm-agent.amazon-ssm-agent 2>/dev/null \
    || systemctl enable --now amazon-ssm-agent 2>/dev/null || true
fi
systemctl start snap.amazon-ssm-agent.amazon-ssm-agent 2>/dev/null \
  || systemctl start amazon-ssm-agent 2>/dev/null || true

# Instance-store NVMe on Nitro reports this exact model string; EBS volumes do
# not. Selecting on the model is what keeps the root volume out of the array.
mapfile -t EPHEMERAL < <(
  lsblk -dno NAME,MODEL 2>/dev/null \
    | awk '/Amazon EC2 NVMe Instance Storage/ {print "/dev/"$1}' \
    | sort
)
echo "found ${#EPHEMERAL[@]} instance-store devices: ${EPHEMERAL[*]}"

if [ "${#EPHEMERAL[@]}" -eq 0 ]; then
  echo "NO_EPHEMERAL_DEVICES" > "$STATUS"
elif [ "${#EPHEMERAL[@]}" -eq 1 ]; then
  mkfs.xfs -f "${EPHEMERAL[0]}" && mkdir -p /mnt/nvme && mount "${EPHEMERAL[0]}" /mnt/nvme
  echo "SINGLE_DEVICE" > "$STATUS"
else
  # RAID0: this is scratch that is always re-derivable from S3, so striping for
  # throughput is the right trade against redundancy.
  mdadm --create --verbose /dev/md0 --level=0 --raid-devices="${#EPHEMERAL[@]}" \
        --force "${EPHEMERAL[@]}"
  mdadm --detail --scan >> /etc/mdadm/mdadm.conf 2>/dev/null || true
  mkfs.xfs -f /dev/md0
  mkdir -p /mnt/nvme
  mount -o noatime /dev/md0 /mnt/nvme
  echo "RAID0_${#EPHEMERAL[@]}_DEVICES" > "$STATUS"
fi

if mountpoint -q /mnt/nvme; then
  mkdir -p /mnt/nvme/corpus /mnt/nvme/runs /mnt/nvme/tmp
  chmod 1777 /mnt/nvme/tmp
  # DLAMI's interactive user; make the scratch usable without sudo.
  chown -R ubuntu:ubuntu /mnt/nvme 2>/dev/null || true
  df -h /mnt/nvme
  echo "OK $(cat "$STATUS")" > "$STATUS"
else
  echo "MOUNT_FAILED" > "$STATUS"
fi

# Persistence-mode keeps the driver resident so the first CUDA context does not
# pay initialisation cost on every one of the 20 runs.
nvidia-smi -pm 1 || true

echo "ms-bootstrap done $(date -u +%FT%TZ) status=$(cat "$STATUS")"
