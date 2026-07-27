#!/bin/bash
# Push the staging kit to the node over SSM and run it.
#
# Ships over SSM rather than S3 so it needs no extra IAM grant on the instance
# role (which is deliberately read-only on corpus/* and write-only on runs/*).
# The whole kit is ~25 KB, comfortably inside the send-command payload limit.
#
#   ./push-stage.sh i-0123456789abcdef0            -> push, then run all stages
#   ./push-stage.sh i-0123456789abcdef0 1          -> push, then run stage 1 only

set -uo pipefail
cd "$(dirname "$0")"

PROFILE=${AWS_PROFILE_NAME}
REGION=us-east-1
IID="${1:?usage: push-stage.sh <instance-id> [stage]}"
ONLY="${2:-all}"

send() {
  local desc="$1"; shift
  local script="$1"
  echo "=== ${desc} ==="

  # Build the request with a real JSON encoder. Hand-rolled sed escaping of a
  # multi-line script into a JSON array corrupts the payload, and AWS-RunShellScript
  # executes under dash, so the corruption surfaces as baffling `set` errors
  # rather than anything that points at quoting. Also force bash explicitly:
  # the staged scripts are bash, not POSIX sh.
  local reqfile
  reqfile=$(mktemp)
  SSM_SCRIPT="$script" SSM_IID="$IID" SSM_DESC="$desc" python3 -c '
import json, os
req = {
    "InstanceIds": [os.environ["SSM_IID"]],
    "DocumentName": "AWS-RunShellScript",
    "Comment": os.environ["SSM_DESC"][:100],
    "TimeoutSeconds": 3600,
    "Parameters": {
        "commands": [os.environ["SSM_SCRIPT"]],
        "executionTimeout": ["3600"],
    },
}
print(json.dumps(req))
' > "$reqfile" || { echo "could not build request"; return 1; }

  CID=$(aws --profile "$PROFILE" ssm send-command --region "$REGION" \
    --cli-input-json "file://${reqfile}" \
    --query 'Command.CommandId' --output text) || { echo "send-command failed"; rm -f "$reqfile"; return 1; }
  rm -f "$reqfile"

  echo "command: $CID"
  while true; do
    ST=$(aws --profile "$PROFILE" ssm get-command-invocation --region "$REGION" \
      --command-id "$CID" --instance-id "$IID" --query 'Status' --output text 2>/dev/null)
    case "$ST" in
      Success|Failed|Cancelled|TimedOut) break ;;
      *) sleep 10 ;;
    esac
  done

  aws --profile "$PROFILE" ssm get-command-invocation --region "$REGION" \
    --command-id "$CID" --instance-id "$IID" \
    --query 'StandardOutputContent' --output text
  ERR=$(aws --profile "$PROFILE" ssm get-command-invocation --region "$REGION" \
    --command-id "$CID" --instance-id "$IID" \
    --query 'StandardErrorContent' --output text)
  [ -n "$ERR" ] && [ "$ERR" != "None" ] && { echo "--- stderr ---"; echo "$ERR"; }
  echo "status: $ST"
  [ "$ST" = "Success" ]
}

echo "packing staging kit"
PAYLOAD=$(tar czf - -C stage . | base64 | tr -d '\n')
echo "payload: ${#PAYLOAD} base64 chars"

send "unpack staging kit" "$(cat <<EOF
set -e
mkdir -p /mnt/nvme/stage
echo '${PAYLOAD}' | base64 -d | tar xzf - -C /mnt/nvme/stage
chmod +x /mnt/nvme/stage/*.sh /mnt/nvme/stage/*.py
ls -la /mnt/nvme/stage
cat /var/log/ms-bootstrap.status
EOF
)" || exit 1

run_stage() {
  send "stage $1" "bash /mnt/nvme/stage/$2 2>&1"
}

case "$ONLY" in
  1)   run_stage 1 01-gpu-verify.sh ;;
  2)   run_stage 2 02-corpus-pull.sh ;;
  3)   run_stage 3 03-nccl-check.sh ;;
  4)   run_stage 4 04-code-package.sh ;;
  all)
    # The corpus transfer is network- and disk-bound; the GPU attestation and
    # NCCL collectives are GPU-bound. Overlapping them reclaims roughly twelve
    # minutes of a block that costs ~$99/hr. Corpus goes first and detached so
    # it is already moving bytes while the fast checks run.
    send "start corpus transfer (detached)" \
      "mkdir -p /mnt/nvme/stage/logs
       nohup bash /mnt/nvme/stage/02-corpus-pull.sh > /mnt/nvme/stage/logs/stage2.log 2>&1 &
       echo \$! > /mnt/nvme/stage/logs/stage2.pid
       sleep 2; echo started pid \$(cat /mnt/nvme/stage/logs/stage2.pid)" || exit 1

    run_stage 1 01-gpu-verify.sh || echo "STAGE 1 reported failure - review before continuing"
    run_stage 4 04-code-package.sh || echo "STAGE 4 reported failure - review before continuing"
    run_stage 3 03-nccl-check.sh || echo "STAGE 3 reported failure - review before continuing"

    send "await corpus transfer" \
      "PID=\$(cat /mnt/nvme/stage/logs/stage2.pid 2>/dev/null)
       while kill -0 \"\$PID\" 2>/dev/null; do sleep 15; done
       cat /mnt/nvme/stage/logs/stage2.log
       grep -q 'STAGE 2 RESULT: PASS' /mnt/nvme/stage/logs/stage2.log" \
      || echo "STAGE 2 reported failure - review the log above"
    ;;
  *) echo "unknown stage: $ONLY"; exit 1 ;;
esac
