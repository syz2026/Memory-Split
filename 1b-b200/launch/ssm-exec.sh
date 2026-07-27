#!/bin/bash
# Run an ad-hoc shell snippet on the node over SSM and print its output.
#   ./ssm-exec.sh <instance-id> 'command string'
#   ./ssm-exec.sh <instance-id> -f script.sh
set -uo pipefail
cd "$(dirname "$0")"

PROFILE=${AWS_PROFILE_NAME}
REGION=us-east-1
IID="${1:?usage: ssm-exec.sh <instance-id> <command|-f file>}"
shift

if [ "${1:-}" = "-f" ]; then
  SCRIPT=$(cat "$2")
else
  SCRIPT="$1"
fi

REQ=$(mktemp)
SSM_SCRIPT="$SCRIPT" SSM_IID="$IID" python3 -c '
import json, os
print(json.dumps({
    "InstanceIds": [os.environ["SSM_IID"]],
    "DocumentName": "AWS-RunShellScript",
    "TimeoutSeconds": 3600,
    "Parameters": {"commands": [os.environ["SSM_SCRIPT"]], "executionTimeout": ["3600"]},
}))' > "$REQ"

CID=$(aws --profile "$PROFILE" ssm send-command --region "$REGION" \
  --cli-input-json "file://${REQ}" --query 'Command.CommandId' --output text) || {
    rm -f "$REQ"; echo "send-command failed"; exit 1; }
rm -f "$REQ"

while true; do
  ST=$(aws --profile "$PROFILE" ssm get-command-invocation --region "$REGION" \
    --command-id "$CID" --instance-id "$IID" --query 'Status' --output text 2>/dev/null)
  case "$ST" in Success|Failed|Cancelled|TimedOut) break ;; *) sleep 5 ;; esac
done

aws --profile "$PROFILE" ssm get-command-invocation --region "$REGION" \
  --command-id "$CID" --instance-id "$IID" --query 'StandardOutputContent' --output text
ERR=$(aws --profile "$PROFILE" ssm get-command-invocation --region "$REGION" \
  --command-id "$CID" --instance-id "$IID" --query 'StandardErrorContent' --output text)
[ -n "$ERR" ] && [ "$ERR" != "None" ] && { echo "--- stderr ---"; echo "$ERR"; }
echo "[status: $ST]"
