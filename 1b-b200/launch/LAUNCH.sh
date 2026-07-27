#!/bin/bash
# One-step launch into Capacity Block ${CAPACITY_BLOCK_ID}.
#
# Re-runs the dry run immediately before the real call, then launches, then
# proves the instance actually consumed reservation capacity rather than
# silently falling through to on-demand at ~$99/hr alongside an idle paid block.
#
#   ./LAUNCH.sh              -> dry run only
#   ./LAUNCH.sh --confirm    -> dry run, then real launch

set -uo pipefail
cd "$(dirname "$0")"

PROFILE=${AWS_PROFILE_NAME}
REGION=us-east-1
CRID=${CAPACITY_BLOCK_ID}
AMI=am${INSTANCE_ID}
SUBNET=subnet-04dd46c921c40074f
SG=sg-048178b0708d33ff2
IAMPROFILE=ms-b200-training-node

run_instances() {
  aws --profile "$PROFILE" ec2 run-instances \
    --region "$REGION" \
    --image-id "$AMI" \
    --instance-type p6-b200.48xlarge \
    --count 1 \
    --subnet-id "$SUBNET" \
    --security-group-ids "$SG" \
    --iam-instance-profile "Name=${IAMPROFILE}" \
    --capacity-reservation-specification "CapacityReservationTarget={CapacityReservationId=${CRID}}" \
    --instance-market-options 'MarketType=capacity-block' \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":200,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
    --metadata-options 'HttpTokens=required,HttpEndpoint=enabled,HttpPutResponseHopLimit=2' \
    --private-dns-name-options 'HostnameType=ip-name,EnableResourceNameDnsARecord=true' \
    --user-data file://user-data.sh \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=ms-135m-b200-node},{Key=Project,Value=MemorySplit},{Key=Cohort,Value=memorysplit-exploratory-v3-135m-aws-n10}]' \
    "$@"
}

echo "=== reservation preflight ==="
aws --profile "$PROFILE" ec2 describe-capacity-reservations \
  --capacity-reservation-ids "$CRID" --region "$REGION" \
  --query 'CapacityReservations[].{State:State,Avail:AvailableInstanceCount,Total:TotalInstanceCount,Start:StartDate}' \
  --output table

STATE=$(aws --profile "$PROFILE" ec2 describe-capacity-reservations \
  --capacity-reservation-ids "$CRID" --region "$REGION" \
  --query 'CapacityReservations[0].State' --output text)

if [ "$STATE" != "active" ]; then
  echo "NOTE: reservation state is '${STATE}', not 'active'."
  echo "      Capacity Block launches only succeed from 16:01:00Z onward."
  [ "${1:-}" = "--confirm" ] && { echo "Refusing to launch before the block is active."; exit 1; }
fi

echo
echo "=== dry run ==="
DRY=$(run_instances --dry-run 2>&1)
echo "$DRY"
if ! echo "$DRY" | grep -q "DryRunOperation"; then
  echo "ABORT: dry run did not return DryRunOperation."
  exit 1
fi
echo "dry run OK"

if [ "${1:-}" != "--confirm" ]; then
  echo
  echo "Dry run only. Re-run with --confirm to launch."
  exit 0
fi

echo
echo "=== LAUNCHING ==="
OUT=$(run_instances --output json) || { echo "ABORT: run-instances failed"; echo "$OUT"; exit 1; }
IID=$(echo "$OUT" | python3 -c 'import json,sys; print(json.load(sys.stdin)["Instances"][0]["InstanceId"])')
echo "instance: $IID"
echo "$OUT" > "launched-${IID}.json"

echo
echo "=== reservation consumption check ==="
# The whole point of the targeted block: if this does not bind, we are paying
# twice. Check the instance side and the reservation side independently.
# describe-instances is eventually consistent and returns None for the first
# few seconds after run-instances even on a correctly bound instance, so this
# retries before concluding anything. Declaring "unbound" too early would
# trigger a spurious terminate of a perfectly good $4.3k allocation.
BOUND=None
for attempt in $(seq 1 12); do
  BOUND=$(aws --profile "$PROFILE" ec2 describe-instances --instance-ids "$IID" --region "$REGION" \
    --query 'Reservations[0].Instances[0].CapacityReservationId' --output text)
  [ "$BOUND" = "$CRID" ] && break
  echo "  [${attempt}/12] binding not yet visible (got '${BOUND}'), retrying..."
  sleep 5
done
echo "instance.CapacityReservationId = ${BOUND}"

aws --profile "$PROFILE" ec2 describe-capacity-reservations \
  --capacity-reservation-ids "$CRID" --region "$REGION" \
  --query 'CapacityReservations[].{State:State,Total:TotalInstanceCount,Avail:AvailableInstanceCount,Alloc:CapacityAllocations}' \
  --output json

if [ "$BOUND" != "$CRID" ]; then
  echo
  echo "*** CRITICAL: instance is NOT bound to ${CRID}. It is running at on-demand"
  echo "*** rates while the paid block sits idle. Terminate and investigate now:"
  echo "***   aws --profile ${PROFILE} ec2 terminate-instances --region ${REGION} --instance-ids ${IID}"
  exit 1
fi
echo "CONFIRMED: instance is consuming the capacity block."

echo
echo "=== waiting for SSM registration ==="
for i in $(seq 1 60); do
  PING=$(aws --profile "$PROFILE" ssm describe-instance-information --region "$REGION" \
    --filters "Key=InstanceIds,Values=${IID}" \
    --query 'InstanceInformationList[0].PingStatus' --output text 2>/dev/null)
  if [ "$PING" = "Online" ]; then echo "SSM Online after ~$((i*10))s"; break; fi
  echo "  [${i}/60] ssm=${PING:-none}"; sleep 10
done

echo
echo "Shell:  aws --profile ${PROFILE} ssm start-session --region ${REGION} --target ${IID}"
echo "Staging: ./push-stage.sh ${IID}"
