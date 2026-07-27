#!/usr/bin/env bash
# One-command day-1 staging, run from the LOCAL machine after
# `bash cluster/connect.sh <sunetid>` (or any live ControlMaster session):
#
#   bash cluster/stage.sh
#
# Does, in order:
#   1. rsync the repo to FarmShare scratch
#   2. idempotent env build + quick unit-test run on the login node
#   3. submit: GPU smoke, gates data build, all full/full1b data builds
#   4. generate the gates manifest and submit the 4 gate training runs
#      chained (afterok) on the gates data build
#   5. print the queue
#
# Later stages (sweep, calib1b, confirm) stay manual on purpose — they are
# gated on human review of gates A-C and the preregistration freeze.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=./config.env
source "$SCRIPT_DIR/config.env"
require_sunet

SOCK=$(cm_socket)
REPO=$(expand_path "$FS_REPO_DIR")
VENV=$(expand_path "$FS_VENV")
DATA=$(expand_path "$FS_DATA")
SSH=(ssh -o ControlPath="$SOCK" -o ConnectTimeout=10 -o BatchMode=yes "$SUNET_ID@$LOGIN_HOST")

echo "== 0/5 session check"
if ! "${SSH[@]}" 'echo "  session ok on $(hostname) as $(whoami)"'; then
    echo "ERROR: no live SSH session. Run: bash cluster/connect.sh $SUNET_ID" >&2
    exit 1
fi

echo "== 1/5 sync repo -> $LOGIN_HOST:$REPO"
bash "$SCRIPT_DIR/sync_push.sh"

echo "== 2/5 env + unit tests on login node"
"${SSH[@]}" "cd $REPO && bash cluster/setup_env.sh >setup_env.log 2>&1 && \
    PYTHONPATH=$REPO $VENV/bin/python -m pytest tests -q 2>&1 | tail -2"

echo "== 3/5 submit smoke + data builds"
SMOKE_JID=$("${SSH[@]}" "cd $REPO && sbatch --parsable cluster/slurm/smoke_gpu.sbatch")
echo "  smoke_gpu: job $SMOKE_JID"
GATES_DATA_JID=$("${SSH[@]}" "cd $REPO && sbatch --parsable \
    --export=ALL,BUILD_ARGS='--stage gates' cluster/slurm/data_prep.sbatch")
echo "  data gates (3 loads, 0.8B tok): job $GATES_DATA_JID"
for load in n50k n200k n800k; do
    jid=$("${SSH[@]}" "cd $REPO && sbatch --parsable --time=08:00:00 \
        --export=ALL,BUILD_ARGS='--stage full --loads $load' cluster/slurm/data_prep.sbatch")
    echo "  data full $load (3.2B tok): job $jid"
done
for load in n800k n4m; do
    jid=$("${SSH[@]}" "cd $REPO && sbatch --parsable --time=12:00:00 --mem=60G \
        --cpus-per-task=16 \
        --export=ALL,BUILD_ARGS='--stage full1b --loads $load' cluster/slurm/data_prep.sbatch")
    echo "  data full1b $load (10B tok): job $jid"
done

echo "== 4/5 gate training runs (chained afterok:$GATES_DATA_JID)"
"${SSH[@]}" "cd $REPO && PYTHONPATH=$REPO $VENV/bin/python scripts/make_manifest.py \
    --stage gates --data-root $DATA"
while IFS= read -r cfg; do
    [ -z "$cfg" ] && continue
    # -n: keep the inner ssh from consuming the loop's stdin
    jid=$(ssh -n -o ControlPath="$SOCK" -o BatchMode=yes "$SUNET_ID@$LOGIN_HOST" \
        "cd $REPO && sbatch --parsable \
        --dependency=afterok:$GATES_DATA_JID \
        --export=ALL,CONFIG='$cfg' cluster/slurm/train_single.sbatch")
    echo "  gate run $cfg: job $jid"
done < <("${SSH[@]}" "cat $REPO/outputs/manifests/gates.tsv")

echo "== 5/5 queue"
"${SSH[@]}" "squeue -u $SUNET_ID -o '%.10i %.14j %.9P %.2t %.10M %.20E'"
echo
echo "Staged. Next human checkpoints:"
echo "  - GPU smoke log: $REPO/slurm-$SMOKE_JID.out"
echo "  - when gate runs finish: scripts/run_evals.py per run, review gates A-C,"
echo "    then preregistration freeze and sweep/calib1b submission (RUNBOOK.md)"
