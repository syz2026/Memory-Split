#!/usr/bin/env bash
# Submit the preregistered battery (through calibration) from any FarmShare
# account. Run ON a login node, from the repo root on scratch:
#
#   bash cluster/run_battery.sh                       # everything (single account)
#   bash cluster/run_battery.sh --role a              # corpus host: builds + seed-0 sweep + calib1b
#   bash cluster/run_battery.sh --role b --data-root /scratch/users/<accountA>/memorysplit_data
#                                                     # reader: seed-1 sweep only, no builds
#
# Role A/B split per HANDOFF-AGENT.md section 9: seed pairs are atomic per
# account; account B trains against account A's corpora (read-only) and
# must not submit until A's builds are complete.
# The 1B confirmation is NOT submitted by this script in any role.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=./config.env
source "$SCRIPT_DIR/config.env"
require_sunet

ROLE=all
DATA_OVERRIDE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --role) ROLE=$2; shift 2 ;;
        --data-root) DATA_OVERRIDE=$2; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done
case "$ROLE" in a|b|all) ;; *) echo "role must be a, b, or all" >&2; exit 1 ;; esac

VENV=$(expand_path "$FS_VENV")
DATA=${DATA_OVERRIDE:-$(expand_path "$FS_DATA")}
EXC="--exclude=wheat-01"

case "$ROLE" in
    a)   SWEEP_FILTER="_s0"; DO_BUILD=1; DO_CALIB=1 ;;
    b)   SWEEP_FILTER="_s1"; DO_BUILD=0; DO_CALIB=0 ;;
    all) SWEEP_FILTER="";    DO_BUILD=1; DO_CALIB=1 ;;
esac

if [ "$ROLE" = "b" ]; then
    if [ -z "$DATA_OVERRIDE" ]; then
        echo "ERROR: role b requires --data-root (account A's memorysplit_data)" >&2
        exit 1
    fi
    for load in n50k n200k n800k; do
        if [ ! -f "$DATA/$load/report.json" ]; then
            echo "ERROR: $DATA/$load/report.json missing — account A's corpora" \
                 "are not ready (or not readable). Wait for A, then retry." >&2
            exit 1
        fi
        head -c 64 "$DATA/$load/dense/train.bin" > /dev/null \
            || { echo "ERROR: cannot read $DATA/$load — permissions?" >&2; exit 1; }
    done
    echo "role b: account A's corpora verified readable at $DATA"
fi

declare -A DEP=()
if [ "$DO_BUILD" = "1" ]; then
    mkdir -p "$DATA"
    echo "== corpus builds (frozen recipe)"
    DEP[n50k]=$(sbatch --parsable --time=04:00:00 $EXC --export=ALL,BUILD_ARGS="--stage full --loads n50k" cluster/slurm/data_prep.sbatch)
    DEP[n200k]=$(sbatch --parsable --time=04:00:00 $EXC --export=ALL,BUILD_ARGS="--stage full --loads n200k" cluster/slurm/data_prep.sbatch)
    DEP[n800k]=$(sbatch --parsable --time=04:00:00 $EXC --export=ALL,BUILD_ARGS="--stage full --loads n800k" cluster/slurm/data_prep.sbatch)
    DEP[n800k_1b]=$(sbatch --parsable --time=12:00:00 --mem=60G --cpus-per-task=16 $EXC --export=ALL,BUILD_ARGS="--stage full1b --loads n800k" cluster/slurm/data_prep.sbatch)
    DEP[n4m_1b]=$(sbatch --parsable --time=12:00:00 --mem=60G --cpus-per-task=16 $EXC --export=ALL,BUILD_ARGS="--stage full1b --loads n4m" cluster/slurm/data_prep.sbatch)
    for k in "${!DEP[@]}"; do echo "  data $k: job ${DEP[$k]}"; done
fi

dep_flag() {
    # dep_flag <load-key> -> "--dependency=afterok:<jid>" or nothing (role b)
    local key=$1
    if [ -n "${DEP[$key]:-}" ]; then
        printf -- "--dependency=afterok:%s" "${DEP[$key]}"
    fi
}

echo "== sweep runs (filter: '${SWEEP_FILTER:-all}')"
PYTHONPATH="$PWD" "$VENV/bin/python" scripts/make_manifest.py --stage sweep --data-root "$DATA"
while IFS= read -r cfg; do
    [ -z "$cfg" ] && continue
    if [ -n "$SWEEP_FILTER" ] && ! printf '%s' "$cfg" | grep -q "$SWEEP_FILTER"; then
        continue
    fi
    case "$cfg" in
        *n50k*) key=n50k ;; *n200k*) key=n200k ;; *n800k*) key=n800k ;;
        *) echo "unmatched config $cfg" >&2; exit 1 ;;
    esac
    jid=$(sbatch --parsable $EXC $(dep_flag "$key") \
        --export=ALL,CONFIG="$cfg" cluster/slurm/train_single.sbatch)
    echo "  sweep $cfg: job $jid"
done < outputs/manifests/sweep.tsv

if [ "$DO_CALIB" = "1" ]; then
    echo "== calib1b (2 short dense-1B runs)"
    PYTHONPATH="$PWD" "$VENV/bin/python" scripts/make_manifest.py --stage calib1b --data-root "$DATA"
    while IFS= read -r cfg; do
        [ -z "$cfg" ] && continue
        case "$cfg" in
            *n800k*) key=n800k_1b ;; *n4m*) key=n4m_1b ;;
            *) echo "unmatched config $cfg" >&2; exit 1 ;;
        esac
        jid=$(sbatch --parsable $EXC $(dep_flag "$key") \
            --export=ALL,CONFIG="$cfg" cluster/slurm/train_single.sbatch)
        echo "  calib $cfg: job $jid"
    done < outputs/manifests/calib1b.tsv
fi

echo
squeue -u "$SUNET_ID" -o "%.9i %.13j %.2t %.20E"
echo "run_battery (role $ROLE): submitted. Evals: run cluster/run_evals_pending.sh"
echo "whenever you check in. Scope ends at calibration; do not submit the 1B"
echo "confirmation from this script."
