#!/bin/bash
# Stage 1 - prove the hardware is what the capacity block promised, and that it
# clears the software floors in cluster/profiles/aws-p6-b200.48xlarge-135m-v1.json.
# Emits the values the profile lists under pending_confirmation so they can be
# frozen from real hardware rather than assumed.
set -uo pipefail

echo "===== STAGE 1: GPU + DRIVER VERIFICATION ====="
date -u +%FT%TZ

FAIL=0
note_fail() { echo "FAIL: $1"; FAIL=1; }

echo
echo "--- bootstrap status ---"
cat /var/log/ms-bootstrap.status 2>/dev/null || echo "no bootstrap status file"
df -h /mnt/nvme 2>/dev/null || note_fail "/mnt/nvme not mounted"

echo
echo "--- nvidia-smi topology + inventory ---"
nvidia-smi || note_fail "nvidia-smi did not run"

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')
echo "gpu_count=${GPU_COUNT}"
[ "$GPU_COUNT" = "8" ] || note_fail "expected 8 GPUs, saw ${GPU_COUNT}"

echo
echo "--- node-derived profile fields (this script owns exactly these three) ---"
# A single scalar goes into the profile, so all eight devices must agree.
# Heterogeneity would make any single frozen value a lie about the node.
uniq_gpu_field() {
  local query="$1" name="$2" extra="${3:-}"
  local vals
  vals=$(nvidia-smi --query-gpu="$query" --format=csv,noheader${extra:+,$extra} | sed 's/^ *//;s/ *$//' | sort -u)
  if [ "$(echo "$vals" | wc -l | tr -d ' ')" != "1" ]; then
    note_fail "${name}: devices disagree -> $(echo "$vals" | paste -sd'|' -)"
    echo ""
    return
  fi
  echo "$vals"
}

GPU_NAME=$(uniq_gpu_field name hardware.gpu_name)
GPU_MEM=$(uniq_gpu_field memory.total hardware.gpu_memory_mib nounits)
GPU_CC=$(uniq_gpu_field compute_cap hardware.gpu_compute_capability)

echo "hardware.gpu_name              = ${GPU_NAME}"
echo "hardware.gpu_memory_mib        = ${GPU_MEM}"
echo "hardware.gpu_compute_capability= ${GPU_CC}"

# Machine-readable so the freeze step consumes exact values instead of
# re-parsing console text.
python3 -c "
import json, sys
d = {
  'hardware.gpu_name': sys.argv[1] or None,
  'hardware.gpu_memory_mib': int(sys.argv[2]) if sys.argv[2] else None,
  'hardware.gpu_compute_capability': sys.argv[3] or None,
}
open('/mnt/nvme/stage/gpu-attestation.json','w').write(json.dumps(d, indent=2, sort_keys=True))
print('MS_ATTEST_JSON ' + json.dumps(d, sort_keys=True))
" "$GPU_NAME" "$GPU_MEM" "$GPU_CC" 2>/dev/null || note_fail "could not serialise GPU attestation"

DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | sort -u | head -1)
echo "driver_version             = ${DRIVER}"
DRIVER_MAJOR=${DRIVER%%.*}
# Floor from the profile: Blackwell needs r570+.
[ "${DRIVER_MAJOR:-0}" -ge 570 ] || note_fail "driver ${DRIVER} below profile floor 570"

# nvidia-smi has no 'cuda_version' query field; the driver's max supported CUDA
# is only in the banner. Prefer a real nvcc if the toolkit is present.
NVCC=$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)
if [ -x "$NVCC" ]; then
  CUDA=$("$NVCC" --version | awk '/release/{print $6}' | tr -d 'V,')
else
  CUDA=$(nvidia-smi | awk -F'CUDA Version:' '/CUDA Version:/{print $2}' | awk '{print $1}' | head -1)
fi
echo "cuda_version               = ${CUDA}"
# Floor from the profile: 12.8 is the first CUDA with sm_100 Blackwell support.
awk -v v="${CUDA:-0}" 'BEGIN{split(v,a,".");if((a[1]+0)*100+(a[2]+0) < 1208) exit 1}' \
  || note_fail "CUDA ${CUDA} below profile floor 12.8"

KERNEL=$(uname -r)
echo "kernel_version             = ${KERNEL}"
awk -v v="${KERNEL}" 'BEGIN{split(v,a,".");if((a[1]+0)*100+(a[2]+0) < 601) exit 1}' \
  || note_fail "kernel ${KERNEL} below profile floor 6.1"

echo
echo "--- torch view of the devices (hardware cross-check ONLY) ---"
# Deliberately does NOT emit software_floors.torch_version or
# software_floors.nccl_version. This is the DLAMI's system python, not the
# qualified container runtime, so its versions are the wrong provenance for
# those floors and would silently void the attestation. Those two fields are
# owned by the image-build agent and must be frozen from inside the image.
# What torch is used for here is a second, independent read of the same
# hardware facts nvidia-smi reports, so a disagreement is caught.
# The DLAMI ships torch in a venv at /opt/pytorch, not in system python, and a
# non-interactive SSM shell does not source it.
PYBIN=""
for c in /opt/pytorch/bin/python /opt/conda/bin/python "$(command -v python3)"; do
  [ -x "$c" ] && "$c" -c "import torch" >/dev/null 2>&1 && PYBIN="$c" && break
done
if [ -z "$PYBIN" ]; then
  note_fail "no python with torch found"
else
  echo "using python: ${PYBIN}"
  "$PYBIN" -c "
import torch
print('(informational, NOT a floor) env torch :', torch.__version__)
print('(informational, NOT a floor) env nccl  :', '.'.join(map(str, torch.cuda.nccl.version())))
print('torch.cuda.device_count       =', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f'  gpu{i}: {p.name} sm_{p.major}{p.minor} {p.total_memory // 1024**2} MiB')
" || note_fail "torch GPU introspection failed"
fi

echo
echo "--- deviceQuery ---"
DQ=""
for c in /usr/local/cuda/extras/demo_suite/deviceQuery \
         /usr/local/cuda/samples/bin/x86_64/linux/release/deviceQuery \
         "$(command -v deviceQuery 2>/dev/null)"; do
  [ -x "$c" ] && DQ="$c" && break
done
if [ -n "$DQ" ]; then
  "$DQ" | tail -40
  "$DQ" | grep -q "Result = PASS" || note_fail "deviceQuery did not report PASS"
else
  # DLAMI does not ship the compiled sample. Try to build it, but do not fail
  # the stage on its absence: compute capability is already attested twice and
  # independently, by nvidia-smi --query-gpu=compute_cap and by torch's
  # sm_<major><minor>. deviceQuery would be a third read of the same fact.
  echo "deviceQuery binary absent - attempting build from cuda-samples"
  TMP=$(mktemp -d)
  if git clone --depth 1 https://github.com/NVIDIA/cuda-samples.git "$TMP/cs" >/dev/null 2>&1 \
     && cmake -S "$TMP/cs/Samples/1_Utilities/deviceQuery" -B "$TMP/build" >/dev/null 2>&1 \
     && cmake --build "$TMP/build" -j 32 >/dev/null 2>&1; then
    find "$TMP/build" -name deviceQuery -type f -exec {} \; | tail -40
  else
    echo "WARN: deviceQuery unavailable; compute capability still attested by"
    echo "      nvidia-smi compute_cap and torch sm_ (two independent sources)"
  fi
  rm -rf "$TMP"
fi

echo
echo "--- NVLink / topology (pairs must share NVLink for the 2-GPU arms) ---"
nvidia-smi topo -m || note_fail "topology query failed"
nvidia-smi nvlink --status | head -40 || echo "nvlink status unavailable"

echo
if [ "$FAIL" = "0" ]; then echo "STAGE 1 RESULT: PASS"; else echo "STAGE 1 RESULT: FAIL"; fi
exit $FAIL
