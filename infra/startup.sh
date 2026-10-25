#!/bin/bash
# VM startup script: run one job, sync results and logs to the bucket, then power off on every exit path.
# Job and config come from instance metadata (set by infra/launch.sh). The HF token is read from Secret Manager
# into an environment variable only; it is never printed, written to disk or put in metadata.
set -uo pipefail

MD=http://metadata.google.internal/computeMetadata/v1/instance
md() { curl -sf -H 'Metadata-Flavor: Google' "$MD/$1"; }
JOB=$(md attributes/job) CONFIG=$(md attributes/config) BUCKET=$(md attributes/bucket) LAUNCH=$(md attributes/launch)
ARGS=$(md attributes/args || true)
WORK=/opt/seeds; mkdir -p "$WORK/out"
LOG="$WORK/run.log"
[[ $JOB == preflight ]] && DEST="gs://$BUCKET/logs/_preflight" || DEST="gs://$BUCKET/logs/$CONFIG"
START=$(date -u +%FT%TZ)

finish() {
  local rc=$?
  trap - EXIT
  echo "exit code $rc"
  printf '{"launch": "%s", "start": "%s", "end": "%s", "exit_code": %d}\n' "$LAUNCH" "$START" "$(date -u +%FT%TZ)" "$rc" > "$WORK/vm_times.json"
  gcloud storage cp "$WORK/vm_times.json" "$DEST/vm_times_$LAUNCH.json" || true
  gcloud storage cp "$LOG" "$DEST/run_$LAUNCH.log" || true
  [[ $JOB == run ]] && { gcloud storage cp "$LOG" "$DEST/run.log" || true; }
  sync
  shutdown -h now
}
trap finish EXIT
trap 'exit 143' TERM INT   # spot preemption sends a shutdown; make sure the EXIT trap still runs

exec > >(tee -a "$LOG") 2>&1
echo "launch $LAUNCH job $JOB config $CONFIG start $START"

for _ in $(seq 60); do nvidia-smi >/dev/null 2>&1 && break; sleep 5; done   # the driver can take a moment after boot
nvidia-smi || exit 1

HF_TOKEN=$(gcloud secrets versions access latest --secret=hf-token) || exit 1
export HF_TOKEN

preflight() {
  echo "hf-token length: ${#HF_TOKEN}"
  python3 - "$WORK/env.json" <<'EOF'
import json, subprocess, sys, urllib.request
def md(p):
    r = urllib.request.Request("http://metadata.google.internal/computeMetadata/v1/instance/" + p, headers={"Metadata-Flavor": "Google"})
    return urllib.request.urlopen(r).read().decode().rsplit("/", 1)[-1]
q = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                   capture_output=True, text=True, check=True).stdout.strip().split(", ")
cuda = next((l.split("CUDA Version:")[1].split()[0] for l in subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout.splitlines() if "CUDA Version" in l), None)
env = {"gpu": q[0], "driver": q[1], "gpu_memory": q[2], "cuda_driver_api": cuda,
       "machine_type": md("machine-type"), "zone": md("zone"), "image": md("image")}
json.dump(env, open(sys.argv[1], "w"), indent=1); print(json.dumps(env))
EOF
  gcloud storage cp "$WORK/env.json" "gs://$BUCKET/logs/_preflight/env.json"
}

run() {
  mkdir -p "$WORK/repo"
  gcloud storage cp "gs://$BUCKET/code/$LAUNCH.tgz" - | tar -xz -C "$WORK/repo"
  cd "$WORK/repo"
  mkdir -p data/benchmark
  gcloud storage cp -r "gs://$BUCKET/benchmark/*" data/benchmark/
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh
  local groups=(--group gen)
  grep -q '^backend: nunchaku' "runner/configs/$CONFIG.yaml" && groups+=(--group nunchaku)
  export HF_HOME=/opt/hf PYTHONUNBUFFERED=1
  # shellcheck disable=SC2086  # ARGS is a list of plain flags, checked by launch.sh
  uv run --python 3.11 "${groups[@]}" runner/run_config.py --config "runner/configs/$CONFIG.yaml" \
    --store "gs://$BUCKET" $ARGS || return 1
  if [[ $ARGS != *--smoke* ]]; then  # DreamSim runs at the end of each GPU job
    uv run --python 3.11 "${groups[@]}" --group eval eval/eval_dreamsim.py --config "$CONFIG" --store "gs://$BUCKET" \
      --cache-dir /opt/dreamsim
  fi
}

case $JOB in
  preflight) preflight ;;
  run) run ;;
  *) echo "unknown job '$JOB'"; exit 2 ;;
esac
