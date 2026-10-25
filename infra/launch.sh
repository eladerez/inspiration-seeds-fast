#!/usr/bin/env bash
# Launch one self-destructing spot GPU VM for one job. Prints the full command and a cost estimate, then asks y/N.
# Usage: infra/launch.sh <config> <hours> [zone] [-- runner args, e.g. --smoke or --write-text-cache]
#   <config>  0..7 or preflight (g4-standard-48)
#   <hours>   whole hours, 1..12, the job's expected time + 30 min margin
#   [zone]    default us-central1-b; us-central1-c and us-central1-f are the fallbacks
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROJECT=inspiration-seed
BUCKET=inspiration-seeds-speed
SA=seeds-runner@inspiration-seed.iam.gserviceaccount.com
IMAGE_FAMILY=common-cu129-ubuntu-2404-nvidia-580   # driver 580; the cu128 families ship 570 and can't drive RTX PRO 6000
IMAGE_PROJECT=deeplearning-platform-release
GCLOUD="${GCLOUD:-gcloud}"
command -v "$GCLOUD" >/dev/null || GCLOUD="$HOME/google-cloud-sdk/bin/gcloud"

die() { echo "launch.sh: $*" >&2; exit 1; }
[[ $# -ge 2 ]] || die "usage: infra/launch.sh <config> <hours> [zone]"
CONFIG=$1 HOURS=$2; shift 2
ZONE=us-central1-b
if [[ $# -gt 0 && $1 != -- ]]; then ZONE=$1; shift; fi
[[ ${1:-} == -- ]] && shift
RUN_ARGS="$*"
[[ $RUN_ARGS =~ ^[a-z\ -]*$ ]] || die "runner args may only contain flags like --smoke, got '$RUN_ARGS'"
[[ $HOURS =~ ^[0-9]+$ ]] && (( HOURS >= 1 && HOURS <= 12 )) || die "hours must be a whole number from 1 to 12, got '$HOURS'"
[[ $ZONE == us-central1-* ]] || die "zone must be in us-central1, got '$ZONE'"

# G4 has no Persistent Disk support, so its boot disk is Hyperdisk Balanced.
case $CONFIG in
  [0-7])        MACHINE=g4-standard-48 DISK=hyperdisk-balanced RATE_VAR=RATE_G4 JOB=run ;;
  preflight)    MACHINE=g4-standard-48 DISK=hyperdisk-balanced RATE_VAR=RATE_G4 JOB=preflight ;;
  *) die "unknown config '$CONFIG'" ;;
esac

# shellcheck source=/dev/null
source "$ROOT/infra/rates.env"
RATE=${!RATE_VAR:-}
[[ -n $RATE ]] || die "$RATE_VAR is empty in infra/rates.env; set the spot rate from the console's price estimate"

if [[ $JOB == run ]]; then
  [[ -f $ROOT/runner/run_config.py && -f $ROOT/runner/configs/$CONFIG.yaml ]] || die "runner/run_config.py or runner/configs/$CONFIG.yaml is missing"
fi

# One VM at a time (rule 4): any instance in the project, running or stopped, blocks a launch.
EXISTING=$("$GCLOUD" compute instances list --project="$PROJECT" --format="value(name,zone.basename(),status)")
[[ -z $EXISTING ]] || die "instances already exist; run infra/check_teardown.sh first:
$EXISTING"

NAME="seeds-${CONFIG//_/-}-$(date -u +%Y%m%d-%H%M)"
CMD=("$GCLOUD" compute instances create "$NAME"
  --project="$PROJECT" --zone="$ZONE" --machine-type="$MACHINE"
  --provisioning-model=SPOT --instance-termination-action=DELETE --max-run-duration="${HOURS}h"
  --maintenance-policy=TERMINATE --no-restart-on-failure
  --service-account="$SA" --scopes=cloud-platform
  --image-family="$IMAGE_FAMILY" --image-project="$IMAGE_PROJECT"
  --boot-disk-size=200GB --boot-disk-type="$DISK" --boot-disk-auto-delete
  --labels="app=seeds,config=${CONFIG//_/-},launch=${NAME}"
  --metadata="^|^job=$JOB|config=$CONFIG|bucket=$BUCKET|launch=$NAME|args=$RUN_ARGS"
  --metadata-from-file=startup-script="$ROOT/infra/startup.sh")

EST=$(python3 -c "print(f'{$HOURS * $RATE:.2f}')")
echo "Command:"
printf '  %q' "${CMD[@]}"; echo
echo
echo "Job: $JOB config $CONFIG ${RUN_ARGS:+($RUN_ARGS)}"
echo "Machine $MACHINE (1 GPU), spot, zone $ZONE, at most $HOURS h × \$$RATE/h = \$$EST (upper bound; the VM stops when the job ends)."
read -r -p "Create this VM? [y/N] " ok
[[ $ok == y || $ok == Y ]] || { echo "Not created."; exit 0; }

if [[ $JOB == run ]]; then  # the VM gets the harness code from the bucket, never from a git remote
  tar -czf - -C "$ROOT" --exclude=.git --exclude=./logs --exclude=./data --exclude=./third_party --exclude=.venv \
    --exclude=__pycache__ --exclude=.pytest_cache . \
    | "$GCLOUD" storage cp - "gs://$BUCKET/code/$NAME.tgz"
fi

"${CMD[@]}"
mkdir -p "$ROOT/logs"
printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$NAME" "$MACHINE" spot "$RATE" "$CONFIG" "$(date -u +%FT%TZ)" >> "$ROOT/logs/launches.tsv"
echo "Launched $NAME. When the job is done, run infra/check_teardown.sh."
