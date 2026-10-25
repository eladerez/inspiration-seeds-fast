#!/usr/bin/env bash
# Wait for a launched VM to stop (or disappear), printing its latest progress line every few minutes, then tear down.
# Usage: infra/watch.sh <instance> <zone>
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GCLOUD="${GCLOUD:-gcloud}"
command -v "$GCLOUD" >/dev/null || GCLOUD="$HOME/google-cloud-sdk/bin/gcloud"
NAME=$1 ZONE=$2
n=0
while true; do
  s=$("$GCLOUD" compute instances describe "$NAME" --zone="$ZONE" --format="value(status)" 2>/dev/null || echo GONE)
  case $s in RUNNING|PROVISIONING|STAGING|STOPPING) ;; *) echo "$(date -u +%T) $NAME status $s"; break ;; esac
  if (( n % 12 == 0 )); then  # every ~2 min: last runner/startup line from the serial console (no secrets are logged)
    line=$("$GCLOUD" compute instances get-serial-port-output "$NAME" --zone="$ZONE" 2>/dev/null \
      | grep -a "startup-script" | grep -a -v -i token | tail -1 | cut -c1-220)
    echo "$(date -u +%T) $line"
  fi
  n=$((n + 1)); sleep 10
done
"$ROOT/infra/check_teardown.sh"
