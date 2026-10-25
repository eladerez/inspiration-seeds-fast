#!/usr/bin/env bash
# After every job: confirm no VM is left and update logs/cost.md.
# A VM that powered itself off is stopped, not deleted (the termination action only fires on preemption or at
# max-run-duration), so stopped VMs with label app=seeds are deleted here. A running VM is reported, never touched.
# Usage: infra/check_teardown.sh      exit 0 = no instances remain, 1 = something is still there
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROJECT=inspiration-seed
BUCKET=inspiration-seeds-speed
GCLOUD="${GCLOUD:-gcloud}"
command -v "$GCLOUD" >/dev/null || GCLOUD="$HOME/google-cloud-sdk/bin/gcloud"
g() { "$GCLOUD" --project="$PROJECT" "$@"; }

while IFS=$'\t' read -r name zone status app; do
  [[ -z $name ]] && continue
  if [[ $app == seeds && ( $status == TERMINATED || $status == STOPPED ) ]]; then
    echo "Deleting stopped VM $name ($zone)."
    # Billed time runs from create to delete (the disk bills until delete), so record both for the cost log.
    created=$(g compute instances describe "$name" --zone="$zone" --format="value(creationTimestamp)")
    printf '%s\t%s\t%s\n' "$name" "$created" "$(date -u +%FT%TZ)" >> "$ROOT/logs/vm_lifetimes.tsv"
    g compute instances delete "$name" --zone="$zone" --quiet
  else
    echo "Still present: $name ($zone) status $status app=${app:-none}; not touching it."
  fi
done < <(g compute instances list --format="value(name,zone.basename(),status,labels.app)")

REMAINING=$(g compute instances list --format="value(name,status)")

# Close every launch in logs/launches.tsv that has a vm_times file in the bucket and no cost row yet.
if [[ -f $ROOT/logs/launches.tsv ]]; then
  while IFS=$'\t' read -r name machine prov rate config start; do
    grep -q "| $name |" "$ROOT/logs/cost.md" && continue
    [[ $config == preflight* ]] && dir=_preflight || dir=$config
    times=$(g storage cat "gs://$BUCKET/logs/$dir/vm_times_$name.json" 2>/dev/null) || { echo "No end time yet for $name."; continue; }
    python3 - "$ROOT/logs/cost.md" "$name" "$machine" "$prov" "$rate" "$times" "$start" "$ROOT/logs/vm_lifetimes.tsv" <<'EOF'
import json, re, sys
from datetime import datetime, timezone
path, name, machine, prov, rate, times, launched, lifetimes = sys.argv[1:]
t = json.loads(times); f = "%Y-%m-%dT%H:%M:%SZ"
t["start"] = min(t["start"], launched)  # billing starts at create, before the startup script runs
try:  # create/delete times recorded at teardown win; vm_times only covers the startup script
    for line in open(lifetimes):
        n, created, deleted = line.rstrip("\n").split("\t")
        if n == name:
            t["start"] = datetime.fromisoformat(created).astimezone(timezone.utc).strftime(f); t["end"] = deleted
except FileNotFoundError:
    pass
hours = (datetime.strptime(t["end"], f) - datetime.strptime(t["start"], f)).total_seconds() / 3600
cost = hours * float(rate)
text = open(path).read()
total = float(re.search(r"Running total: \$([\d.]+)", text).group(1)) + cost
row = f"| {name} | {machine} | {prov} | {t['start']} | {t['end']} | {hours:.2f} | {float(rate):.2f} | {cost:.2f} |\n"
head, _ = text.split("\nRunning total:")
text = head.rstrip("\n") + "\n" + row + f"\nRunning total: ${total:.2f}\n"
open(path, "w").write(text)
print(f"Logged {name}: {hours:.2f} GPU-hours, ${cost:.2f}; running total ${total:.2f}")
EOF
  done < "$ROOT/logs/launches.tsv"
fi

if [[ -z $REMAINING ]]; then
  echo "VMs running: 0"
  grep "Running total" "$ROOT/logs/cost.md"
else
  echo "Instances still present:"; echo "$REMAINING"; exit 1
fi
