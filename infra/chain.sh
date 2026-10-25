#!/usr/bin/env bash
# Run several configs back to back, unattended:
# for each config: launch (zones b, c, f), watch, tear down, check the run, then score it with Gemini in the
# background while the next VM runs. Stops on the first failed run or once the running total passes $100.
# Usage: infra/chain.sh <hours per VM> <config> [<config> ...]
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
GCLOUD="${GCLOUD:-gcloud}"
command -v "$GCLOUD" >/dev/null || GCLOUD="$HOME/google-cloud-sdk/bin/gcloud"
BUCKET=inspiration-seeds-speed
HOURS=$1; shift
total() { grep -o 'Running total: \$[0-9.]*' logs/cost.md | grep -o '[0-9.]*$'; }

score() {  # Gemini, twice, then log its cost; runs on the Jetson, no GPU
  local c=$1
  for r in 1 2; do
    env -u PYTHONPATH .venv/bin/python eval/eval_gemini.py --config "$c" --run "$r" --store "gs://$BUCKET" --workers 8 \
      2>&1 | grep -v -i -E "warning|api.key" | tail -1
  done
  env -u PYTHONPATH .venv/bin/python scripts/log_gemini_cost.py --store "gs://$BUCKET"
}

for c in "$@"; do
  if awk "BEGIN{exit !($(total) >= 100)}"; then echo "Running total \$$(total) >= \$100: stopping (budget check)."; exit 1; fi
  name="" zone=""
  for z in us-central1-b us-central1-c us-central1-f; do
    out=$(echo y | infra/launch.sh "$c" "$HOURS" "$z" 2>&1)
    if grep -q "^Launched" <<<"$out"; then name=$(grep -o 'seeds-[^ .]*' <<<"$out" | tail -1); zone=$z; break; fi
    echo "config $c: $z unavailable ($(grep -o 'STOCKOUT' <<<"$out" | head -1 || echo error))"
  done
  [[ -n $name ]] || { echo "config $c: no zone had capacity; stopping."; exit 1; }
  echo "$(date -u +%T) config $c: $name in $zone"
  infra/watch.sh "$name" "$zone" | tail -3
  rc=$("$GCLOUD" storage cat "gs://$BUCKET/logs/$c/vm_times_$name.json" 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin)['exit_code'])" 2>/dev/null || echo missing)
  n=$("$GCLOUD" storage ls "gs://$BUCKET/outputs/$c/" 2>/dev/null | wc -l)
  echo "$(date -u +%T) config $c: exit $rc, $n images, total \$$(total)"
  git add logs && git commit -qm "Config $c run: exit $rc, $n images, running total \$$(total)" >/dev/null 2>&1
  [[ $rc == 0 && $n -ge 396 ]] || { echo "config $c failed; see gs://$BUCKET/logs/$c/run_$name.log. Stopping."; exit 1; }
  score "$c" &
done
wait
git add logs && git commit -qm "Chain done; Gemini scoring logged, running total \$$(total)" >/dev/null 2>&1
echo "chain done; running total \$$(total)"
