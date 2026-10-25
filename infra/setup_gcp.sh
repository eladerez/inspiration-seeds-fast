#!/usr/bin/env bash
# One-time setup, run by the project owner: secrets, the runner service account and its IAM bindings.
#
# Before running: accept the FLUX.1 Kontext [dev] license on Hugging Face
# (https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) with the SAME account the HF token belongs to,
# or the VM's weight download will fail with 403.
#
# Usage: bash infra/setup_gcp.sh
set -euo pipefail

PROJECT=inspiration-seed
BUCKET=inspiration-seeds-speed
REGION=us-central1
SA_NAME=seeds-runner
SA="${SA_NAME}@${PROJECT}.iam.gserviceaccount.com"
GCLOUD="${GCLOUD:-gcloud}"
command -v "$GCLOUD" >/dev/null || GCLOUD="$HOME/google-cloud-sdk/bin/gcloud"
g() { "$GCLOUD" --project="$PROJECT" "$@"; }

# Reads a secret from the terminal (never echoed, never on disk) and stores it as a new secret or a new version.
put_secret() {
  local name=$1 prompt=$2 value
  read -rs -p "$prompt (input hidden): " value; echo
  if [[ -z "$value" ]]; then echo "Empty input for $name; nothing stored." >&2; exit 1; fi
  if g secrets describe "$name" >/dev/null 2>&1; then
    printf '%s' "$value" | g secrets versions add "$name" --data-file=- >/dev/null
    echo "Added a new version of secret $name."
  else
    printf '%s' "$value" | g secrets create "$name" --replication-policy=user-managed --locations="$REGION" --data-file=- >/dev/null
    echo "Created secret $name."
  fi
  unset value
}

g storage buckets describe "gs://$BUCKET" >/dev/null   # fails here if the bucket is missing

put_secret hf-token "Hugging Face token"
put_secret gemini-api-key "Gemini API key"

if g iam service-accounts describe "$SA" >/dev/null 2>&1; then
  echo "Service account $SA already exists."
else
  g iam service-accounts create "$SA_NAME" --display-name="Inspiration Seeds runner (GPU VMs)" >/dev/null
  echo "Created service account $SA."
  for _ in 1 2 3 4 5 6; do g iam service-accounts describe "$SA" >/dev/null 2>&1 && break; sleep 5; done  # new accounts take a moment to appear
fi

# Least privilege: objects in this bucket only, the HF token only (the VM never needs the Gemini key), log writing.
g storage buckets add-iam-policy-binding "gs://$BUCKET" --member="serviceAccount:$SA" --role=roles/storage.objectAdmin >/dev/null
g secrets add-iam-policy-binding hf-token --member="serviceAccount:$SA" --role=roles/secretmanager.secretAccessor >/dev/null
g projects add-iam-policy-binding "$PROJECT" --member="serviceAccount:$SA" --role=roles/logging.logWriter --condition=None >/dev/null

echo
echo "Service account: $SA"
echo "Bucket gs://$BUCKET roles for it:"
g storage buckets get-iam-policy "gs://$BUCKET" --format=json \
  | python3 -c "import json, sys; [print(b['role']) for b in json.load(sys.stdin)['bindings'] if 'serviceAccount:$SA' in b['members']]"
echo "Secret hf-token roles for it:"
g secrets get-iam-policy hf-token --flatten="bindings[].members" --filter="bindings.members:serviceAccount:$SA" --format="value(bindings.role)"
echo "Project roles for it:"
g projects get-iam-policy "$PROJECT" --flatten="bindings[].members" --filter="bindings.members:serviceAccount:$SA" --format="value(bindings.role)"
echo "Secrets present (names only):"
g secrets list --format="value(name)"
