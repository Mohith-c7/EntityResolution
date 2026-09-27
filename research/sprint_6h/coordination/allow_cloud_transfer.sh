#!/usr/bin/env bash
# Temporary, source-restricted SSH allowance for the two task-owned Azure VMs.
set -euo pipefail
az network nsg rule create \
  --resource-group er-expcpu-0927-e70007-rg \
  --nsg-name er-expcpu-0927-e70007-nsg \
  --name sprint-transfer-from-er-cpu-01 \
  --priority 1100 --direction Inbound --access Allow --protocol Tcp \
  --source-address-prefixes 20.219.186.115/32 --source-port-ranges '*' \
  --destination-address-prefixes '*' --destination-port-ranges 22 \
  --description 'Temporary transfer from the existing task VM; remove after verified copy' \
  --output none
