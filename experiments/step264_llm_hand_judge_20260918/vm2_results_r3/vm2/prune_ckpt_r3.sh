#!/bin/bash
# STEP264 r3 checkpoint pruner.  Same pattern as vm_bundle_20260918/prune_ckpt.sh, retargeted at
# runs/r3 and at --save-steps 5.  Start it BEFORE the real run:
#     nohup bash vm2/prune_ckpt_r3.sh > /workspace/prune.log 2>&1 &
#
# Keeps a permanent copy of every 20th checkpoint under keep/, and keeps only the last 2 live
# checkpoint-* directories.  The r1 box ran out of disk; this is why.
RUN_DIR="${1:-/workspace/step264/runs/r3}"
while true; do
  cd "$RUN_DIR" 2>/dev/null || { sleep 60; continue; }
  for d in $(ls -d checkpoint-* 2>/dev/null | sort -t- -k2 -n); do
    n=${d#checkpoint-}
    if [ $((n % 20)) -eq 0 ] && [ ! -d "keep/$d" ]; then
      mkdir -p "keep/$d"
      cp "$d/adapter_model.safetensors" "$d/adapter_config.json" "keep/$d/" 2>/dev/null
    fi
  done
  for d in $(ls -d checkpoint-* 2>/dev/null | sort -t- -k2 -n | head -n -2); do rm -rf "$d"; done
  sleep 120
done
