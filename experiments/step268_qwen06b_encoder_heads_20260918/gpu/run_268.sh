#!/usr/bin/env bash
# STEP268 -- the whole GPU-box run, in order.
#
# NOTHING IN THIS DIRECTORY HAS EVER RUN ON A GPU.  This script was written on a laptop with no
# torch and no transformers; only the --dry-run paths and py_compile have been executed.  Read
# RUNBOOK_268.txt (and its "VERIFY ON THE GPU BOX" list) before the first real command.
#
# Expects, in the working directory:
#   universe_docs.jsonl.gz  universe_meta.jsonl  instruction.txt
#   heldout_scope57_uids.txt  heldout_subset_uids.txt
#   encode_hands.py  fit_heads.py  finetune_head.py
# keys_universe.parquet MUST NOT BE HERE.

set -euo pipefail

PY=${PY:-python}
BATCH=${BATCH:-64}
MAXLEN=${MAXLEN:-2048}

for f in universe_docs.jsonl.gz universe_meta.jsonl instruction.txt \
         heldout_scope57_uids.txt heldout_subset_uids.txt; do
  [ -f "$f" ] || { echo "MISSING INPUT: $f" >&2; exit 1; }
done
if [ -e keys_universe.parquet ]; then
  echo "REFUSING TO RUN: keys_universe.parquet is on this box; it maps uid -> pair_id/hand_id" >&2
  exit 1
fi

echo "=== 0. dry runs (no GPU, no torch) ============================================="
$PY encode_hands.py --dry-run --docs universe_docs.jsonl.gz --instruction instruction.txt \
    --out dry_emb.npy --uids dry_uids.txt --limit 500
$PY fit_heads.py --dry-run --emb dry_emb.npy --uids dry_uids.txt --out heads_dry/ || true
rm -f dry_emb.npy dry_uids.txt dry_emb_summary.json
echo

echo "=== 1. PRIMARY ARM: Qwen3-Embedding-0.6B, last-token pooling ==================="
$PY encode_hands.py --model Qwen/Qwen3-Embedding-0.6B \
    --docs universe_docs.jsonl.gz --instruction instruction.txt \
    --out emb_universe.npy --uids emb_uids.txt \
    --batch "$BATCH" --max-len "$MAXLEN" --pooling last --bf16 2>&1 | tee log_encode_emb.txt

echo "=== 2. heads =================================================================="
$PY fit_heads.py --emb emb_universe.npy --uids emb_uids.txt --meta universe_meta.jsonl \
    --scope57 heldout_scope57_uids.txt --subset heldout_subset_uids.txt \
    --out heads/ --seeds 268,269,270 --tag qwen3emb06b_last 2>&1 | tee log_fit_heads.txt

echo "=== 3. SECOND ARM (only if time): Qwen3-0.6B base, mean pooling ==============="
echo "    $PY encode_hands.py --model Qwen/Qwen3-0.6B --docs universe_docs.jsonl.gz \\"
echo "        --instruction instruction.txt --out emb_base.npy --uids emb_base_uids.txt \\"
echo "        --batch $BATCH --max-len $MAXLEN --pooling mean --bf16"
echo "    $PY fit_heads.py --emb emb_base.npy --uids emb_base_uids.txt --meta universe_meta.jsonl \\"
echo "        --scope57 heldout_scope57_uids.txt --subset heldout_subset_uids.txt \\"
echo "        --out heads_base/ --seeds 268,269,270 --tag qwen3base06b_mean"
echo "    (uncomment when arm 1 is done and the numbers are back on the laptop)"

echo "=== 4. GATED LAST ARM: LoRA fine-tune ========================================="
echo "    SKIPPED unless STEP267's FT_A standalone AP is >= .55; then:"
echo "    $PY finetune_head.py --step267-ft-ap <THE NUMBER> --out ft/"

echo
echo "=== COPY BACK (and NOTHING ELSE) =============================================="
echo "  heads/        heads_base/   ft/ (if it ran)"
echo "  emb_universe_summary.json   log_encode_emb.txt   log_fit_heads.txt"
echo "NEVER copy back: emb_universe.npy / emb_base.npy (80k x 1024) or any model weights."
