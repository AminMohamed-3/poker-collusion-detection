#!/usr/bin/env bash
# STEP267 arm ladder, in registration order.  NOTHING HERE HAS EVER RUN ON A GPU.
#
#   Z      zero-shot: Qwen3-Reranker-0.6B (agnostic and max3) and bge-reranker-v2-m3
#   FT_A   Qwen3-Reranker-0.6B + LoRA r16 on the yes/no logit difference, BCE, 2 epochs, 3 seeds
#   FT_B   bge-reranker-v2-m3 fine-tuned (BCE on the single logit), 2 epochs, 3 seeds
#   ENS    mean of the FT_A seed mean and the FT_B seed mean
#   Z_4B / FT_C  optional, only if GPU time allows
#
# Every arm scores BOTH held-out files: scope57 (the 57-query fusion scope - the only file the
# gates are read on) and subset (the STEP264 detection eval).  Each arm writes to runs/<arm>/.
# The tar at the end packs SCORES, SUMMARIES and LOGS only - NEVER weights or adapters.
#
# Usage:  bash run_ladder.sh              run everything
#         DRY=1 bash run_ladder.sh        print the commands with --dry-run appended
#         ARMS="FT_A ENS" bash run_ladder.sh   run a subset
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG="${PKG:-$(cd "$HERE/.." && pwd)}"          # the package root: data/, queries.json, gpu/
DATA="${DATA:-$PKG/data}"
QUERIES="${QUERIES:-$PKG/queries.json}"
INSTRUCTION="${INSTRUCTION:-$PKG/instruction.txt}"
RUNS="${RUNS:-$PKG/runs}"
PY="${PY:-python}"
SCOPE="${SCOPE:-$DATA/heldout_scope57_rr.jsonl}"
SUBSET="${SUBSET:-$DATA/heldout_subset_rr.jsonl}"
TRAIN="${TRAIN:-$DATA/train_rr.jsonl}"

M_QWEN="${M_QWEN:-Qwen/Qwen3-Reranker-0.6B}"
M_BGE="${M_BGE:-BAAI/bge-reranker-v2-m3}"
M_QWEN4B="${M_QWEN4B:-Qwen/Qwen3-Reranker-4B}"
BATCH="${BATCH:-32}"
TRAIN_BATCH="${TRAIN_BATCH:-16}"
MAXLEN="${MAXLEN:-2048}"
EPOCHS="${EPOCHS:-2}"
LR="${LR:-1e-4}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LOSS="${LOSS:-bce}"
NEG_PER_POS="${NEG_PER_POS:-7}"
QMODE="${QMODE:-agnostic}"
QKEYS="${QKEYS:-agnostic,directed_transfer,soft_play,coordinated_isolation}"
ARMS="${ARMS:-Z_qwen06_agnostic Z_qwen06_max3 Z_bge FT_A ENS_A FT_B ENS_B ENS}"
DRYFLAG=""
if [ "${DRY:-0}" = "1" ]; then DRYFLAG="--dry-run"; fi

mkdir -p "$RUNS"
have() { case " $ARMS " in *" $1 "*) return 0;; *) return 1;; esac }

score () {   # score <arm> <model> <arch> <primary> [adapter]
  local arm="$1" model="$2" arch="$3" primary="$4" adapter="${5:-}"
  local ad=(); [ -n "$adapter" ] && ad=(--adapter "$adapter")
  mkdir -p "$RUNS/$arm"
  for split in scope57 subset; do
    local f="$SCOPE"; [ "$split" = subset ] && f="$SUBSET"
    echo "=== $arm / $split ==="
    $PY "$HERE/score_reranker.py" --model "$model" --arch "$arch" "${ad[@]}" \
      --data "$f" --queries "$QUERIES" --instruction "$INSTRUCTION" \
      --query-keys "$QKEYS" --primary "$primary" \
      --out "$RUNS/$arm/${split}_scores.jsonl" --batch "$BATCH" --max-len "$MAXLEN" \
      $DRYFLAG 2>&1 | tee -a "$RUNS/$arm/log.txt"
  done
}

finetune () {   # finetune <arm> <model> <arch>
  local arm="$1" model="$2" arch="$3"
  mkdir -p "$RUNS/$arm"
  echo "=== train $arm ==="
  $PY "$HERE/finetune_reranker.py" --model "$model" --arch "$arch" \
    --train "$TRAIN" --eval "$SCOPE" --queries "$QUERIES" --instruction "$INSTRUCTION" \
    --query-mode "$QMODE" --out "$RUNS/$arm" --epochs "$EPOCHS" --lr "$LR" \
    --lora-r "$LORA_R" --lora-alpha "$LORA_ALPHA" --batch "$TRAIN_BATCH" --grad-accum 1 \
    --max-len "$MAXLEN" --seed "${SEED:-267}" --loss "$LOSS" --neg-per-pos "$NEG_PER_POS" \
    $DRYFLAG 2>&1 | tee -a "$RUNS/$arm/log.txt"
}

# ---------------------------------------------------------------- Z: zero-shot
have Z_qwen06_agnostic && score Z_qwen06_agnostic "$M_QWEN" qwen3 agnostic
have Z_qwen06_max3     && score Z_qwen06_max3     "$M_QWEN" qwen3 max3
have Z_bge             && score Z_bge             "$M_BGE"  bge   agnostic

# ---------------------------------------------------------------- FT_A: Qwen3-0.6B + LoRA
if have FT_A; then
  for s in 267 268 269; do
    SEED=$s finetune "FT_A_s$s" "$M_QWEN" qwen3
    score "FT_A_s$s" "$M_QWEN" qwen3 agnostic "$RUNS/FT_A_s$s/final"
  done
fi
if have ENS_A; then
  for split in scope57 subset; do
    $PY "$HERE/ensemble_scores.py" $DRYFLAG \
      --inputs "$RUNS"/FT_A_s26{7,8,9}/"${split}_scores.jsonl" \
      --out "$RUNS/ENS_A/${split}_scores.jsonl"
  done
fi

# ---------------------------------------------------------------- FT_B: bge-reranker-v2-m3
if have FT_B; then
  for s in 267 268 269; do
    SEED=$s LORA_R=0 finetune "FT_B_s$s" "$M_BGE" bge
    score "FT_B_s$s" "$RUNS/FT_B_s$s/final" bge agnostic
  done
fi
if have ENS_B; then
  for split in scope57 subset; do
    $PY "$HERE/ensemble_scores.py" $DRYFLAG \
      --inputs "$RUNS"/FT_B_s26{7,8,9}/"${split}_scores.jsonl" \
      --out "$RUNS/ENS_B/${split}_scores.jsonl"
  done
fi

# ---------------------------------------------------------------- ENS: the two architectures
if have ENS; then
  for split in scope57 subset; do
    $PY "$HERE/ensemble_scores.py" $DRYFLAG \
      --inputs "$RUNS/ENS_A/${split}_scores.jsonl" "$RUNS/ENS_B/${split}_scores.jsonl" \
      --out "$RUNS/ENS/${split}_scores.jsonl"
  done
fi

# ---------------------------------------------------------------- optional 4B arms
have Z_qwen4b && score Z_qwen4b "$M_QWEN4B" qwen3 agnostic
if have FT_C; then
  SEED=267 TRAIN_BATCH=4 finetune FT_C_s267 "$M_QWEN4B" qwen3
  score FT_C_s267 "$M_QWEN4B" qwen3 agnostic "$RUNS/FT_C_s267/final"
fi

# ---------------------------------------------------------------- pack: NO WEIGHTS
TAR="$PKG/step267_results_$(date -u +%Y%m%dT%H%M%SZ).tar.gz"
PACK=$(cd "$PKG" && find runs -type f \
      \( -name '*_scores.jsonl' -o -name '*_summary.json' -o -name 'train_log.jsonl' \
         -o -name 'log.txt' -o -name 'args.json' \) 2>/dev/null | sort)
if [ -z "$PACK" ]; then
  echo "nothing to pack (no score/summary/log files under $RUNS)"
  exit 0
fi
tar -czf "$TAR" -C "$PKG" $PACK
echo "packed $TAR"
echo "COPY BACK ONLY THAT TAR.  Never copy runs/*/final (adapters / weights)."
