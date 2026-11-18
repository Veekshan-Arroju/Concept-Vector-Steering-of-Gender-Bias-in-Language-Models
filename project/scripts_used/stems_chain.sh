#!/usr/bin/env bash
cd /DATA/deeksha/Veekshan/mtp_v5 && source env.sh
export CUDA_DEVICE_ORDER=PCI_BUS_ID
note () { echo "[stems] $(date +%F\ %T) $*" >> logs/stems_chain.log; }
run_job () {
  local name=$1 exp=$2 tag=$3; shift 3
  local log=logs/calibrate_${name}.log prefix t
  if [ -e "$log" ]; then note "skip: $log exists"; return 0; fi
  for t in $(seq 1 240); do
    if prefix=$(bash scripts/pick_gpus.sh); then
      note "starting $name on: $prefix"
      env $prefix python -u calibrate_steering.py --vector pooled__misogyny --experiment $exp --positions generated --signs both --prompts-file evaluation_prompts/en/stem_benchmark_prompts.txt --max-coef 0.8 --step 0.2 --max-new-tokens 80 --tag $tag "$@" > "$log" 2>&1
      note "finished $name (exit $?)"; return 0
    fi
    note "waiting for a GPU for $name (try $t)"; sleep 60
  done
}
note "chain started"
( run_job claims_stems claims_raw stems --n-prompts 80; run_job random_stems claims_raw stems_random --n-prompts 40 --skip-learned --random-seeds 0 ) &
sleep 180
( run_job old_stems old_v3 stems --n-prompts 80; run_job stance_stems stance_raw stems --n-prompts 80 ) &
wait
note "chain done"
