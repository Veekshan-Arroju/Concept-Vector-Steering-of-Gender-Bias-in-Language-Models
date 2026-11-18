#!/usr/bin/env bash
cd /DATA/deeksha/Veekshan/mtp_v5 && source env.sh
export CUDA_DEVICE_ORDER=PCI_BUS_ID
note () { echo "[seeds] $(date +%F\ %T) $*" >> logs/seeds_chain.log; }
run_seed () {
  local seed=$1 log=logs/calibrate_random_stems_seed${1}.log prefix t
  if [ -e "$log" ]; then note "skip: $log exists"; return 0; fi
  for t in $(seq 1 240); do
    if prefix=$(bash scripts/pick_gpus.sh); then
      note "starting seed $seed on: $prefix"
      env $prefix python -u calibrate_steering.py --vector pooled__misogyny --experiment claims_raw --positions generated --signs both --prompts-file evaluation_prompts/en/stem_benchmark_prompts.txt --n-prompts 40 --max-coef 0.8 --step 0.2 --max-new-tokens 80 --skip-learned --random-seeds $seed --tag stems_random${seed} > "$log" 2>&1
      note "finished seed $seed (exit $?)"; return 0
    fi
    note "waiting for a GPU for seed $seed (try $t)"; sleep 60
  done
}
note "chain started"
run_seed 1 &
sleep 180
run_seed 2 &
wait
note "chain done"
