#!/usr/bin/env bash
cd /DATA/deeksha/Veekshan/mtp_v5 && source env.sh
export CUDA_DEVICE_ORDER=PCI_BUS_ID
note () { echo "[opinion] $(date +%F\ %T) $*" >> logs/opinion_chain.log; }
run_job () {
  local exp=$1 log=logs/calibrate_${1}_opinion.log prefix t
  if [ -e "$log" ]; then note "skip: $log exists"; return 0; fi
  for t in $(seq 1 240); do
    if prefix=$(bash scripts/pick_gpus.sh); then
      note "starting $exp on: $prefix"
      env $prefix python -u calibrate_steering.py --vector pooled__misogyny --experiment $exp --positions generated --signs positive --prompts-file evaluation_prompts/en/opinion_probe_prompts.txt --n-prompts 12 --max-coef 1.2 --step 0.2 --tag opinion > "$log" 2>&1
      note "finished $exp (exit $?)"; return 0
    fi
    note "waiting for a GPU for $exp (try $t)"; sleep 60
  done
}
note "chain started"
run_job old_v3 &
sleep 180
run_job stance_raw &
sleep 180
run_job claims_raw &
wait
note "chain done"
