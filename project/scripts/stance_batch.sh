#!/usr/bin/env bash
# Runs the two remaining steering tests one after the other, unattended. Before EACH job it waits until the GPUs have room
# (scripts/pick_gpus.sh), so a neighbour's job cannot make it fail or squeeze anyone. Run from the project root, after `source env.sh`:
#     setsid nohup bash scripts/stance_batch.sh > /dev/null 2>&1 < /dev/null &
# Progress:  cat logs/stance_batch.log     and     grep -ac "\[stance\]" logs/stance_allayers.log     (126 passes per job; the all-layers log counts one extra header line, so 127)
export CUDA_DEVICE_ORDER=PCI_BUS_ID
WAIT=${BATCH_WAIT_SECONDS:-120}
TRIES=${BATCH_MAX_ATTEMPTS:-90}                       # 90 x 120 s = 3 hours of waiting per job
mkdir -p logs
note () { echo "[batch] $(date +%F\ %T) $*" >> logs/stance_batch.log; }
run_job () {                                          # $1 = log file, the rest = arguments of stance_logit_test.py
  local log=$1; shift
  local prefix rc t
  for t in $(seq 1 "$TRIES"); do
    if prefix=$(bash scripts/pick_gpus.sh); then
      note "starting $log on: $prefix"
      env $prefix python -u stance_logit_test.py "$@" > "$log" 2>&1
      rc=$?
      note "finished $log (exit $rc)"
      return $rc
    fi
    note "waiting for free GPUs for $log (try $t of $TRIES)"
    sleep "$WAIT"
  done
  note "gave up waiting for GPUs for $log"
  return 1
}
# (C) the paper's configuration: the vector at ALL 31 layers, coefficients as a fraction of each layer's typical norm
run_job logs/stance_allayers.log --vector pooled__misogyny --experiment matched --relative \
    --coefs -0.4 -0.2 -0.1 0.1 0.2 0.4 --with-reversed --tag allrel
# (D) exploratory: the layers where the content test saw a signal
run_job logs/stance_content.log --vector pooled__misogyny --experiment matched --layers -7 -8 -9 -10 \
    --coefs -6 -4 -2 2 4 6 --with-reversed --tag content
note "batch done"
