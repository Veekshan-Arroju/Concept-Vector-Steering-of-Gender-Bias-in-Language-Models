#!/usr/bin/env bash
# Prints how to run on the GPUs that are free RIGHT NOW (the GPUs are shared and the neighbours come and go):
#   CUDA_VISIBLE_DEVICES=1                         one GPU with at least NEED_ONE_MIB (default 17000) free: the one with the most room
#   MAX_GPU_GIB=9 CUDA_VISIBLE_DEVICES=0,1         else two GPUs with at least NEED_SPLIT_MIB (default 10000) free each: the model is split, <= 9 GiB on each
# Exit status 1 and no output if neither is possible. Indices are nvidia-smi's (the scripts use CUDA_DEVICE_ORDER=PCI_BUS_ID).
need_one=${NEED_ONE_MIB:-17000}
need_split=${NEED_SPLIT_MIB:-10000}
mapfile -t free < <(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
best=-1; bestfree=0; ok=()
for i in "${!free[@]}"; do
  f=${free[$i]}
  [[ "$f" =~ ^[0-9]+$ ]] || continue
  if [ "$f" -ge "$need_one" ] && [ "$f" -gt "$bestfree" ]; then best=$i; bestfree=$f; fi
  if [ "$f" -ge "$need_split" ]; then ok+=("$i"); fi
done
if [ "$best" -ge 0 ]; then echo "CUDA_VISIBLE_DEVICES=$best"; exit 0; fi
if [ "${#ok[@]}" -ge 2 ]; then echo "MAX_GPU_GIB=9 CUDA_VISIBLE_DEVICES=${ok[0]},${ok[1]}"; exit 0; fi
exit 1
