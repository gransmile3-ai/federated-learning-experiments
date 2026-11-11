#!/bin/bash
# OCI 단독 FL: SuperLink + SuperNode 2개 (CPU 2코어 → 노드당 스레드 1개)
export PATH=$HOME/fl/.venv/bin:$PATH VIRTUAL_ENV=$HOME/fl/.venv OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
L=$HOME/NonIID/sweep30; mkdir -p $L
nohup flower-superlink --insecure --host 100.91.47.16 --port 8000 --disable-runtime-dependency-installation > $L/superlink.log 2>&1 < /dev/null &
sleep 6
for p in 0 1; do
  nohup flower-supernode --insecure --superlink 100.91.47.16:8000 --host 127.0.0.1 --port $((9094+p)) --node-config "partition-id=$p num-partitions=2" > $L/supernode$p.log 2>&1 < /dev/null &
done
sleep 10
grep -c "Activated node_id" $L/superlink.log
