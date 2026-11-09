#!/bin/bash
cd ~/FL/NonIID
F=~/FL/.venv-1/bin/flwr
run() { tag=$1; shift; echo "=== $tag $(date +%T)"; $F run . aws --stream --run-config "$* run-tag=\"$tag\"" > logs/$tag.txt 2>&1; echo "done $tag rc=$? $(date +%T)"; }
run noniid_fedavg  'strategy="fedavg"'
run noniid_fedprox_mu0.1 'strategy="fedprox" proximal-mu=0.1'
run noniid_fedprox_mu0.01 'strategy="fedprox" proximal-mu=0.01'
run noniid_fedprox_mu1 'strategy="fedprox" proximal-mu=1.0'
