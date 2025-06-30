#!/bin/bash
# This script creates multiple copy of ibmb_batch.yaml 
# file in the current directory and name them based on batch size and sprs value and sprs_rate value 
# batch size will be 4, 8, 16, 32
# sprs will be 0 1 2 
# for sprs 2 the sprs_rate will be 0.2 0.4 0.6 0.8
# for sprs 1 the sprs_rate will be 1
# for sprs 0 the sprs_rate will be 1
# The file name will be ibmb_batch_{batch_size}_{sprs}_{sprs_rate}.yaml
# in each file number 32 will be replaced with batch size 
# sprs: 0 will be replaced with sprs value
# sprs_rate: 0.2 will be replaced with sprs_rate value

#store the name of the first {input} arg
input=$1

for batch_size in 300 600 1200 2400 
do
  for sprs in 0 1 2 3
  do
    for sprs_method in SCAN LSIM
    do
      if [ $sprs -gt 1 ]
      then
        for sprs_rate in 0.2 0.4 0.6 0.8
        do
          #cp ibmb_batch.yaml ibmb_batch_${batch_size}_${sprs}_${sprs_method}_${sprs_rate}.yaml
          cp ${input}.yaml ${input}_${batch_size}_${sprs}_${sprs_method}_${sprs_rate}.yaml 
          sed -i "s/num_batches: [ [0-9]*,/num_batches: [ $batch_size,/g" ${input}_${batch_size}_${sprs}_${sprs_method}_${sprs_rate}.yaml
          sed -i "s/sprs: 0/sprs: $sprs/g" ${input}_${batch_size}_${sprs}_${sprs_method}_${sprs_rate}.yaml
          sed -i "s/sprs_rate: 1/sprs_rate: $sprs_rate/g" ${input}_${batch_size}_${sprs}_${sprs_method}_${sprs_rate}.yaml
          sed -i "s/sprs_method: 'SCAN'/sprs_method: '$sprs_method'/g" ${input}_${batch_size}_${sprs}_${sprs_method}_${sprs_rate}.yaml
        done
      elif [ $sprs -eq 1 ]
      then
        cp ${input}.yaml ${input}_${batch_size}_${sprs}_${sprs_method}_1.yaml
        sed -i "s/num_batches: [ [0-9]*,/num_batches: [ $batch_size,/g" ${input}_${batch_size}_${sprs}_${sprs_method}_1.yaml
        sed -i "s/sprs: 0/sprs: $sprs/g" ${input}_${batch_size}_${sprs}_${sprs_method}_1.yaml
        sed -i "s/sprs_rate: 1/sprs_rate: 1/g" ${input}_${batch_size}_${sprs}_${sprs_method}_1.yaml
        sed -i "s/sprs_method: 'SCAN'/sprs_method: '$sprs_method'/g" ${input}_${batch_size}_${sprs}_${sprs_method}_1.yaml
      else
        cp ${input}.yaml ${input}_${batch_size}_${sprs}_SCAN_1.yaml
        #replace a pattern like [ *, with the batch size and * can be any number
        sed -i "s/num_batches: [ [0-9]*,/num_batches: [ $batch_size,/g" ${input}_${batch_size}_${sprs}_SCAN_1.yaml
        sed -i "s/sprs: 0/sprs: $sprs/g" ${input}_${batch_size}_${sprs}_SCAN_1.yaml
        sed -i "s/sprs_rate: 1/sprs_rate: 1/g" ${input}_${batch_size}_${sprs}_SCAN_1.yaml
      fi
    done
  done
done

