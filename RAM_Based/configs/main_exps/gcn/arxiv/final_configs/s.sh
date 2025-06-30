#!/bin/bash

cp ibmb_batch_2_0_SCAN_1.yaml ibmb_batch_1_0_SCAN_1.yaml
cp ibmb_batch_2_2_LSIM_0.2.yaml ibmb_batch_1_2_LSIM_0.2.yaml
cp ibmb_batch_2_2_LSIM_0.2.yaml ibmb_batch_1_2_LSIM_0.4.yaml
cp ibmb_batch_2_2_LSIM_0.4.yaml ibmb_batch_1_2_LSIM_0.4.yaml
cp ibmb_batch_2_2_LSIM_0.6.yaml ibmb_batch_1_2_LSIM_0.6.yaml
cp ibmb_batch_2_2_LSIM_0.8.yaml ibmb_batch_1_2_LSIM_0.8.yaml
cp ibmb_batch_2_2_SCAN_0.8.yaml ibmb_batch_1_2_SCAN_0.8.yaml
cp ibmb_batch_2_2_SCAN_0.8.yaml ibmb_batch_1_2_SCAN_0.6.yaml
cp ibmb_batch_2_2_SCAN_0.6.yaml ibmb_batch_1_2_SCAN_0.6.yaml
cp ibmb_batch_2_2_SCAN_0.4.yaml ibmb_batch_1_2_SCAN_0.4.yaml
cp ibmb_batch_2_2_SCAN_0.2.yaml ibmb_batch_1_2_SCAN_0.2.yaml
sed -i 's/\[ 2,/\[ 1,/g' ibmb_batch_1_*

