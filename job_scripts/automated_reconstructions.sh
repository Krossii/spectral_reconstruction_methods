#!/bin/bash

cd ~/Desktop/spec_rec_methods
for B in 0 2 4 6 8 10 12
do
    python3 reconstruction.py --config params.json --default_model_file "../neuralFit/outputs/emconduc_recs/mem_priors/RhoOverOmega_data_wilson_emconduc_48_16_b6.872_B${B}_x.txt" --correlatorFile "../dat/data_wilson_emconduc_48_16_b6.872_B${B}_x.txt"
done