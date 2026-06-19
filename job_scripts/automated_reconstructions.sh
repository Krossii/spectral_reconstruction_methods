#!/bin/bash

cd ~/Desktop/spec_rec_methods
for B in 0 2 4 6 8 10 12
do
    python3 reconstruction.py --config params.json --correlatorFile "../dat/data_wilson_emconduc_48_16_b6.872_B${B}_z.txt"
done