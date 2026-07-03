#!/bin/bash

cd ~/Desktop/spec_rec_methods
for B in 10 12
do
    python3 reconstruction.py --config params.json --correlatorFile "../dat/data_wilson_emconduc_24_36_b5.845_B${B}_x.txt"
done