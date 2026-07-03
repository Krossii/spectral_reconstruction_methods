#!/bin/bash

cd ~/Desktop/spec_rec_methods
output="./dat/zero_T_reconstructions/unsupervised/econduct_B_x.dat"
> "$output"   # clear output file


for B in 0 2 4 6 8
do
    cd ~/Desktop/spec_rec_methods/neuralFit/outputs/emconduc_recs/mem_priors/24x36/
    file="./RhoOverOmega_data_wilson_emconduc_24_36_b5.845_B${B}_x.txt"
    pwd

    value=$(awk 'NR==2 {print $2, $3}' "$file")
    cd ~/Desktop/spec_rec_methods
    printf "%s\t%s\n" "$B" "$value" >> "$output"
done
