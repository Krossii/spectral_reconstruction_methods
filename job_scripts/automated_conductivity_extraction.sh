#!/bin/bash

cd ~/Desktop/spec_rec_methods
output="./dat/finite_T_reconstructions/mem_neural/econduct_B_x.dat"
> "$output"   # clear output file


for B in 0 2 4 6 8 10 12
do
    cd ~/Desktop/spec_rec_methods/mem/outputs/emconduc_recs/neural_prior
    file="./RhoOverOmega_finite_T_prior_file_data_wilson_emconduc_48_16_b6.872_B${B}_x.txt"
    pwd

    value=$(awk 'NR==2 {print $2, $3}' "$file")
    cd ~/Desktop/spec_rec_methods
    printf "%s\t%s\n" "$B" "$value" >> "$output"
done
