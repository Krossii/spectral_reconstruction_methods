#!/bin/bash

cd ~/Desktop/spec_rec_methods
for noise in 2 3 4
do
    python3 reconstruction.py --config params.json --default_model_file "../neuralFit/outputs/mock_data/Nt16/RhoOverOmega_mock_corr_BW_Nt16_noise${noise}.dat" --correlatorFile "../../mock-data-main/finite_T/uncorrelated_data/BW/mock_corr_BW_Nt16_noise${noise}.dat"
done