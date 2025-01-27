#!/usr/bin/env bash

#* Loop through numbers from 128 to 4096 in powers of 2
for number in 128 256 512 1024 2048 4096; do
#* Reverse order
# for number in 4096 2048 1024 512 256 128; do
    #* Loop through the suffix options: with and without "_checked"
    # for checked in "" "_checked"; do
    for checked in ""; do
        #* Loop through the file type: "_attack" and "_normal"
        for type in "_attack" "_normal"; do
            #* Construct the file names and command
            dataset_file="data/cidds_wk2${type}_10k.csv"
            rules_file="rules/cidds/dc/learned_${number}${checked}.pl"

            echo anuta -validate -dataset=cidds -data="${dataset_file}" -rules="${rules_file}"
            # python anuta -validate -dataset=cidds -data="${dataset_file}" -rules="${rules_file}"
        done
    done
done