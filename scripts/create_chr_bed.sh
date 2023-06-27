#!/bin/bash

# Load the modules
module load bedops/2.4.39-wrl
module load bedtools/2.29.2-wrl

# Define some paths
base_dir="$MYTEAM/maxatac"
data_dir="$base_dir/training_data"
full_bed_dir="$base_dir/maxatac/training_data/ChIP_Peaks/ChIP_Peaks"

# Define the relevant tfs
tfs="LEF1 TCF7 TCF7L2"

# Define the chromosomes (choose a random bed file)
chrs=$(bedextract --list-chr "$full_bed_dir/D1_mesendoderm_LEF1_sorted.bdg")

for tf in ${tfs[@]};
do
    for chr in ${chrs[@]};
    do
        echo "Working on $tf for $chr"
        chr_bed_dir="$data_dir/bed_files_$chr"
        mkdir $chr_bed_dir
        
        # Run bedextract to get the bed files with only one chr
        for file in $(ls $full_bed_dir);
        do
            full_file_dir="$full_bed_dir/$file"
            #echo $full_file_dir
            bedextract $chr $full_file_dir > "$chr_bed_dir/$file"
        done
    done
done


