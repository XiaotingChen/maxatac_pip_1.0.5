#!/bin/bash

mode="peak"
train_tfs="HEK293T K562 HepG2 GM12878 MCF-7 Panc1 HEK293"
zorn_tfs="D3_definitive_endoderm_SOX17KO D3_definitive_endoderm_WT Pluripotent_ES_cells D1_mesendoderm D2_endoderm_progenitors D3_neuromesodermal_progenitors_SOX2KD D3_neuromesodermal_progenitors_WT"

for tr in $train_tfs;
do
    for zo in $zorn_tfs;
    do
        job="
        #BSUB -W 4:00
        #BSUB -n 2
        #BSUB -M 32000
        #BSUB -R 'span[ptile=2]'
        #BSUB -e logs/visualize_%J.err
        #BSUB -o logs/visualize_%J.out
        
        # load modules
        module load bedtools/2.29.2-wrl
        module load samtools/1.6-wrl
        module load pigz/2.6.0
        module load ucsctools
        source activate maxatac
        
        maxatac phuc --compare_training_and_zorn peak $tr $zo
        "
        echo "$job" | bsub

    done
done
