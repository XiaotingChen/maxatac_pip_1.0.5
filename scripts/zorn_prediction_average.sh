#!/bin/bash

base_dir="$MYTEAM/maxatac/zorn/Zorn_hESC_CHIP/outputs/biorep_peaks_cat"
cell_types=$(ls "$MYTEAM/maxatac/zorn/Zorn_hESC_ATAC/outputs" | grep -vE "average_predictions")
chr_for_average="chr1"
average_chip_seq_dir="$MYTEAM/maxatac/zorn/Zorn_hESC_ATAC/outputs/average_predictions"

#for cell_type in $cell_types;
#do
#    predictions=$(ls $MYTEAM/maxatac/zorn/Zorn_hESC_ATAC/outputs/$cell_type/maxatac/predictions/*/*.bw)
#    job="
#    #BSUB -W 6:00
#    #BSUB -n 2
#    #BSUB -M 24000
#    #BSUB -R 'span[ptile=2]'
#    #BSUB -e logs/average_zorn_predict_%J.err
#    #BSUB -o logs/average_zorn_predict_%J.out
#
#    module load bedtools/2.29.2-wrl
#    module load samtools/1.6-wrl
#    module load pigz/2.6.0
#    module load ucsctools
#    source activate maxatac
#
#    maxatac average -i "${predictions[@]}" \\
#    --prefix \"${cell_type}_${chr_for_average}\" \\
#    --output \"$average_chip_seq_dir\" \\
#    --chromosomes $chr_for_average"
#    echo "$job" | bsub
#done



