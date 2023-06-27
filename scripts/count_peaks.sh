#!/bin/bash

bw_pred=$1
bw_zorn=$2
con_mat_csv=$3

job="
#BSUB -W 6:00
#BSUB -n 2
#BSUB -M 24000
#BSUB -R 'span[ptile=2]'
#BSUB -e $MYTEAM/maxatac/logs/count_peaks_%J.err
#BSUB -o $MYTEAM/maxatac/logs/count_peaks_%J.out
        
module load bedtools/2.29.2-wrl
module load samtools/1.6-wrl
module load pigz/2.6.0
module load ucsctools
source activate maxatac
        
maxatac phuc \\
--count_peaks $bw_pred $bw_zorn $con_mat_csv"

echo "$job" | bsub