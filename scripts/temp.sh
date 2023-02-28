#!/bin/bash

model_dir="$MYTEAM/maxatac/runs/$1"    # example: /data/weirauchlab/team/ngun7t/maxatac/runs/run-transformer-4
tf=$2           # example: CTCF 
cell_line=$3    # example: GM12878
chromosomes=$4  # example: chr1
gold_standard="$MYTEAM/maxatac/training_data/ChIP_Peaks/ChIP_Peaks/${cell_line}__$tf.bw"

# Add the code to automatically generate meta file

# Submit a job that does both maxatac predict and maxatac benchmark
job_head="
#BSUB -W 72:00
#BSUB -n 16
#BSUB -M 80000
#BSUB -R "span[ptile=16]"
#BSUB -e logs/ztemp_%J.err
#BSUB -o logs/ztemp_%J.out
#BSUB -q amdgpu
#BSUB -gpu "num=1"

# load modules
module load bedtools/2.29.2-wrl
module load gcc/9.3.0
module load cuda/11.7
module load samtools/1.6-wrl
module load pigz/2.6.0
module load ucsctools
source activate maxatac
cd $MYTEAM/maxatac/runs
"

best_model=$(cat ${model_dir}/*txt)

predict_job="
$job_head
maxatac predict \\
--model ${best_model} \\
--signal $MYTEAM/maxatac/training_data/ATAC_Signal_File/ATAC_Signal_File/${cell_line}_RP20M_minmax_percentile99.bw \\
--batch_size 100 \\
--chromosomes $chromosomes \\
--prefix ${tf}_${cell_line}_${chromosomes} \\
--multiprocessing False \\
--output ${model_dir}/prediction"

echo "$predict_job" | bsub
