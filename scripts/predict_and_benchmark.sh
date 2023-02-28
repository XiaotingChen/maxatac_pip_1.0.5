#!/bin/bash

model_dir="$MYTEAM/maxatac/runs/$1"    # example: /data/weirauchlab/team/ngun7t/maxatac/runs/run-transformer-4
tf=$2           # example: CTCF 
cell_line=$3    # example: GM12878
chromosomes=$4  # example: chr1
best_model=$(cat ${model_dir}/*txt)
gold_standard="/data/weirauchlab/team/ngun7t/maxatac/training_data/ChIP_Peaks/ChIP_Peaks/"$cell_line"__"$tf".bw"

# Submit a job that does both maxatac predict and maxatac benchmark
job="
#BSUB -W 6:00
#BSUB -n 4
#BSUB -M 20000
#BSUB -R 'span[hosts=1]'
#BSUB -e logs/predict_benchmark_${cell_line}_${tf}_${chromosomes}_%J.err
#BSUB -o logs/predict_benchmark_${cell_line}_${tf}_${chromosomes}_%J.out
#BSUB -q amdgpu
#BSUB -gpu 'num=1'

# load modules
module load bedtools/2.29.2-wrl
module load samtools/1.6-wrl
module load gcc/9.3.0
module load cuda/11.7
module load pigz/2.6.0
module load ucsctools

source activate maxatac
cd /data/weirauchlab/team/ngun7t/maxatac/runs

maxatac predict \\
--model $best_model \\
--train_json $model_dir/cmd_args.json \\
--signal /data/weirauchlab/team/ngun7t/maxatac/training_data/ATAC_Signal_File/ATAC_Signal_File/${cell_line}_RP20M_minmax_percentile99.bw \\
--batch_size 100 \\
--chromosomes chr1 \\
--prefix ${tf}_${cell_line}_${chromosomes} \\
--multiprocessing False \\
--output ${model_dir}/prediction

maxatac benchmark \\
--prediction ${model_dir}/prediction/${tf}_${cell_line}_${chromosomes}.bw \\
--gold_standard ${gold_standard} \\
--prefix ${tf}_${cell_line}_${chromosomes} \\
--chromosomes chr1 \\
--output ${model_dir}/benchmark"

echo "$job" | bsub
