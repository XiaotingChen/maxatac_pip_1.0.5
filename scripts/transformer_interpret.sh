#!/bin/bash

mode="shap"
chromosome="chr1"
meta_file="/data/weirauchlab/team/ngun7t/maxatac/meta_file_for_interpreting_LEF1.tsv"
output_dir="/data/weirauchlab/team/ngun7t/maxatac/scratch"
cell_type="D1_mesendoderm"
model_base_dir="/data/weirauchlab/team/ngun7t/maxatac/runs/rpe_LEF1"

job="
#BSUB -W 6:00
#BSUB -n 2
#BSUB -M 32000
#BSUB -R 'span[ptile=2]'
#BSUB -e logs/transformer_interpret_%J.err
#BSUB -o logs/transformer_interpret_%J.out
#BSUB -q amdgpu
#BSUB -gpu 'num=1'

# load modules
module load bedtools/2.29.2-wrl
module load gcc/9.3.0
module load cuda/11.7
module load samtools/1.6-wrl
module load pigz/2.6.0
module load ucsctools
source activate maxatac
cd $MYTEAM/maxatac/runs

maxatac transformer \\
--analysis $mode \\
--chromosome $chromosome \\
--meta_file $meta_file \\
--output_dir $output_dir \\
--cell_type $cell_type \\
--model_base_dir $model_base_dir
"

echo "$job" | bsub