#!/bin/bash

tf="TCF7L2"           # example: CTCF
model_dir="$MYTEAM/maxatac/runs/rpe_$tf"    # example: /data/weirauchlab/team/ngun7t/maxatac/runs/run-transformer-4
cell_line="GM12878"    # example: GM12878
chromosomes="chr1"  # example: chr1
gold_standard="$MYTEAM/maxatac/training_data/ChIP_Peaks/ChIP_Peaks/${cell_line}__$tf.bw"

# Add the code to automatically generate meta file

# Submit three jobs one for 
job="
#BSUB -W 72:00
#BSUB -n 8
#BSUB -M 80000
#BSUB -R 'span[ptile=8]'
#BSUB -e logs/train_pipeline_${cell_line}_${tf}_${chromosomes}_%J.err
#BSUB -o logs/train_pipeline_${cell_line}_${tf}_${chromosomes}_%J.out
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

maxatac train \\
--genome hg38 \\
--arch Transformer_phuc \\
--sequence $HOME/opt/maxatac/data/hg38/hg38.2bit \\
--meta_file $MYTEAM/maxatac/training_data/meta_file_$tf.tsv \\
--output $model_dir \\
--prefix transformer \\
--epochs 200

best_model=\$(cat ${model_dir}/*txt)

#maxatac predict \\
#--model \"\${best_model}\" \\
#--train_json $model_dir/cmd_args.json \\
#--signal $MYTEAM/maxatac/training_data/ATAC_Signal_File/ATAC_Signal_File/${cell_line}_RP20M_minmax_percentile99.bw \\
#--batch_size 100 \\
#--chromosomes $chromosomes \\
#--prefix ${tf}_${cell_line}_${chromosomes} \\
#--multiprocessing False \\
#--output ${model_dir}/prediction
#
#maxatac benchmark \\
#--prediction ${model_dir}/prediction/${tf}_${cell_line}_${chromosomes}.bw \\
#--gold_standard ${gold_standard} \\
#--prefix ${tf}_${cell_line}_${chromosomes} \\
#--chromosomes $chromosomes \\
#--output ${model_dir}/benchmark"

echo "$job" | bsub
