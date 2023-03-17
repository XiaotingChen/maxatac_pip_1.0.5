#!/bin/bash

gold_standard_dir="/data/weirauchlab/team/ngun7t/maxatac/zorn/Zorn_hESC_CHIP/outputs/biorep_peaks_cat/gold_standard_bigwig"
maxatac_pred_base_dir="/data/weirauchlab/team/ngun7t/maxatac/zorn/Zorn_hESC_ATAC/outputs"
chromosomes="chr1"
base_output_dir="/data/weirauchlab/team/ngun7t/maxatac/runs/benchmarking_results_"$chromosomes


for file in $(ls ${gold_standard_dir});
do
  cell_line=${file%_*}
  tf=$(echo "${file##*_}" | awk -F . '{print $1}')

  maxatac_pred_file=$maxatac_pred_base_dir"/"$cell_line"/maxatac/predictions/"$tf"/"$cell_line"_"$tf".bw"
  gold_standard_full_file=$gold_standard_dir"/"$file
  output_dir=$base_output_dir"/"$cell_line"_"$tf
  job="
  #BSUB -W 3:00
  #BSUB -n 4
  #BSUB -M 20000
  #BSUB -R "span[hosts=1]"
  #BSUB -e logs/benchmark_${cell_line}_${tf}_${chromosomes}_%J.err
  #BSUB -o logs/benchmark_${cell_line}_${tf}_${chromosomes}_%J.out
  
  # load modules
  module load bedtools/2.29.2-wrl
  module load samtools/1.6-wrl
  module load pigz/2.6.0
  module load ucsctools
  
  source activate maxatac
  cd /data/weirauchlab/team/ngun7t/maxatac/runs
  
  # the main command
  maxatac benchmark --prediction ${maxatac_pred_file} \\
  --gold_standard ${gold_standard_full_file} \\
  --prefix maxatac_benchmark \\
  --chromosomes ${chromosomes} \\
  --output_directory ${output_dir}
  "
  echo "$job" | bsub
done