#!/bin/bash

# The pipeline for normalizing ChIP-seq signals, taking the average of them, and benchmark
# The whole pipeline may be a little to complicated, so probably run each step first
base_dir="$MYTEAM/maxatac/zorn/Zorn_hESC_CHIP/outputs/biorep_peaks_cat"
chip_seq_dir="$base_dir/gold_standard_bigwig"
normalized_chip_seq_dir="$base_dir/gold_standard_bigwig_normalized_minmax"
average_chip_seq_dir="$base_dir/gold_standard_bigwig_average_minmax"
chr_for_average="chr1"
benchmark_dir="$MYTEAM/maxatac/runs/benchmarking_results_avg_chip_minmax_$chr_for_average"

if [[ -d "$normalized_chip_seq_dir" ]]; then
    echo "ChIP seq already normalized"  
else
    mkdir "$normalized_chip_seq_dir"
    # normalize the bw files
    for file in $(ls $chip_seq_dir);
    do
        # get the cell line and tf from the name of file
        cell_line_and_tf="${file%.*}"
        full_dir="$chip_seq_dir/$file"
        job="
        #BSUB -W 6:00
        #BSUB -n 2
        #BSUB -M 24000
        #BSUB -R 'span[ptile=2]'
        #BSUB -e logs/average_chip_pipeline_%J.err
        #BSUB -o logs/average_chip_pipeline_%J.out
    
        module load bedtools/2.29.2-wrl
        module load samtools/1.6-wrl
        module load pigz/2.6.0
        module load ucsctools
        source activate maxatac
    
        maxatac normalize \\
        --signal \"$full_dir\" \\
        --output \"$normalized_chip_seq_dir\" \\
        --prefix \"$cell_line_and_tf\" \\
        --method min-max"
        echo "$job" | bsub
    done
fi

if [[ -d "$average_chip_seq_dir" ]]; then
    echo "ChIP seq already averaged"
else
    mkdir $average_chip_seq_dir
    # average the bw files, the way the average is done is that
    # for example, we want the average file of CL_TF
    # we choose all files *_TF excluding CL_TF itself, then we create a list of these files
    # we pass this list to maxatac average
    # we then do this for all CL_TF.bw files
    for file in $(ls $normalized_chip_seq_dir/*.bw);
    do
        file_name="${file##*/}"
        cell_line="${file_name%_*}"
        tf="${file##*_}"
        tf="${tf%.*}"
        files_for_average=$(find $normalized_chip_seq_dir -type f -name "*${tf}*" -name "*.bw" ! -name "*${cell_line}*")
        job="
        #BSUB -W 6:00
        #BSUB -n 2
        #BSUB -M 24000
        #BSUB -R 'span[ptile=2]'
        #BSUB -e logs/average_chip_pipeline_%J.err
        #BSUB -o logs/average_chip_pipeline_%J.out
    
        module load bedtools/2.29.2-wrl
        module load samtools/1.6-wrl
        module load pigz/2.6.0
        module load ucsctools
        source activate maxatac
    
        maxatac average -i "${files_for_average[@]}" \\
        --prefix \"${cell_line}_${tf}_${chr_for_average}\" \\
        --output \"$average_chip_seq_dir\" \\
        --chromosomes $chr_for_average"
        echo "$job" | bsub
    done
fi

# benchmark using the avg signal track
if [[ -d "$benchmark_dir" ]]; then
    echo "Avg ChIP-seq already benchmarked"
else
    mkdir $benchmark_dir
    for file in $(ls ${chip_seq_dir});
    do
      cell_line=${file%_*}
      tf=$(echo "${file##*_}" | awk -F . '{print $1}')
    
      avg_chip_file="$average_chip_seq_dir/${cell_line}_${tf}_${chr_for_average}.bw"
      gold_standard_full_file="$chip_seq_dir/$file"
      output_dir="$benchmark_dir/${cell_line}_$tf"
      job="
      #BSUB -W 6:00
      #BSUB -n 2
      #BSUB -M 20000
      #BSUB -R 'span[ptile=2]'
      #BSUB -e logs/average_chip_pipeline_%J.err
      #BSUB -o logs/average_chip_pipeline_%J.out
      
      # load modules
      module load bedtools/2.29.2-wrl
      module load samtools/1.6-wrl
      module load pigz/2.6.0
      module load ucsctools
      
      source activate maxatac
      cd /data/weirauchlab/team/ngun7t/maxatac/runs
      
      # the main command
      maxatac benchmark --prediction ${avg_chip_file} \\
      --gold_standard ${gold_standard_full_file} \\
      --prefix maxatac_benchmark \\
      --chromosomes ${chr_for_average} \\
      --output_directory ${output_dir}"
      echo "$job" | bsub
    done
fi



