#!/bin/bash

# The pipeline for normalizing ChIP-seq signals, taking the average of them, and benchmark
# The whole pipeline may be a little to complicated, so probably run each step first
chip_seq_dir="$MYTEAM/maxatac/training_data/ChIP_Peaks/ChIP_Peaks"
normalized_chip_seq_dir="$MYTEAM/maxatac/training_data/ChIP_normalized_signals"
average_chip_seq_dir="$MYTEAM/maxatac/training_data/ChIP_average_signals"
chr_for_average="chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 chr21 chr22" 
gold_standard_dir="$MYTEAM/maxatac/zorn/Zorn_hESC_CHIP/outputs/biorep_peaks_cat/gold_standard_bigwig"

chr_for_benchmark="chr1"
benchmark_dir="$MYTEAM/maxatac/runs/benchmark_avg_chip_true_$chr_for_benchmark"
steps="c"

if [[ $steps == "a" ]]; then

    # normalize the bw files
    for file in $(ls $chip_seq_dir);
    do
        cell_line_and_tf=$(echo $file | cut -d '.' -f 1)
        # get the cell line and tf from the name of file
        if [[ $file == *"LEF1"* ]] || [[ $file == *"TCF7"* ]] || [[ $file == *"TCF7L2"* ]]; then
            echo $cell_line_and_tf
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
        fi
    done

fi

if [[ $steps == "b" ]]; then

    # average the bw files, the way the average is done is that
    # for example, we want the average file of CL_TF
    # we choose all files *_TF excluding CL_TF itself, then we create a list of these files
    # we pass this list to maxatac average
    # we then do this for all CL_TF.bw files
    tfs="LEF1 TCF7 TCF7L2"
    for tf in $tfs;
    do
        for chr in ${chr_for_average[@]};
        do
            files=$(ls $normalized_chip_seq_dir/*$tf.bw)
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
        
            maxatac average -i "${files[@]}" \\
            --prefix \"${tf}_${chr}\" \\
            --output \"$average_chip_seq_dir\" \\
            --chromosomes $chr"
            echo "$job" | bsub
        done
    done
fi

# benchmark using the avg signal track
if [[ $steps == "c" ]]; then

    mkdir $benchmark_dir
    for file in $(ls ${gold_standard_dir});
    do
      cell_line=${file%_*}
      tf=$(echo "${file##*_}" | awk -F . '{print $1}')
    
      avg_chip_file="$average_chip_seq_dir/${tf}_${chr_for_benchmark}.bw"
      gold_standard_full_file="$gold_standard_dir/$file"
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
      --chromosomes ${chr_for_benchmark} \\
      --output_directory ${output_dir}"
      echo "$job" | bsub
    done
fi



