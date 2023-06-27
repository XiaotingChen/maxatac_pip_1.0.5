#!/bin/bash

# zorn cell types:
cell_types_1="D3_definitive_endoderm_SOX17KO D3_definitive_endoderm_WT Pluripotent_ES_cells"
cell_types_2="D1_mesendoderm D2_endoderm_progenitors D3_neuromesodermal_progenitors_SOX2KD D3_neuromesodermal_progenitors_WT"

# change these variables to match the demanded need
model_dir="$MYTEAM/maxatac/runs/multiinput_GATA3"    # example: /data/weirauchlab/team/ngun7t/maxatac/runs/run-transformer-4
tf="GATA3"           # example: CTCF 
cell_line="A549"    # example: GM12878 (ignore when $dataset = zorn)
chromosomes="chr1"  # example: chr1
best_model=$(cat ${model_dir}/*txt)
dataset="train"

if [[ $dataset == "train" ]]; then
    # train dataset
    output_dir="$model_dir/prediction"
    benchmark_dir="$model_dir/benchmark"
    gold_standard="$MYTEAM/maxatac/training_data/ChIP_Peaks/ChIP_Peaks/${cell_line}__$tf.bw"
    atac_signal="$MYTEAM/maxatac/training_data/ATAC_Signal_File/ATAC_Signal_File/${cell_line}_RP20M_minmax_percentile99.bw"

    job="
    #BSUB -W 6:00
    #BSUB -n 2
    #BSUB -M 32000
    #BSUB -R 'span[ptile=2]'
    #BSUB -e logs/predict_benchmark_${cell_line}_${tf}_${chromosomes}_${dataset}_%J.err
    #BSUB -o logs/predict_benchmark_${cell_line}_${tf}_${chromosomes}_${dataset}_%J.out
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
    --signal $atac_signal \\
    --batch_size 100 \\
    --chromosomes $chromosomes \\
    --prefix ${tf}_${cell_line}_${chromosomes} \\
    --multiprocessing False \\
    --output $output_dir
    
    maxatac benchmark \\
    --prediction $output_dir/${tf}_${cell_line}_${chromosomes}.bw \\
    --gold_standard $gold_standard \\
    --prefix ${tf}_${cell_line}_${chromosomes} \\
    --chromosomes $chromosomes \\
    --output $benchmark_dir"

    echo "$job" | bsub

else
    # zorn dataset
    for cell_type in ${cell_types_2[@]}; do
        output_dir="$model_dir/prediction_zorn"
        benchmark_dir="$model_dir/benchmark_zorn"
        gold_standard="$MYTEAM/maxatac/zorn/Zorn_hESC_CHIP/outputs/biorep_peaks_cat/gold_standard_bigwig/${cell_type}_$tf.bw"
        atac_signal="$MYTEAM/maxatac/zorn/Zorn_hESC_ATAC/outputs/$cell_type/maxatac/normalize_bigwig/$cell_type.bw"
    
        job="
        #BSUB -W 6:00
        #BSUB -n 2
        #BSUB -M 32000
        #BSUB -R 'span[hosts=1]'
        #BSUB -e logs/predict_benchmark_${cell_type}_${tf}_${chromosomes}_${dataset}_%J.err
        #BSUB -o logs/predict_benchmark_${cell_type}_${tf}_${chromosomes}_${dataset}_%J.out
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
        --signal $atac_signal \\
        --batch_size 100 \\
        --chromosomes $chromosomes \\
        --prefix ${tf}_${cell_type}_${chromosomes} \\
        --multiprocessing False \\
        --output $output_dir
        
        maxatac benchmark \\
        --prediction $output_dir/${tf}_${cell_type}_${chromosomes}.bw \\
        --gold_standard $gold_standard \\
        --prefix ${tf}_${cell_type}_${chromosomes} \\
        --chromosomes $chromosomes \\
        --output $benchmark_dir"
        
        echo "$job" | bsub
    done
fi

# Submit a job that does both maxatac predict and maxatac benchmark

