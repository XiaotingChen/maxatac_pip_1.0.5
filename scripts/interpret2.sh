tf=$1
cell_lines="D3_definitive_endoderm_SOX17KO D2_endoderm_progenitors D3_neuromesodermal_progenitors_SOX2KD D3_definitive_endoderm_WT Pluripotent_ES_cells D1_mesendoderm D3_neuromesodermal_progenitors_WT"
model_weight=$(ls /users/ngun7t/opt/maxatac/data/models/${tf}/*.h5)
for cell_line in $cell_lines; do

    job="#BSUB -W 24:00
    #BSUB -n 4
    #BSUB -M 32000
    #BSUB -R \"span[ptile=16]\"
    #BSUB -e logs/interpret2_${cell_line}_${tf}_%J.err
    #BSUB -o logs/interpret2_${cell_line}_${tf}_%J.out
    #BSUB -q gpu-v100
    #BSUB -gpu \"num=1\"
    
    module load bedtools/2.29.2-wrl
    module load gcc/8.3.0
    module load cuda/11.5
    source activate maxatac-tfmodisco-2
    
    cd /data/weirauchlab/team/ngun7t/maxatac
    
    maxatac interpret \
    --arch DCNN_V2 \
    --sequence /users/ngun7t/opt/maxatac/data/hg38/hg38.2bit  \
    --meta_file /data/weirauchlab/team/ngun7t/maxatac/meta_file_for_interpreting_${tf}.tsv \
    --output /data/weirauchlab/team/ngun7t/maxatac/runs/interpreting_results/${tf}_${cell_line} \
    --prefix interpreting_analysis \
    --interpreting_cell_line ${cell_line} \
    --weights ${model_weight} \
    --threads 96 \
    --sample_size  5000  \
    --channels  11 15"

    echo "$job" | bsub
    
done