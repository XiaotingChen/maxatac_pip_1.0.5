#!/bin/bash
module load ucsctools
module load ucsc_userapps
module load bedtools/2.29.2-wrl 

zorn_version="zorn2"
input_dir="/data/weirauchlab/team/ngun7t/maxatac/runs/moods/zorn/${zorn_version}"
zorn_bed_dir="/data/weirauchlab/team/ngun7t/maxatac/runs/moods/zorn/zorn_bed"
hg38_file="/users/ngun7t/opt/maxatac/data/hg38/hg38.chrom.sizes"
genome_fasta="/data/weirauchlab/team/ngun7t/maxatac/zorn/Genome_fasta/Homo_sapiens.GRCh38.fa"
blacklist_file="/users/ngun7t/opt/maxatac/data/hg38/hg38_maxatac_blacklist.bed"
cell_type_specific_dir="/data/weirauchlab/team/ngun7t/maxatac/runs/moods/zorn/${zorn_version}_specific"
zorn_atac="/data/weirauchlab/team/ngun7t/maxatac/zorn/Zorn_hESC_ATAC/outputs"

# biorep_peaks and biorep_peaks_cat has some differences in the naming system
# Files in biorep_peaks/gold_standard are *_peaks.bdg, while files in biorep_peaks_cat/gold_standard do not have _peaks in their names

steps="b"

if [[ $steps == "a" ]]; then

    echo "Preprocess bed files"
    for file in $(ls $zorn_bed_dir);
    do
        tf=${file#*_}
        tf=${tf%.*}
        grep -v "^chrM\s" "$zorn_bed_dir/$file" | cut -f1-3,5,7- > "$input_dir/zorn_${tf}_noMcut.bed"
        #bedRemoveOverlap "$input_dir/zorn_${tf}_noMcut.bed" "$input_dir/zorn_${tf}_noMcut_nooverlap.bed"
        bedtools makewindows -g "$hg38_file" -w 200 | \
        bedtools intersect -a - -b "$input_dir/zorn_${tf}_noMcut.bed" -c | \
        bedtools intersect -a - -b $blacklist_file -v > "$input_dir/zorn_${tf}_blacklisted.bed"
        # First intersect the bed file with the blacklist
        # bedtools intersect -a "$input_dir/zorn_${tf}_noMcut_nooverlap.bed" -b $blacklist_file -v > "$input_dir/zorn_${tf}_blacklisted.bed"
        # Later, when running on specific cell types, intersect one more with the cell-specific ATACseq narrowPeak

    done

    echo "Convert bed files to bigwig"
    for file in $(ls $input_dir/*blacklisted*);
    do
        for cell in $(ls $zorn_atac);
        do
            name_file=$(basename $file)
            tf=${name_file#*_}
            tf=${name_file%_*}
            temp=( $(ls $zorn_atac/$cell/replicate_data/*/peaks/maxatac/*.narrowPeak) )
            atac_bed=${temp[0]}
    
            #sorted_bed="$input_dir/${tf}_sorted.bed"
            final_bw="$cell_type_specific_dir/${cell}_${tf}.bw"
            bedtools intersect -a $file -b $atac_bed | sort -k1,1 -k2,2n > "$cell_type_specific_dir/${cell}_${tf}.bed"
            bedGraphToBigWig "$cell_type_specific_dir/${cell}_${tf}.bed" $hg38_file $final_bw
            echo "Finished with "$name_file
        done
    done

fi


if [[ $steps == "b" ]]; then
    
    gold_standard_dir="/data/weirauchlab/team/ngun7t/maxatac/zorn/Zorn_hESC_CHIP/outputs/biorep_peaks_cat/gold_standard_bigwig"
    pred_base_dir="/data/weirauchlab/team/ngun7t/maxatac/runs/moods/zorn/zorn2_specific"
    chromosomes="chr1"
    base_output_dir="/data/weirauchlab/team/ngun7t/maxatac/runs/benchmarking_results_mood_"$chromosomes

    for file in $(ls $pred_base_dir/*.bw);
    do
        name_file=$(basename $file)
        cell_line=${name_file%_*}
        cell_line=${cell_line%_*}
        tf=${name_file##*_}
        tf=${tf%.*}
        output_dir="$base_output_dir/${cell_line}_${tf}"
        gold="$gold_standard_dir/${cell_line}_${tf}.bw"
        job="
        #BSUB -W 3:00
        #BSUB -n 2
        #BSUB -M 20000
        #BSUB -R "span[hosts=1]"
        #BSUB -e logs/benchmark_${tf}_${chromosomes}_%J.err
        #BSUB -o logs/benchmark_${tf}_${chromosomes}_%J.out
        
        # load modules
        module load bedtools/2.29.2-wrl
        module load samtools/1.6-wrl
        module load pigz/2.6.0
        module load ucsctools
        
        source activate maxatac
        cd /data/weirauchlab/team/ngun7t/maxatac/runs
        
        # the main command
        maxatac benchmark --prediction ${file} \\
        --gold_standard ${gold} \\
        --prefix maxatac_benchmark \\
        --chromosomes ${chromosomes} \\
        --output_directory ${output_dir}
        "
        echo "$job" | bsub
    done
fi