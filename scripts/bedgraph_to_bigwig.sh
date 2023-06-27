module load ucsctools

folder="biorep_peaks_cat"
out_middle_dir="/data/weirauchlab/team/ngun7t/maxatac/zorn/Zorn_hESC_CHIP/outputs/"${folder}"/gold_standard_temp"
out_final_dir="/data/weirauchlab/team/ngun7t/maxatac/zorn/Zorn_hESC_CHIP/outputs/"${folder}"/gold_standard_bigwig"
input_dir="/data/weirauchlab/team/ngun7t/maxatac/zorn/Zorn_hESC_CHIP/outputs/"${folder}"/gold_standard"
hg38_file="/users/ngun7t/opt/maxatac/data/hg38/hg38.chrom.sizes"

# make 2 new dirs
mkdir $out_middle_dir
mkdir $out_final_dir

# biorep_peaks and biorep_peaks_cat has some differences in the naming system
# Files in biorep_peaks/gold_standard are *_peaks.bdg, while files in biorep_peaks_cat/gold_standard do not have _peaks in their names

for file in $(ls ${input_dir});
do
if [ $folder == "biorep_peaks" ]; then
  file_name=${file%_*}
else
  file_name=${file%.*}
fi
input_file_full_dir=$input_dir"/"$file
sorted_bdg_full_dir=$out_middle_dir"/"$file_name"_sorted.bdg"
sort -k1,1 -k2,2n $input_file_full_dir > $sorted_bdg_full_dir
final_bw_full_dir=$out_final_dir"/"$file_name".bw"
bedGraphToBigWig $sorted_bdg_full_dir $hg38_file $final_bw_full_dir
echo "Finished with "$file_name
done