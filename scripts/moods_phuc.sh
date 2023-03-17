#!/bin/bash

######## MOODS.sh ########
# This script is a wrapper around MOODS and will run MOODS using the motif, sequence, and cutoff specified

# INPUT: This script takes as input:

# INPUT_FASTA: A fasta file
# INPUT_MOTIF_DIR: directory with motifs
# OUTPUT_DIR: directory were to write MOODS output
# PVAL: Pval cutoff to use
# TF_META: A tf meta file that associates a TF gene symbol to a motif
# TF_LIST: A list of TFs that are specific for the cell type of interest
# PROJECT: Project name

##########################

# OUTPUT: 

# MOODS FILE: The output file from MOODS
# Predictions: BED file with predicted TF binding events

##########################

# Change these params based on your purpose
INPUT_FASTA="/data/wlarchive/databank/allelic/genome/hg38/hg38.fa"
INPUT_MOTIF_DIR="/data/weirauchlab/team/ngun7t/maxatac/zorn/CisBP_pfm"
OUTPUT_DIR="/data/weirauchlab/team/ngun7t/maxatac/runs/moods"
PVAL=0.0001
PROJECT="zorn"
TFS="LEF1 TCF7 TCF7L2"

INPUT_BASENAME=`basename ${1} .fa`
MOODS_PARSER="/Users/caz3so/workspaces/tacazares/scripts/python/moods/MOODS2TFBED.py"
PROJECT_DIR=${OUTPUT_DIR}/${PROJECT}
MOODS_OUTPUT=${PROJECT_DIR}/MOODS/${PROJECT}.mood

echo "Input Fasta: " ${INPUT_FASTA}
echo "Motif Directory: " ${INPUT_MOTIF_DIR}
echo "Output Directory: " ${OUTPUT_DIR}
echo "MOODS pvalue cutoff: " ${PVAL}
echo "Basename: " ${INPUT_BASENAME}
echo "Python script to parse MOODS output: " ${MOODS_PARSER}
echo "Project Name: " ${PROJECT}
echo "Project Directory: " ${PROJECT_DIR}

# Make the output directory if it does not exist
mkdir -p ${PROJECT_DIR}
mkdir -p ${PROJECT_DIR}/motifs
mkdir -p ${PROJECT_DIR}/MOODS

cp -r $INPUT_MOTIF_DIR/* ${PROJECT_DIR}/motifs

# When using this script you must change into the directory with the motifs to prevent "an argument list too long" error
echo "Change to the motif directory"

cd ${PROJECT_DIR}/motifs

# Run MOODS
echo "Run moods-dna.py"
for tf in $TFS; do
    mood_output=${PROJECT_DIR}/MOODS/${PROJECT}_${tf}.mood
    moods-dna.py -m $tf* -s ${INPUT_FASTA} -p ${PVAL} --batch -o ${mood_output}
done

# The MOODS output file is not in the format that we want it in so we have to parse it. 
#echo "Parse the MOODS output and write a bed file for each TF"
#python ${MOODS_PARSER} -i ${PROJECT_DIR}/MOODS/${PROJECT}.mood -o ${PROJECT_DIR}/TF_BEDS -T CTCF

echo "Run moods2tfbed.py"
for tf in $TFS; do
    mood_output=${PROJECT_DIR}/MOODS/${PROJECT}_${tf}.mood
    python /data/weirauchlab/team/ngun7t/maxatac/python_scripts/moods2tfbed_phuc.py -i $mood_output -o ${PROJECT_DIR}/MOODS
done

echo "DONE!"