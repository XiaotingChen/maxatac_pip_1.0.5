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

# Parameters
INPUT_FASTA="/data/weirauchlab/team/ngun7t/maxatac/zorn/Genome_fasta/Homo_sapiens.GRCh38.fa"
INPUT_MOTIF_DIR="/data/weirauchlab/team/ngun7t/maxatac/zorn/HOCOMOCO_motifs"
OUTPUT_DIR="/data/weirauchlab/team/ngun7t/maxatac/runs/moods"
PVAL=0.0001
TF_META=${5}
TF=${6}
PROJECT="zorn"

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
echo "TF Meta File: " ${TF_META}
echo "TFs to predict: " ${TF}
echo "Project Name: " ${PROJECT}
echo "Project Directory: " ${PROJECT_DIR}

# Make the output directory if it does not exist
mkdir -p ${PROJECT_DIR}
mkdir -p ${PROJECT_DIR}/motifs
mkdir -p ${PROJECT_DIR}/MOODS

echo "TF_Name\tMotif_ID\tDatabase" > ${PROJECT_DIR}/${PROJECT}_meta_file.tsv

# Get the meta information for the TFs in the list
grep -w ${TF} ${TF_META} | sort | uniq >> ${PROJECT_DIR}/${PROJECT}_meta_file.tsv

# copy the motifs of interest into the output directory 
for i in $(tail -n +2 ${PROJECT_DIR}/${PROJECT}_meta_file.tsv | cut -f2 | sort | uniq);
    do
        cp ${INPUT_MOTIF_DIR}/${i} ${PROJECT_DIR}/motifs/${i}
    done

# When using this script you must change into the directory with the motifs to prevent "an argument list too long" error
echo "Change to the motif directory"

cd ${PROJECT_DIR}/motifs

# Run MOODS
echo "Run MOODS"
moods-dna.py -m * -s ${INPUT_FASTA} -p ${PVAL} --batch -o ${MOODS_OUTPUT}

# The MOODS output file is not in the format that we want it in so we have to parse it. 
#echo "Parse the MOODS output and write a bed file for each TF"
#python ${MOODS_PARSER} -i ${PROJECT_DIR}/MOODS/${PROJECT}.mood -o ${PROJECT_DIR}/TF_BEDS -T CTCF

# python /data/weirauchlab/team/ngun7t/maxatac/python_scripts/moods2tfbed.py -i /data/weirauchlab/team/ngun7t/maxatac/runs/moods/zorn/MOODS/zorn.mood -o /data/weirauchlab/team/ngun7t/maxatac/runs/moods/zorn/MOODS

echo "DONE!"