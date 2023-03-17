#!/bin/bash

# Load the modules
module load bedops/2.4.39-wrl
module load bedtools/2.29.2-wrl

# This script is run as a subprocess to a python program that generates
# all combinations of 2 bed files with similar TF
first_bed=$1
second_bed=$2
bedtools jaccard -a $first_bed -b $second_bed

