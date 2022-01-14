# Prepare

The `prepare` function will convert a BAM file to Tn5 cut sites that are smoothed with a specific slop size. The files are converted to bigwig signal tracks and then min-max normalized. You must have `samtools`, `bedtools`, `pigz`, and `bedGraphToBigWig` installed to run this function. 

## Example

```bash
maxatac prepare -i SRX2717911.bam -o ./output -prefix SRX2717911
```

## Required Arguments

### `-i, --input`

The input file to be processed. The input file can be either:

* `.bam`: Bulk ATAC-seq BAM file. We will do a simple check to see if duplicates are reported.
* `.tsv`: 10X scATAC fragments file. Must end in `.tsv` or `.tsv.gz`.

### `-o, --output`

The output directory path.

### `-prefix`

This argument is used to set the logging level. Currently, the only working logging level is `ERROR`.

## Optional Arguments

### `-slop, --slop`

The slop size to use around the Tn5 cut sites.

### `-rpm, --rpm_factor`

The RPM factor to use. Example 1000000, 20000000

### `--blacklist_bed`

The path to the blacklist bed file.

### `--blacklist_bw`

The path to the blacklist bigwig file

### `--chrom_sizes`

The chromosome sizes file.

### `-chroms, --chromosomes`

The chromosomes to use for the final output.

### `-threads`

The number of threads to use.

### `--loglevel`

The log level to use for printing messages.