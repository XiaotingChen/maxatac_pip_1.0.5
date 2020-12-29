import numpy as np
import pandas as pd
import sys
import random
import pybedtools

from maxatac.utilities.helpers import load_bigwig, safe_load_bigwig, load_2bit
from maxatac.utilities.constants import (
    INPUT_CHANNELS,
    INPUT_LENGTH,
    BATCH_SIZE,
    VAL_BATCH_SIZE,
    CHR_POOL_SIZE,
    BP_ORDER,
    TRAIN_SCALE_SIGNAL
)

def get_significant(data, min_threshold):
    selected = np.concatenate(([0], np.greater_equal(data, min_threshold).view(np.int8), [0]))
    breakpoints = np.abs(np.diff(selected))
    ranges = np.where(breakpoints == 1)[0].reshape(-1, 2)  # [[s1,e1],[s2,e2],[s3,e3]]
    expanded_ranges = list(map(lambda a : list(range(a[0], a[1])), ranges))
    mask = sum(expanded_ranges, [])  # to flatten
    starts = mask.copy()  # copy list just in case
    ends = [i + 1 for i in starts]
    return mask, starts, ends
    

def get_input_matrix(
    rows,
    cols,
    batch_size,          # make sure that cols % batch_size == 0
    signal_stream,
    average_stream,
    sequence_stream,
    bp_order,
    chrom,
    start,               # end - start = cols
    end,
    reshape=True,
    scale_signal=None,   # (min, max) ranges to scale signal
    filters_stream=None  # defines regions that should be set to 0
):
 
    input_matrix = np.zeros((rows, cols))
    for n, bp in enumerate(bp_order):
        input_matrix[n, :] = get_one_hot_encoded(
            sequence_stream.sequence(chrom, start, end),
            bp
        )
    signal_array = np.array(signal_stream.values(chrom, start, end))
    avg_array = np.array(average_stream.values(chrom, start, end))

    if filters_stream is not None:
        exclude_mask = np.array(filters_stream.values(chrom, start, end)) <= 0
        signal_array[exclude_mask] = 0
        avg_array[exclude_mask] = 0

    input_matrix[4, :] = signal_array
    input_matrix[5, :] = input_matrix[4, :] - avg_array

    if scale_signal is not None:
        scaling_factor = random.random() * (scale_signal[1] - scale_signal[0]) + \
            scale_signal[0]
        input_matrix[4, :] = input_matrix[4, :] * scaling_factor

    input_matrix = input_matrix.T

    if reshape:
        input_matrix = np.reshape(
            input_matrix,
            (batch_size, round(cols/batch_size), rows)
        )

    return input_matrix
        

def get_splitted_chromosomes(
    chroms,
    tchroms,
    vchroms,
    proportion
):
    """
    Doesn't take regions into account.
    May produce not correct results if inputs are not received from
    get_synced_chroms with ignore_regions=True
    """
    free_chrom_set = set(chroms) - set(tchroms) - set(vchroms)

    n = round(len(free_chrom_set) * proportion)

    # need sorted list for random.sample to reproduce in tests
    tchrom_set = set(random.sample(sorted(free_chrom_set), n))
    vchrom_set = free_chrom_set.difference(tchrom_set).union(set(vchroms))
    tchrom_set = tchrom_set.union(set(tchroms))

    extended_tchrom = {
        chrom_name: {
            "length": chroms[chrom_name]["length"],
            "region": chroms[chrom_name]["region"]
        } for chrom_name in tchrom_set}

    extended_vchrom = {
        chrom_name: {
            "length": chroms[chrom_name]["length"],
            "region": chroms[chrom_name]["region"]
        } for chrom_name in vchrom_set}

    return extended_tchrom, extended_vchrom
    
class RandomRegionsPool:

    def __init__(
        self,
        chroms,            # in a form of {"chr1": {"length": 249250621, "region": [0, 249250621]}}, "region" is ignored
        chrom_pool_size,
        region_length,
        preferences=None   # bigBed file with ranges to limit random regions selection
    ):

        
        self.chroms = chroms
        self.chrom_pool_size = chrom_pool_size
        self.region_length = region_length
        self.preferences = preferences

        #self.preference_pool = self.__get_preference_pool()  # should be run before self.__get_chrom_pool()
        self.preference_pool = False
        
        self.chrom_pool = self.__get_chrom_pool()

        self.__idx = 0


    def get_region(self):
        
        
        if self.__idx == self.chrom_pool_size:
            random.shuffle(self.chrom_pool)
            self.__idx = 0

        chrom_name, chrom_length = self.chrom_pool[self.__idx]
        self.__idx += 1

        if self.preference_pool:
            preference = random.sample(self.preference_pool[chrom_name], 1)[0]
            start = round(
                random.randint(
                    preference[0],
                    preference[1] - self.region_length
                )
            )
        else:
            start = round(
                random.randint(
                    0,
                    chrom_length - self.region_length
                )
            )
        end = start + self.region_length

        return (chrom_name, start, end)


    def __get_preference_pool(self):
        preference_pool = {}
        if self.preferences is not None:
            with load_bigwig(self.preferences) as input_stream:
                for chrom_name, chrom_data in self.chroms.items():
                    for entry in input_stream.entries(
                        chrom_name,
                        0,
                        chrom_data["length"],
                        withString=False
                    ):
                        if entry[1] - entry[0] < self.region_length:
                            continue
                        preference_pool.setdefault(
                            chrom_name, []
                        ).append(list(entry[0:2]))
        return preference_pool


    def __get_chrom_pool(self):
        """
        TODO: rewrite to produce exactly the same number of items
        as chrom_pool_size regardless of length(chroms) and
        chrom_pool_size
        """
        
        chroms = {
            chrom_name: chrom_data
            for chrom_name, chrom_data in self.chroms.items()
            #if not self.preference_pool or (chrom_name in self.preference_pool)
        }

        sum_lengths = sum(map(lambda v: v["length"], chroms.values()))
        frequencies = {
            chrom_name: round(
                chrom_data["length"] / sum_lengths * self.chrom_pool_size
            )
            for chrom_name, chrom_data in chroms.items()
        }
        labels = []
        for k, v in frequencies.items():
            labels += [(k, chroms[k]["length"])] * v
        random.shuffle(labels)
        
        return labels

def get_roi_pool_predict(seq_len=None, roi=None, shuffle=False, tf=None, cl=None):
    roi_df = pd.read_csv(roi, sep="\t", header=0, index_col=None)
    temp = roi_df['Stop'] - roi_df['Start']
    ##############################
    
    #Temporary Workaround. Needs to be deleted later 
    roi_ok = (temp == seq_len)
    temp_df = roi_df[roi_ok==True]
    roi_df = temp_df
    ###############################

    #roi_ok = (temp == seq_len).all()
    #if not roi_ok:
        
        #sys.exit("ROI Length Does Not Match Input Length")
    roi_df['TF'] = tf
    roi_df['Cell_Line'] = cl
    if shuffle:
        roi_df = roi_df.sample(frac=1)
    return roi_df


def get_roi_pool(seq_len=None, roi=None, shuffle=False):
    roi_df = pd.read_csv(roi, sep="\t", header=0, index_col=None)
    temp = roi_df['Stop'] - roi_df['Start']
    ##############################
    #Temporary Workaround. Needs to be deleted later 
    roi_ok = (temp == seq_len)
    temp_df = roi_df[roi_ok==True]
    roi_df = temp_df
    ###############################
    if shuffle:
        roi_df = roi_df.sample(frac=1)
    return roi_df


def get_one_hot_encoded(sequence, target_bp):
    one_hot_encoded = []
    for s in sequence:
        if s.lower() == target_bp.lower():
            one_hot_encoded.append(1)
        else:
            one_hot_encoded.append(0)
    return one_hot_encoded

def get_pc_input_matrix(
        rows,
        cols,
        batch_size,  # make sure that cols % batch_size == 0
        signal_stream,
        average_stream,
        sequence_stream,
        bp_order,
        chrom,
        start,  # end - start = cols
        end,
        reshape=True,
        
):
    
    input_matrix = np.zeros((rows, cols))
    for n, bp in enumerate(bp_order):
        input_matrix[n, :] = get_one_hot_encoded(
            sequence_stream.sequence(chrom, start, end),
            bp
        )
                    
    signal_array = np.array(signal_stream.values(chrom, start, end))
    avg_array = np.array(average_stream.values(chrom, start, end))
    input_matrix[4, :] = signal_array
    input_matrix[5, :] = input_matrix[4, :] - avg_array
    input_matrix = input_matrix.T

    if reshape:
        input_matrix = np.reshape(
            input_matrix,
            (batch_size, round(cols / batch_size), rows)
        )
    
    return input_matrix


def make_pc_pred_batch(
        batch_idxs,
        sequence,
        average,
        meta_table,
        roi_pool,
        bp_resolution=1,
        filters=None
):
    roi_size = roi_pool.shape[0]
    #batch_idx=0
    #n_batches = int(roi_size/BATCH_SIZE)
    #Here I will process by row, if performance is bad then process by cell line
    
    with \
            safe_load_bigwig(filters) as filters_stream, \
            load_bigwig(average) as average_stream, \
            load_2bit(sequence) as sequence_stream:
        
        inputs_batch, targets_batch = [], []
        batch_meta_df = pd.DataFrame()
        batch_gold_vals = []
        for row_idx in batch_idxs:
            roi_row = roi_pool.iloc[row_idx,:]
            cell_line = roi_row['Cell_Line']
            tf = roi_row['TF']
            chrom_name = roi_row['Chr']
            start = int(roi_row['Start'])
            end = int(roi_row['Stop'])
            meta_row = meta_table[((meta_table['Cell_Line'] == cell_line) & (meta_table['TF'] == tf))]
            meta_row = meta_row.reset_index(drop=True)
            meta_row["Start"] = start
            meta_row["Stop"] = end
            meta_row["Chr"] = chrom_name
            try:
                signal = meta_row.loc[0, 'ATAC_Signal_File']
                binding = meta_row.loc[0, 'Binding_File']
            except:
                print(roi_row)
                
                #sys.exit("Here=1. Error while creating input batch")
    
            
            with \
                    load_bigwig(binding) as binding_stream, \
                    load_bigwig(signal) as signal_stream:
                try:
                    input_matrix = get_pc_input_matrix( rows=INPUT_CHANNELS,
                                                        cols=INPUT_LENGTH,
                                                        batch_size=1,                  # we will combine into batch later
                                                        reshape=False,
                                                        bp_order=BP_ORDER,
                                                        signal_stream=signal_stream,
                                                        average_stream=average_stream,
                                                        sequence_stream=sequence_stream,
                                                        chrom=chrom_name,
                                                        start=start,
                                                        end=end
                                                    )
                    inputs_batch.append(input_matrix)
                    batch_meta_df = pd.concat([batch_meta_df, meta_row], axis='index', ignore_index=True)
                    target_vector = np.array(binding_stream.values(chrom_name, start, end)).T
                    target_vector = np.nan_to_num(target_vector, 0.0)
                    n_bins = int(target_vector.shape[0] / bp_resolution)
                    split_targets = np.array(np.split(target_vector, n_bins, axis=0))
                    bin_sums = np.sum(split_targets, axis=1)
                    bin_vector = np.where(bin_sums > 0.5*bp_resolution, 1.0, 0.0)
                    batch_gold_vals.append(bin_vector)
                    
                except:
                    print(roi_row)
                    continue
                    #sys.exit("Error while creating input batch")
        batch_meta_df = batch_meta_df.drop(['ATAC_Signal_File', 'Binding_File'], axis='columns')
        batch_meta_df.reset_index(drop=True)
        return (np.array(inputs_batch), np.array(batch_gold_vals), batch_meta_df)       
            

def create_roi_batch(
    sequence,
    average,
    meta_table,
    roi_pool,
    n_roi,
    train_tf,
    tchroms,
    bp_resolution=1,
    filters=None
    ):
        
        
        while True:
            inputs_batch, targets_batch = [], []
            roi_size = roi_pool.shape[0]
        
            curr_batch_idxs = random.sample(range(roi_size), n_roi)
    
            #Here I will process by row, if performance is bad then process by cell line
            for row_idx in curr_batch_idxs:
                roi_row = roi_pool.iloc[row_idx,:]
                cell_line = roi_row['Cell_Line']
                tf = train_tf
                chrom_name = roi_row['Chr']
                try:
                    assert chrom_name in tchroms, \
                            "Chromosome in roi file not in tchroms list. Exiting"
                except:
                    #print("Skipped {0} because it is not in tchroms".format(chrom_name))
                    continue
                start = int(roi_row['Start'])
                end = int(roi_row['Stop'])
                meta_row = meta_table[((meta_table['Cell_Line'] == cell_line) & (meta_table['TF'] == tf))]
                meta_row = meta_row.reset_index(drop=True)
                try:
                    signal = meta_row.loc[0, 'ATAC_Signal_File']
                    binding = meta_row.loc[0, 'Binding_File']
                except:
                    print("could not read meta_row. row_idx = {0}".format(row_idx))
                    continue
                with \
                safe_load_bigwig(filters) as filters_stream, \
                load_bigwig(average) as average_stream, \
                load_2bit(sequence) as sequence_stream, \
                load_bigwig(signal) as signal_stream, \
                load_bigwig(binding) as binding_stream:
                    try:
                        input_matrix = get_pc_input_matrix(
                            rows=INPUT_CHANNELS,
                            cols=INPUT_LENGTH,
                            batch_size=1,                  # we will combine into batch later
                            reshape=False,
                            bp_order=BP_ORDER,
                            signal_stream=signal_stream,
                            average_stream=average_stream,
                            sequence_stream=sequence_stream,
                            chrom=chrom_name,
                            start=start,
                            end=end
                        )
                        inputs_batch.append(input_matrix)
                        target_vector = np.array(binding_stream.values(chrom_name, start, end)).T
                        target_vector = np.nan_to_num(target_vector, 0.0)
                        n_bins = int(target_vector.shape[0] / bp_resolution)
                        split_targets = np.array(np.split(target_vector, n_bins, axis=0))
                        bin_sums = np.sum(split_targets, axis=1)
                        bin_vector = np.where(bin_sums > 0.5*bp_resolution, 1.0, 0.0)
                        targets_batch.append(bin_vector)

                    except:
                        here = 3
                        print(roi_row)
                        continue
        
            yield (np.array(inputs_batch), np.array(targets_batch))

def create_random_batch(
    sequence,
    average,
    meta_table,
    train_cell_lines,
    n_rand,
    train_tf,
    regions_pool,
    bp_resolution=1,
    filters=None
):
    
    while True:
        inputs_batch, targets_batch = [], []
        for idx in range(n_rand):
            cell_line = random.choice(train_cell_lines) #Randomly select a cell line
            chrom_name, seq_start, seq_end = regions_pool.get_region()  # returns random region (chrom_name, start, end) 
            meta_row = meta_table[((meta_table['Cell_Line'] == cell_line) & (meta_table['TF'] == train_tf))] #get meta table row corresponding to selected cell line
            meta_row = meta_row.reset_index(drop=True)
            signal = meta_row.loc[0, 'ATAC_Signal_File']
            binding = meta_row.loc[0, 'Binding_File']
            with \
                safe_load_bigwig(filters) as filters_stream, \
                load_bigwig(average) as average_stream, \
                load_2bit(sequence) as sequence_stream, \
                load_bigwig(signal) as signal_stream, \
                load_bigwig(binding) as binding_stream:
                    try:
                        input_matrix = get_input_matrix(
                            rows=INPUT_CHANNELS,
                            cols=INPUT_LENGTH,
                            batch_size=1,                  # we will combine into batch later
                            reshape=False,
                            bp_order=BP_ORDER,
                            signal_stream=signal_stream,
                            average_stream=average_stream,
                            sequence_stream=sequence_stream,
                            chrom=chrom_name,
                            start=seq_start,
                            end=seq_end,
                            scale_signal=TRAIN_SCALE_SIGNAL,
                            filters_stream=filters_stream
                        )
                        inputs_batch.append(input_matrix)
                        target_vector = np.array(binding_stream.values(chrom_name, seq_start, seq_end)).T
                        target_vector = np.nan_to_num(target_vector, 0.0)
                        n_bins = int(target_vector.shape[0] / bp_resolution)
                        split_targets = np.array(np.split(target_vector, n_bins, axis=0))
                        bin_sums = np.sum(split_targets, axis=1)
                        bin_vector = np.where(bin_sums > 0.5*bp_resolution, 1.0, 0.0)
                        targets_batch.append(bin_vector)
                    except:
                        here = 2
                        continue
        
        yield (np.array(inputs_batch), np.array(targets_batch))    
        
        
def pc_train_generator(
        sequence,
        average,
        meta_table,
        roi_pool,
        train_cell_lines,
        rand_ratio,
        train_tf,
        tchroms,
        bp_resolution=1,
        filters=None
):
    n_roi = round(BATCH_SIZE*(1. - rand_ratio))
    
    n_rand = round(BATCH_SIZE - n_roi)
    
    train_random_regions_pool = RandomRegionsPool(
        chroms=tchroms,
        chrom_pool_size=CHR_POOL_SIZE,
        region_length=INPUT_LENGTH,
        preferences=False                          # can be None
    )

    roi_gen = create_roi_batch( sequence,
                                average,
                                meta_table,
                                roi_pool,
                                n_roi,
                                train_tf,
                                tchroms,
                                bp_resolution=bp_resolution,
                                filters=None
                              )
    
    rand_gen = create_random_batch(  sequence,
                                     average,
                                     meta_table,
                                     train_cell_lines,
                                     n_rand,
                                     train_tf,
                                     train_random_regions_pool,
                                     bp_resolution=bp_resolution,
                                     filters=None
                                  )
                                            
    while True:
        
        #roi_batch.shape = (n_samples, 1024, 6)
        if rand_ratio > 0. and rand_ratio < 1.:
            roi_input_batch, roi_target_batch = next(roi_gen)
            rand_input_batch, rand_target_batch = next(rand_gen)
            inputs_batch = np.concatenate((roi_input_batch, rand_input_batch), axis=0)
            targets_batch = np.concatenate((roi_target_batch, rand_target_batch), axis=0)
        
        elif rand_ratio == 1.:
            rand_input_batch, rand_target_batch = next(rand_gen)
            inputs_batch = rand_input_batch
            targets_batch = rand_target_batch
        
        else:
            roi_input_batch, roi_target_batch = next(roi_gen)
            inputs_batch = roi_input_batch
            targets_batch = roi_target_batch
        
        yield (inputs_batch, targets_batch)
 
      
def create_val_generator(
        sequence,
        average,
        meta_table,
        train_cell_lines,
        train_tf,
        all_val_regions,
        bp_resolution=1,
        filters=None
):

    while True:
        
        inputs_batch, targets_batch = [], []
        n_val_batches = round(all_val_regions.shape[0]/VAL_BATCH_SIZE)
        all_batch_idxs = np.array_split(np.arange(all_val_regions.shape[0]), n_val_batches)  
        
        for idx, batch_idxs in enumerate(all_batch_idxs):
            inputs_batch, targets_batch = [], []
            for row_idx in batch_idxs:
                roi_row = all_val_regions.iloc[row_idx,:]
                cell_line = roi_row['Cell_Line']
                chrom_name = roi_row['Chr']
                seq_start = int(roi_row['Start'])
                seq_end = int(roi_row['Stop'])
                meta_row = meta_table[((meta_table['Cell_Line'] == cell_line) & (meta_table['TF'] == train_tf))]
                meta_row = meta_row.reset_index(drop=True)
                try:
                    signal = meta_row.loc[0, 'ATAC_Signal_File']
                    binding = meta_row.loc[0, 'Binding_File']
                except:
                    print(roi_row)
                   
                with \
                    safe_load_bigwig(filters) as filters_stream, \
                    load_bigwig(average) as average_stream, \
                    load_2bit(sequence) as sequence_stream, \
                    load_bigwig(signal) as signal_stream, \
                    load_bigwig(binding) as binding_stream:
                        input_matrix = get_input_matrix(
                            rows=INPUT_CHANNELS,
                            cols=INPUT_LENGTH,
                            batch_size=1,                  # we will combine into batch later
                            reshape=False,
                            bp_order=BP_ORDER,
                            signal_stream=signal_stream,
                            average_stream=average_stream,
                            sequence_stream=sequence_stream,
                            chrom=chrom_name,
                            start=seq_start,
                            end=seq_end,
                            scale_signal=None,
                            filters_stream=filters_stream
                        )
                        inputs_batch.append(input_matrix)
                        target_vector = np.array(binding_stream.values(chrom_name, seq_start, seq_end)).T
                        target_vector = np.nan_to_num(target_vector, 0.0)
                        n_bins = int(target_vector.shape[0] / bp_resolution)
                        split_targets = np.array(np.split(target_vector, n_bins, axis=0))
                        bin_sums = np.sum(split_targets, axis=1)
                        bin_vector = np.where(bin_sums > 0.5*bp_resolution, 1.0, 0.0)
                        targets_batch.append(bin_vector)
                        
            yield (np.array(inputs_batch), np.array(targets_batch))    

def get_roi_pool(seq_len=None, roi=None, shuffle=False):
    roi_df = pd.read_csv(roi, sep="\t", header=None)

    roi_df.columns = ['chr', 'start', 'stop']

    temp = roi_df['stop'] - roi_df['start']

    roi_ok = (temp == seq_len).all()

    if not roi_ok:
        sys.exit("ROI Length Does Not Match Input Length")
        
    if shuffle:
        roi_df = roi_df.sample(frac=1)

    return roi_df

def window_prediction_intervals(df, number_intervals=32):
    # Create BedTool object from the dataframe
    df_css_bed = pybedtools.BedTool.from_dataframe(df[['chr', 'start', 'stop']])

    # Window the intervals into 32 bins
    pred_css_bed = df_css_bed.window_maker(b=df_css_bed, n=number_intervals)

    # Create a dataframe from the BedTool object 
    return pred_css_bed.to_dataframe()

def write_df2bigwig(output_filename, interval_df, chromosome_length_dictionary, chrom):
    with dump_bigwig(output_filename) as data_stream:
        header = [(chrom, int(chromosome_length_dictionary[chrom]))]
        data_stream.addHeader(header)

        data_stream.addEntries(
        chroms = interval_df["chr"].tolist(),
        starts = interval_df["start"].tolist(),
        ends = interval_df["stop"].tolist(),
        values = interval_df["score"].tolist()
        )

def get_batch(
    signal,
    sequence,
    roi_pool,
    bp_resolution=1
):
    inputs_batch, targets_batch = [], []
    roi_size = roi_pool.shape[0]
    with \
        load_bigwig(signal) as signal_stream, \
        load_2bit(sequence) as sequence_stream:
            for row_idx in range(roi_size):
                row = roi_pool.loc[row_idx,:]
                chrom_name = row[0]
                start = int(row[1])
                end = int(row[2])
                input_matrix = get_input_matrix(
                    rows=INPUT_CHANNELS,
                    cols=INPUT_LENGTH,
                    batch_size=1,                  # we will combine into batch later
                    reshape=False,
                    scale_signal=TRAIN_SCALE_SIGNAL,
                    bp_order=BP_ORDER,
                    signal_stream=signal_stream,
                    sequence_stream=sequence_stream,
                    chrom=chrom_name,
                    start=start,
                    end=end
                )
                inputs_batch.append(input_matrix)
    return (np.array(inputs_batch))   
