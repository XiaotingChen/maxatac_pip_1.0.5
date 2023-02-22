import os
import re
import json

cell_lines = [name.split("_")[0] for name in os.listdir("/data/weirauchlab/team/ngun7t/maxatac/training_data/ATAC_Peaks/ATAC_Peaks")]
tfs = list(set([
    re.search("/_(.*?)\\./", name).group() for name in os.listdir("/data/weirauchlab/team/ngun7t/maxatac/training_data/ChIP_Binding_File/ChIP_Binding_File")
]))

cell_lines_to_tf = {cell_line: [] for cell_line in cell_lines}
tf_to_cell_lines = {tf: [] for tf in tfs}

for name in os.listdir("/data/weirauchlab/team/ngun7t/maxatac/training_data/ChIP_Binding_File/ChIP_Binding_File"):
    cell_line = name.split("_")[0]
    tf = re.search("/_(.*?)\\./", name).group()
    cell_lines_to_tf[cell_line].append(tf)
    tf_to_cell_lines[tf].append(cell_line)

with open("/data/weirauchlab/team/ngun7t/maxatac/training_data/cell_line_to_tf.json", "w") as f:
    json.dump(cell_lines_to_tf, f)

with open("/data/weirauchlab/team/ngun7t/maxatac/training_data/tf_to_cell_line.json", "w") as f:
    json.dump(tf_to_cell_lines, f)
