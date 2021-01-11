import os
import numpy as np
import csv
from pathlib import Path

sample_rate = 32000
test_data_folder = 'gunshots_holdout/'
labels_csv_path = 'panns_data/class_labels_indices_gunshots.csv'
labels = ['BoltAction22', 'Colt1911', 'Glock9', 'Glock45', 'HKUSP', 'Kimber45', 'Lorcin380', 'M16',
          'MP40', 'Remington700', 'Ruger22', 'Ruger357', 'Sig9', 'Smith&Wesson22', 'Smith&Wesson38special',
          'SportKing22', 'WASR-10', 'WinchesterM14']

# Load label
with open(labels_csv_path, 'r') as f:
    reader = csv.reader(f, delimiter=',')
    lines = list(reader)

labels = []
ids = []    # Each label has a unique id such as "/m/068hy"
for i1 in range(1, len(lines)):
    id = lines[i1][1]
    label = lines[i1][2]
    ids.append(id)
    labels.append(label)

classes_num = len(labels)

lb_to_ix = {label : i for i, label in enumerate(labels)}
ix_to_lb = {i : label for i, label in enumerate(labels)}

id_to_ix = {id : i for i, id in enumerate(ids)}
ix_to_id = {i : id for i, id in enumerate(ids)}