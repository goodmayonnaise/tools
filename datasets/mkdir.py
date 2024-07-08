import os
import numpy as np

root = 'dataset/sequences'
sequences = [str(i).zfill(2) for i in range(22)]
print(sequences)
sequences = [os.path.join(root, s) for s in sequences ] 
print(sequences)

print([os.path.exists(s) for s in sequences])

dirs = ['image_2', 
        'proj_range',
        'proj_mask',
        'proj_xyz',
        'unproj_range',
        'proj_x', 'proj_y',
        'npoint',
        'proj_sem_label',
        'proj_sem_color',
        'sem_label']
h, w = 384, 1248*5
split = 5


for seq in sequences:
    for dir in dirs:
        os.mkdirs(os.path.join(seq, f"{dir}_{h}x{w}_split{split}"))
        
    
