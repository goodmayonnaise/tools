

import os
import shutil
from time import sleep


if __name__ == "__main__":

    train_sequences = ['0'+str(i) for i in range(8)]
    train_sequences.append('09')
    train_sequences.append('10')
    val_sequences = ['08']

    path = '/mnt/data/home/team_gh/data/kitti/dataset/sequences'
    
    # train 
    for s in train_sequences:
        input_dir = os.path.join(path, s, 'velodyne')
        inputs = os.listdir(input_dir)
        label_dir = os.path.join(path, s, 'labels')
        labels = os.listdir(label_dir)

        inputs, labels = sorted(inputs), sorted(labels)
        
        print(len(inputs), len(labels))


        for input in inputs:
            before_input = os.path.join(input_dir, input)
            label = input[:-4]+'.label'
            before_label = os.path.join(label_dir, label)
            
            after_input = f'/mnt/data/home/team_gh/jyjeon/vit-adapter-kitti/data/semantic_kitti/pcd/train/{s}_{input}'
            after_label = f"/mnt/data/home/team_gh/jyjeon/vit-adapter-kitti/data/semantic_kitti/label/train/{s}_{label}"

            shutil.copy(before_input, after_input)
            shutil.copy(before_label, after_label)
            # print(before_input, after_input)
            # print(before_label, after_label)

        print(f'END sequences {s}-------------------------------------------------------------------')


    # val 
    for s in val_sequences:
        input_dir = os.path.join(path, s, 'velodyne')
        inputs = os.listdir(input_dir)
        label_dir = os.path.join(path, s, 'labels')
        labels = os.listdir(label_dir)

        for input in inputs:
            before_input = os.path.join(input_dir, input)
            label = input[:-4]+'.label'
            before_label = os.path.join(label_dir, label)

            after_input = f"/mnt/data/home/team_gh/jyjeon/vit-adapter-kitti/data/semantic_kitti/pcd/val/{s}_{input}"
            after_label = f"/mnt/data/home/team_gh/jyjeon/vit-adapter-kitti/data/semantic_kitti/label/val/{s}_{label}"
            shutil.copy(before_input, after_input)
            shutil.copy(before_label, after_label)
            # print(before_input, after_input)
            # print(before_label, after_label)
            
        print(f'END sequences {s}-------------------------------------------------------------------')

