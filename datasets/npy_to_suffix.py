import os
import numpy as np
import cv2

def roof(sequences_path, dir_path, dataset, suffix):
    if dataset == "train":
        sequences = ['0'+str(i) for i in range(8)]
        sequences.append('09')
    else:
        sequences = ['08']
        
    for s in sequences:

        print(f'start sequence {s} ------------------------------------------')
        print(s)
        path = os.path.join(sequences_path, s)
            
        if dataset == 'train':
            path = path + '/input_projection/remission'

        else:
            path = path + '/label_projection/label'
        
        npy_names = os.listdir(path)
        print(npy_names)

        for name in npy_names:
            npy_path = os.path.join(path, name)
            npy = np.load(npy_path)
            arr = np.array(npy)
            arr_norm1000 = cv2.normalize(arr, None, 0, 1000, cv2.NORM_MINMAX)
            arr_norm100 = cv2.normalize(arr, None, 0, 1000, cv2.NORM_MINMAX)        
            
            arr_path = f'./semantic_kitti/{dir_path}/{dataset}/{s}_{name[:-4]}'+suffix
            arr_norm100_path = f'./semantic_kitti_norm100/{dir_path}/{dataset}/{s}_{name[:-4]}'+suffix
            arr_norm1000_path = f'./semantic_kitti_norm1000/{dir_path}/{dataset}/{s}_{name[:-4]}'+suffix
            print(f'./semantic_kitti/{dir_path}/{dataset}/{s}_{name[:-4]}'+suffix)
            # print(name)
            cv2.imwrite(arr_path, arr)
            cv2.imwrite(arr_norm100_path, arr_norm100)
            cv2.imwrite(arr_norm1000_path, arr_norm1000)


if __name__ == "__main__":
    
    sequences_path = '/mnt/team_gh/kitti/dataset/sequences'
    dir_path = 'img_dir' # img_dir ann_dir
    dataset = 'val' # train val

    if dir_path == 'img_dir':
        suffix = '.png'
        roof(sequences_path, dir_path, dataset, suffix)
        
    else:
        suffix = '.jpg'
        roof(sequences_path, dir_path, dataset, suffix)


