

import os
import cv2
import time

if __name__ == "__main__":
    sequences_path = '../../data/kitti/dataset/sequences'
    val_sequences = ['0'+str(i) for i in range(9)]
    val_sequences.append('09')
    val_sequences.append('10')
    

    for s in val_sequences:
        print(f"sequence {s} START --------------------------")

        s_path = os.path.join(sequences_path,s,'image_2')
        
        if s in ['00','01','02']:
            first_shape = (376, 1241, 3)
        elif s == '03':
            first_shape = (375, 1242, 3)
        else:
            first_shape = (370, 1226, 3)
        
        print(f"sequence {s} first shape : {first_shape}" )

        samples = os.listdir(s_path)
        for sample in samples:
            sample = os.path.join(s_path, sample)
            sample_path = cv2.imread(sample)
            if first_shape != sample_path.shape:
                print(os.path.join(s_path, sample))

            # print(f'sequence {s} shape {sample.shape}')

        print(f"sequence {s} END --------------------------")
        # time.sleep(1)
        
    
