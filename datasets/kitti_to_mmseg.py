'''
sequneces와 image_2, image_3 별로 train/val 구분 
'''

import os 
import shutil
mode = 'val'

if mode == 'val':
    sequences = '08'
elif mode == 'train':
    sequences = ['0'+str(i) for i in range(8)]
    sequences.append('09')
    sequences.append('10')
    sequences = sorted(sequences)
print(sequences)

path = '/mnt/data/home/team_gh/data/kitti/dataset/sequences'
all_sequences = os.listdir(path)
all_sequences = sorted(all_sequences)

for s in all_sequences:
    if s in sequences:
        s_path = os.path.join(path, s)
        image2 = os.path.join(s_path, 'image_2')
        image3 = os.path.join(s_path, 'image_3')
        label2 = os.path.join(s_path, 'label_projection_front_1x1_image2')
        label3 = os.path.join(s_path, 'label_projection_front_1x1_image3')
        # image2 = sorted(image2)
        # print(image2, image3)

        # label pcd
        labels = os.listdir(label2)
        for l in labels:
            if not os.path.exists(os.path.join(label2, l)):
                if not os.path.exists(os.path.join(label3, l)):
                    print(os.path.exists(os.path.join(label2, l)))
        
            # print(os.path.join(label2, l))
            print(f'/mnt/data/home/team_gh/jyjeon/vit-adapter/data/kitti/label_dir/{mode}/{s}_2_{l}')
            label2_before, label2_after = os.path.join(label2, l), f'/mnt/data/home/team_gh/jyjeon/vit-adapter/data/kitti/label_dir/{mode}/{s}_2_{l}'
            print(label2_before, label2_after)
            label3_before, label3_after = os.path.join(label3, l), f'/mnt/data/home/team_gh/jyjeon/vit-adapter/data/kitti/label_dir/{mode}/{s}_3_{l}'
            shutil.copy(label2_before, label2_after)
            shutil.copy(label3_before, label3_after)

        # input rgb
        # images = os.listdir(image2)
        # for i in images:
        #     if not os.path.exists(os.path.join(image2,i)):
        #         print(os.path.join(image2,i))
        #         if not os.path.isfile(os.path.join(image3,i)):
        #             print(os.path.join(image3,i))
        #     print(os.path.join(image2,i))
        #     print(os.path.join(image3,i))
        #     print(f'/mnt/data/home/team_gh/jyjeon/vit-adapter/data/kitti/input_dir/{mode}/{s}_2_{i}')
        #     print(f'/mnt/data/home/team_gh/jyjeon/vit-adapter/data/kitti/input_dir/{mode}/{s}_3_{i}')
                 
        #     before2, after2 = os.path.join(image2,i), f'/mnt/data/home/team_gh/jyjeon/vit-adapter/data/kitti/input_dir/{mode}/{s}_2_{i}'
        #     before3, after3 = os.path.join(image3,i), f'/mnt/data/home/team_gh/jyjeon/vit-adapter/data/kitti/input_dir/{mode}/{s}_3_{i}'
        #     shutil.copy(before2, after2)
        #     shutil.copy(before3, after3)
        print(f'{s} END ------------------------------------------------')
            
        
        
        

