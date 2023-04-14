
import os

if __name__ == "__main__":
    phase = 'val' # train val     
    semantic_kitti = 'semantic_kitti_norm100'

    f = open(f'./{semantic_kitti}/{phase}.csv', 'w')
    img_path = f'/home/jyjeon/code/mmsegmentation/data/{semantic_kitti}/img_dir/{phase}'
    ann_path = f'/home/jyjeon/code/mmsegmentation/data/{semantic_kitti}/ann_dir/{phase}'
    img_names = os.listdir(img_path)
    ann_names = os.listdir(ann_path)
    img_names, ann_names = sorted(img_names), sorted(ann_names)
    print(img_names[:5], ann_names[:5])
    
    for img, ann in zip(img_names, ann_names):
        f.write(os.path.join(img_path, img)+','+os.path.join(ann_path, ann)+'\n')
        
    f.close()
    
    
