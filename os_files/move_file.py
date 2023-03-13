import os, shutil
from tqdm import tqdm 


if __name__ == "__main__":

    root_path = 'cityscapes_origin_aug'
    moved_path = '/mnt/team_gh/cityscapes_origin_aug'

    img_label = os.listdir(root_path)
    for dir in img_label:
        root_path1 = os.path.join(root_path, dir, 'train')

        classes = os.listdir(root_path1)
        # print(classes)

        for class_ in classes:
            root_path2 = os.path.join(root_path1, class_)     

            # print(os.listdir(root_path2))
            # print(len(os.listdir(root_path2)))
            files = os.listdir(root_path2)

            for file in tqdm(files):
                # print(os.path.join(root_path2, file))
                before_path = os.path.join(root_path2,file)
                # print(before_path)
                after_path = os.path.join(moved_path, dir, 'train', class_, file)
                shutil.move(before_path, after_path)
                print(before_path, after_path)
    print('\nclear')

