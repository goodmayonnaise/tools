
import os

def sort_list(lst):
    
    lst.sort()
    return lst


if __name__ =="__main__":

    imgs_path  = '/mnt/team_gh/cityscapes_origin_aug/leftImg8bit/train'
    labels_path  = '/mnt/team_gh/cityscapes_origin_aug/gtFine/train'
    save_path = './train_aug.csv'

    img_dirs = os.listdir(imgs_path)
    label_dirs = os.listdir(labels_path)

    img_lst, label_lst = [], []
    with open(save_path, 'w') as f:
        for img_dir in img_dirs:
            img_dir_paths = os.path.join(imgs_path, img_dir)
            # print(img_dir_paths, label_dir_paths)
            img_names = os.listdir(img_dir_paths)
            # print(img_names[0], label_names[0])
            print(len(img_names))
            # print(len(label_names))
            for img_name in img_names:
                # print(os.path.join(labels_path, label_dir, label_name))
                img_lst.append(os.path.join(imgs_path, img_dir, img_name))

        for label_dir in label_dirs:
            # img_dir_paths = os.path.join(imgs_path, img_dir)
            label_dir_paths = os.path.join(labels_path, label_dir)
            # print(img_dir_paths, label_dir_paths)
            # img_names = os.listdir(img_dir_paths)
            label_names = os.listdir(label_dir_paths)
            for label_name in label_names:
                if "labelIds" in label_name:
                    label_lst.append(os.path.join(labels_path, label_dir, label_name))
    
        sort_list(img_lst)
        sort_list(label_lst)
        print(len(img_lst), len(label_lst))
        file = open('./train_aug.csv','w')
        file.write('img,label')
        for img, label in zip(img_lst, label_lst):
            file.write(f'\n{img},{label}')
            print(img, label )

                    # print(img_path, label_path)
                    # file.write(file_path+'\n')
    file.close()
