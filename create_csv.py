
import os

if __name__ =="__main__":

    root_path = './cityscapes_origin_aug/leftImg8bit/train'
    save_path = './train_aug.csv'

    train_dirs = os.listdir(root_path)
    with open(save_path, 'w') as f:
        file = open('./train_aug.csv','w')
        file.write('img,label\n')

        for train_dir in train_dirs:
            train_dir_paths = os.path.join(root_path, train_dir)
            print(train_dir_paths)
            file_names = os.listdir(train_dir_paths)
            for file_name in file_names:
                file_path = os.path.join(root_path,train_dir,file_name)
                print(file_path)
                file.write(file_path+'\n')

    file.close()
