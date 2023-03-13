import os 

if __name__ == "__main__":

    root_path = '/mnt/team_gh/cityscapes_origin_aug'

    img_label = ['leftImg8bit', 'gtFine']

    for dir in img_label:
        root_path1 = os.path.join(root_path, dir, 'train')
        # print(root_path1)
        classes = os.listdir(root_path1)
        # print(classes)

        for class_ in classes:
            cnt_per_class = 0   

            root_path2 = os.path.join(root_path1, class_)     
            print(root_path2)
            # print(os.listdir(root_path2))
            # print(len(os.listdir(root_path2)))
            files = os.listdir(root_path2)

            for file in files:
                # print(os.path.join(root_path2, file))
                final_path = os.path.join(root_path2, file)
                # print(final_path[-3:])
                # print(file)
                if final_path[-3:] == 'png' and "labelIds" in final_path :
                    cnt_per_class += 1

            print(cnt_per_class, cnt_per_class/4)
            
