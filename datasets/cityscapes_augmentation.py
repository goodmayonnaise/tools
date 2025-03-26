import os, csv, random
from torchvision import utils
from torchvision import transforms as T
import albumentations
import torch
def data_path_load(root_path="cityscapes_origin_aug", phase="train"): 

    img_data_dir = os.path.join(root_path, "leftImg8bit")
    gt_data_dir = os.path.join(root_path, "gtFine")
    
    if phase == "train":
        img_train_dir = os.path.join(img_data_dir, "train")
        img_train_fns = os.listdir(img_train_dir)
        
        # img_val_dir = os.path.join(img_data_dir, "val")
        # img_val_fns = os.listdir(img_val_dir)   
        
        gt_train_dir = os.path.join(gt_data_dir, "train")
        gt_train_fns = os.listdir(gt_train_dir)
        
        # gt_val_dir = os.path.join(gt_data_dir, "val")
        # gt_val_fns = os.listdir(gt_val_dir)

        # img_train_paths, img_val_paths = [], []
        img_train_paths = []

        for train_fn in img_train_fns:
            filenames = os.listdir(os.path.join(img_train_dir, train_fn))
            for filename in filenames:
                img_train_paths.append(os.path.join(img_data_dir, img_train_dir, train_fn, filename))

        # for val_fn in img_val_fns:
        #     filenames = os.listdir(os.path.join(img_val_dir, val_fn))
        #     for filename in filenames:
        #         img_val_paths.append(os.path.join(img_data_dir, img_val_dir, val_fn, filename))
        
        # gt_train_paths, gt_val_paths = [], []
        gt_train_paths = []
        for train_fn in gt_train_fns:
            filenames = os.listdir(os.path.join(gt_train_dir, train_fn))
            for filename in filenames:
                if 'labelIds.png' in filename:
                    gt_train_paths.append(os.path.join(gt_data_dir, gt_train_dir, train_fn, filename))

        # for val_fn in gt_val_fns:
        #     filenames = os.listdir(os.path.join(gt_val_dir, val_fn))
        #     for filename in filenames:
        #         if 'labelIds.png' in filename:
        #             gt_val_paths.append(os.path.join(gt_data_dir, gt_val_dir, val_fn, filename))
        # return img_train_paths, img_val_paths, gt_train_paths, gt_val_paths
        return img_train_paths, gt_train_paths 

    elif phase == "test":

        img_test_dir = os.path.join(img_data_dir, "test")
        img_test_fns = os.listdir(img_test_dir)
        gt_test_dir = os.path.join(gt_data_dir, "test")
        gt_test_fns = os.listdir(gt_test_dir)

        img_test_paths, gt_test_paths = [], []
        for test_fn in img_test_fns:
            filenames = os.listdir(os.path.join(img_test_dir, test_fn))
            for filename in filenames:
                img_test_paths.append(os.path.join(img_data_dir, img_test_dir, test_fn, filename))

        for test_fn in gt_test_fns:
            filenames = os.listdir(os.path.join(gt_test_dir, test_fn))
            for filename in filenames:
                gt_test_paths.append(os.path.join(gt_data_dir, gt_test_dir, test_fn, filename))

        return img_test_paths, gt_test_paths


def save_ft(data_path, data, augmentation):
    first_path = '/home/jyjeon/sota'
    folder_name = 'cityscapes_origin_aug'
    file_name = data_path.split('/')[-1][:-4]+'_'+augmentation+data_path.split('/')[-1][-4:]
    mid_path = data_path.split('/')[4:-1]
    mid_path = '/'.join(mid_path)
    save_path = os.path.join(first_path, folder_name, mid_path, file_name)

    # dir_path2 = os.path.dirname(data_path)[24:]
    # file_name = os.path.basename(data_path)
    # final_path = os.path.join(first_path, folder_name, augmentation, dir_path2)
    # full_path = os.path.join(final_path, file_name)
    # make path

    if augmentation != "norm":
        print()
    try:
        if not os.path.exists('/'.join(save_path.split('/')[:-1])):
            os.makedirs('/'.join(save_path.split('/')[:-1]))
    except:
        print('exist file folder')
    # utils.save_image(data, './test1.png')
    utils.save_image(data, save_path)
    return 

def aug_ft(data, augmentation):
    if augmentation == 'center_crop':
        center_crop = T.Compose([T.ToTensor(), T.CenterCrop((256,512))])
        augmented_data = center_crop(data)
    # elif augmentation == 'horizontal_flip':
    #     Horizontal_Flip = albumentations.HorizontalFlip(p=1)
    #     Totensor = T.Compose([T.ToTensor()])
    #     augmented_data = Totensor(Horizontal_Flip(image=data)['image'])
    elif augmentation == 'vertical_flip':
        Vertical_Flip = albumentations.VerticalFlip(p=1)
        Totensor = T.Compose([T.ToTensor()])
        augmented_data = Totensor(Vertical_Flip(image=data)['image'])
    elif augmentation == 'norm':
        # norm_data = albumentations.normalize(img=data, mean=np.mean(data), std=np.std(data), max_pixel_value=1)
        Totensor = T.Compose([T.ToTensor()])
        tensor_data = Totensor(data)
        norm = T.Compose([T.Normalize(mean=(torch.mean(tensor_data[0,:,:]), torch.mean(tensor_data[1,:,:]), torch.mean(tensor_data[2,:,:])), 
                                      std=(torch.std(tensor_data[0,:,:]), torch.std(tensor_data[1,:,:]), torch.std(tensor_data[2,:,:])))])
        augmented_data = norm(tensor_data)
    return augmented_data

def to_csv(phase='train'):
    root_dir = "cityscapes_origin_aug"

    # Image dir
    orign_img = os.path.join(root_dir, "leftImg8bit")
    # crop_img = os.path.join(root_dir, "leftImg8bit_crop")
    # hrzntl_img = os.path.join(root_dir, "leftImg8bit_hrzntl")
    # norm_img = os.path.join(root_dir, "leftImg8bit_norm")
    
    # label dir
    orign_label = os.path.join(root_dir, "gtFine_regen")
    # crop_label = os.path.join(root_dir, "gtFine_crop")
    # hrzntl_label = os.path.join(root_dir, "gtFine_hrzntl")
    # norm_label = os.path.join(root_dir, "gtFine_norm")

    org_img_list, org_lab_list = [], []
    for cityname in os.listdir(os.path.join(orign_img, phase)):
        img_city_path = os.path.join(orign_img, phase, cityname)
        label_city_path = os.path.join(orign_label, phase, cityname)
        print(len(os.listdir(img_city_path)))
        for file_name, label_file_name in zip(os.listdir(img_city_path), os.listdir(label_city_path)):
            org_img_list.append(os.path.join(orign_img, phase, cityname, file_name))
            org_lab_list.append(os.path.join(orign_label, phase, cityname, label_file_name))
            org_img_list.sort()
            org_lab_list.sort()

    # crop_img_list, crop_lab_list = [], []
    # for cityname in os.listdir(os.path.join(crop_img, phase)):
    #     img_city_path = os.path.join(crop_img, phase, cityname)
    #     label_city_path = os.path.join(crop_label, phase, cityname)
    #     for file_name, label_file_name in zip(os.listdir(img_city_path), os.listdir(label_city_path)):
    #         crop_img_list.append(os.path.join(crop_img, phase, cityname, file_name))
    #         crop_lab_list.append(os.path.join(crop_label, phase, cityname, label_file_name))
    #         crop_img_list.sort()
    #         crop_lab_list.sort()

    # hrzntl_img_list, hrzntl_label_list = [], []
    # for cityname in os.listdir(os.path.join(hrzntl_img, phase)):
    #     img_city_path = os.path.join(hrzntl_img, phase, cityname)
    #     label_city_path = os.path.join(hrzntl_label, phase, cityname)
    #     for file_name, label_file_name in zip(os.listdir(img_city_path), os.listdir(label_city_path)):
    #         hrzntl_img_list.append(os.path.join(hrzntl_img, phase, cityname, file_name))
    #         hrzntl_label_list.append(os.path.join(hrzntl_label, phase, cityname, label_file_name))
    #         hrzntl_img_list.sort()
    #         hrzntl_label_list.sort()
    
    # norm_img_list, norm_label_list = [], []
    # for cityname in os.listdir(os.path.join(norm_img, phase)):
    #     img_city_path = os.path.join(norm_img, phase, cityname)
    #     label_city_path = os.path.join(norm_label, phase, cityname)
    #     for file_name, label_file_name in zip(os.listdir(img_city_path), os.listdir(label_city_path)):
    #         norm_img_list.append(os.path.join(norm_img, phase, cityname, file_name))
    #         norm_label_list.append(os.path.join(norm_label, phase, cityname, label_file_name))
    #         norm_img_list.sort()
    #         norm_label_list.sort()
    
    
    org_train_list, crop_train_list, hrz_train_list, norm_train_list = [],[],[],[]
    for img, label in zip(org_img_list, org_lab_list):
        org_train_list.append([img, label])
    random.shuffle(org_train_list)
    # for img, label in zip(crop_img_list, crop_lab_list):
    #     crop_train_list.append([img, label])
    # random.shuffle(crop_train_list)
    # for img, label in zip(hrzntl_img_list, hrzntl_label_list):
    #     hrz_train_list.append([img, label])
    # random.shuffle(hrz_train_list)
    # for img, label in zip(norm_img_list, norm_label_list):
    #     norm_train_list.append([img, label])
    # random.shuffle(norm_train_list)

    train_list = []
    for i in org_train_list:
        train_list.append(i)
    # for i in crop_train_list:
    #     train_list.append(i)
    # for i in hrz_train_list:
    #     train_list.append(i)
    # for i in norm_train_list:
    #     train_list.append(i)

    with open('./city_train_org.csv', 'w', newline='') as f:
        wirter = csv.writer(f)
        for i in train_list:
            wirter.writerow(i)
    return 

if __name__ == '__main__':
    augmentation = ['norm', 'center_crop', 'vertical_flip'] # center_crop or HorizontalFlip or HorizontalFlip***
    # img_train_paths, img_val_paths, gt_train_paths, gt_val_paths = data_path_load(phase='train')
    img_train_paths, gt_train_paths= data_path_load(phase='train')
   
    import cv2
    from tqdm import tqdm
    for i in augmentation:
        print(i)
        for path in tqdm(img_train_paths):
            data = cv2.imread(path)
            if data.max() == 0:
                print()
            aug_data = aug_ft(data, i)
            save_ft(path, aug_data, i)
            
        # for path in img_val_paths:
        #     data = cv2.imread(path) 
        #     aug_data = aug_ft(data, i)
        #     save_ft(path, aug_data, i)

        for path in tqdm(gt_train_paths):
            data = cv2.imread(path) 
            aug_data = aug_ft(data, i)
            save_ft(path, aug_data, i)
            
        # for path in gt_val_paths:
        #     data = cv2.imread(path)
        #     aug_data = aug_ft(data, i)
        #     save_ft(path, aug_data, i)
            
        
    # to_csv()
    
    
