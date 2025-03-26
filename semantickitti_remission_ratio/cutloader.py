
import yaml, os, cv2
from glob import glob
import numpy as np
from einops import rearrange
from tqdm import tqdm 

import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torch.nn.functional import one_hot


class SemnaticKITTI(Dataset):
    def __init__(self, input_file_pathes, label_file_pathes, input_shape, swap_dict, num_cls, train_phase=True, **kwargs):
        # super(Dataset,self).__init__(**kwargs)
        self.remission_pathes = input_file_pathes
        self.label_pathes = label_file_pathes
        self.swap_dict = swap_dict
        self.num_cls = num_cls
        self.input_shape = input_shape     
        self.train_phase = train_phase

    def replace_with_dict(self, ar, dic):
        # Extract out keys and values
        k = np.array(list(dic.keys()))
        v = np.array(list(dic.values()))

        # Get argsort indices
        sidx = k.argsort()
        
        # Drop the magic bomb with searchsorted to get the corresponding
        # places for a in keys (using sorter since a is not necessarily sorted).
        # Then trace it back to original order with indexing into sidx
        # Finally index into values for desired output.
        return v[sidx[np.searchsorted(k,ar,sorter=sidx)]]     

    def __len__(self):
        return len(self.remission_pathes)

    def __getitem__(self, idx):

        x_rem = self.remission_pathes[idx]
        x_rem = np.load(x_rem)
        x_rem = np.array(x_rem)
        x_rem = cv2.resize(x_rem, (self.input_shape[1], self.input_shape[0]))
        x_rem = np.expand_dims(x_rem, axis=-1)
        x_rem = torch.FloatTensor(x_rem)
        x_rem = rearrange(x_rem, 'h w c -> c h w')
        x_rem = x_rem[:,118:,:]

        y_rem = self.label_pathes[idx]
        y_rem = np.load(y_rem)
        y_rem = cv2.resize(y_rem, (self.input_shape[1], self.input_shape[0]))
        y_rem = np.expand_dims(y_rem, axis=-1)
        y_rem = torch.FloatTensor(y_rem)
        y_rem = rearrange(y_rem, 'h w c -> c h w')
        y_rem = y_rem[:,118:,:]



        return {'X_rem':x_rem,'Y_rem':y_rem}




if __name__ == "__main__":
    semantickitti_path = '/mnt/team_gh/kitti/dataset/sequences'

    cfg_path = './config/semantic-kitti.yaml'
    try:
        print("Opening config file %s" % "config/semantic-kitti.yaml")
        CFG = yaml.safe_load(open(cfg_path, 'r'))
    except Exception as e:
        print(e)
        print("Error opening yaml file.")
        quit()
    learning_map = CFG['learning_map']
    sequences = CFG['split']["train"]
    sequences = [str(i).zfill(2) for i in sequences]

    rem_x_paths = [os.path.join(semantickitti_path, sequence_num, "projection_front", "remission") for sequence_num in sequences]
    rem_y_paths =  [os.path.join(semantickitti_path, sequence_num, "projection_front", "depth") for sequence_num in sequences]
    rem_x_names = []
    rem_y_names = []
    for rem_x_path, rem_y_path in zip(rem_x_paths, rem_y_paths):
        rem_x_names = rem_x_names + glob(str(os.path.join(os.path.expanduser(rem_x_path),"*.npy")))
        rem_y_names = rem_y_names + glob(str(os.path.join(os.path.expanduser(rem_y_path),"*.npy")))

    rem_x_names.sort()
    rem_y_names.sort()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    gpus = os.environ["CUDA_VISIBLE_DEVICES"]
    num_workers = len(gpus.split(",")) * 2

    device = "cuda" if torch.cuda.is_available() else 'cpu'
    
    torch.manual_seed(777) # 정확한 테스트를 위한 random seed 고정 
    if device == 'cuda':
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]= gpus 
        torch.cuda.manual_seed_all(777)
    

    dataset = SemnaticKITTI(input_file_pathes=rem_x_names, 
                            label_file_pathes=rem_y_names, 
                            input_shape=(384, 1280),
                            swap_dict=learning_map, 
                            num_cls=20, 
                            train_phase=True) 
    dataset_size = len(dataset)
    data_loader = DataLoader(dataset, num_workers=0, batch_size=1, shuffle=True)

    x_check, y_check = [], []
    # with torch.set_grad_enabled(True):
    for iter, batch in enumerate(tqdm(data_loader)):
        inputs_rem = Variable(batch['X_rem'].to(device))
        labels_rem = Variable(batch['Y_rem'].to(device))
