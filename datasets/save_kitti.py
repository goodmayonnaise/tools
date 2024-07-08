import os, yaml, cv2
import numpy as np 
from einops import rearrange

from torch.utils.data import Dataset


class SemanticKITTI(Dataset):
    
    def __init__(self, path, mode, h, w, split):
        self.path = path
        self.mode = mode
        self.split = split
        self.load_path()
        CFG = self.load_config()

        color_dict = CFG['color_map']
        self.learning_map = CFG['learning_map']
        self.learning_map_inv = CFG['learning_map_inv']
        self.color_dict = {self.learning_map[key]:color_dict[self.learning_map_inv[self.learning_map[key]]] for key, value in color_dict.items()}

            
    def __len__(self):
        return len(self.rgb_paths)
        
    def find_dir(self, dir, path):
        if 'unproj' in dir:
            self.unproj_range_paths.append(path)
        elif 'proj_range' in dir:
            self.proj_range_paths.append(path)
        elif 'npoint' in dir:
            self.npoint_paths.append(path)
        elif 'mask' in dir:
            self.proj_mask_paths.append(path)
        elif 'rdm' in dir:
            self.proj_rdm_paths.append(path)
        elif 'remission' in dir:
            self.proj_remission_paths.append(path)
        elif 'pad' in dir:
            self.proj_sem_label_pad_paths.append(path)
        elif 'color' in dir:
            self.proj_sem_color_paths.append(path)
        elif 'proj_sem_label' in dir :
            self.proj_sem_label_paths.append(path)
        elif 'proj_xyz' in dir:
            self.proj_xyz_paths.append(path)
        elif 'proj_x' in dir:
            self.proj_x_paths.append(path)
        elif 'proj_y' in dir:
            self.proj_y_paths.append(path)
        elif 'sem_label' in dir:
            self.sem_label_paths.append(path)

    def load_config(self):
        cfg_path = './semantic-kitti.yaml'
        try:
            print("Opening config file %s" % cfg_path)
            CFG = yaml.safe_load(open(cfg_path, 'r'))
        except Exception as e:
            print(e)
            print("Error opening yaml file.")
            quit()
        return CFG
     
    def load_path(self):

        if self.mode == 'train':
            self.seqs =  [str(i).zfill(2) for i in range(11)]
            self.seqs.remove('08') 
        elif self.mode == 'val':
            self.seqs = ['08']
        elif self.mode == 'test':
            self.seqs = [str(i).zfill(2) for i in range(11,22)]
            
        # seqs = sorted(os.listdir(path))
        dirs = sorted(os.listdir(os.path.join(self.path, self.seqs[0])))
        rgbdir = dirs[0]
        otherdir = dirs[1:]
        
        # other_npys = sorted(os.listdir(os.path.join(self.path,seqs[0], dirs[-1])))
                
        self.rgb_paths = []
        self.proj_rdm_paths = []
        self.proj_remission_paths = []
        self.proj_range_paths = []
        self.proj_mask_paths = []
        self.proj_xyz_paths = []
        self.unproj_range_paths = []
        self.proj_x_paths = []
        self.proj_y_paths = []
        self.npoint_paths = []
        if self.mode != 'test':
            self.proj_sem_label_paths = []
            self.proj_sem_label_pad_paths = []
            self.proj_sem_color_paths = []
            self.sem_label_paths = []
                
        for seq in self.seqs:
            rgb_npys = sorted(os.listdir(os.path.join(self.path, seq, dirs[0])))

            for rgb in rgb_npys:
                self.rgb_paths.append(os.path.join(self.path, seq, rgbdir, rgb))
                for dir in otherdir:
                    for i in range(self.split):
                        p = os.path.join(self.path, seq, dir, rgb).replace('.', f'_{i}.')
                        self.find_dir(dir, p)
        
        self.rgb_paths.sort()
        self.proj_rdm_paths.sort()
        self.proj_remission_paths.sort()
        self.proj_range_paths.sort()
        self.proj_mask_paths.sort()
        self.proj_xyz_paths.sort()
        self.unproj_range_paths.sort()
        self.proj_x_paths.sort()
        self.proj_y_paths.sort()
        self.npoint_paths.sort()
        
    def convert_color(self, arr):
        result = np.zeros((*arr.shape, 3))
        for c in range(20):
            j = np.where(arr==c) # coord x, y
            try:
                xs, ys = j[0], j[1]
            except:
                xs = j[0]
            rgb = self.color_dict[c] # rgb 

            if len(xs) == 0:
                continue
            for x, y in zip(xs, ys):
                result[x, y] = rgb
        return result
    
    def __getitem__(self, idx) :
        
        rgb = np.load(self.rgb_paths[idx])
        rgb = rearrange(rgb, 'h w c -> c h w')
        proj_rdm = np.stack([np.load(self.proj_rdm_paths[idx]) for i in range(self.split)],0) # 5 3 384 1248
        proj_sem_label_pad = np.stack([np.load(self.proj_sem_label_pad_paths[idx+i]) for i in range(self.split)], 0)
        
        return {'rgb':rgb, 'proj_rdm':proj_rdm, 'proj_sem_label_pad':proj_sem_label_pad}
        

if __name__=="__main__":
    h, w, split = 384, 1248, 5

    path = f'dataset/sequences_{h}x{w*split}_split{split}/'
    mode = 'train'
    # dirs = ['rgb', 'proj_rdm', 'proj_remission', 'proj_range', 'proj_mask',
    #         'proj_xyz', 'unproj_range', 'proj_x', 'proj_y', 'npoint',
    #         'proj_sem_label', 'proj_sem_label_pad', 'proj_sem_color', 'sem_label'] # test X
    
    dataset = SemanticKITTI(path, mode, h, w, split)
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset=dataset, batch_size=1, num_workers=0, shuffle=False)
    
    for idx, batch in enumerate(loader):
        print(batch)
    
     
     
        
