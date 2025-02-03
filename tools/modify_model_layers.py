import torch 
pth_path = '/mmrotate/configs/checkpoint.pth'

pth = torch.load(pth_path)
dicts = pth['state_dict']
save_pth = pth
keys_to_remove = [key for key in dicts.keys() if 'head_module' in key]

# 변경 전 layer name 확인 필수 

for key in keys_to_remove:
    value = dicts[key]
    del dicts[key]
    new_key = f"bbox_head{key[21:]}"
    dicts[new_key] = value 
      
for key, value in pth['state_dict'].items():
    if 'head' in key:
        print(key)
        
torch.save(pth, 'rename_checkpoint.pth')
      
