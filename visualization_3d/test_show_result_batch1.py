import os, cv2
import numpy as np
from time import time, sleep
from einops import rearrange

from utils import AverageMeter, ProgressMeter, set_device, load_config, colorize_sample
from metrics import iou, pixel_acc, uv_iou
from losses import total_loss
from model import new_model
from DataLoaders.SemanticKITTI import load_semanticKITTI

import torch
from torch.optim import lr_scheduler, Adam


def test(test_loader, model, use_gpu, criterion, device):
    batch_time = AverageMeter('test time', ':6.3f')
    loss_running = AverageMeter('test Loss', ':.4f')
    iou_running = AverageMeter('test mIoU', ':.4f')
    acc_running = AverageMeter('test pAcc', ':.4f')
    uv_running = AverageMeter('test_uv_mIoU',':.4f')
    progress = ProgressMeter(
                             len(test_loader),
                             [batch_time, loss_running, acc_running, iou_running, uv_running],
                             prefix=f"epoch {0+1} Test")
                            
    uv_ious = []
    
    state = torch.load('./weights/best_weight/pytorch_model.bin')
    model.load_state_dict(state)
    
    with torch.no_grad():
        end = time()
        model.eval()
        for iter, batch in enumerate(test_loader):
            # data_time.update(time()-end)
            if use_gpu:
                    inputs = batch['X'].to(device)
                    labels = batch['segment_label'].to(device)
                    inputs_rem = batch['X_rem'].to(device)
                    uv_labels = batch['uv_label'].to(device)
                    us, vs = batch['u'], batch['v']
                    label = batch['label']
            else:
                inputs, labels = batch['X'], batch['segment_label']
                inputs_rem, uv_labels = batch['X_rem'], batch['uv_label']
                
            result_rem = inputs_rem[0][0].cpu().numpy() # 1 256 1280
            
            result_rem2 = cv2.normalize(result_rem2, None, 0, 1000, cv2.NORM_MINMAX)
            sleep(0.001)   
            
            cv2.imwrite('./samples/rem.png', result_rem)

            segment_out, uv_out = model(inputs, inputs_rem)
            
            if save_result:
                result_2d = segment_out[0].cpu().numpy()
                result_2d = np.argmax(result_2d, axis=0)
                show_2d = np.zeros([*result_2d.shape, 3])
                for c in color_map.keys():
                    show_2d[result_2d==c] = color_map[c][::-1]
                
                cv2.imwrite(f"./samples/rgb_out.png", show_2d)
                
                result_3d = uv_out[0].cpu().numpy()
                result_3d = rearrange(result_3d, 'c (h w) -> c h w', h=256)
                result_3d = np.argmax(result_3d, axis=0)
                
                show_3d = np.zeros([*result_3d.shape])
                for x, y in zip(us, vs):
                    show_3d[y, x] = result_3d[y, x]
                
                uv_pred, uv_label = colorize_sample(show_3d, us[0].cpu().numpy(), vs[0].cpu().numpy(), label[0].cpu().numpy(), CFG)
                cv2.imwrite(f"./samples/uv_pred.png", uv_pred)
                cv2.imwrite(f"./samples/uv_label.png", uv_label)
            
            loss = criterion(segment_out, labels, uv_out, uv_labels)
            bs = inputs.size(0) # current batch size
            loss = loss.item()
            loss_running.update(loss, bs)

            batch_time.update(time() - end)
            end = time()

            if iter%50==0:
                progress.display(iter)

            miou = iou(segment_out, labels)
            iou_running.update(miou)  
            
            p_acc = pixel_acc(segment_out, labels)
            acc_running.update(p_acc) 

            uv_ious = uv_iou(uv_out, uv_labels)
            uv_running.update(uv_ious)

            # gpu memory 비우기 
            del batch
            torch.cuda.empty_cache()

    print('\ntest loss {:.4f} | test pAcc {:.4f}  | test miou {:.4f} | test uv miou {:.4f} '.format(loss_running.avg, acc_running.avg, iou_running.avg, uv_running.avg))   
  
if __name__ == "__main__":
    # -------------------------------------------------------------------------- setting parameter ---------------------------------------------------------------------------
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    gpus = os.environ["CUDA_VISIBLE_DEVICES"]
    num_workers = len(gpus.split(",")) * 2
    device = set_device(gpus) 
    use_gpu = torch.cuda.is_available()
    # use_gpu = False
    num_gpu = list(range(torch.cuda.device_count()))

    fusion_lev = "mid_stage2" # none, early, mid_stage1~4, late 
    save_result = True

    num_workers = 0

    phase = "test" # train /transfer_learning / test 
    freeze_cnt = 150
    dataset = "semantic_kitti" # cityscapes / kitti / city_kitti / semantic_kitti
    n_class    = 20
    input_shape = (256, 1280) # 96 312
    criterion = total_loss(reduction="mean") # setting loss 
    transfer_learning = False # 2D detach
    epochs     = 500
    lr         = 1e-4
    momentum   = 0
    w_decay    = 1e-5
    step_size  = 50
    gamma      = 0.5
    
    CFG = load_config()
    learning_map = CFG['learning_map']
    learning_map_inv = CFG['learning_map_inv']
    color_map = CFG['color_map']
    learning_map = CFG['learning_map']
    sequences = CFG['split'][phase]
    sequences = [str(i).zfill(2) for i in sequences]


    
    print("test data loading...")

    batch_size = 1
    configs    = "{}_batch{}_epoch{}_Adam_scheduler-step{}-gamma{}_lr{}_momentum{}_w_decay{}".format(dataset, batch_size, epochs, step_size, gamma, lr, momentum, w_decay)
    semantickitti_path = '/mnt/team_gh/kitti/dataset/sequences'
    test_loader = load_semanticKITTI(batch_size, phase, semantickitti_path, num_workers, input_shape, learning_map, sequences)

    model = new_model(input_shape=input_shape)

    if use_gpu:
        ts = time()
        model = model.cuda()
        # model = nn.DataParallel(model, device_ids=num_gpu)
        print("Finish cuda loading, time elapsed {}".format(time() - ts))
    
    optimizer = Adam(model.to(device).parameters(), lr=lr)
    # optimizer = Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer,gamma=0.99)
    
    test(test_loader, model, use_gpu, criterion, device)           

   
