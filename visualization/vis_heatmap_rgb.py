import argparse
import os
import PIL.Image as Image

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

import numpy as np
import cv2 

from efficientnet_pytorch import EfficientNet

class ImageFolderWithPath(ImageFolder):
    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        path, _ = self.samples[index]
        return image, target, path
    

feature_maps = {}

def save_feature(name):
    def hook(module, input, output):
        feature_maps[name] = output.detach()

    return hook

parser = argparse.ArgumentParser()
parser.add_argument('data', type=str)
parser.add_argument('--arch', default="efficientnet-b3")
parser.add_argument('--weight', type=str, required=True)
parser.add_argument('--batch-size', type=int, default=1)
parser.add_argument('--gpu', type=int, default=0)
# parser.add_argument('--out', type=str, default='heatmap')

def main():
    args = parser.parse_args()
    args.out = os.path.join(*args.weight.split('/')[:-1], *args.data.split('/'))
    os.makedirs(args.out, exist_ok=True)
    model = EfficientNet.from_name(args.arch)
    model._fc = nn.Linear(model._fc.in_features, 2)
    model = nn.DataParallel(model)

    checkpoint = torch.load(args.weight, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'], strict=True)

    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)
    model.eval()

    # vis feature setting
    block_list = [2, 6, 12]
    block_names = []
    for idx in block_list:
        model.module._blocks[idx].register_forward_hook(
            save_feature(f"block{idx}")
        )
        block_names.append(f"block{idx}")
    model.module._conv_head.register_forward_hook(save_feature("conv_head"))
    block_names.append("conv_head")

    cudnn.benchmark = True

    image_size = EfficientNet.get_image_size(args.arch)
    
    transform = transforms.Compose([
        transforms.Resize(image_size, interpolation=Image.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    dataset = ImageFolderWithPath(args.data, transform)
    dataset.class_to_idx = {"defect":0, "false":1}

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    idx_to_class = {v: k for k, v in loader.dataset.class_to_idx.items()}


    with torch.no_grad():
        for image, target, path in loader:
            image = image.cuda(args.gpu)

            logits = model(image)
            preds = logits.argmax(1)
            
            image, pred, target, path = image[0], preds[0], target[0], path[0]
            
            # path =
            img = Image.open(path).convert("RGB")
            img = np.array(img.resize((image_size, image_size)))
            
            results = [img]
            
            for bname in block_names:
                
                fmap = feature_maps[bname]
                fmap = fmap.mean(1)
                fmap = fmap.cpu().numpy()
                
                cam = fmap[0]
                cam = np.maximum(cam, 0)
                cam = cam / (cam.max() + 1e-6)
                cam = cv2.resize(cam, (image_size, image_size))

                heatmap = np.uint8(255 * cam)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

                overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
                
                # test
                label = f"Layer: {bname}"
                gt_label = f"GT: {idx_to_class[target.item()]}"
                pred_label = f"Pred: {idx_to_class[pred.item()]}"

                h, w, _ = overlay.shape
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale, thickness = 0.45, 1
                (layer_w, layer_h), baseline1 = cv2.getTextSize(
                    label, font, font_scale, thickness
                )
                (gt_w, gt_h), baseline2 = cv2.getTextSize(
                    gt_label, font, font_scale, thickness
                )
                (pred_w, pred_h), baseline3 = cv2.getTextSize(
                    pred_label, font, font_scale, thickness
                )

                x_layer, x_gt, x_pred = (w-layer_w)//2, (w-gt_w)//2, (w-pred_w)//2
                y_pred = h - 5 
                y_gt = y_pred - pred_h - 5
                y_layer = y_gt - layer_h - 5

                cv2.rectangle(
                    overlay, 
                    (x_layer-2, y_layer-layer_h-2),
                    (x_layer+layer_w+2, y_layer+baseline1+2),
                    (0,0,0), -1
                )
                cv2.rectangle(
                    overlay, (x_gt-2, y_gt-gt_h-2),
                    (x_gt+gt_w+2, y_gt+baseline2+2),
                    (0,0,0), -1
                )
                cv2.rectangle(
                    overlay,  (x_pred-2, y_pred-pred_h-2),
                    (x_pred+pred_w+2, y_pred+baseline3+2),
                    (0,0,0),-1
                )
                cv2.putText(
                    overlay, label, (x_layer, y_layer),
                    font, font_scale, (255,255,255), thickness,
                    cv2.LINE_AA
                
                )
                cv2.putText(
                    overlay, gt_label, (x_gt,y_gt), font,
                    font_scale, (255,255,255), thickness,
                    cv2.LINE_AA
                )
                cv2.putText(
                    overlay, pred_label, (x_pred,y_pred),
                    font, font_scale, (255,255,255), thickness,
                    cv2.LINE_AA
                )
                results.append(overlay[:, :, ::-1])

            fname = os.path.basename(path)
            out_path = os.path.join(args.out, fname)
            results = np.concatenate(results, 1)
            cv2.imwrite(out_path, results)
            print(f"[SAVE] {out_path}\t| GT : {target.item()} PRED : {pred.item()} | correct : {target.item()==pred.item()}")


    print("Done.")
                

if __name__ == "__main__":
    main()
