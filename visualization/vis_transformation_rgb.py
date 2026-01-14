import torch
import torchvision.transforms as transforms

import matplotlib.pyplot as plt 
from PIL import Image

def denormalize(tensor, mean, std):
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return tensor * std + mean

def visualize_step_by_step(img_path, transform_steps):
    img = Image.open(img_path).convert("RGB")
    current = img
    
    total = len(transform_steps) + 1
    n_cols = 4
    n_rows = (total + n_cols - 1) // n_cols
    plt.figure(figsize=(4 * n_cols, 3 * n_rows))
    plt.subplot(n_rows, n_cols, 1)
    plt.imshow(img)
    plt.title("Original", fontsize=8)
    plt.axis("off")
    
    for i, (name, tf) in enumerate(transform_steps):
        current = tf(current)

        if isinstance(current, torch.Tensor):
            vis = current
            if name == "Normalize" or "Erasing" in name:
                vis = denormalize(
                    vis,
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            vis = vis.permute(1, 2, 0).clamp(0, 1)
        else:
            vis = current
        
        plt.subplot(n_rows, n_cols, i+2)
        plt.imshow(vis)
        plt.title(name, fontsize=8)
        plt.axis("off")
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    image_size = 300
    transform_steps = [
        ("CenterCrop",
        transforms.CenterCrop(image_size)),
        ("RandomResizedCrop",
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.9, 1.1))),
        ("RandomHorizontalFlip", 
        transforms.RandomHorizontalFlip(p=1.0)),
        ("RandomVerticalFlip",
        transforms.RandomVerticalFlip(p=1.0)),
        ("RandomAffine", 
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))),
        ("ToTensor",
        transforms.ToTensor()),
        ("Normalize",
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )),
        ("RandomErasing",
        transforms.RandomErasing(p=1.0, scale=(0.02, 0.1)))
    ]

    # for _ in range(5):
    #     visualize_step_by_step(
    #         img_path=r"",
    #         transform_steps=transform_steps
    #     )
visualize_step_by_step(img_path=r"", transform_steps=transform_steps)

