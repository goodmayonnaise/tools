## ref
* https://github.com/solomontesema/SIGtor?tab=readme-ov-file


## Data Structure 
```
.\data
├─new
│  ├─defect
│  │  ├─augmented_images
│  │  └─augmented_masks
│  └─labels
│      └─aug_defect
│          ├─augmented_images
│          └─augmented_masks
├─true_false
   ├─false
   └─true
```

## generate
```python sigtor\scripts\generate.py --config config.yaml --source_ann_file data\true_fasle\annotations.txt```
