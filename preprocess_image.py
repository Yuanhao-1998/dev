import json
import cv2
import albumentations as A
import os
import numpy as np
import matplotlib.pyplot as plt
import copy

dataset_path = r'c:\projects\oil-level-prediction\data\data_dev\mask_rcnn_acc'
ann_file = "annotation.json"
output_file = "annotation_aug.json"

with open(os.path.join(dataset_path, 'annotations', ann_file), 'r') as f:
    coco = json.load(f)
coco_aug = {'licenses': coco['licenses'], 'info': coco['info'], 'categories': coco['categories'], 'images': [], 'annotations': []}

# transform parameters
transform_flip = A.Compose([A.HorizontalFlip(p=1)], 
    bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']),
    )     # no MaskParams

transform_crop1 = A.Compose([A.CenterCrop(height=600, width=600, p=1)],
    bbox_params=A.BboxParams(format='coco', label_fields=['category_ids'])                   # no MaskParams
    )     # for image size 1066*600
transform_crop2 = A.Compose([A.CenterCrop(height=2040, width=2040, p=1)],
    bbox_params=A.BboxParams(format='coco', label_fields=['category_ids'])
    )     # for image size 2040*3064
transform_crop3 = A.Compose([A.CenterCrop(height=1080, width=1080, p=1)],
    bbox_params=A.BboxParams(format='coco', label_fields=['category_ids'])
    )     # for image size 2040*3064

'''
NOTE:
Albumentations supports two methods to match the bboxes and categories:
1. Assign 'label_fields=['category_ids']' in 'bbox_params' of 'A.Compose', and input 'bboxes=bboxes, category_ids=categories' in the transform function.
2. Insert categories to each bbox, the format of bboxes: [[514.4, 222.88, 66.58, 117.88, 1], [549.7, 295.8, 8.31, 39.4, 2]].
   don't need to specify 'label_fields'and only input bboxes

Only the second method is supported, when assign additional_targets of 'bboxes' type: 'imag_aug': 'image', 'bboxes_aug': 'bboxes', 'masks_aug': 'masks'.
Reason: additional_targets only supports 'image', 'bboxes', and 'masks' types, not 'category_ids', the label fields of the bboxes_aug cannot be specified in the transform function.
'''


for img_info in coco['images']:
    img_id = img_info['id']
    filename=img_info['file_name']
    
    img_path = os.path.join(dataset_path, 'images_orig', filename)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Image {filename} not exist or cannot be read.")
        continue
    
    # find annotations for the current image
    bboxes=[]
    categories=[]
    masks=[]
    
    for ann in coco['annotations']:
        if ann['image_id'] == img_id:
            bboxes.append(ann['bbox'])
            categories.append(ann['category_id'])
            
            # Create a mask from polygon segmentation
            mask = np.zeros(img.shape[0:2], dtype=np.uint8)
            for seg in ann['segmentation']:
                pts = np.array(seg).reshape(-1, 2)
                pts = pts.round().astype(np.int32)
                cv2.fillPoly(mask, [pts], 1)
            masks.append(mask)
    
    if not bboxes:
        print(f"No annotations for image {filename}")
        continue

    # Determine the appropriate crop transform based on image size
    if min(img.shape[:2]) == 600:
        transform_crop = transform_crop1
        print('using transform_crop1')
    elif min(img.shape[:2]) == 2040:
        transform_crop = transform_crop2
        print('using transform_crop2')
    elif min(img.shape[:2]) == 1080:
        transform_crop = transform_crop3
        print('using transform_crop3')
    else:
        raise ValueError(f"Unexpected image size {img.shape} for image {filename}")
    
    try:
        transformed = transform_flip(image=img, bboxes=bboxes, masks=masks, category_ids=categories)
        img_flip = transformed['image']
        bboxes_flip = transformed['bboxes']
        masks_flip = transformed['masks']
        #categories_flip = transformed['category_ids']
        
        transformed1 = transform_crop(image=img, bboxes=bboxes, masks=masks, category_ids=categories)
        transformed2 = transform_crop(image=img_flip, bboxes=bboxes_flip, masks=masks_flip, category_ids=categories)

        img_aug = [transformed1['image'], transformed2['image']]
        filename_aug = [filename, f'flipped_{filename}']
        bboxes_aug = [transformed1['bboxes'], transformed2['bboxes']]
        masks_aug = [transformed1['masks'], transformed2['masks']]
        #categories_aug = transformed2['category_ids']
        
        
        # Save the cropped images and create annotations
        output_path = os.path.join(dataset_path, 'images')
        os.makedirs(output_path, exist_ok=True)
        current_img_id = len(coco_aug['images']) + 1
        for i, img in enumerate(img_aug):
            cv2.imwrite(os.path.join(output_path, filename_aug[i]), img)
            coco_aug['images'].append({
                "id": current_img_id + i,
                "width": img.shape[1],
                "height": img.shape[0],
                "file_name": filename_aug[i],
                "license": 0,
                "flickr_url": "",
                "coco_url": "",
                "date_captured": 0
            })

            # Create new ann["annotations"]
            current_ann_id = len(coco_aug['annotations']) + 1
            for j in range(len(bboxes_aug[i])):
                # recover polygon segmentation from masks
                mask_aug = masks_aug[i][j].astype(np.uint8)
                contours, _ = cv2.findContours(mask_aug, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                segmentation = []
                for contour in contours:
                    contour = contour.flatten().tolist()
                    if len(contour) >= 6:   # polygon must have at least 3 points
                        segmentation.append(contour)
                bbox_aug= bboxes_aug[i][j]

                coco_aug['annotations'].append({
                    'id': current_ann_id + j,
                    'image_id': current_img_id + i,
                    'bbox': bbox_aug,
                    'category_id': categories[j],
                    'area': round(bbox_aug[2] * bbox_aug[3], 2),
                    'segmentation': segmentation, 
                    'iscrowd': 0,
                    "attributes": {"occluded": 'false'}
                })

    except Exception as e:
        print(f"Error processing image {filename}: {e}")
        continue

with open(os.path.join(dataset_path, 'annotations', output_file), 'w') as f:
    json.dump(coco_aug, f, indent=4, ensure_ascii=False)
