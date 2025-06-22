import json
import cv2
import albumentations as A
import os
import numpy as np
import matplotlib.pyplot as plt
import copy

dataset_path = r'c:\projects\oil-level-prediction\test_data\mask_rcnn'
img_folder= "images"
ann_file = "annotation_aug.json"
output_file = "annotations_maskrcnn_aug1.json"

with open(os.path.join(dataset_path,'annotations', ann_file), 'r', encoding='utf-8') as f:
    coco = json.load(f)
coco_new= copy.deepcopy(coco)   #deepcopy to avoid modifying the original coco

# transform parameters
transform_flip = A.Compose([A.HorizontalFlip(p=1)], 
    bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))     # no MaskParams
transform_affine = A.Compose([
        A.ShiftScaleRotate(
            shift_limit=0.2,
            scale_limit=0.1,
            rotate_limit=3,
            p=1)], 
    bbox_params=A.BboxParams(format='coco'),            # no label_fields, because additional_targets is used
    additional_targets={'image_aug': 'image', 'bboxes_aug': 'bboxes', 'masks_aug': 'masks'}
    )     # no MaskParams


for img_info in coco['images']:
    img_id = img_info['id']
    filename=img_info['file_name']
    
    img_path = os.path.join(dataset_path, img_folder, filename)
    img = cv2.imread(img_path)
    if img is None:
        coco_new['images'].remove(img_info)            # remove the image info if the image cannot be read
        print(f"Image {filename} not exist or cannot be read.")
        continue
    
    # find annotations for the current image
    bboxes=[]
    masks=[]
    for ann in coco['annotations']:
        if ann['image_id'] == img_id:
            bboxes.append(ann['bbox']+ [ann['category_id']])  # insert category_id to bbox
            
            # create mask from polygon segmentation
            mask = np.zeros(img.shape[0:2], dtype=np.uint8)
            for seg in ann['segmentation']:
                pts = np.array(seg).reshape(-1, 2)
                pts = pts.astype(np.int32)
                cv2.fillPoly(mask, [pts], 1)
            masks.append(mask)

    try:
        transformed = transform_flip(image=img, bboxes=bboxes, masks=masks)
        img_flip = transformed['image']
        bboxes_flip = transformed['bboxes']
        masks_flip = transformed['masks']

        transformed = transform_affine(image=img_flip, bboxes=bboxes_flip, masks=masks_flip,
                    image_aug=img, bboxes_aug=bboxes, masks_aug=masks)
        img_aug = [img_flip, transformed['image'], transformed['image_aug']]
        filename_aug = [f'flipped_{filename}', f'affined_{filename}', f'affined_flipped_{filename}']
        bboxes_aug = [bboxes_flip, transformed['bboxes'], transformed['bboxes_aug']]
        masks_aug = [masks_flip, transformed['masks'], transformed['masks_aug']]
        
        current_img_id = coco_new['images'][-1]['id'] + 1
        for i, img in enumerate(img_aug):
            output_path = os.path.join(dataset_path, img_folder, filename_aug[i])
            cv2.imwrite(output_path, img)
            coco_new['images'].append({
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
            current_ann_id = coco_new['annotations'][-1]['id'] + 1
            for j in range(len(bboxes_aug[i])):
                # unpack category_id from bbox
                bbox_aug = bboxes_aug[i][j][:-1]
                category = bboxes_aug[i][j][-1]
                
                # recover polygon segmentation from masks
                mask_aug = masks_aug[i][j].astype(np.uint8)
                contours, _ = cv2.findContours(mask_aug, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                segmentation = []
                for contour in contours:
                    contour = contour.flatten().tolist()
                    if len(contour) >= 6:   # polygon must have at least 3 points
                        segmentation.append(contour)

                coco_new['annotations'].append({
                    'id': current_ann_id + j,
                    'image_id': current_img_id + i,
                    'bbox': bbox_aug,
                    'category_id': category,
                    'area': round(bbox_aug[2] * bbox_aug[3], 2),
                    'segmentation': segmentation, 
                    'iscrowd': 0,
                    "attributes": {"occluded": 'false'}
                })
            
    except Exception as e:
        print(f"Error processing image {img_info['file_name']}: {e}")
        continue

with open(os.path.join(dataset_path,'annotations', output_file), 'w', encoding='utf-8') as f:
    json.dump(coco_new, f, indent=4, ensure_ascii=False)
