import json
import os

ann_path=r'c:\projects\oil-level-prediction\test_data\annotations'
ann_file = "instances_default.json"
output_file = "instances_mask.json"

# Load the JSON file
with open(os.path.join(ann_path, ann_file), 'r', encoding='utf-8') as f:
    data = json.load(f)
with open(os.path.join(ann_path, output_file), 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4, ensure_ascii=False)


#json.dump(data, f, indent=4, ensure_ascii=False, sort_keys=True) #对key排序
#json.dumps  转换成字符串
 