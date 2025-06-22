import os
import shutil

# 경로 설정
image_list_path = '/mnt/d/EEG_saliency_data/image_list.txt'
source_root = '/mnt/d/EEG_saliency_data/imagenet_raw/train'
target_root = '/mnt/d/EEG_saliency_data/imagenet_select'

with open(image_list_path, 'r') as f:
    target_ids = [line.strip() for line in f if line.strip()]  # 예: ['n01440764_18', 'n02951358_31190', ...]

count = 0
missing = []

for img_id in target_ids:
    class_name = img_id.split('_')[0]
    fname = f"{img_id}.JPEG"
    src_path = os.path.join(source_root, class_name, fname)
    dst_path = os.path.join(target_root, fname)

    if os.path.exists(src_path):
        os.makedirs(target_root, exist_ok=True)
        shutil.copy2(src_path, dst_path)
        count += 1
    else:
        missing.append(img_id)

print(f"{count} files copied.")
if missing:
    print(f"⚠️ {len(missing)} files missing:", missing[:10], "...")