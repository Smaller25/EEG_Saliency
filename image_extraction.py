import os
import shutil

# 경로 설정
image_list_path = '/mnt/d/EEG_saliency_data/image_list.txt'
source_root = '/mnt/d/EEG_saliency_data/imagenet_raw/train'
target_root = '/mnt/d/EEG_saliency_data/imagenet_select'

# 이미지 리스트 로딩
with open(image_list_path, 'r') as f:
    target_ids = set(line.strip() for line in f if line.strip())

count = 0

# 모든 class 디렉토리 순회
for class_name in os.listdir(source_root):
    try:
        class_dir = os.path.join(source_root, class_name)
        if not os.path.isdir(class_dir):
            continue

        for fname in os.listdir(class_dir):
            img_id = os.path.splitext(fname)[0]  # 확장자 제거
            if img_id in target_ids:
                src_path = os.path.join(class_dir, fname)
                dst_class_dir = os.path.join(target_root, class_name)
                os.makedirs(dst_class_dir, exist_ok=True)
                dst_path = os.path.join(dst_class_dir, fname)
                shutil.copy2(src_path, dst_path)
                count += 1
    except:
        print(class_name)

print(f"[완료] 총 {count}개의 이미지를 복사했습니다.")