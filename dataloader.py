import torch
from torch.utils.data import DataLoader

import random
from collections import defaultdict
from torchvision import transforms
from torch.utils.data import Sampler
from PIL import Image
import os


# Data augmentation
# ImageNet 정규화 기준
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

# Sampler : sampling different classes' items to one batch
class LabelBasedBatchSampler(Sampler):
    def __init__(self, labels, batch_size):
        """
        Args:
            labels (List[int] or Tensor): Class label of each image sample 
            batch_size (int): 배치 내에 포함될 클래스 수 (즉, 유니크 클래스 수 == 배치 크기)
        """
        self.labels = labels
        self.batch_size = batch_size
        self.class_to_indices = defaultdict(list)

        for idx, label in enumerate(labels):
            self.class_to_indices[label].append(idx)

        self.unique_classes = list(self.class_to_indices.keys())
        assert batch_size <= len(self.unique_classes), "Batch size can't be bigger than unique number of classes"

    def __iter__(self):
        random.shuffle(self.unique_classes)
        batches = []

        for i in range(0, len(self.unique_classes), self.batch_size):
            batch_classes = self.unique_classes[i:i+self.batch_size]
            if len(batch_classes) < self.batch_size:
                continue  # batch size 도달 못하는 경우 제외
            batch_indices = []
            for cls in batch_classes:
                idx = random.choice(self.class_to_indices[cls])
                batch_indices.append(idx)
            batches.append(batch_indices)

        return iter(batches)

    def __len__(self):
        return len(self.unique_classes) // self.batch_size


# Dataset
class EEGDataset:
    def __init__(self, eeg_signals_path, image_root='images/', subject=0, time_low=20, time_high=460, image_transform=None):
        loaded = torch.load(eeg_signals_path)
        if subject != 0:
            self.data = [d for d in loaded['dataset'] if d['subject'] == subject]
        else:
            self.data = loaded['dataset']
        self.labels = loaded["labels"]
        self.images = loaded["images"]
        self.time_low = time_low
        self.time_high = time_high
        self.image_root = image_root
        self.image_transform = image_transform or transforms.Compose([
            transforms.Resize((330, 330)),                # 먼저 리사이즈
            transforms.RandomCrop(299),                   # 랜덤 crop (하나만)
            transforms.RandomHorizontalFlip(p=0.5),       # 50% 확률로 좌우 반전
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
        ])
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        eeg = self.data[i]["eeg"].float().t()
        eeg = eeg[self.time_low:self.time_high, :]
        label = self.data[i]['label']
        image_index = self.images[self.data[i]['image']]
        
        # 이미지 로딩
        image_path = os.path.join(self.image_root, f"{image_index}.JPEG")
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.image_transform(image)

        return eeg, label, image_index, image_tensor

# split into train/test/val with pre-defined index list 
class Splitter:
    def __init__(self, dataset, split_path, split_num=0, split_name="test"):
        self.dataset = dataset
        loaded = torch.load(split_path)
        self.split_idx = loaded["splits"][split_num][split_name]
        self.split_idx = [i for i in self.split_idx if 450 <= self.dataset.data[i]["eeg"].size(1) <= 600]

    def __len__(self):
        return len(self.split_idx)

    def __getitem__(self, i):
        eeg, label, image_index, image_tensor = self.dataset[self.split_idx[i]]
        return eeg, label, image_index, image_tensor
    
# Dataloader 
#############################################
# dataset, dataloader 코드
# Q1.2 code 변형 연습 - dataloder 변형하여 사용
#############################################
def get_eeg_test_dataloader(eeg_path, split_path, subject=0, split_name='test', 
                            batch_size=1, time_low=20, time_high=460, image_root='images/'):
    dataset = EEGDataset(
        eeg_signals_path=eeg_path,
        image_root=image_root,
        subject=subject,
        time_low=time_low,
        time_high=time_high
    )
    test_split = Splitter(dataset, split_path, split_num=0, split_name=split_name)
    loader = DataLoader(test_split, batch_size=batch_size, shuffle=False, num_workers=0)
    return loader

# sampling 방식 변경 : 논문의 방식으로 sampler 추가함 
def get_eeg_train_dataloader(eeg_path, split_path, subject=0, split_name='train', 
                             batch_size=10, time_low=20, time_high=460, image_root='images/',sampler='original'):
    
    dataset = EEGDataset(
        eeg_signals_path=eeg_path,
        image_root=image_root,
        subject=subject,
        time_low=time_low,
        time_high=time_high
    )

    split_dataset = Splitter(dataset, split_path, split_num=0, split_name=split_name)

    if sampler == 'original':
        # 여기서 split 이후 label들을 추출함
        labels = [split_dataset[i][1] for i in range(len(split_dataset))]
        batch_sampler = LabelBasedBatchSampler(labels, batch_size)
        loader = DataLoader(split_dataset, batch_sampler=batch_sampler, num_workers=0)
    else:
        raise ValueError("sampler must be 'original', anything else isn't implemented")

    return loader

