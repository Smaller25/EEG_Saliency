import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
from models import EEGNet, Inception_ImageEncoder
from sklearn.metrics import roc_auc_score
import argparse
from PIL import Image
from torchvision import transforms
import time
from torch.utils.data import Sampler
import random
from collections import defaultdict

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
        label = self.data[i]["label"]
        image_index = self.data[i]["image"]
        
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

    if sampler = 'original':
        # 여기서 split 이후 label들을 추출함
        labels = [split_dataset[i][1] for i in range(len(split_dataset))]
        batch_sampler = LabelBasedBatchSampler(labels, batch_size)
        loader = DataLoader(test_split, batch_sampler=sampler, batch_size=batch_size, shuffle=False, num_workers=0)
    else:
        raise ValueError("sampler must be 'original', anything else isn't implemented")

    return loader



######################################################
######################################################
# utils : compatibility (dot product)
def compatibility(eeg_embed, img_embed):
    return torch.sum(eeg_embed * img_embed, dim=1)

# utils : compute_saliency (with multi-scale)
def compute_saliency(eeg, image, eeg_enc, img_enc, scale=16):
    device = image.device
    eeg_embed = eeg_enc(eeg)
    full_embed = img_enc(image)
    full_score = compatibility(eeg_embed, full_embed)

    _, _, H, W = image.shape
    saliency_map = torch.zeros((H, W), device=device)

    for y in range(0, H, scale):
        for x in range(0, W, scale):
            masked = image.clone()
            masked[:, :, y:y+scale, x:x+scale] = 0
            masked_embed = img_enc(masked)
            masked_score = compatibility(eeg_embed, masked_embed)
            delta = (full_score - masked_score).item()
            saliency_map[y:y+scale, x:x+scale] = delta

    saliency_map = saliency_map.cpu().numpy()
    saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-6) # normalization image-by-image
    return saliency_map

class StructuredHingeLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(StructuredHingeLoss, self).__init__()
        self.margin = margin

    def forward(self, eeg_emb, img_emb, labels):
    batch_size = eeg_emb.size(0)
    compatibility = torch.matmul(eeg_emb, img_emb.T)  # (B, B)

    correct_scores = torch.diag(compatibility).unsqueeze(1)  # (B, 1)

    # Label 기반 inter-class masking
    label_matrix = labels.unsqueeze(1).expand(-1, batch_size)
    label_mask = label_matrix != label_matrix.T  # True where labels differ

    # imposter scores: only other-class embeddings
    imposter_scores_e = compatibility.masked_fill(~label_mask, float('-inf')).max(dim=1)[0].unsqueeze(1)
    imposter_scores_v = compatibility.T.masked_fill(~label_mask.T, float('-inf')).max(dim=1)[0].unsqueeze(1)

    loss_e = F.relu(self.margin + imposter_scores_e - correct_scores).mean()
    loss_v = F.relu(self.margin + imposter_scores_v - correct_scores).mean()

    return (loss_e + loss_v) / 2

def train_model(eeg_encoder, image_encoder, train_loader, optimizer, device,
                margin=0.2, epochs=100, base_save_dir='./checkpoints'):
    criterion = StructuredHingeLoss(margin)
    eeg_encoder.train()
    image_encoder.train()

    # save directory
    timestamp = time.strftime('%Y%m%d%H%M')
    save_dir = os.path.join(base_save_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)

    # saving hyperparms
    if hasattr(eeg_encoder, 'hparams'):
        with open(os.path.join(save_dir, 'eeg_encoder_hparams.json'), 'w') as f:
            json.dump(eeg_encoder.hparams, f, indent=2)
    if hasattr(image_encoder, 'hparams'):
        with open(os.path.join(save_dir, 'image_encoder_hparams.json'), 'w') as f:
            json.dump(image_encoder.hparams, f, indent=2)

    for epoch in range(epochs):
        epoch_loss = 0.0
        best_loss = 0.0
        for eegs, images, _ in train_loader:
            eegs, images = eegs.to(device), images.to(device)
            eeg_emb = eeg_encoder(eegs)
            img_emb = image_encoder(images)

            loss = criterion(eeg_emb, img_emb, None)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader):.4f}")

        # saving model
        if (epoch + 1) % 20 == 0 or (epoch + 1) == epochs:
            eeg_path = os.path.join(save_dir, f"eeg_encoder_epoch{epoch+1}.pth")
            img_path = os.path.join(save_dir, f"image_encoder_epoch{epoch+1}.pth")
            torch.save(eeg_encoder.state_dict(), eeg_path)
            torch.save(image_encoder.state_dict(), img_path)
            print(f"[Saving Model] {eeg_path}, {img_path}")


# 사용 안할 예정
def evaluate(saliency_map, fixation_map):
    saliency_flat = saliency_map.flatten()
    fixation_flat = fixation_map.flatten()
    if fixation_flat.sum() == 0 or fixation_flat.sum() == len(fixation_flat):
        return 0.0
    return roc_auc_score(fixation_flat, saliency_flat)


# ---------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train EEG_Saliency Model')
    parser.add_argument('--eeg_dataset', type=str, default='/mnt/d/EEG_saliency_data/eeg_5_95_std.pth', help='Path to EEG dataset')
    parser.add_argument('--splits_path', type=str, default='/mnt/d/EEG_saliency_data/block_splits_by_image_all.pth', help='Path to split .pth file')
    parser.add_argument('--time_low', type=int, default=20, help='Start index of EEG time slice (in samples)')
    parser.add_argument('--time_high', type=int, default=460, help='End index of EEG time slice (in samples)')
    parser.add_argument('--subject', type=int, default=0, help='Subject ID (1~6) or 0 for all')
    parser.add_argument('--image_root', type=str, default='/mnt/d/EEG_saliency_data/imagenet_select/', help='Path to directory containing stimulus images')
    parser.add_argument('--epochs', default=10, type=int, help='number of max epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='number of data samples in a batch')
    parser.add_argumnet('--sampler', type=str, default='original', help='choose sampler, default is original method in paper:visual saliency detection guided by neural signals')
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 임의 dataset
    # eeg = torch.randn(1, 1, 32, 250).to(device)
    # image = torch.randn(1, 3, 128, 128).to(device)
    # fixation_map = np.zeros((128, 128))
    # fixation_map[40:60, 40:60] = 1

    # real dataset 
    test_loader = get_eeg_test_dataloader(
    eeg_path=args.eeg_dataset,
    split_path=args.splits_path,
    subject=args.subject,
    split_name='test',
    batch_size=1,
    time_low=args.time_low,
    time_high=args.time_high
    )

    train_loader = get_eeg_train_dataloader(
    eeg_path=args.eeg_dataset,
    split_path=args.splits_path,
    subject=args.subject,
    split_name='train',
    batch_size=args.batch_size,
    time_low=args.time_low,
    time_high=args.time_high
    sampler=args.sampler
    )
        
    # model 선언
    eeg_encoder = EEGNet().to(device)
    img_encoder = Inception_ImageEncoder().to(device)

    optimizer = torch.optim.Adam(list(eeg_encoder.parameters()) + list(img_encoder.parameters()), lr=1e-4)
    
    # Train the model
    train_model(eeg_encoder, img_encoder, train_loader, optimizer, device, epochs=args.epochs)

    # Evaluate on test sample
    eeg_encoder.eval()
    img_encoder.eval()
    for eeg, label, image_index, image_tensor in test_loader:
        eeg = eeg.to(device)
        image_tensor = image_tensor.to(device)
        saliency_map = compute_saliency(eeg, image_tensor, eeg_encoder, img_encoder)
        #################################
        # Q3. visualize saliency map
        #################################
        plt.imshow(saliency_map, cmap='hot')
        plt.title("Saliency Map")
        plt.axis('off')
        plt.show()
        break