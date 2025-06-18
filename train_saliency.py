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

# Data augmentation
# ImageNet 정규화 기준
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]


# dataloader
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
    
    
def get_eeg_test_dataloader(eeg_path, split_path, subject=0, split_name='test', batch_size=1, time_low=20, time_high=460, image_root='images/'):
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

def get_eeg_train_dataloader(eeg_path, split_path, subject=0, split_name='train', 
                             batch_size=10, time_low=20, time_high=460, image_root='images/'):
    
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



# -------------------------------------
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

        correct_scores = compatibility[torch.arange(batch_size), torch.arange(batch_size)].unsqueeze(1)  # (B, 1)

        # Mask out diagonal (correct pairs)
        mask = torch.ones_like(compatibility, dtype=torch.bool)
        mask.fill_diagonal_(False)

        # Calculate max imposter score for each example
        max_imposter_scores = compatibility.masked_fill(~mask, float('-inf')).max(dim=1)[0].unsqueeze(1)

        loss = F.relu(self.margin + max_imposter_scores - correct_scores).mean()
        return loss

def train_model(eeg_encoder, image_encoder, train_loader, optimizer, device, margin=0.2, epochs=100):
    criterion = StructuredHingeLoss(margin)
    eeg_encoder.train()
    image_encoder.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        for eegs, images, _ in train_loader:
            eegs, images = eegs.to(device), images.to(device)
            eeg_emb = eeg_encoder(eegs)     # (B, D)
            img_emb = image_encoder(images) # (B, D)

            loss = criterion(eeg_emb, img_emb, None)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader):.4f}")

        
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
    parser.add_argument('--image_root', type=str, default='/mnt/d/EEG_saliency_data/images/', help='Path to directory containing stimulus images')
    parser.add_argument('--epochs', default=10, type=int, help='number of max epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='number of data samples in a batch')
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
        plt.imshow(saliency_map, cmap='hot')
        plt.title("Saliency Map")
        plt.axis('off')
        plt.show()
        break