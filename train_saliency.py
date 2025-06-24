import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt

from models import EEGNet, Inception_ImageEncoder
from utils import compute_saliency, visualize_and_save_saliency
from dataloader import get_eeg_test_dataloader, get_eeg_train_dataloader

import argparse
from PIL import Image
import time
import json
import torchvision.transforms.functional as TF
import sys


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

def train_model(subject, eeg_encoder, image_encoder, train_loader, optimizer, device,
                margin=0.2, epochs=100, base_save_dir='./checkpoints'):
    criterion = StructuredHingeLoss(margin)
    eeg_encoder.train()
    image_encoder.train()

    # save directory
    timestamp = time.strftime('%Y%m%d%H%M')
    save_dir = os.path.join(base_save_dir, f'{timestamp}_{subject}')
    os.makedirs(save_dir, exist_ok=True)

    # saving hyperparms
    if hasattr(eeg_encoder, 'hparams'):
        with open(os.path.join(save_dir, 'eeg_encoder_hparams.json'), 'w') as f:
            json.dump(eeg_encoder.hparams, f, indent=2)
    if hasattr(image_encoder, 'hparams'):
        with open(os.path.join(save_dir, 'image_encoder_hparams.json'), 'w') as f:
            json.dump(image_encoder.hparams, f, indent=2)

    # saving log
    log_file = open(os.path.join(save_dir, 'train_log.txt'), 'w')
    sys.stdout = log_file
    # sys.stderr = log_file

    for epoch in range(epochs):
        epoch_loss = 0.0
        best_loss = 0.0
        for eegs, labels, _, images in train_loader:
            eegs = eegs.permute(0, 2, 1).unsqueeze(1).to(device)  # [B, 1, 128, 440]
            images = images.to(device)
            labels = labels.to(device)

            eeg_emb = eeg_encoder(eegs)
            img_emb = image_encoder(images)

            loss = criterion(eeg_emb, img_emb, labels)

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
    
    return save_dir




# ---------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train EEG_Saliency Model')
    parser.add_argument('--eeg_dataset', type=str, default='/mnt/d/EEG_saliency_data/eeg_5_95_std.pth', help='Path to EEG dataset')
    parser.add_argument('--splits_path', type=str, default='/mnt/d/EEG_saliency_data/block_splits_by_image_all.pth', help='Path to split .pth file')
    parser.add_argument('--time_low', type=int, default=20, help='Start index of EEG time slice (in samples)')
    parser.add_argument('--time_high', type=int, default=460, help='End index of EEG time slice (in samples)')
    parser.add_argument('--subject', type=int, default=0, help='Subject ID (1~6) or 0 for all')
    parser.add_argument('--image_root', type=str, default='/mnt/d/EEG_saliency_data/imagenet_select/', help='Path to directory containing stimulus images')
    parser.add_argument('--epochs', default=100, type=int, help='number of max epochs')
    parser.add_argument('--batch_size', type=int, default=10, help='number of data samples in a batch')
    parser.add_argument('--sampler', type=str, default='original', help='choose sampler, default is original method in paper:visual saliency detection guided by neural signals')
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 임의 dataset
    # eeg = torch.randn(1, 1, 32, 250).to(device)
    # image = torch.randn(1, 3, 128, 128).to(device)
    # fixation_map = np.zeros((128, 128))
    # fixation_map[40:60, 40:60] = 1      

    # Train
    # dataset 
    test_loader = get_eeg_test_dataloader(
    eeg_path=args.eeg_dataset,
    split_path=args.splits_path,
    subject=args.subject,
    split_name='test',
    batch_size=1,
    time_low=args.time_low,
    time_high=args.time_high,
    image_root=args.image_root
    )

    train_loader = get_eeg_train_dataloader(
    eeg_path=args.eeg_dataset,
    split_path=args.splits_path,
    subject=args.subject,
    split_name='train',
    batch_size=args.batch_size,
    time_low=args.time_low,
    time_high=args.time_high,
    sampler=args.sampler,
    image_root=args.image_root
    )
        
    # model 선언
    eeg_encoder = EEGNet().to(device)
    img_encoder = Inception_ImageEncoder(out_dim=128).to(device)
    print("model is ready")

    optimizer = torch.optim.Adam(list(eeg_encoder.parameters()) + list(img_encoder.parameters()), lr=1e-4)
    
    # Train the model
    save_dir = train_model(args.subject, eeg_encoder, img_encoder, train_loader, optimizer, device, epochs=args.epochs)

    # Evaluate on test sample
    eeg_encoder.eval()
    img_encoder.eval()
    i = 0
    for eeg, label, image_index, image_tensor in test_loader:
        if i > 3:
            break
        image_tensor = image_tensor.to(device)
        saliency_map = compute_saliency(eeg, image_tensor, eeg_encoder, img_encoder)
        #################################
        # Q3. visualize saliency map
        #################################
        visualize_and_save_saliency(image_tensor.squeeze(0), saliency_map, str(label.item()), save_dir)
        i +=1
        break