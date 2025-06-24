import torch
from sklearn.metrics import roc_auc_score
import numpy as np
import os 
import matplotlib.pyplot as plt 
from PIL import Image


def compatibility(eeg_embed, img_embed):
    return torch.sum(eeg_embed * img_embed, dim=1)

# utils : compute_saliency (with multi-scale)
def compute_saliency(eeg, image, eeg_enc, img_enc, scale=16):
    device = image.device
    eeg = eeg.permute(0, 2, 1).unsqueeze(1).to(device) # [B, 1, 128, 440] 형태
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


# 이건 label이 없어서 사용 못할 듯
def evaluate(saliency_map, fixation_map):
    saliency_flat = saliency_map.flatten()
    fixation_flat = fixation_map.flatten()
    if fixation_flat.sum() == 0 or fixation_flat.sum() == len(fixation_flat):
        return 0.0
    return roc_auc_score(fixation_flat, saliency_flat)


def visualize_and_save_saliency(image_tensor, saliency_map, label, save_dir):
    # image_tensor: (3, H, W) → convert to (H, W, 3)
    image_np = image_tensor.detach().cpu().numpy()
    image_np = np.transpose(image_np, (1, 2, 0))  # C, H, W → H, W, C
    image_np = np.clip(image_np, 0, 1)  # Just in case

    # Resize saliency map to match image resolution
    H, W = image_np.shape[:2]
    if saliency_map.shape != (H, W):
        saliency_resized = resize(saliency_map, (H, W), mode='reflect', anti_aliasing=True)
    else:
        saliency_resized = saliency_map

    saliency_resized = np.clip(saliency_resized, 0, 1)

    # Generate heatmap (cmap is RGBA)
    cmap = plt.get_cmap('hot')
    heatmap = cmap(saliency_resized)[:, :, :3]  # Drop alpha

    # Overlay
    overlay = np.clip(0.6 * image_np + 0.4 * heatmap, 0, 1)

    # Plot and save
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(image_np)
    axs[0].set_title('Original Image')
    axs[1].imshow(saliency_resized, cmap='gray')
    axs[1].set_title('Saliency Map')
    axs[2].imshow(overlay)
    axs[2].set_title('Overlay')
    for ax in axs:
        ax.axis('off')

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{label}.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)