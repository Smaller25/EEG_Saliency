import torch
import os
import argparse
from models import EEGNet, Inception_ImageEncoder
from torchvision import transforms
from utils import compute_saliency, visualize_and_save_saliency
from dataloader import get_eeg_test_dataloader

# Argument parser
parser = argparse.ArgumentParser(description='Visualize Saliency from Pretrained EEG-Image Model')
parser.add_argument('--eeg_dataset', type=str, default='/mnt/d/EEG_saliency_data/eeg_5_95_std.pth', help='Path to EEG dataset')
parser.add_argument('--splits_path', type=str, default='/mnt/d/EEG_saliency_data/block_splits_by_image_all.pth', help='Path to split .pth file')
parser.add_argument('--time_low', type=int, default=20, help='Start index of EEG time slice (in samples)')
parser.add_argument('--time_high', type=int, default=460, help='End index of EEG time slice (in samples)')
parser.add_argument('--subject', type=int, default=0, help='Subject ID (1~6) or 0 for all')
parser.add_argument('--image_root', type=str, default='/mnt/d/EEG_saliency_data/imagenet_select/', help='Path to directory containing stimulus images')
parser.add_argument('--model_dir', type=str, required=True, help='Directory containing pretrained models')
parser.add_argument('--num_visualize', type=int, default=4, help='Number of saliency maps to visualize')
parser.add_argument('--output_dir', type=str, default=None, help='Where to save saliency maps')
args = parser.parse_args()

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load test set
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

# Load models
eeg_encoder = EEGNet().to(device)
img_encoder = Inception_ImageEncoder(out_dim=128).to(device)
eeg_encoder.load_state_dict(torch.load(os.path.join(args.model_dir, 'eeg_encoder_epoch100.pth'), map_location=device))
img_encoder.load_state_dict(torch.load(os.path.join(args.model_dir, 'image_encoder_epoch100.pth'), map_location=device))
eeg_encoder.eval()
img_encoder.eval()

# Output directory
save_dir = args.output_dir or args.model_dir
os.makedirs(save_dir, exist_ok=True)

# Visualize
for i, (eeg, label, image_index, image_tensor) in enumerate(test_loader):
    if i >= args.num_visualize:
        break
    image_tensor = image_tensor.to(device)
    saliency_map = compute_saliency(eeg, image_tensor, eeg_encoder, img_encoder)
    visualize_and_save_saliency(image_tensor.squeeze(0), saliency_map, str(label.item()), save_dir)
