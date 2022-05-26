from models import GeneratorRRDB
import torch
import argparse
import os
from torchvision import transforms
import torch.nn as nn
import cv2
import numpy as np
import torch.nn.functional as F
import random
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--image_path", type=str, required=True, help="Path to image")
parser.add_argument("--checkpoint_model", type=str, required=True, help="Path to checkpoint model")
parser.add_argument("--output_dir", type=str, required=True, help="Path to save output")
parser.add_argument("--channels", type=int, default=3, help="Number of image channels")
parser.add_argument("--residual_blocks", type=int, default=23, help="Number of residual blocks in G")
opt = parser.parse_args()
print(opt)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(opt.output_dir, exist_ok=True)

# Define model and load model checkpoint
generator = GeneratorRRDB()
generator.load_state_dict(torch.load(opt.checkpoint_model, map_location=device))
generator.to(device).eval()

criterion_psnr = nn.MSELoss().to(device)

# Prepare input
if os.path.isdir(opt.image_path):
    imgs_path = sorted([os.path.join(opt.image_path, im) for im in os.listdir(opt.image_path)])
else:
    imgs_path = [opt.image_path]

tm = datetime.now().strftime("%Y%m%d%H%M%S")
start = datetime.now()

for i, img_path in enumerate(imgs_path):
    output_path = os.path.join(opt.output_dir, "tmo_" + os.path.basename(img_path)[:-4]+".png")

    with torch.no_grad():
        HDR = cv2.imread(img_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        # Rescale to 0-1.
        if (".png" in img_path):
            HDR = HDR / 65536.0

        HDR = torch.tensor(HDR).permute(2, 0, 1).unsqueeze(0).to(device, dtype=torch.float)

        # TMO
        gen_SDR = generator(HDR)

        # convert tensor to numpy
        np_gen_SDR = gen_SDR.squeeze(0).permute(1, 2, 0).cpu().numpy()

        # Rescale values in range 0-255
        np_gen_SDR = 255 * (np_gen_SDR + 1.0) / 2.0

    # Save image
    print("Saving to ", output_path)
    cv2.imwrite(output_path, np_gen_SDR,
                [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF])
    
    HDR = gen_SDR = np_gen_SDR = 0

    print("Single image running time:", (datetime.now() - start) / (i + 1))

print("Total running time:", datetime.now() - start)

