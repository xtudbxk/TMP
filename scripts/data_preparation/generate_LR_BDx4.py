"""
Generate BDx4 LR data (PyTorch port of the MATLAB BD degradation script).

Recipe:
    1. fspecial('gaussian', 12, 1.6) (normalized)
    2. MATLAB imfilter(im, k, 'replicate')  # correlation
    3. im(2:4:end-2, 2:4:end-2)             # x4 decimation

NOTE on even-kernel anchoring (the only pitfall):
    MATLAB imfilter anchors an even kernel (12-tap) at the 0-indexed center 5,
    covering input spans -5..+6. The faithful PyTorch equivalent is
    F.pad(img, (5, 6, 5, 6), 'replicate') + F.conv2d(groups=3), then [:, :, 1::4, 1::4].
    A naive symmetric pad (or scipy origin=0) gives spans -6..+5, which shifts the LR
    by 1 px (= 4 px in HR space) and severely misaligns the SR output against GT.
    BI degradation (imresize with antialiasing) has no such ambiguity.

Usage (repo root):
    python scripts/data_preparation/generate_LR_BDx4.py --gt_root <GT dir> --save_root <out dir>
Recursively processes all .png under gt_root, preserving the relative directory structure.
"""
import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

K_SIZE, SIGMA, SCALE = 12, 1.6, 4


def fspecial_gaussian(k=K_SIZE, sigma=SIGMA):
    # equivalent to MATLAB fspecial('gaussian', k, sigma): normalized 2-D Gaussian
    x = torch.arange(k, dtype=torch.float64) - (k - 1) / 2.0
    g = torch.exp(-(x ** 2) / (2.0 * sigma ** 2))
    ker = torch.outer(g, g)
    return (ker / ker.sum()).float()


def bd_degrade(img):
    """img: (1,3,H,W) float in [0,1] (H/W multiples of 4) -> (1,3,H/4,W/4)"""
    ker = fspecial_gaussian().to(img.device)
    w = ker.view(1, 1, K_SIZE, K_SIZE).repeat(3, 1, 1, 1)
    # MATLAB imfilter (correlation) + replicate: even kernel anchored at 0-idx center 5,
    # spans -5..+6 -> asymmetric pad (5, 6)
    x = F.pad(img, (5, 6, 5, 6), mode='replicate')
    blur = F.conv2d(x, w, groups=3)
    return blur[:, :, 1::4, 1::4]  # MATLAB im(2:4:end-2, 2:4:end-2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt_root', required=True, help='GT image root dir (recursively scan .png)')
    ap.add_argument('--save_root', required=True, help='BDx4 output root dir (same relative structure)')
    args = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device:', dev)
    n = 0
    for dirpath, dirnames, filenames in os.walk(args.gt_root):
        dirnames.sort()
        for fn in sorted(filenames):
            if not fn.lower().endswith('.png'):
                continue
            src = os.path.join(dirpath, fn)
            dst = os.path.join(args.save_root, os.path.relpath(src, args.gt_root))
            if os.path.exists(dst):
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            bgr = cv2.imread(src).astype(np.float32) / 255.0
            h, wd = bgr.shape[:2]
            h, wd = h - h % SCALE, wd - wd % SCALE  # modcrop 4 (no-op if already aligned)
            bgr = bgr[:h, :wd]
            t = torch.from_numpy(bgr).permute(2, 0, 1).unsqueeze(0).to(dev)
            lr = bd_degrade(t).squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            cv2.imwrite(dst, (lr * 255).round().astype(np.uint8))
            n += 1
            if n % 500 == 0:
                print('%d done' % n, flush=True)
    print('TOTAL %d LR frames -> %s' % (n, args.save_root))


if __name__ == '__main__':
    main()
