import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm


try:
    import nibabel as nb

    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False

try:
    import cv2

    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
from scipy.ndimage import binary_erosion


def compute_dice(y_true, y_pred):
    """Binary Dice, strict empty handling"""
    gt_sum = np.sum(y_true)
    pred_sum = np.sum(y_pred)

    if gt_sum == 0:
        return np.nan if pred_sum == 0 else 0.0

    inter = np.sum(y_true * y_pred)
    return (2. * inter) / (gt_sum + pred_sum + 1e-6)


def compute_iou(y_true, y_pred):
    """Binary IoU, same logic"""
    gt_sum = np.sum(y_true)
    pred_sum = np.sum(y_pred)

    if gt_sum == 0:
        return np.nan if pred_sum == 0 else 0.0

    inter = np.sum(y_true * y_pred)
    union = gt_sum + pred_sum - inter
    return inter / (union + 1e-6)


def get_surface_points(mask):
    eroded = binary_erosion(mask)
    surface = np.logical_and(mask, ~eroded)
    return np.argwhere(surface)


def compute_nsd_3d(gt, pred, spacing=(1, 1, 1), tau=3.0):
    gt_pts = get_surface_points(gt)
    pred_pts = get_surface_points(pred)

    if len(gt_pts) == 0:
        return np.nan if len(pred_pts) == 0 else 0.0
    if len(pred_pts) == 0:
        return 0.0

    gt_pts_mm = gt_pts * np.array(spacing)
    pred_pts_mm = pred_pts * np.array(spacing)

    tree_gt = KDTree(gt_pts_mm)
    tree_pred = KDTree(pred_pts_mm)

    d_gt2pred = tree_pred.query(gt_pts_mm, k=1)[0]
    d_pred2gt = tree_gt.query(pred_pts_mm, k=1)[0]

    ok_gt = (d_gt2pred <= tau).sum()
    ok_pred = (d_pred2gt <= tau).sum()

    return (ok_gt + ok_pred) / (len(gt_pts) + len(pred_pts))


def compute_nsd_2d(gt, pred, tau=3.0):
    if not HAS_OPENCV:
        return np.nan
    kernel = np.ones((3, 3), np.uint8)
    gt_e = cv2.erode(gt.astype(np.uint8), kernel, 1)
    pred_e = cv2.erode(pred.astype(np.uint8), kernel, 1)

    gt_surf = np.argwhere((gt > 0) & (gt_e == 0))
    pred_surf = np.argwhere((pred > 0) & (pred_e == 0))

    if len(gt_surf) == 0:
        return np.nan if len(pred_surf) == 0 else 0.0
    if len(pred_surf) == 0:
        return 0.0

    d_gt2pred = cdist(gt_surf, pred_surf, 'euclidean').min(axis=1)
    d_pred2gt = cdist(pred_surf, gt_surf, 'euclidean').min(axis=1)

    ok_gt = (d_gt2pred <= tau).sum()
    ok_pred = (d_pred2gt <= tau).sum()

    return (ok_gt + ok_pred) / (len(gt_surf) + len(pred_surf))


def process_2d_binary(gt_path, seg_path, save_path):
    if not HAS_OPENCV:
        print("OpenCV not found. Cannot process 2D imagesTs.")
        return

    valid_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    seg_files = sorted([f for f in os.listdir(seg_path) if f.lower().endswith(valid_ext)])

    if not seg_files:
        print("No valid image files found in seg_path.")
        return

    results = {'filename': [], 'DSC': [], 'IOU': [], 'NSD': []}

    print(f"Processing {len(seg_files)} 2D imagesTs (binary mode)...")
    for fname in tqdm(seg_files):
        base = os.path.splitext(fname)[0].replace('_0000', '')


        gt_candidates = [
                            base + ext for ext in valid_ext
                        ] + [
                            base + '_label' + ext for ext in valid_ext
                        ] + [
                            fname.replace(ext, '_label' + ext) for ext in valid_ext if fname.endswith(ext)
                        ]

        gt_name = None
        for cand in gt_candidates:
            if os.path.exists(os.path.join(gt_path, cand)):
                gt_name = cand
                break

        if not gt_name:
            print(f"Warning: No GT found for {fname}")
            continue

        gt_img = cv2.imread(os.path.join(gt_path, gt_name), cv2.IMREAD_GRAYSCALE)
        seg_img = cv2.imread(os.path.join(seg_path, fname), cv2.IMREAD_GRAYSCALE)

        if gt_img is None or seg_img is None or gt_img.shape != seg_img.shape:
            continue

        gt_bin = (gt_img > 0).astype(np.uint8)
        seg_bin = (seg_img > 0).astype(np.uint8)

        dsc = compute_dice(gt_bin, seg_bin)
        iou = compute_iou(gt_bin, seg_bin)
        nsd = compute_nsd_2d(gt_bin, seg_bin)

        results['filename'].append(fname)
        results['DSC'].append(dsc)
        results['IOU'].append(iou)
        results['NSD'].append(nsd)

    if not results['filename']:
        print("No valid pairs processed.")
        return

    df = pd.DataFrame(results)
    df.to_csv(save_path, index=False)
    print(f"\nResults saved to: {save_path}")

    valid_dsc = df['DSC'].dropna()
    valid_iou = df['IOU'].dropna()
    valid_nsd = df['NSD'].dropna()

    print("\nBinary Segmentation Results (excluding true negatives):")
    print(f"  DSC  : {valid_dsc.mean():.4f} ± {valid_dsc.std():.4f}  (n={len(valid_dsc)})")
    print(f"  IoU  : {valid_iou.mean():.4f} ± {valid_iou.std():.4f}  (n={len(valid_iou)})")
    print(f"  NSD  : {valid_nsd.mean():.4f} ± {valid_nsd.std():.4f}  (n={len(valid_nsd)})")


def process_3d_binary(gt_path, seg_path, save_path):
    if not HAS_NIBABEL:
        print("nibabel not found. Cannot process 3D volumes.")
        return

    files = [f for f in os.listdir(seg_path) if f.endswith(('.nii', '.nii.gz'))]
    files.sort()

    if not files:
        print("No .nii/.nii.gz files found.")
        return

    results = {'filename': [], 'DSC': [], 'IOU': [], 'NSD': []}

    print(f"Processing {len(files)} 3D volumes (binary mode)...")
    for fname in tqdm(files):
        gt_name = fname
        gt_file = os.path.join(gt_path, gt_name)
        seg_file = os.path.join(seg_path, fname)

        if not os.path.exists(gt_file):
            print(f"GT not found: {gt_name}")
            continue

        try:
            gt_nii = nb.load(gt_file)
            seg_nii = nb.load(seg_file)
            gt = gt_nii.get_fdata().astype(np.uint8)
            seg = seg_nii.get_fdata().astype(np.uint8)

            if gt.shape != seg.shape:
                continue


            gt_bin = (gt > 0).astype(np.uint8)
            seg_bin = (seg > 0).astype(np.uint8)

            dsc = compute_dice(gt_bin, seg_bin)
            iou = compute_iou(gt_bin, seg_bin)

            spacing = gt_nii.header.get_zooms()[:3]
            nsd = compute_nsd_3d(gt_bin, seg_bin, spacing=spacing)

            results['filename'].append(fname)
            results['DSC'].append(dsc)
            results['IOU'].append(iou)
            results['NSD'].append(nsd)

        except Exception as e:
            print(f"Error processing {fname}: {e}")

    if not results['filename']:
        print("No valid volumes processed.")
        return

    df = pd.DataFrame(results)
    df.to_csv(save_path, index=False)
    print(f"\nResults saved to: {save_path}")

    valid_dsc = df['DSC'].dropna()
    valid_iou = df['IOU'].dropna()
    valid_nsd = df['NSD'].dropna()

    print("\nBinary Segmentation Results (excluding true negatives):")
    print(f"  DSC  : {valid_dsc.mean():.4f} ± {valid_dsc.std():.4f}  (n={len(valid_dsc)})")
    print(f"  IoU  : {valid_iou.mean():.4f} ± {valid_iou.std():.4f}  (n={len(valid_iou)})")
    print(f"  NSD  : {valid_nsd.mean():.4f} ± {valid_nsd.std():.4f}  (n={len(valid_nsd)})")


def main():
    parser = argparse.ArgumentParser(description="Binary segmentation evaluation (DSC/IoU/NSD)")
    parser.add_argument('--gt_path', required=True, help="Ground truth folder")
    parser.add_argument('--seg_path', required=True, help="Prediction folder")
    parser.add_argument('--save_path', default='binary_results.csv', help="Output csv path")
    args = parser.parse_args()

    seg_files = os.listdir(args.seg_path)
    if any(f.endswith(('.nii', '.nii.gz')) for f in seg_files):
        process_3d_binary(args.gt_path, args.seg_path, args.save_path)
    else:
        process_2d_binary(args.gt_path, args.seg_path, args.save_path)


if __name__ == '__main__':
    main()
