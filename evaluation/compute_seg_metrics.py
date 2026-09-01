import os
import argparse
import numpy as np
import pandas as pd
from collections import OrderedDict
from tqdm import tqdm
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree


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

def compute_dice_coefficient(y_true, y_pred):

    y_true_sum = np.sum(y_true)
    y_pred_sum = np.sum(y_pred)


    if y_true_sum == 0:
        if y_pred_sum == 0:
            return np.nan
        else:
            return 0.0


    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (y_true_sum + y_pred_sum + 1e-6)


def compute_iou(y_true, y_pred):

    y_true_sum = np.sum(y_true)
    y_pred_sum = np.sum(y_pred)


    if y_true_sum == 0:
        if y_pred_sum == 0:
            return np.nan
        else:
            return 0.0

    intersection = np.sum(y_true * y_pred)
    union = y_true_sum + y_pred_sum - intersection
    return intersection / (union + 1e-6)


def get_surface_points(mask):

    from scipy.ndimage import binary_erosion
    eroded = binary_erosion(mask)
    surface = np.logical_and(mask, ~eroded)
    return np.argwhere(surface)


def compute_nsd_3d_fast(gt, seg, spacing=(1, 1, 1), tau=2.0):

    gt_points = get_surface_points(gt)
    seg_points = get_surface_points(seg)


    if len(gt_points) == 0:
        if len(seg_points) == 0:
            return np.nan
        else:
            return 0.0

    if len(seg_points) == 0:
        return 0.0

    gt_points_mm = gt_points * np.array(spacing)
    seg_points_mm = seg_points * np.array(spacing)

    tree_gt = KDTree(gt_points_mm)
    tree_seg = KDTree(seg_points_mm)

    dists_gt_to_seg, _ = tree_seg.query(gt_points_mm, k=1)
    dists_seg_to_gt, _ = tree_gt.query(seg_points_mm, k=1)

    num_gt_ok = (dists_gt_to_seg <= tau).sum()
    num_seg_ok = (dists_seg_to_gt <= tau).sum()

    return (num_gt_ok + num_seg_ok) / (len(gt_points) + len(seg_points))


def compute_nsd_2d(gt, seg, tau=3.0):
    """2D NSD"""
    if not HAS_OPENCV: return np.nan
    kernel = np.ones((3, 3), np.uint8)
    gt_eroded = cv2.erode(gt, kernel, iterations=1)
    seg_eroded = cv2.erode(seg, kernel, iterations=1)
    gt_surface = np.argwhere((gt > 0) & (gt_eroded == 0))
    seg_surface = np.argwhere((seg > 0) & (seg_eroded == 0))


    if len(gt_surface) == 0:
        if len(seg_surface) == 0:
            return np.nan
        else:
            return 0.0

    if len(seg_surface) == 0:
        return 0.0
    dist_gt_to_seg = cdist(gt_surface, seg_surface, 'euclidean').min(axis=1)
    dist_seg_to_gt = cdist(seg_surface, gt_surface, 'euclidean').min(axis=1)

    num_gt_ok = (dist_gt_to_seg <= tau).sum()
    num_seg_ok = (dist_seg_to_gt <= tau).sum()
    return (num_gt_ok + num_seg_ok) / (len(gt_surface) + len(seg_surface))


def print_class_average_stats(df, metrics_suffix=['DSC', 'IOU', 'NSD']):
    print("\n" + "=" * 80)
    print(f"📊 Evaluation Report (Total Samples: {len(df)})")
    print("NOTE: 'True Negative' imagesTs (Empty GT & Empty Pred) are EXCLUDED (NaN) from averages.")
    print("-" * 80)

    for metric in metrics_suffix:
        cols = [c for c in df.columns if c.endswith(f'_{metric}')]
        if not cols: continue

        print(f"\n--- {metric} Statistics ---")
        print(f"{'Class Name':<20} | {'Mean ± Std':<25} | {'Valid Samples':<15}")
        print("-" * 80)

        class_means = []
        for col in cols:
            data = df[col].dropna()
            count = len(data)
            if count == 0:
                print(f"{col:<20} | No Valid Data")
                continue
            mean_val = data.mean()
            std_val = data.std()
            class_means.append(mean_val)
            display_name = col.replace(f'_{metric}', '')
            print(f"{display_name:<20} | {mean_val:.4f} ± {std_val:.4f}     | {count}/{len(df)}")

        if class_means:
            macro_avg = np.mean(class_means)
            print("-" * 80)
            print(f"{'⚠️ Class Average':<20} | {macro_avg:.4f}")
    print("=" * 80 + "\n")

def find_gt_file(gt_path, seg_name, valid_exts=None):
    if os.path.exists(os.path.join(gt_path, seg_name)):
        return seg_name
    if '_0000' in seg_name:
        clean_name = seg_name.replace('_0000', '')
        if os.path.exists(os.path.join(gt_path, clean_name)):
            return clean_name
    if valid_exts:
        base_seg = os.path.splitext(seg_name)[0].replace('_0000', '')
        for f in os.listdir(gt_path):
            if f.lower().endswith(valid_exts):
                base_gt = os.path.splitext(f)[0]
                if base_gt == base_seg:
                    return f
    return None


def scan_classes_3d(gt_path, filenames):
    unique_values = set()
    print(f"🔍 Auto-detecting classes from ALL {len(filenames)} samples (3D Mode)...")
    for name in tqdm(filenames, desc="Scanning Classes"):
        gt_name = find_gt_file(gt_path, name)
        if gt_name:
            try:
                img_path = os.path.join(gt_path, gt_name)
                img_proxy = nb.load(img_path)
                data = img_proxy.get_fdata()
                unique_values.update(np.unique(data.astype(int)))
            except Exception as e:
                pass
    if 0 in unique_values: unique_values.remove(0)
    sorted_vals = sorted(list(unique_values))
    if not sorted_vals: return [1]
    return sorted_vals

def process_3d_nifti(gt_path, seg_path, save_path):
    if not HAS_NIBABEL:
        print("Error: nibabel is not installed.")
        return

    print(f">>> Mode: NIfTI Evaluation (Supports 3D Volume & 2D Slices in .nii.gz)")
    join = os.path.join
    filenames = sorted([x for x in os.listdir(seg_path) if x.endswith(('.nii.gz', '.nii'))])

    if not filenames:
        print(f"Error: No .nii files found in {seg_path}")
        return

    classes = scan_classes_3d(gt_path, filenames)
    print(f"✅ Detected Classes: {classes}")


    metrics_data = OrderedDict([('Name', [])])
    for c in classes:
        metrics_data[f'Class_{c}_DSC'] = []
        metrics_data[f'Class_{c}_IOU'] = []
        metrics_data[f'Class_{c}_NSD'] = []

    for name in tqdm(filenames, desc="Processing NIfTI Files"):
        gt_name = find_gt_file(gt_path, name)
        if gt_name is None: continue

        gt_full = join(gt_path, gt_name)
        try:
            gt_nii = nb.load(gt_full)
            gt_data = np.uint8(gt_nii.get_fdata())
            seg_nii = nb.load(join(seg_path, name))
            seg_data = np.uint8(seg_nii.get_fdata())

            if gt_data.shape != seg_data.shape:
                if gt_data.shape[-1] == 1: gt_data = gt_data.squeeze()
                if seg_data.shape[-1] == 1: seg_data = seg_data.squeeze()

            if gt_data.shape != seg_data.shape:
                continue

            header = gt_nii.header
            zooms = header.get_zooms()
            spacing = zooms[:gt_data.ndim]

            metrics_data['Name'].append(name)

            for c in classes:
                gt_bin = (gt_data == c)
                seg_bin = (seg_data == c)


                dsc = compute_dice_coefficient(gt_bin, seg_bin)
                iou = compute_iou(gt_bin, seg_bin)
                metrics_data[f'Class_{c}_DSC'].append(dsc)
                metrics_data[f'Class_{c}_IOU'].append(iou)


                nsd = compute_nsd_3d_fast(gt_bin, seg_bin, spacing=spacing, tau=2.0)
                metrics_data[f'Class_{c}_NSD'].append(nsd)

        except Exception as e:
            print(f"Error processing {name}: {e}")
            continue

    df = pd.DataFrame(metrics_data)
    if not df.empty:
        df.round(4).to_csv(save_path, index=False)
        print(f"Results saved to: {save_path}")
        print_class_average_stats(df, ['DSC', 'IOU', 'NSD'])
    else:
        print("No valid results computed.")


def scan_classes_in_gt(gt_path, gt_file_list):
    unique_values = set()
    print(f"🔍 Scanning ALL {len(gt_file_list)} GTs for classes (2D Image Mode)...")
    for gt_name in tqdm(gt_file_list, desc="Scanning Classes"):
        path = os.path.join(gt_path, gt_name)
        if os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                unique_values.update(np.unique(img))
    if 0 in unique_values: unique_values.remove(0)
    sorted_vals = sorted(list(unique_values))
    if len(sorted_vals) == 1 and sorted_vals[0] == 255: return [1]
    return sorted_vals if sorted_vals else [1]


def process_2d_image(gt_path, seg_path, save_path):
    if not HAS_OPENCV:
        print("Error: opencv not installed.")
        return

    print(f">>> Mode: 2D Image Evaluation (png, jpg, tif, etc.)")
    join = os.path.join
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    filenames = sorted([x for x in os.listdir(seg_path) if x.lower().endswith(valid_exts)])

    if not filenames:
        print(f"Error: No 2D image files found.")
        return

    gt_files_map = {}
    valid_gt_files = []
    for f in os.listdir(gt_path):
        if not f.lower().endswith(valid_exts): continue
        full_base = os.path.splitext(f)[0]
        filename = f
        if full_base.endswith('_label'):
            key = full_base[:-6]
        else:
            key = full_base
        gt_files_map[key] = filename
        gt_files_map[full_base] = filename
        valid_gt_files.append(filename)

    if not valid_gt_files: return

    classes = scan_classes_in_gt(gt_path, valid_gt_files)
    print(f"✅ Detected Classes: {classes}")

    is_multi = len(classes) > 1 or (len(classes) == 1 and classes[0] != 1)


    metrics = OrderedDict([('Name', [])])
    class_names = []
    for c in classes:
        c_name = f"Class_{c}"
        class_names.append((c, c_name))
        metrics[f'{c_name}_DSC'] = []
        metrics[f'{c_name}_IOU'] = []
        metrics[f'{c_name}_NSD'] = []

    for name in tqdm(filenames, desc="Processing Images"):
        base_noext = os.path.splitext(name)[0]
        base_clean = base_noext.replace('_0000', '')

        gt_name = None
        for key in [base_clean, base_noext, base_clean + '_label', base_noext + '_label']:
            if key in gt_files_map:
                gt_name = gt_files_map[key]
                break
        if not gt_name: continue

        metrics['Name'].append(name)
        gt_img = cv2.imread(join(gt_path, gt_name), cv2.IMREAD_GRAYSCALE)
        seg_img = cv2.imread(join(seg_path, name), cv2.IMREAD_GRAYSCALE)

        if gt_img is None or seg_img is None or gt_img.shape != seg_img.shape:
            for _, c_name in class_names:
                metrics[f'{c_name}_DSC'].append(np.nan)
                metrics[f'{c_name}_IOU'].append(np.nan)
                metrics[f'{c_name}_NSD'].append(np.nan)
            continue

        if not is_multi and classes[0] == 1:
            if np.max(gt_img) > 1: gt_img = (gt_img > 127).astype(np.uint8)
            if np.max(seg_img) > 1: seg_img = (seg_img > 127).astype(np.uint8)

        for c_val, c_name in class_names:
            gt_bin = (gt_img == c_val).astype(np.uint8)
            seg_bin = (seg_img == c_val).astype(np.uint8)


            dsc = compute_dice_coefficient(gt_bin, seg_bin)
            iou = compute_iou(gt_bin, seg_bin)
            nsd = compute_nsd_2d(gt_bin, seg_bin, tau=2.0)

            metrics[f'{c_name}_DSC'].append(dsc)
            metrics[f'{c_name}_IOU'].append(iou)
            metrics[f'{c_name}_NSD'].append(nsd)

    df = pd.DataFrame(metrics)
    if not df.empty:
        df.round(4).to_csv(save_path, index=False)
        print(f"Results saved to: {save_path}")
        print_class_average_stats(df, ['DSC', 'IOU', 'NSD'])
    else:
        print("No matches found.")


def main():
    parser = argparse.ArgumentParser(description="Unified Evaluation (DSC, IOU, NSD)")
    parser.add_argument('--gt_path', required=True, help="Ground truth folder")
    parser.add_argument('--seg_path', required=True, help="Prediction folder")
    parser.add_argument('--save_path', default='results.csv', help="Output CSV path")
    args = parser.parse_args()

    files = [f for f in os.listdir(args.seg_path) if not f.startswith('.')]
    if not files:
        print("Error: Prediction folder is empty!")
        return

    has_nifti = any(f.endswith(('.nii', '.nii.gz')) for f in files)
    if has_nifti:
        process_3d_nifti(args.gt_path, args.seg_path, args.save_path)
    else:
        process_2d_image(args.gt_path, args.seg_path, args.save_path)


if __name__ == '__main__':
    main()
