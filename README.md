# Mac-UMamba

Official implementation of **MAC-UMamba: Multi-scale Asymmetric Channel-calibrated Hybrid Architecture for Generalizable Biomedical Segmentation**.

The implementation contains the MAC-UMamba network and its nnU-Net v2 training integration. The model code is in `umamba/nnunetv2/nets/MACUmamba.py`, with the trainer entry point in `umamba/nnunetv2/training/nnUNetTrainer/nnUNetTrainerMACUmamba.py`.

## Setup

```bash
pip install -r requirements.txt
```

The repository includes dataset metadata and conversion utilities only. Raw datasets, trained weights, experiment outputs, and IDE files are intentionally excluded.

## Citation

If you use this code, please cite the associated MAC-UMamba paper.
