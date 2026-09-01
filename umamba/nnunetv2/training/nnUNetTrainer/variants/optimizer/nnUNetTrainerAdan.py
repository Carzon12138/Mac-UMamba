import torch

from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from torch.optim.lr_scheduler import CosineAnnealingLR
try:
    from adan_pytorch import Adan
except ImportError:
    Adan = None


class nnUNetTrainerAdan(nnUNetTrainer):
    def configure_optimizers(self):
        if Adan is None:
            raise RuntimeError('This trainer requires adan_pytorch to be installed, install with "pip install adan-pytorch"')
        optimizer = Adan(self.network.parameters(),
                         lr=self.initial_lr,

                         weight_decay=self.weight_decay)


        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler


class nnUNetTrainerAdan1en3(nnUNetTrainerAdan):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 1e-3


class nnUNetTrainerAdan3en4(nnUNetTrainerAdan):

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 3e-4


class nnUNetTrainerAdan1en1(nnUNetTrainerAdan):

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 1e-1


class nnUNetTrainerAdanCosAnneal(nnUNetTrainerAdan):





    def configure_optimizers(self):
        if Adan is None:
            raise RuntimeError('This trainer requires adan_pytorch to be installed, install with "pip install adan-pytorch"')
        optimizer = Adan(self.network.parameters(),
                         lr=self.initial_lr,

                         weight_decay=self.weight_decay)


        lr_scheduler = CosineAnnealingLR(optimizer, T_max=self.num_epochs)
        return optimizer, lr_scheduler
