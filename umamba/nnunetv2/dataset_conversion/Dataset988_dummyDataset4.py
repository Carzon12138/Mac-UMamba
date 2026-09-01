import os

from batchgenerators.utilities.file_and_folder_operations import *

from nnunetv2.paths import nnUNet_raw
from nnunetv2.utilities.utils import get_filenames_of_train_images_and_targets

if __name__ == '__main__':

    source_dataset = 'Dataset004_Hippocampus'

    target_dataset = 'Dataset987_dummyDataset4'
    target_dataset_dir = join(nnUNet_raw, target_dataset)
    maybe_mkdir_p(target_dataset_dir)

    dataset = get_filenames_of_train_images_and_targets(join(nnUNet_raw, source_dataset))






    for k in dataset.keys():
        dataset[k]['label'] = os.path.relpath(dataset[k]['label'], target_dataset_dir)
        dataset[k]['images'] = [os.path.relpath(i, target_dataset_dir) for i in dataset[k]['images']]


    dataset_json = load_json(join(nnUNet_raw, source_dataset, 'dataset.json'))
    dataset_json['dataset'] = dataset


    save_json(dataset_json, join(target_dataset_dir, 'dataset.json'), sort_keys=False)
