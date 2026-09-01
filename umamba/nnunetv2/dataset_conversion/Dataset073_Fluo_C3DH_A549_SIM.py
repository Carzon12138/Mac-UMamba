from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed
import tifffile
from batchgenerators.utilities.file_and_folder_operations import *
import shutil


if __name__ == '__main__':
    """
    This is going to be my test dataset for working with tif as input and output images

    All we do here is copy the files and rename them. Not file conversions take place
    """
    dataset_name = 'Dataset073_Fluo_C3DH_A549_SIM'

    imagestr = join(nnUNet_raw, dataset_name, 'imagesTr')
    imagests = join(nnUNet_raw, dataset_name, 'imagesTs')
    labelstr = join(nnUNet_raw, dataset_name, 'labelsTr')
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelstr)



    train_source = '/home/fabian/Downloads/Fluo-C3DH-A549-SIM_train'
    test_source = '/home/fabian/Downloads/Fluo-C3DH-A549-SIM_test'









    spacing = (1, 0.126, 0.126)


    for seq in ['01', '02']:
        images_dir = join(train_source, seq)
        seg_dir = join(train_source, seq + '_GT', 'SEG')


        images = subfiles(images_dir, suffix='.tif', sort=True, join=False)
        segs = subfiles(seg_dir, suffix='.tif', sort=True, join=False)
        for i, (im, se) in enumerate(zip(images, segs)):
            target_name = f'{seq}_image_{i:03d}'


            shutil.copy(join(images_dir, im), join(imagestr, target_name + '_0000.tif'))

            save_json({'spacing': spacing}, join(imagestr, target_name + '.json'))
            shutil.copy(join(seg_dir, se), join(labelstr, target_name + '.tif'))

            save_json({'spacing': spacing}, join(labelstr, target_name + '.json'))


    for seq in ['01', '02']:
        images_dir = join(test_source, seq)
        images = subfiles(images_dir, suffix='.tif', sort=True, join=False)
        for i, im in enumerate(images):
            target_name = f'{seq}_image_{i:03d}'
            shutil.copy(join(images_dir, im), join(imagests, target_name + '_0000.tif'))

            save_json({'spacing': spacing}, join(imagests, target_name + '.json'))


    generate_dataset_json(
        join(nnUNet_raw, dataset_name),
        {0: 'fluorescence_microscopy'},
        {'background': 0, 'cell': 1},
        60,
        '.tif'
    )


    caseids = [i[:-4] for i in subfiles(labelstr, suffix='.tif', join=False)]
    splits = []
    splits.append(
        {'train': [i for i in caseids if i.startswith('01_')], 'val': [i for i in caseids if i.startswith('02_')]}
    )
    splits.append(
        {'train': [i for i in caseids if i.startswith('02_')], 'val': [i for i in caseids if i.startswith('01_')]}
    )
    save_json(splits, join(nnUNet_preprocessed, dataset_name, 'splits_final.json'))
