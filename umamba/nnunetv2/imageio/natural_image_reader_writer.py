














from typing import Tuple, Union, List
import numpy as np
from nnunetv2.imageio.base_reader_writer import BaseReaderWriter
from skimage import io


class NaturalImage2DIO(BaseReaderWriter):
    """
    ONLY SUPPORTS 2D IMAGES!!!
    """


    supported_file_endings = [
        '.png',


        '.bmp',
        '.tif'
    ]

    def read_images(self, image_fnames: Union[List[str], Tuple[str, ...]]) -> Tuple[np.ndarray, dict]:
        images = []
        for f in image_fnames:
            npy_img = io.imread(f)
            if npy_img.ndim == 3:


                assert npy_img.shape[-1] == 3 or npy_img.shape[-1] == 4, "If image has three dimensions then the last " \
                                                                         "dimension must have shape 3 or 4 " \
                                                                         f"(RGB or RGBA). Image shape here is {npy_img.shape}"

                images.append(npy_img.transpose((2, 0, 1))[:, None])
            elif npy_img.ndim == 2:

                images.append(npy_img[None, None])

        if not self._check_all_same([i.shape for i in images]):
            print('ERROR! Not all input images have the same shape!')
            print('Shapes:')
            print([i.shape for i in images])
            print('Image files:')
            print(image_fnames)
            raise RuntimeError()
        return np.vstack(images).astype(np.float32), {'spacing': (999, 1, 1)}

    def read_seg(self, seg_fname: str) -> Tuple[np.ndarray, dict]:
        return self.read_images((seg_fname, ))

    def write_seg(self, seg: np.ndarray, output_fname: str, properties: dict) -> None:
        io.imsave(output_fname, seg[0].astype(np.uint8), check_contrast=False)


if __name__ == '__main__':
    images = ('/media/fabian/data/nnUNet_raw/Dataset120_RoadSegmentation/imagesTr/img-11_0000.png',)
    segmentation = '/media/fabian/data/nnUNet_raw/Dataset120_RoadSegmentation/labelsTr/img-11.png'
    imgio = NaturalImage2DIO()
    img, props = imgio.read_images(images)
    seg, segprops = imgio.read_seg(segmentation)
