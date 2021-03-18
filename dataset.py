#!/usr/bin/env python
import os
import glob

from tqdm import tqdm
from PIL import Image, ImageOps
import numpy as np
import torch
from torch.utils.data import Dataset


class DirDataset(Dataset):
    def __init__(self, img_dir, mask_dir, scale=1,pytorch=True):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.scale = scale
        self.pytorch = pytorch

        try:
            self.ids = [s.split('.')[0] for s in os.listdir(self.img_dir)]
        except FileNotFoundError:
            self.ids = []

    def __len__(self):
        return len(self.ids)

    
    def preprocess_original(self, img):
        # w, h = img.size
        # # _h = int(h * self.scale)
        # # _w = int(w * self.scale)

        # _h = int(h * self.scale)
        # _w = int(w * self.scale)
        # assert _w > 0
        # assert _h > 0

        # _img = img.resize((_w, _h))

        _img = img.resize((256, 256))
        _img = np.array(_img)
        if len(_img.shape) == 2:  ## gray/mask images
            _img = np.expand_dims(_img, axis=-1)

        # hwc to chw
        _img = _img.transpose((2, 0, 1))
        if _img.max() > 1:
            _img = _img / 255.
        return _img
    
    # def preprocess(self, img, type):
    #     # w, h = img.size
    #     # _h = int(h * self.scale)
    #     # _w = int(w * self.scale)
    #     # assert _w > 0
    #     # assert _h > 0
    #     # _img = img.resize((_w, _h))

    #     _img = img.resize((256, 256))
    #     _img = np.array(_img)
    #     if len(_img.shape) == 2:  ## gray/mask images
    #         _img = np.expand_dims(_img, axis=-1)

    #     # hwc to chw
    #     _img = _img.transpose((2, 0, 1))
    #     if _img.max() > 1:
    #         if type == "mask":
    #             _img = np.where(_img > 120, 1, 0)
    #         else:
    #             _img = _img / 255.
    #     return _img

    def open_as_array(self, i, invert=False):
        idx = self.ids[i]

        img_files = glob.glob(os.path.join(self.img_dir, idx + '.*'))
        assert len(img_files) == 1, f'{idx}: {img_files}'
        _img = Image.open(img_files[0])
        raw_rgb = np.array(_img.resize((256,256)))

        if invert:
            raw_rgb = raw_rgb.transpose((2,0,1))

        # normalize
        return (raw_rgb / np.iinfo(raw_rgb.dtype).max)

    def open_mask(self, i, add_dims=False):
        idx = self.ids[i]
        mask_files = glob.glob(os.path.join(self.mask_dir, idx + '.*'))
        assert len(mask_files) == 1, f'{idx}: {mask_files}'
        _mask = Image.open(mask_files[0])

        raw_mask = np.array(_mask.getchannel(0).resize((256,256)))

        # plt.figure()
        # image = tf_2_tensor(Image.open(self.files[idx]['gt']))
        # inp = np.array(tf_2_PIL(image))
        # plt.imshow(inp)
        # print("inp",inp.shape)

        raw_mask = np.where(raw_mask==255, 1, 0)

        return np.expand_dims(raw_mask, 0) if add_dims else raw_mask

    def __getitem__(self, i):


        # img_files = glob.glob(os.path.join(self.img_dir, idx + '.*'))
        # mask_files = glob.glob(os.path.join(self.mask_dir, idx + '.*'))
        # assert len(img_files) == 1, f'{idx}: {img_files}'
        # assert len(mask_files) == 1, f'{idx}: {mask_files}'

        # use Pillow's Image to read .gif mask
        # https://answers.opencv.org/question/185929/how-to-read-gif-in-python/

        # img = Image.open(img_files[0])
        # mask = Image.open(mask_files[0])

        # assert img.size == mask.size, f'{img.shape} # {mask.shape}'

        xx = torch.tensor(self.open_as_array(i, invert=self.pytorch),
                          dtype=torch.float32)
        yy = torch.tensor(self.open_mask(i, add_dims=False),
                          dtype=torch.torch.int64)

        return xx, yy

        # img = self.preprocess(img, "image")
        # mask = self.preprocess(ImageOps.grayscale(mask), "mask")
        # x = torch.from_numpy(img).float()
        # y = torch.from_numpy(mask).float()

        # return x,y
