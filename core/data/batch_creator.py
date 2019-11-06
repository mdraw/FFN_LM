from torch.utils import data
from typing import Sequence, List
import h5py
from .utils import *

import scipy.ndimage
from scipy import ndimage
from scipy import misc
import random
import random

import imageio
import imgaug as ia
import numpy as np
from imgaug import augmenters as iaa

class BatchCreator(data.Dataset):
    def __init__(
            self,
            input_h5data: List[str],
            input_size: Sequence[int],
            delta: Sequence[int],
            threshold: float = 0.9,
            train: bool = True):
        super(BatchCreator, self).__init__()

        self.shifts = []
        self.deltas = delta
        for dx in (-self.deltas[0], 0, self.deltas[0]):
            for dy in (-self.deltas[1], 0, self.deltas[1]):
                for dz in (-self.deltas[2], 0, self.deltas[2]):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    self.shifts.append((dx, dy, dz))
        random.shuffle(self.shifts)

        self.input_data = []
        self.label_data = []
        self.seed = []
        self.coor = []
        self.parse_h5py(input_h5data)

        # Early checks
        if len(self.input_data) != len(self.label_data):
            raise ValueError("input_h5data and target_h5data must be lists of same length!")

        self.input_size = np.array(input_size)
        self.seed_shape = self.input_size + np.array(delta) * 2
        self.label_radii = self.seed_shape // 2

        self.data_idx = 0
        self.coor_idx = 0

        self.image_patch = None
        self.label_patch = None
        self.seed_patch = None
        self.offset = None

        self.threshold = threshold
        self.num_classes = 2
        self.reset = True

    def parse_h5py(self, input):
        """解析数据"""
        for input_data in input:
            with h5py.File(input_data, 'r') as raw:
                self.input_data.append((raw['image'][()].astype(np.float32)-128) / 33.0)
                self.label_data.append(raw['label'][()])
                #coords_m = np.array([[143, 196, 62], [219, 147, 183], [218, 197, 149], [245, 70, 87], [220, 157, 124], [199, 182, 72],[138, 120, 70], [156, 178, 110]])
                self.coor.append(raw['coor'][()])#.astype(np.uint8))
                self.seed.append(logit(np.full(list(raw['label'][()].shape), 0.05, dtype=np.float32)))

    #def flip(self, data):

    def geometric_transform(self, data1, data2):
        t1 = random.choice([np.fliplr, np.flipud])
        data1 = t1(data1)
        data2 = t1(data2)

        t2 = random.choice([np.fliplr, np.flipud])
        data1 = t2(data1)
        data2 = t2(data2)
        degree = np.random.choice([90, 270, 180])
        data1 = ndimage.rotate(data1, degree)
        data2 = ndimage.rotate(data2, degree)

        return data1, data2

    def hue_transform1(self, image):
        z = image.shape[0]
        para = random.randrange(6, 14, 1)
        hue_m = para / 10
        aug = iaa.MultiplyHueAndSaturation((hue_m, hue_m))
        for stack in range(z):
            image[stack] = aug.augment_image(image[stack])

        return image

    def hue_transform2(self, image):
        z = image.shape[0]
        para = random.randrange(-20, 20, 1)
        hue_2 = para
        aug = iaa.AddToHueAndSaturation((hue_2, hue_2))
        for stack in range(z):
            image[stack] = aug.augment_image(image[stack])

        return image

    def brightness_transform1(self,image):
        z = image.shape[0]
        y = image.shape[1]
        x = image.shape[2]

        for zc in range(z):
            for yc in range(y):
                for xc in range(x):
                    pixel_data = image[zc, yc, xc]
                    intensity = pixel_data[0] + pixel_data[1] + pixel_data[2]
                    if intensity <= 90:
                        continue
                    if (intensity >= 90) & (intensity <= 360):
                        image[zc, yc, xc] = image[zc, yc, xc] + 50
                    if (intensity >= 540) & (intensity <= 785):
                        image[zc, yc, xc] = image[zc, yc, xc] - 50

        return image

    def brightness_transform2(self, image):

        delta = random.randrange(-70, 70, 1)
        z = image.shape[0]
        y = image.shape[1]
        x = image.shape[2]

        for zc in range(z):
            for yc in range(y):
                for xc in range(x):
                    pixel_data = image[zc, yc, xc]

                    intensity = pixel_data[0] + pixel_data[1] + pixel_data[2]

                    if intensity <= 60:
                        image[zc, yc, xc] = image[zc, yc, xc]
                    else:
                        image[zc, yc, xc] = image[zc, yc, xc] + delta
                        for channel in range(3):
                            if image[zc, yc, xc][channel] >= 255:
                                image[zc, yc, xc][channel] = 255

                        if (image[zc, yc, xc][0] + image[zc, yc, xc][1] + image[zc, yc, xc][2]) <= 120:
                            image[zc, yc, xc] = pixel_data

        return image
    def __getitem__(self, idx):

        self.coor_patch = self.coor[self.data_idx][idx]

        self.image_patch = center_crop_and_pad(self.input_data[self.data_idx], self.coor_patch, self.seed_shape).transpose(3, 0, 1, 2)

        self.label_patch = center_crop_and_pad(self.label_data[self.data_idx], self.coor_patch, self.seed_shape)




        self.label_patch = np.logical_and(self.label_patch > 0, np.equal(self.label_patch, self.label_patch[tuple(self.label_radii)]))
        self.label_patch = np.where(self.label_patch, np.ones(self.label_patch.shape)*0.95, np.ones(self.label_patch.shape)*0.05)

        self.seed_patch = center_crop_and_pad(self.seed[self.data_idx], self.coor_patch, self.seed_shape)
        self.seed_patch[tuple(self.label_radii)] = logit(0.95)

        return self.image_patch, self.label_patch, self.seed_patch, self.coor_patch

    def __len__(self):
        return len(self.coor[self.data_idx])
