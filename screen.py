import numpy as np
from sources import *
from mask import generate_mask
from util import *
import warnings
from tqdm import tqdm


class Screen:
    def __init__(self, resolution=(1920, 1080), partition=(1, 1), source_list=None, sources_gain=1, grey_levels=256,
                 mask_levels=None, mask_list=None):
        self.resolution = resolution
        self.partition = partition
        self.source_plane_shape = (int(np.floor(resolution[Coordinate.X] * partition[0] / np.sum(partition))),
                                   resolution[Coordinate.Y])
        self.source_plane = SourcePlane(shape=self.source_plane_shape)
        if source_list:
            for source in tqdm(source_list):
                self.source_plane.add_source(**source)
            print("Sources have been successfully added.\n")
        self.sources_gain = sources_gain
        self.mask_shape = (resolution[Coordinate.X] - self.source_plane_shape[Coordinate.X],
                           resolution[Coordinate.Y])
        self.grey_levels = grey_levels
        self.mask_levels = mask_levels if mask_levels else grey_levels
        self.mask_list = mask_list if mask_list else [{}]
        self.current_mask_number = 0
        self.mask = None
        self.swap = False

    def update_mask(self):
        self.current_mask_number += 1
        if self.current_mask_number > len(self.mask_list):
            return False
        self.mask = generate_mask(self.mask_shape, **self.mask_list[self.current_mask_number - 1]) / (2 * np.pi) * \
                    self.mask_levels
        print("Mask has been updated.\n")
        return True

    def catch_radiation(self):
        mod, remainder = np.divmod(self.sources_gain * self.source_plane.show_radiation(), self.grey_levels)
        if mod.any() > 0:
            warnings.warn("Sources' intensity is out of bounds.\n")
        return remainder

    def update_screen(self):
        if self.swap:
            return np.hstack([self.mask, self.catch_radiation()])
        return np.hstack([self.catch_radiation(), self.mask])

    def swap_screen(self):
        self.swap = not self.swap
        print("Screen has been swapped.\n")
