import os
import typing
import torch
import pytorch_lightning as pl

from datasets import GenACDC

class ACDCDataModuleCreator(pl.LightningDataModule):
    def __init__(self, 
            data_dir: str='./',             # path to ACDC nii data
            slice_num: int=-1,              # cardiac slice number, {0, ..., 6} | -1 is for all slices
            save_data_dir: str='./',        # path to save processed frames
            data_mode: str='labeled',       # labeled || unlabeled
            resolution: float=1.37          # MRI resolution, default value 1.37
        ):
        super().__init__()
        self.data_dir = data_dir
        self.slice_num = slice_num
        self.data_mode = data_mode
        self.save_data_dir = save_data_dir
        self.resolution = resolution

    def setup(self,
            stage: str=None
        ) -> None:
        dataset = GenACDC(
                    data_dir=self.data_dir,
                    slice_num=self.slice_num,
                    data_mode=self.data_mode,
                    resolution=self.resolution
                )
        if self.data_mode == 'labeled':
            dataset.create_labeled_dataset(
                                    path_to_dir=self.save_data_dir + os.sep + self.data_mode  + '_' + str(self.slice_num),
                                    slice_num=self.slice_num
                                )
        else:
            dataset.create_unlabeled_dataset(
                                    path_to_dir=self.save_data_dir + os.sep + self.data_mode  + '_' + str(self.slice_num),
                                    slice_num=self.slice_num
                                )
