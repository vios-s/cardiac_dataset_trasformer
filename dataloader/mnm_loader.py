import os
import typing
import torch
import pytorch_lightning as pl

from datasets import GenMNM

class MNMDataModuleCreator(pl.LightningDataModule):
    def __init__(self, 
            vendor: str,                    # vendor name: vendorA | vendorB | vendorC | vendorD
            data_dir: str='./',             # path to MnM nii data
            slice_num: int=-1,              # cardiac slice number, {0, ..., 6} | -1 is for all slices
            save_data_dir: str='./',        # path to save processed frames
            data_mode: str='labeled',       # labeled || unlabeled
        ):
        super().__init__()
        self.data_dir = data_dir
        self.vendor = vendor
        self.slice_num = slice_num
        self.data_mode = data_mode
        self.save_data_dir = save_data_dir

    def setup(self,
            stage: str=None
        ) -> None:
        dataset = GenMNM(
                    data_dir=self.data_dir,
                    slice_num=self.slice_num,
                    data_mode=self.data_mode
                )
        if self.data_mode == 'labeled':
            dataset.create_labeled_dataset(
                                    path_to_dir=self.save_data_dir + os.sep + self.data_mode + '_' + self.vendor + '_' + str(self.slice_num),
                                    slice_num=self.slice_num,
                                    vendor=self.vendor
                                )
        else:
            dataset.create_unlabeled_dataset(
                                    path_to_dir=self.save_data_dir + os.sep + self.data_mode + '_' + self.vendor + '_' + str(self.slice_num),
                                    slice_num=self.slice_num,
                                    vendor=self.vendor
                                )
