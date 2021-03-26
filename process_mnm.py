import sys
import torch
import pytorch_lightning as pl

from dataloader import MNMDataModuleCreator
from options.base_options import parse_arguments

if __name__ == "__main__":
    opt, uknown = parse_arguments(sys.argv)
    dataset = MNMDataModuleCreator(
                            data_dir=opt.data_dir,
                            slice_num=opt.slice_num,
                            data_mode=opt.data_mode,
                            save_data_dir=opt.save_data_dir,
                            vendor=opt.vendor
                        )
    dataset.setup()

