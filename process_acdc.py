import sys
import torch
import pytorch_lightning as pl

from dataloader import ACDCDataModuleCreator
from options.base_options import parse_arguments

if __name__ == "__main__":
    opt, uknown = parse_arguments(sys.argv)
    dataset = ACDCDataModuleCreator(
                            data_dir=opt.data_dir,
                            slice_num=opt.slice_num,
                            save_data_dir=opt.save_data_dir,
                            data_mode=opt.data_mode,
                            resolution=opt.resolution
                        )
    dataset.setup()

