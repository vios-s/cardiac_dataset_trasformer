import argparse

def parse_arguments(args):
    usage_text = (
        "SDNet Pytorch Implementation"
        "Usage:  python train.py [options],"
        "   with [options]:"
    )
    parser = argparse.ArgumentParser(description=usage_text)
    parser.add_argument("--data_dir", type=str, default='/home/sthermos/idcom_imaging/data/Cardiac/ACDC/segmentation/training', help='Path to nii format dataset.')
    parser.add_argument("--save_data_dir", type=str, default='/home/sthermos/idcom_imaging/data/Cardiac/ACDC/processed/dataset1', help='Path to save the processed dataset.')
    parser.add_argument("--slice_num", type=int, default=-1, help='Cardiac slice index. Default -1 is used to load all slices.')
    parser.add_argument("--data_mode", type=str, default='labeled', help='Type of data to load. labeled | unlabeled')
    parser.add_argument("--resolution", type=float, default=1.37, help='MRI resolution. NxN')
    parser.add_argument("--vendor", type=str, default='vendorA', help='Vendor name: vendorA | vendorB | vendorC | vendorD')
    return parser.parse_known_args(args)