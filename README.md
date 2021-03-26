# Cardiac dataset transformer
Several medical datasets comprise data in DICOM format. Although this format includes a lot of valuable information (data + metadata in the header), it cannot be easily exploited with traditional image-based dataloaders (e.g. torchvision). Most of existing custom dataloaders load the DICOM data directly as tensors, even if we do not need all the information included in the "nii" file (e.g. all slices of an MRI), **overloading memory** and introducing **extra overhead** during training a DNN. In this repo, we propose dataset transformers that enable an image-based dataset generation from cardiac DICOM data. The generated datasets are ready-to-use with any generic image dataloader.


## Prerequisites
The architecture has been implemented using the following:
- Python 3.7.1
- PyTorch Lightning
- PyTorch 1.7


## Supported datasets
- **ACDC** - this dataset consists of cardiac images acquired from MRI scanners across 100 subjects, and provides pathology annotations (5 classes) for all images and pixel-level segmentation masks for end-systole and end-diastole per subject. ACDC was introduced as a challenge - [challenge URL](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html)
- **M&Ms** - this dataset consists of cardiac-MR images acquired from 4 different (known) sites, across 345 subjects. It provides end-systole and end-diastole pixel-level segmentation masks annotations for 320 out of 345 subjects and pathology annotations for all images. In our M&Ms transformer we use the 4 most dominant pathology classes. M&Ms was introduced as a challenge - [challenge URL](https://www.ub.edu/mnms/)


## Transforming ACDC

```
python process_acdc.py --data_dir /path/to/ACDC
```

## Transforming M&Ms

```
python process_mnm.py --data_dir /path/to/MnM
```

## License
