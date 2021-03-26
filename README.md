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
We assume the following structure when downloading ACDC from the provider.
```
acdc
	|
	|-----patient001
	|		|-----Info.cfg
	|		|-----patient001_4d.nii
	|		|-----patient001_frame01.nii
	|		|-----patient001_frame01_gt.nii
	|		|-----patient001_frame12.nii
	|		|-----patient001_frame12_gt.nii  
	|
	|-----patientXXX
	|		|-----Info.cfg
	|		|-----patientXXX_4d.nii
	|		|-----patientXXX_frameXX.nii
	|		|-----patientXXX_frameXX_gt.nii
	|		|-----patientXXX_frameXX.nii
	|		|-----patientXXX_frameXX_gt.nii  
	|
```
More details about the data structure:
- "Info.cfg" - configuration file that contains metadata, such as the pathology/disease class.
- "patientXXX_4d.nii" - the full sequence of the MRI in NIFTI format.
- "patientXXX_frameXX.nii" - end-systole or end-diastole frame in NIFTI format.
- "patientXXX_frameXX_gt.nii" - pixel-level annotation of the end-systole or end-diastole frame in NIFTI format. The annotation covers 3 semantic classes: left ventricular cavity (LV), myocardium (MYO) of the LV, and right ventricle (RV).

The structure of the generated dataset will be as follows.

```
python process_acdc.py --data_dir /path/to/ACDC
```

## Transforming M&Ms

```
python process_mnm.py --data_dir /path/to/MnM
```

## License
