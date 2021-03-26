import os
import sys
import json
import typing
import torch
import numpy as np
import albumentations
import nibabel as nib

from skimage import transform
from torch.utils.data import random_split
from torch.utils.data.dataset import Dataset
from warnings import simplefilter
from matplotlib import pyplot as plt

simplefilter(action='ignore', category=FutureWarning)

class GenACDC(Dataset):
    def __init__(self,
        data_dir: str,              # path to ACDC nii data
        slice_num: int,
        data_mode: str,
        resolution: float
    ) -> None:
        super(GenACDC, self).__init__()
        self.data_dir = data_dir
        self.slice_num = slice_num
        self.data_mode = data_mode
        self.res = resolution
        self.transform = albumentations.Compose([
            albumentations.augmentations.Normalize(mean=0.5, std=0.5, max_pixel_value=1.0)]
        )
        if self.data_mode == 'labeled':
            self.data = self._load_labeled_data()
        else:
            self.data = self._load_unlabeled_data(include_all=True)

    def __getitem__(self,
            index: int
        ) -> typing.Tuple[typing.Any, typing.Any]:
        img, mask, label = self.data['images'][index], self.data['masks'][index], self.data['labels'][index]
        augmented_img = self.transform(image=img.numpy().transpose(1, 2, 0))
        img = torch.from_numpy(augmented_img['image'].transpose(2, 0, 1))
        return img, mask, label

    def __len__(self) -> int:
        return len(self.data['images'])

    def create_labeled_dataset(self,
        path_to_dir: str,
        slice_num: int
    ) -> None:
        try:
            os.mkdir(path_to_dir)
            os.mkdir(path_to_dir + os.sep + 'images')
            os.mkdir(path_to_dir + os.sep + 'masks')
            os.mkdir(path_to_dir + os.sep + 'labels')
        except OSError:
            print ("Creation of directories in %s failed" % path_to_dir)
        if slice_num == -1:
            images = self.data['images']
            masks = self.data['masks']
            targets = []
            for i in range(images.shape[0]):
                for slice_num in range(images[i].shape[0]):
                    self._save_intensity_image(images[i][slice_num].squeeze(0).numpy(), path_to_dir, self.data['subject_idx'][i], self.data['frame_idx'][i], slice_num)
                    self._save_mask(masks[i][slice_num].squeeze(0).numpy(), path_to_dir, self.data['subject_idx'][i], self.data['frame_idx'][i], slice_num)
                    targets.append(self.data['labels'][i].item())
        else:
            images = self.data['images'][:, slice_num]
            masks = self.data['masks'][:, slice_num]
            for i in range(images.shape[0]):
                self._save_intensity_image(images[i].squeeze(0).numpy(), path_to_dir, self.data['subject_idx'][i], self.data['frame_idx'][i], slice_num)
                self._save_mask(masks[i].squeeze(0).numpy(), path_to_dir, self.data['subject_idx'][i], self.data['frame_idx'][i], slice_num)
            targets = self.data['labels'].tolist()
        with open(path_to_dir + '/labels/' + 'labels.json', 'w') as outfile:
            outfile.write(
                '[' +
                ',\n'.join(json.dumps(i) for i in targets) +
                ']\n'
            )

    def create_unlabeled_dataset(self,
        path_to_dir: str,
        slice_num: int
    ) -> None:
        try:
            os.mkdir(path_to_dir)
            os.mkdir(path_to_dir + os.sep + 'images')
            os.mkdir(path_to_dir + os.sep + 'labels')
        except OSError:
            print ("Creation of directories in %s failed" % path_to_dir)
        if slice_num == -1:
            images = self.data['images']
            targets = []
            for i in range(images.shape[0]):
                for slice_num in range(images[i].shape[0]):
                    self._save_intensity_image(images[i][slice_num].squeeze(0).numpy(), path_to_dir, self.data['subject_idx'][i], self.data['frame_idx'][i], slice_num)
                    targets.append(self.data['labels'][i].item())
        else:
            images = self.data['images'][:, slice_num]
            for i in range(images.shape[0]):
                self._save_intensity_image(images[i].squeeze(0).numpy(), path_to_dir, self.data['subject_idx'][i], self.data['frame_idx'][i], slice_num)
            targets = self.data['labels'].tolist()
        with open(path_to_dir + '/labels/' + 'labels.json', 'w') as outfile:
            outfile.write(
                '[' +
                ',\n'.join(json.dumps(i) for i in targets) +
                ']\n'
            )

    def _load_labeled_data(self) -> typing.Dict[str, np.array]:
        td = {}
        images, masks, labels, subject_idx, frame_idx = self._load_raw_labeled_data()
        td = {
            "images": torch.from_numpy(np.float32(images)),
            "masks": torch.from_numpy(np.float32(masks)),
            "labels": torch.from_numpy(np.float32(labels)),
            "subject_idx": torch.from_numpy(subject_idx),
            "frame_idx": torch.from_numpy(frame_idx)
        }
        return td

    def _load_raw_labeled_data(self) -> typing.List[np.array]:
        images, masks_lv, masks_rv, masks_myo, labels = [], [], [], [], []
        subject_idx, frame_idx = [], []
        volumes = list(range(1, 101))
        for patient_i in volumes:
            patient = 'patient%03d' % patient_i
            patient_folder = os.path.join(self.data_dir, patient)
            # retrieve pathology label from patient's Info.cfg file
            cfg = [f for f in os.listdir(patient_folder) if 'cfg' in f and f.startswith('Info')]
            label_file = open(os.path.join(patient_folder, cfg[0]), mode = 'r')
            lines = label_file.readlines()
            label_file.close()
            label_char = ''
            for line in lines:
                line = line.split(' ')
                if line[0] == 'Group:':
                    label_char = line[1]
            if label_char == 'NOR\n':
                label = 0
            elif label_char == 'MINF\n':
                label = 1
            elif label_char == 'DCM\n':
                label = 2
            elif label_char == 'HCM\n':
                label = 3
            else: # RV
                label = 4
            gt = [f for f in os.listdir(patient_folder) if 'gt' in f and f.startswith(patient + '_frame')]
            ims = [f.replace('_gt', '') for f in gt]
            for i in range(len(ims)):
                subject_idx.append(patient_i)
                frame_idx.append(int(ims[i].split('.')[0].split('frame')[-1]))
                im = self._process_raw_image(ims[i], patient_folder)
                im = np.expand_dims(im, axis=-1)

                m = self._resample_raw_image(gt[i], patient_folder, binary=True)
                m = np.expand_dims(m, axis=-1)
                images.append(im)

                # convert 3-dim mask array to 3 binary mask arrays for lv, rv, myo
                m_lv = m.copy()
                m_lv[m != 3] = 0
                m_lv[m == 3] = 1
                masks_lv.append(m_lv)

                m_rv = m.copy()
                m_rv[m != 1] = 0
                m_rv[m == 1] = 1
                masks_rv.append(m_rv)

                m_myo = m.copy()
                m_myo[m != 2] = 0
                m_myo[m == 2] = 1
                masks_myo.append(m_myo)

                labels.append(label)

        # move slice axis to the first position
        images = [np.moveaxis(im, 2, 0) for im in images]
        masks_lv = [np.moveaxis(m, 2, 0) for m in masks_lv]
        masks_rv = [np.moveaxis(m, 2, 0) for m in masks_rv]
        masks_myo = [np.moveaxis(m, 2, 0) for m in masks_myo]

        # normalize images
        for i in range (len(images)):
            images[i] = (images[i] / 757.4495) * 255.0

        # crop images and masks to the same pixel dimensions and concatenate all data
        images_cropped, masks_lv_cropped = self._crop_same(images, masks_lv, (224, 224))
        _, masks_rv_cropped = self._crop_same(images, masks_rv, (224, 224))
        _, masks_myo_cropped = self._crop_same(images, masks_myo, (224, 224))

        # images_cropped = np.expand_dims(images_cropped[:], axis=0)
        images_cropped = [np.expand_dims(image, axis=0) for image in images_cropped]
        images_cropped = np.concatenate(images_cropped, axis=0)
        masks_cropped = np.concatenate([masks_myo_cropped, masks_lv_cropped, masks_rv_cropped], axis=-1)
        labels = np.array(labels)
        subject_idx = np.array(subject_idx)
        frame_idx = np.array(frame_idx)

        return images_cropped.transpose(0,1,4,2,3), masks_cropped.transpose(0,1,4,2,3), labels, subject_idx, frame_idx 

    def _load_unlabeled_data(self,
        include_all: bool=False
        ) -> typing.Dict[str, torch.Tensor]:
        td = {}
        images, labels, subject_idx, frame_idx = self._load_raw_unlabeled_data(include_all)
        td = {
            "images": torch.from_numpy(np.float32(images)),
            "labels": torch.from_numpy(np.float32(labels)),
            "subject_idx": torch.from_numpy(subject_idx),
            "frame_idx": torch.from_numpy(frame_idx)
        }
        return td

    def _load_raw_unlabeled_data(self,
            include_all: bool
        ) -> np.array:
        images, labels = [], []
        subject_idx, frame_idx = [], []
        volumes = list(range(1, 101))
        more_than_10_cnt = 0
        for patient_i in volumes:
            patient = 'patient%03d' % patient_i
            patient_folder = os.path.join(self.data_dir, patient)
            # retrieve pathology label from patient's Info.cfg file
            cfg = [f for f in os.listdir(patient_folder) if 'cfg' in f and f.startswith('Info')]
            label_file = open(os.path.join(patient_folder, cfg[0]), mode = 'r')
            lines = label_file.readlines()
            label_file.close()
            label_char = ''
            for line in lines:
                line = line.split(' ')
                if line[0] == 'Group:':
                    label_char = line[1]
            if label_char == 'NOR\n':
                label = 0
            elif label_char == 'MINF\n':
                label = 1
            elif label_char == 'DCM\n':
                label = 2
            elif label_char == 'HCM\n':
                label = 3
            else: #RV
                label = 4
            im_name = patient + '_4d.nii.gz'
            im = self._process_raw_image(im_name, patient_folder)
            frames = range(im.shape[-1])
            if include_all:
                gt = [f for f in os.listdir(patient_folder) if 'gt' in f and not f.startswith('._')]
                gt_ims = [f.replace('_gt', '') for f in gt if not f.startswith('._')]
                exclude_frames = [int(gt_im.split('.')[0].split('frame')[1]) for gt_im in gt_ims]
                frames = [f for f in range(im.shape[-1]) if (f > exclude_frames[0] and f < exclude_frames[1]) or f == exclude_frames[0] or f == exclude_frames[1]]
            else:
                gt = [f for f in os.listdir(patient_folder) if 'gt' in f and not f.startswith('._')]
                gt_ims = [f.replace('_gt', '') for f in gt if not f.startswith('._')]
                exclude_frames = [int(gt_im.split('.')[0].split('frame')[1]) for gt_im in gt_ims]
                frames = [f for f in range(im.shape[-1]) if f not in exclude_frames and f > exclude_frames[0] and f < exclude_frames[1]]
            for frame in frames:
                subject_idx.append(patient_i)
                frame_idx.append(frame)
                im_res = im[:, :, :, frame]
                if im_res.sum() == 0:
                    print('Skipping blank images')
                    continue
                im_res = np.expand_dims(im_res, axis=-1)
                images.append(im_res)
                labels.append(label)
        images = [np.moveaxis(im, 2, 0) for im in images]
        # normalize images
        for i in range (len(images)):
            images[i] = np.round((images[i] / 757.4495) * 255.0)
        zeros = [np.zeros(im.shape) for im in images]
        images_cropped, _ = self._crop_same(images, zeros, (224, 224))
        images_cropped = np.concatenate(np.expand_dims(images_cropped, axis=0), axis=0)#[..., 0]
        labels = np.array(labels)
        subject_idx = np.array(subject_idx)
        frame_idx = np.array(frame_idx)
        return images_cropped.transpose(0,1,4,2,3), labels, subject_idx, frame_idx
    
    def _resample_raw_image(self,   # Load raw data (image/mask) and resample to fixed resolution.
            mask_fname: str,        # filename of mask
            patient_folder: str,    # folder containing patient data
            binary: bool=False      # boolean to define binary masks or not
        )-> np.array:
        m_nii_fname = os.path.join(patient_folder, mask_fname)
        new_res = (self.res, self.res)
        im_nii = nib.load(m_nii_fname)
        im_data = im_nii.get_data()
        voxel_size = im_nii.header.get_zooms()
        sform_matrix = im_nii.header.get_sform()
        scale_vector = [voxel_size[i] / new_res[i] for i in range(len(new_res))]
        order = 0 if binary else 1
        result = []
        dims = im_data.shape
        if len(dims) < 4:
            for i in range(im_data.shape[-1]):
                if i > 5:
                    break
                im = im_data[..., i]
                rescaled = transform.rescale(im, scale_vector, order=order, preserve_range=True, mode='constant')
                rotated = transform.rotate(rescaled, 270.0)
                result.append(np.expand_dims(np.flip(rotated, axis=0), axis=-1))
        else:
            for i in range(im_data.shape[-1]):
                inner_im_data = im_data[..., i]
                all_slices = []
                for j in range(inner_im_data.shape[-1]):
                    if j > 5:
                        break
                    im = inner_im_data[..., j]
                    rescaled = transform.rescale(im, scale_vector, order=order, preserve_range=True, mode='constant')
                    rotated = transform.rotate(rescaled, 270.0)
                    all_slices.append(np.expand_dims(rotated, axis=-1))
                result.append(np.expand_dims(np.concatenate(all_slices, axis=-1), axis=-1))
        return np.concatenate(result, axis=-1)

    def _process_raw_image(self,     # Normalise and crop extreme values of an image
            im_fname: str,           # filename of the image
            patient_folder: str,     # folder of patient data
            value_crop: bool=True    # True/False to crop values between 5/95 percentiles
        ) -> typing.List:
        im = self._resample_raw_image(im_fname, patient_folder, binary=False)
        # crop to 5-95%
        if value_crop:
            p5 = np.percentile(im.flatten(), 5)
            p95 = np.percentile(im.flatten(), 95)
            im = np.clip(im, p5, p95)
        return im

    def _crop_same(self,
            image_list: list,       # List of images. Each element should be 4-dimensional, (slice,height,width,channel)
            mask_list: list,        # List of masks. Each element should be 4-dimensional, (slice,height,width,channel)
            size: tuple,            # Dimensions to crop the images to.
            mode: str='equal',      # [equal, left, right]. Denotes where to crop pixels from. Defaults to middle.
            pad_mode: str='edge',   # ['edge', 'constant']. 'edge' pads using the values of the edge pixels, 'constant' pads with a constant value
            image_only: bool=False
        ) -> typing.List[np.array]:
        min_w = np.min([im.shape[1] for im in image_list]) if size[0] is None else size[0]
        min_h = np.min([im.shape[2] for im in image_list]) if size[1] is None else size[1]

        if image_only:
            img_result = []
            for i in range(len(image_list)):
                im = image_list[i]
                if im.shape[1] > min_w:
                    im = self._crop(im, 1, min_w, mode)
                if im.shape[1] < min_w:
                    im = self._pad(im, 1, min_w, pad_mode)
                if im.shape[2] > min_h:
                    im = self._crop(im, 2, min_h, mode)
                if im.shape[2] < min_h:
                    im = self._pad(im, 2, min_h, pad_mode)
                img_result.append(im)
            return img_result
        else:
            img_result, msk_result = [], []
            for i in range(len(mask_list)):
                im = image_list[i]
                m = mask_list[i]
                if m.shape[1] > min_w:
                    m = self._crop(m, 1, min_w, mode)
                if im.shape[1] > min_w:
                    im = self._crop(im, 1, min_w, mode)
                if m.shape[1] < min_w:
                    m = self._pad(m, 1, min_w, pad_mode)
                if im.shape[1] < min_w:
                    im = self._pad(im, 1, min_w, pad_mode)
                if m.shape[2] > min_h:
                    m = self._crop(m, 2, min_h, mode)
                if im.shape[2] > min_h:
                    im = self._crop(im, 2, min_h, mode)
                if m.shape[2] < min_h:
                    m = self._pad(m, 2, min_h, pad_mode)
                if im.shape[2] < min_h:
                    im = self._pad(im, 2, min_h, pad_mode)
                img_result.append(im)
                msk_result.append(m)
            return img_result, msk_result

    def _crop(self,
            image: list,
            dim: int,
            nb_pixels: int,
            mode: str
        ) -> typing.Union[None, list]:
        diff = image.shape[dim] - nb_pixels
        if mode == 'equal':
            l = int(np.ceil(diff / 2))
            r = image.shape[dim] - l
        elif mode == 'right':
            l = 0
            r = nb_pixels
        elif mode == 'left':
            l = diff
            r = image.shape[dim]
        else:
            raise 'Unexpected mode: %s. Expected to be one of [equal, left, right].' % mode

        if dim == 1:
            return image[:, l:r, :, :]
        elif dim == 2:
            return image[:, :, l:r, :]
        else:
            return None

    def _pad(self,
            image: list,
            dim: int,
            nb_pixels: int,
            mode: str='edge'
        ) -> list:
        diff = nb_pixels - image.shape[dim]
        l = int(diff / 2)
        r = int(diff - l)
        if dim == 1:
            pad_width = ((0, 0), (l, r), (0, 0), (0, 0))
        elif dim == 2:
            pad_width = ((0, 0), (0, 0), (l, r), (0, 0))
        else:
            return None
        if mode == 'edge':
            new_image = np.pad(image, pad_width, 'edge')
        elif mode == 'constant':
            new_image = np.pad(image, pad_width, 'constant', constant_values=np.min(image))
        else:
            raise Exception('Invalid pad mode: ' + mode)

        return new_image

    def _save_intensity_image(self,
            y: np.array,
            path: str,
            subject_idx: int,
            frame_idx: int,
            slice_idx: int
        ) -> None:
        subject_idx = '%03d' % subject_idx
        frame_idx = '%02d' % frame_idx
        slice_idx = '%02d' % slice_idx
        plt.imsave(
                path + os.sep + 'images' + os.sep + 'subject' + subject_idx + '_frame' + frame_idx + '_slice' + slice_idx + '.png',
                y,
                cmap='gray'
            )

    def _save_mask(self,
            y: np.array,
            path: str,
            subject_idx: int,
            frame_idx: int,
            slice_idx: int
        ) -> None:
        subject_idx = '%03d' % subject_idx
        frame_idx = '%02d' % frame_idx
        slice_idx = '%02d' % slice_idx
        plt.imsave(
                path + os.sep + 'masks' + os.sep + 'subject' + subject_idx + '_frame' + frame_idx + '_slice' + slice_idx + '_MYO' + '.png',
                y[0],
                cmap='gray'
            )
        plt.imsave(
                path + os.sep + 'masks' + os.sep + 'subject' + subject_idx + '_frame' + frame_idx + '_slice' + slice_idx + '_LV' + '.png',
                y[1],
                cmap='gray'
            )
        plt.imsave(path + os.sep + 'masks' + os.sep + 'subject' + subject_idx + '_frame' + frame_idx + '_slice' + slice_idx + '_RV' + '.png',
                y[2],
                cmap='gray'
            )