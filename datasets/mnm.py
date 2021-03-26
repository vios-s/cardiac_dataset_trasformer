import os
import sys
import json
import typing
import torch
import csv
import numpy as np
import albumentations
import nibabel as nib

from skimage import transform
from torch.utils.data.dataset import Dataset
from warnings import simplefilter
from matplotlib import pyplot as plt

simplefilter(action='ignore', category=FutureWarning)

class GenMNM(Dataset):
    def __init__(self,
        data_dir: str,
        slice_num: int,
        data_mode: str,
        equalize: bool=True,
    ) -> None:
        super(GenMNM, self).__init__()
        self.data_dir = data_dir
        self.slice_num = slice_num
        self.data_mode = data_mode
        targets = {}
        targets_file = os.path.join(self.data_dir, 'diagnosis.csv')
        with open(targets_file, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for i, row in enumerate(spamreader):
                if i == 0:
                    continue
                subject, _, _, _, t, _, _, _, _ = row[0].split(',')
                if t == 'NOR':
                    target = 0
                elif t == 'DCM':
                    target = 1
                elif t == 'HCM':
                    target = 2
                elif t == 'RV':
                    target = 3
                else: # any other class
                    target = 99
                targets[subject] = {}
                targets[subject]['target'] = {}
                targets[subject]['target'] = target
        labels = {}
        centers = {}
        labels_file = os.path.join(self.data_dir, 'mnms_dataset_info.csv')
        with open(labels_file, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for i, row in enumerate(spamreader):
                if i == 0:
                    continue
                subject, _, _, c, ed, es = row[0].split(',')
                labels[subject] = {}
                labels[subject]['ED'] = {}
                labels[subject]['ED'] = ed
                labels[subject]['ES'] = {}
                labels[subject]['ES'] = es
                centers[subject] = {}
                centers[subject]['center'] = c
        self.data = {}
        self.data_dir += os.sep + 'Labeled'
        for vendor in os.listdir(self.data_dir):
            if 'xlsx' in vendor or 'csv' in vendor:
                continue
            v_path = os.path.join(self.data_dir, vendor)
            for subject in os.listdir(v_path):
                s_path = os.path.join(v_path, subject)
                if subject not in self.data:
                    self.data[subject] = {}
                    self.data[subject]['name'] = {}
                    self.data[subject]['name'] = subject + '-' + vendor
                    self.data[subject]['vendor'] = {}
                    self.data[subject]['vendor'] = vendor
                    for v in centers:
                        if v == subject:
                            self.data[subject]['center'] = {}
                            self.data[subject]['center'] = centers[v]['center']
                    for v in labels:
                        if v == subject:
                            self.data[subject]['ED'] = {}
                            self.data[subject]['ED'] = labels[v]['ED']
                            self.data[subject]['ES'] = {}
                            self.data[subject]['ES'] = labels[v]['ES']
                    for v in targets:
                        if v == subject:
                            self.data[subject]['target'] = {}
                            self.data[subject]['target'] = targets[v]['target']
                    for sample in os.listdir(s_path):
                        fname, _, _ = sample.split('.')
                        if fname[-2:] == 'gt':
                            masks = os.path.join(s_path, sample)
                            self.data[subject]['masks'] = {}
                            self.data[subject]['masks'] = masks
                        elif fname == 'old':
                            continue
                        else:
                            images = os.path.join(s_path, sample)
                            self.data[subject]['images'] = {}
                            self.data[subject]['images'] = images
        root_path = self.data_dir.split('Labeled')[0]
        root_path += 'Unlabeled'
        for vendor in os.listdir(root_path):
            if 'xlsx' in vendor or 'csv' in vendor:
                continue
            v_path = os.path.join(root_path, vendor)
            for subject in os.listdir(v_path):
                s_path = os.path.join(v_path, subject)
                if subject not in self.data:
                    self.data[subject] = {}
                    self.data[subject]['name'] = {}
                    self.data[subject]['name'] = subject + '-' + vendor
                    self.data[subject]['vendor'] = {}
                    self.data[subject]['vendor'] = vendor
                    self.data[subject]['ED'] = {}
                    self.data[subject]['ED'] = '99'
                    self.data[subject]['ES'] = {}
                    self.data[subject]['ES'] = '99'
                    for v in centers:
                        if v == subject:
                            self.data[subject]['center'] = {}
                            self.data[subject]['center'] = centers[v]['center']
                    for v in targets:
                        if v == subject:
                            self.data[subject]['target'] = {}
                            self.data[subject]['target'] = targets[v]['target']
                    for sample in os.listdir(s_path):
                        fname, _, _ = sample.split('.')
                        if fname == 'old':
                            continue
                        else:
                            images = os.path.join(s_path, sample)
                            self.data[subject]['images'] = {}
                            self.data[subject]['images'] = images

    def __getitem__(self, idx):
        pass

    def __len__(self) -> int:
        return len(self.data)

    def create_labeled_dataset(self,
        path_to_dir: str,
        slice_num: int,
        vendor: str
    ) -> None:
        try:
            os.mkdir(path_to_dir)
            os.mkdir(path_to_dir + os.sep + 'images')
            os.mkdir(path_to_dir + os.sep + 'masks')
            os.mkdir(path_to_dir + os.sep + 'labels')
        except OSError:
            print ("Creation of directories in %s failed" % path_to_dir)
        targets_list = []
        if slice_num == -1:
            for sample in self.data:
                if self.data[sample]['ED'] != '99' and vendor == self.data[sample]['vendor']:
                    images = self._resample_raw_image(self.data[sample]['images'])
                    if self.data[sample]['vendor'] == 'vendorA':
                        images = (images / 704.77) * 255.0
                    elif self.data[sample]['vendor'] == 'vendorB':
                        if self.data[sample]['center'] == '2':
                            images = (images / 347.424) * 255.0
                        else:
                            images = (images / 1503.283) * 255.0
                    elif self.data[sample]['vendor'] == 'vendorC':
                        images = (images / 1709.734) * 255.0
                    else:
                        images = (images / 7507.834) * 255.0
                    masks = self._resample_raw_image(self.data[sample]['masks'], binary=True)
                    images_cropped, masks_cropped = self._crop_same(images.transpose(2,0,1,3), masks.transpose(2,0,1,3), (224, 224))
                    images_cropped = torch.from_numpy(np.float32(images_cropped.transpose(3,0,1,2)))
                    masks_cropped = torch.from_numpy(np.float32(masks_cropped.transpose(3,0,1,2).copy()))
                    for slice_idx in range(images_cropped.shape[1]):
                        image_ed = images_cropped[int(self.data[sample]['ED'])][slice_idx]
                        image_es = images_cropped[int(self.data[sample]['ES'])][slice_idx]
                        masks_ed = masks_cropped[int(self.data[sample]['ED'])][slice_idx]
                        masks_es = masks_cropped[int(self.data[sample]['ES'])][slice_idx]
                        masks_list = [masks_ed, masks_ed]
                        masks = torch.zeros(3, masks_ed.shape[0], masks_ed.shape[1])
                        for i, m in enumerate(masks_list):
                            if i == 0:
                                image = image_ed
                                frame = int(self.data[sample]['ED'])
                            else:
                                image = image_es
                                frame = int(self.data[sample]['ES'])
                            mask_LV, mask_MYO, mask_RV = m.clone(), m.clone(), m.clone()
                            mask_MYO[mask_MYO != 1] = 0
                            mask_LV[mask_LV != 2] = 0
                            mask_LV[mask_LV == 2] = 1
                            mask_RV[mask_RV != 3] = 0
                            mask_RV[mask_RV == 3] = 1
                            masks[0] = mask_LV
                            masks[1] = mask_MYO
                            masks[2] = mask_RV
                            if 'target' in self.data[sample]:
                                targets_list.append(self.data[sample]['target'])
                            else:
                                targets_list.append(99)
                            self._save_mask(masks.numpy(), path_to_dir, self.data[sample]['name'], frame, slice_idx)
                            self._save_intensity_image(image.numpy(), path_to_dir, self.data[sample]['name'], frame, slice_idx)
        else:
            for sample in self.data:
                if self.data[sample]['ED'] != '99' and vendor == self.data[sample]['vendor']:
                    images = self._resample_raw_image(self.data[sample]['images'])
                    if self.data[sample]['vendor'] == 'vendorA':
                        images = (images / 704.77) * 255.0
                    elif self.data[sample]['vendor'] == 'vendorB':
                        if self.data[sample]['center'] == '2':
                            images = (images / 347.424) * 255.0
                        else:
                            images = (images / 1503.283) * 255.0
                    elif self.data[sample]['vendor'] == 'vendorC':
                        images = (images / 1709.734) * 255.0
                    else:
                        images = (images / 7507.834) * 255.0
                    masks = self._resample_raw_image(self.data[sample]['masks'], binary=True)
                    images_cropped, masks_cropped = self._crop_same(images.transpose(2,0,1,3), masks.transpose(2,0,1,3), (224, 224))
                    images_cropped = torch.from_numpy(np.float32(images_cropped.transpose(3,0,1,2)))
                    masks_cropped = torch.from_numpy(np.float32(masks_cropped.transpose(3,0,1,2).copy()))
                    image_ed = images_cropped[int(self.data[sample]['ED'])][slice_num]
                    image_es = images_cropped[int(self.data[sample]['ES'])][slice_num]
                    masks_ed = masks_cropped[int(self.data[sample]['ED'])][slice_num]
                    masks_es = masks_cropped[int(self.data[sample]['ES'])][slice_num]
                    masks_list = [masks_ed, masks_ed]
                    masks = torch.zeros(3, masks_ed.shape[0], masks_ed.shape[1])
                    for i, m in enumerate(masks_list):
                        if i == 0:
                            image = image_ed
                            frame = int(self.data[sample]['ED'])
                        else:
                            image = image_es
                            frame = int(self.data[sample]['ES'])
                        mask_LV, mask_MYO, mask_RV = m.clone(), m.clone(), m.clone()
                        mask_MYO[mask_MYO != 1] = 0
                        mask_LV[mask_LV != 2] = 0
                        mask_LV[mask_LV == 2] = 1
                        mask_RV[mask_RV != 3] = 0
                        mask_RV[mask_RV == 3] = 1
                        masks[0] = mask_LV
                        masks[1] = mask_MYO
                        masks[2] = mask_RV
                        if 'target' in self.data[sample]:
                            targets_list.append(self.data[sample]['target'])
                        else:
                            targets_list.append(99)
                        self._save_mask(masks.numpy(), path_to_dir, self.data[sample]['name'], frame, slice_num)
                        self._save_intensity_image(image.numpy(), path_to_dir, self.data[sample]['name'], frame, slice_num)
        with open(path_to_dir + '/labels/' + 'labels.json', 'w') as outfile:
            outfile.write(
                '[' +
                ',\n'.join(json.dumps(i) for i in targets_list) +
                ']\n'
            )

    def create_unlabeled_dataset(self,
        path_to_dir: str,
        slice_num: int,
        vendor: str
    ) -> None:
        try:
            os.mkdir(path_to_dir)
            os.mkdir(path_to_dir + os.sep + 'images')
            os.mkdir(path_to_dir + os.sep + 'labels')
        except OSError:
            print ("Creation of directories in %s failed" % path_to_dir)
        targets_list = []
        if slice_num == -1:
            for sample in self.data:
                if vendor == self.data[sample]['vendor']:
                    images = self._resample_raw_image(self.data[sample]['images'])
                    if self.data[sample]['vendor'] == 'vendorA':
                        images = (images / 704.77) * 255.0
                    elif self.data[sample]['vendor'] == 'vendorB':
                        if self.data[sample]['center'] == '2':
                            images = (images / 347.424) * 255.0
                        else:
                            images = (images / 1503.283) * 255.0
                    elif self.data[sample]['vendor'] == 'vendorC':
                        images = (images / 1709.734) * 255.0
                    else:
                        images = (images / 7507.834) * 255.0
                    images_cropped = self._crop_same(images.transpose(2,0,1,3), images, (224, 224), image_only=True)
                    images_cropped = torch.from_numpy(np.float32(images_cropped.transpose(3,0,1,2)))
                    for frame_idx in range(images_cropped.shape[0]):
                        for slice_idx in range(images_cropped.shape[1]):
                            image = images_cropped[frame_idx][slice_idx]
                            if 'target' in self.data[sample]:
                                targets_list.append(self.data[sample]['target'])
                            else:
                                targets_list.append(99)
                            self._save_intensity_image(image.numpy(), path_to_dir, self.data[sample]['name'], frame_idx, slice_idx)
        else:
            for sample in self.data:
                if vendor == self.data[sample]['vendor']:
                    images = self._resample_raw_image(self.data[sample]['images'])
                    if self.data[sample]['vendor'] == 'vendorA':
                        images = (images / 704.77) * 255.0
                    elif self.data[sample]['vendor'] == 'vendorB':
                        if self.data[sample]['center'] == '2':
                            images = (images / 347.424) * 255.0
                        else:
                            images = (images / 1503.283) * 255.0
                    elif self.data[sample]['vendor'] == 'vendorC':
                        images = (images / 1709.734) * 255.0
                    else:
                        images = (images / 7507.834) * 255.0
                    images_cropped = self._crop_same(images.transpose(2,0,1,3), images, (224, 224), image_only=True)
                    images_cropped = torch.from_numpy(np.float32(images_cropped.transpose(3,0,1,2)))
                    for frame_idx in range(images_cropped.shape[0]):
                        image = images_cropped[frame_idx][slice_num]
                        if 'target' in self.data[sample]:
                            targets_list.append(self.data[sample]['target'])
                        else:
                            targets_list.append(99)
                        self._save_intensity_image(image.numpy(), path_to_dir, self.data[sample]['name'], frame_idx, slice_num)
        with open(path_to_dir + '/labels/' + 'labels.json', 'w') as outfile:
            outfile.write(
                '[' +
                ',\n'.join(json.dumps(i) for i in targets_list) +
                ']\n'
            )

    def _resample_raw_image(self,       # Load raw data (image/mask) and resample to fixed resolution.
            m_nii_fname: str,
            value_crop: bool=True,
            binary: bool=False,         # boolean to define binary masks or not
            res: float=1.2
    )-> np.array:
        new_res = (res, res)
        im_nii = nib.load(m_nii_fname)
        im_data = im_nii.get_data()
        voxel_size = im_nii.header.get_zooms()
        scale_vector = [voxel_size[i] / new_res[i] for i in range(len(new_res))]
        order = 0 if binary else 1
        scale_vector = [ 1.10, 1.10, 1, 1]
        dims = im_data.shape
        rescaled = transform.rescale(im_data, scale_vector, order=order, preserve_range=True, mode='constant')
        rotated = np.swapaxes(rescaled[:,:,:10,:], 0, 1)[:,::-1,:,:]
        if value_crop and binary == False:
            p5 = np.percentile(rotated.flatten(), 5)
            p95 = np.percentile(rotated.flatten(), 95)
            rotated = np.clip(rotated, p5, p95)
        return rotated

    def _crop_same(self,
        im: np.array,               # List of images. Each element should be 4-dimensional, (slice,height,width,channel)
        m: np.array,                # List of masks. Each element should be 4-dimensional, (slice,height,width,channel)
        size: tuple,                # Dimensions to crop the images to.
        mode: str='equal',          # [equal, left, right]. Denotes where to crop pixels from. Defaults to middle.
        pad_mode: str='edge',       # ['edge', 'constant']. 'edge' pads using the values of the edge pixels, 'constant' pads with a constant value
        image_only: bool=False
    ) -> typing.List[np.array]:
        min_w, min_h = size[0], size[1]
        if im.shape[1] > min_w:
            im = self._crop(im, 1, min_w, mode)
        if im.shape[1] < min_w:
            im = self._pad(im, 1, min_w, pad_mode)
        if im.shape[2] > min_h:
            im = self._crop(im, 2, min_h, mode)
        if im.shape[2] < min_h:
            im = self._pad(im, 2, min_h, pad_mode)
        if image_only:
            return im
        else:
            if m.shape[1] > min_w:
                m = self._crop(m, 1, min_w, mode)
            if m.shape[1] < min_w:
                m = self._pad(m, 1, min_w, pad_mode)
            if m.shape[2] > min_h:
                m = self._crop(m, 2, min_h, mode)
            if m.shape[2] < min_h:
                m = self._pad(m, 2, min_h, pad_mode)
            return im, m

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
        subject_idx: str,
        frame_idx: int,
        slice_idx: int
    ) -> None:
        frame_idx = '%02d' % frame_idx
        slice_idx = '%02d' % slice_idx
        plt.imsave(
                path + os.sep + 'images' + os.sep + 'subject' + subject_idx + '_frame' + frame_idx + '_slice' + slice_idx + '.png',
                y,
                vmin=0,
                vmax=255,
                cmap='gray'
            )

    def _save_mask(self,
            y: np.array,
            path: str,
            subject_idx: str,
            frame_idx: int,
            slice_idx: int
    ) -> None:
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
        plt.imsave(
                path + os.sep + 'masks' + os.sep + 'subject' + subject_idx + '_frame' + frame_idx + '_slice' + slice_idx + '_RV' + '.png',
                y[2],
                cmap='gray'
            )
