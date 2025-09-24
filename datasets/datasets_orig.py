import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import json
import ipdb
from collections import defaultdict
DATASET_NAMES = [
    'BIPED',
    'MDBD',
    'CLASSIC',
    'C3VDv2'
]  # 8


def dataset_info(dataset_name):
    config = {
        'BIPED': {
            'img_height': 720, #720 # 1088
            'img_width': 1280, # 1280 5 1920
            'test_list': 'test_pair.lst',
            'train_list': 'train_rgb.lst',
            'data_dir': 'Data/BIPEDv2/BIPED/',  # mean_rgb
            'yita': 0.5
        },
        'MDBD': {
            'img_height': 720,
            'img_width': 1280,
            'test_list': 'test_pair.lst',
            'train_list': 'train_pair.lst',
            'data_dir': '/opt/dataset/MDBD',  # mean_rgb
            'yita': 0.3
        },
        'CLASSIC': {
            'img_height': 512,
            'img_width': 512,
            'test_list': None,
            'train_list': None,
            'data_dir': 'data',  # mean_rgb
            'yita': 0.5
        },
        'C3VDv2': {
            'img_height': 512,
            'img_width': 512,
            'data_dir': 'Data/c3vdv2/',
            'train_list': 'train_pairs_rel.json',
            # Use test pairs for parameter tuning, val pairs for final reporting (switching val & test)
            'test_list': 'val_pairs_rel.json',
            'yita': 0.5
        },
    }
    return config[dataset_name]


class TestDataset(Dataset):
    def __init__(self,
                 data_root,
                 test_data,
                 mean_bgr,
                 img_height,
                 img_width,
                 test_list=None,
                 arg=None,
                 size_limit = None,
                 ):
        if test_data not in DATASET_NAMES:
            raise ValueError(f"Unsupported dataset: {test_data}")

        self.data_root = data_root
        self.test_data = test_data
        self.test_list = test_list
        self.args=arg
        # self.arg = arg
        # self.mean_bgr = arg.mean_pixel_values[0:3] if len(arg.mean_pixel_values) == 4 \
        #     else arg.mean_pixel_values
        self.mean_bgr = mean_bgr
        self.img_height = img_height
        self.img_width = img_width
        self.data_index = self._build_index()
        if size_limit is not None:
            self.data_index = self.data_index[:size_limit]

        print(f"mean_bgr: {self.mean_bgr}")

    def _build_index(self):
        sample_indices = []
        if self.test_data == "CLASSIC":
            # for single image testing
            images_path = os.listdir(self.data_root)
            labels_path = None
            sample_indices = [images_path, labels_path]
        else:
            # image and label paths are located in a list file

            if not self.test_list:
                raise ValueError(
                    f"Test list not provided for dataset: {self.test_data}")

            list_name = os.path.join(self.data_root, self.test_list)
            if self.test_data.upper()=='BIPED':

                with open(list_name) as f:
                    files = json.load(f)
                for pair in files:
                    tmp_img = pair[0]
                    tmp_gt = pair[1]
                    sample_indices.append(
                        (os.path.join(self.data_root, tmp_img),
                         os.path.join(self.data_root, tmp_gt),))
            else:
                # with open(list_name, 'r') as f:
                #     files = f.readlines()
                # files = [line.strip() for line in files]
                # pairs = [line.split() for line in files]

                # for pair in pairs:
                #     tmp_img = pair[0]
                #     tmp_gt = pair[1]
                #     sample_indices.append(
                #         (os.path.join(self.data_root, tmp_img),
                        #  os.path.join(self.data_root, tmp_gt),))

                with open(list_name) as f:
                    files = json.load(f)
                for pair in files:
                    tmp_img = pair['image']
                    tmp_gt = pair['gt']

                    img_path = os.path.join(self.data_root, tmp_img)
                    label_path = os.path.join(self.data_root, tmp_gt)
                    
                    # verify if the image exists
                    if not os.path.exists(img_path):
                        print("Image not found: ", img_path)
                    if not os.path.exists(label_path):
                        print("Label not found: ", label_path)

                    sample_indices.append(
                        (img_path,
                         label_path))
        #ipdb.set_trace()
        return sample_indices

    def __len__(self):
        return len(self.data_index[0]) if self.test_data.upper()=='CLASSIC' else len(self.data_index)

    def __getitem__(self, idx):
        # get data sample
        # image_path, label_path = self.data_index[idx]
        if self.data_index[1] is None:
            image_path = self.data_index[0][idx]
        else:
            image_path = self.data_index[idx][0]
        label_path = None if self.test_data == "CLASSIC" else self.data_index[idx][1]
        img_name = os.path.basename(image_path)
        file_name = os.path.splitext(img_name)[0] + ".png"

        # base dir
        if self.test_data.upper() == 'BIPED':
            img_dir = os.path.join(self.data_root, 'imgs', 'test')
            gt_dir = os.path.join(self.data_root, 'edge_maps', 'test')
        elif self.test_data.upper() == 'CLASSIC':
            img_dir = self.data_root
            gt_dir = None
        else:
            img_dir = self.data_root
            gt_dir = self.data_root

        # load data
        image = cv2.imread(os.path.join(img_dir, image_path), cv2.IMREAD_COLOR)
        if not self.test_data == "CLASSIC":
            label = cv2.imread(os.path.join(
                gt_dir, label_path), cv2.IMREAD_COLOR)
        else:
            label = None

        im_shape = [image.shape[0], image.shape[1]]
        image, label = self.transform(img=image, gt=label)

        return dict(images=image, labels=label, file_names=file_name, image_shape=im_shape)

    def transform(self, img, gt):
        # gt[gt< 51] = 0 # test without gt discrimination
        if self.test_data == "CLASSIC":
            img_height = self.img_height
            img_width = self.img_width
            print(
                f"actual size: {img.shape}, target size: {( img_height,img_width,)}")
            # img = cv2.resize(img, (self.img_width, self.img_height))
            img = cv2.resize(img, (img_width,img_height))
            gt = None

        # Make images and labels at least 512 by 512
        elif img.shape[0] < 512 or img.shape[1] < 512:
            img = cv2.resize(img, (self.args.test_img_width, self.args.test_img_height)) # 512
            gt = cv2.resize(gt, (self.args.test_img_width, self.args.test_img_height)) # 512

        # Make sure images and labels are divisible by 2^4=16
        elif img.shape[0] % 16 != 0 or img.shape[1] % 16 != 0:
            img_width = ((img.shape[1] // 16) + 1) * 16
            img_height = ((img.shape[0] // 16) + 1) * 16
            img = cv2.resize(img, (img_width, img_height))
            gt = cv2.resize(gt, (img_width, img_height))
        else:
            img_width =self.args.test_img_width
            img_height =self.args.test_img_height
            img = cv2.resize(img, (img_width, img_height))
            gt = cv2.resize(gt, (img_width, img_height))

        # if self.yita is not None:
        #     gt[gt >= self.yita] = 1
        img = np.array(img, dtype=np.float32)
        # if self.rgb:
        #     img = img[:, :, ::-1]  # RGB->BGR
        # img=cv2.resize(img, (400, 464))
        # img -= self.mean_bgr
        
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img.copy()).float()

        img_mean=(0.485, 0.456, 0.406)
        img_std=(0.229, 0.224, 0.225)
        img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
        img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]
        
        img /= 255.
        img -= img_mean
        img /= img_std
       

        if self.test_data == "CLASSIC":
            gt = np.zeros((img.shape[:2]))
            gt = torch.from_numpy(np.array([gt])).float()
        else:
            gt = np.array(gt, dtype=np.float32)
            if len(gt.shape) == 3:
                gt = gt[:, :, 0]
            gt = 255. - gt
            gt /= 255.
            gt = torch.from_numpy(np.array([gt])).float()

        return img, gt

# def check_image_size(image_path, expected_width, expected_height):

#     img = cv2.imread(image_path)  

#     if img is not None:  # Check if image loaded successfully

#         if img.shape[:2] == (expected_height, expected_width):  # Compare height and width

#             return True  

#         else:

#             print(f"Image size mismatch: Expected {expected_width}x{expected_height}, got {img.shape[1]}x{img.shape[0]}")

#     else:

#         print(f"Image not found: {image_path}")

#     return False


class TestDatasetOutput(Dataset):
    def __init__(self,
                 data_root,
                 test_data,
                 mean_bgr,
                 img_height,
                 img_width,
                 num_frames=3,  # NEW: Number of temporal frames
                 test_list=None,
                 arg=None,
                 size_limit = None,
                 ):
        if test_data not in DATASET_NAMES:
            raise ValueError(f"Unsupported dataset: {test_data}")

        self.data_root = os.path.abspath(data_root)
        self.test_data = test_data
        self.test_list = test_list
        self.args=arg
        # self.arg = arg
        # self.mean_bgr = arg.mean_pixel_values[0:3] if len(arg.mean_pixel_values) == 4 \
        #     else arg.mean_pixel_values
        self.mean_bgr = mean_bgr
        self.img_height = img_height
        self.img_width = img_width
        self.num_frames = num_frames
        self.final_results_path = "checkpoints_SAM_mem/Final_results/Validate/"
        self.data_index = self._build_index()
        if size_limit is not None:
            self.data_index = self.data_index[:size_limit]

        print(f"mean_bgr: {self.mean_bgr}")

    def _build_index(self):
        sample_indices = []
        if not self.test_list:
            raise ValueError(
                f"Test list not provided for dataset: {self.test_data}")

        file_path = os.path.join(self.data_root, self.args.test_list)
        print(f"Loading test list from: {file_path}")
        with open(file_path) as f:
            files = json.load(f)
        
        # Group frames by video sequence
        video_sequences = defaultdict(list)
        for pair in files:
            img_path = pair['image']
            # Extract video name from filename pattern: [video_name]_[4-digit-frame].png
            filename = os.path.basename(img_path)
            video_name = "_".join(filename.split('_')[:-1])  # Remove frame number
            
            # For rebuttal expt
            # Only include the video if the name ends with v2
            if video_name.endswith('v3'):
                video_sequences[video_name].append(pair)

        sequences = []
        for video_name, frames in video_sequences.items():
            # Sort frames numerically by their 4-digit suffix
            frames.sort(key=lambda x: int(os.path.basename(x['image']).split('_')[-1].split('.')[0]))
            
            # Create sliding window of frame sequences
            # Non-overlapping window, skips the late partial window
            for i in range(0, len(frames) - self.num_frames + 1, self.num_frames):
                # sequence = [
                #     (os.path.join(self.data_root, frames[j]['image']),
                #     os.path.join(self.data_root, frames[j]['gt']))
                #     for j in range(i, i+self.num_frames)
                # ]
                # "/home/durrlab/Desktop/Colonoscopy/HaustralFoldDetection/Data/haustral_fold_c3vdv2/Test/c1_ascending_p1_v2/image/c1_ascending_p1_v2_0001.png"
                # /home/durrlab/Desktop/Colonoscopy/HaustralFoldDetection/SAMDexiNed/checkpoints_SAM_mem/Final_results/Test/c1_ascending_p1_v2/c1_ascending_p1_v2_0000.png
                
                # def replacePath(img_path):
                #     filename = os.path.basename(img_path)
                #     img_path = os.path.join(self.final_results_path, video_name, filename)
                #      # Check if file exists or print error
                #     if not os.path.exists(img_path):
                #         print(f"File not found: {img_path}")
                #     return img_path
                
                # Refined
                sequence = [
                    # (replacePath(frames[j]['image']),
                    (frames[j]['image'],
                    frames[j]['gt'],
                    frames[j]['flow'])
                    for j in range(i, i+self.num_frames)
                ]
                sequences.append(sequence)

        return sequences

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        frame_paths = self.data_index[idx]
        
        # Load sequence of frames
        frames = []
        labels = []
        filenames = []
        flows = []
        for path in frame_paths:
            img = cv2.imread(os.path.join(self.data_root, path[0]), cv2.IMREAD_GRAYSCALE)
            gt = cv2.imread(os.path.join(self.data_root, path[1]), cv2.IMREAD_GRAYSCALE)
            flow = cv2.imread(os.path.join(self.data_root, path[2]), cv2.IMREAD_COLOR)
            img, gt, flow = self.transform(img, gt, flow)
            file_name = os.path.basename(path[0])
            filenames.append(file_name)
            frames.append(img)
            labels.append(gt)
            flows.append(flow)

        im_shape = [img.shape[0], img.shape[1]]
        # Create 4D tensor (T, C, H, W)
        video_tensor = torch.stack(frames, dim=0)  
        label_tensor = torch.stack(labels, dim=0)
        flow_tensor = torch.stack(flows, dim=0)
        return {'images': video_tensor, 'labels': label_tensor, 'flows':flow_tensor,'file_names': filenames, 'image_shape': im_shape} 

    def transform(self, img, gt, flow):
        # gt[gt< 51] = 0 # test without gt discrimination
        
        img_width =self.args.test_img_width
        img_height =self.args.test_img_height
        img = cv2.resize(img, (img_width, img_height))
        gt = cv2.resize(gt, (img_width, img_height))
        flow = cv2.resize(flow, (img_width, img_height))

        # if self.yita is not None:
        #     gt[gt >= self.yita] = 1
        # img = np.array(img, dtype=np.float32)
        
        flow = flow[:,:,1:]
        flow = np.array(flow, dtype=np.float32)
        # if self.rgb:
        #     img = img[:, :, ::-1]  # RGB->BGR
        # img=cv2.resize(img, (400, 464))
        # img -= self.mean_bgr
        
        # Img is output gt
        img = np.array(img, dtype=np.float32)
        if len(gt.shape) == 3:
            img = img[:, :, 0]
        # gt = 255. - gt
        img /= 255.
        img = np.clip(img, 0., 1.)
        img = torch.from_numpy(np.array([gt])).float()
       
        
        gt = np.array(gt, dtype=np.float32)
        if len(gt.shape) == 3:
            gt = gt[:, :, 0]
        # gt = 255. - gt
        gt /= 255.
        gt = np.clip(gt, 0., 1.)
        gt = torch.from_numpy(np.array([gt])).float()

        flow = flow.transpose((2, 0, 1))
        flow = torch.from_numpy(flow.copy()).float()
        flow = flow/255.0 * 40.0 - 20.0 

        return img, gt, flow



class BiDataset(Dataset):
    train_modes = ['train', 'test', ]
    dataset_types = ['rgbr', ]
    data_types = ['aug', ]

    def __init__(self,
                 data_root='../../Data/haustral_fold_c3vdv2',
                 img_height=512,
                 img_width=512,
                 mean_bgr=[103.939,116.779,123.68] ,
                 args=None,
                 dataset_type='rgbr',
                 #  is_scaling=None,
                 # Whether to crop image or otherwise resize image to match image height and width.
                 crop_img=False,
                 train_mode='train',
                 size_limit = None,
                 ):

        self.data_root = data_root
        self.train_mode = train_mode
        self.dataset_type = dataset_type
        self.data_type = 'aug'  # be aware that this might change in the future
        self.img_height = img_height
        self.img_width = img_width
        self.mean_bgr = mean_bgr
        self.crop_img = crop_img
        self.arg = args
        
        self.data_index = self._build_index()
        
        if size_limit is not None:
            self.data_index = self.data_index[:size_limit]

    def _build_index(self):
        assert self.train_mode in self.train_modes, self.train_mode
        assert self.dataset_type in self.dataset_types, self.dataset_type
        assert self.data_type in self.data_types, self.data_type

        data_root = os.path.abspath(self.data_root)
        sample_indices = []
        # ipdb.set_trace()
        if self.arg.train_data.lower()=='biped':

            images_path = os.path.join(data_root,
                                       'edges/imgs',
                                       self.train_mode,
                                       self.dataset_type,
                                       self.data_type)
            labels_path = os.path.join(data_root,
                                       'edges/edge_maps',
                                       self.train_mode,
                                       self.dataset_type,
                                       self.data_type)

            for directory_name in os.listdir(images_path):
                image_directories = os.path.join(images_path, directory_name)
                for file_name_ext in os.listdir(image_directories):
                    file_name = os.path.splitext(file_name_ext)[0]
                    sample_indices.append(
                        (os.path.join(images_path, directory_name, file_name + '.jpg'),
                         os.path.join(labels_path, directory_name, file_name + '.png'),)
                    )
        else:
            file_path = os.path.join(data_root, self.arg.train_list)
            if self.arg.train_data.lower()=='bsds':

                with open(file_path, 'r') as f:
                    files = f.readlines()
                files = [line.strip() for line in files]

                pairs = [line.split() for line in files]
                for pair in pairs:
                    tmp_img = pair[0]
                    tmp_gt = pair[1]
                    sample_indices.append(
                        (os.path.join(data_root,tmp_img),
                         os.path.join(data_root,tmp_gt),))
            else:
                with open(file_path) as f:
                    files = json.load(f)
                for pair in files:
                    tmp_img = pair['image']
                    tmp_gt = pair['gt']

                    img_path = os.path.join(data_root, tmp_img)
                    label_path = os.path.join(data_root, tmp_gt)
                    
                    # check_image_size(img_path, 512, 512)
                    # check_image_size(label_path, 512, 512)

                    sample_indices.append(
                        (img_path,
                         label_path))
        
        return sample_indices

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        # get data sample
        image_path, label_path = self.data_index[idx]

        # load data
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        image, label = self.transform(img=image, gt=label)
        return dict(images=image, labels=label)

    def transform(self, img, gt):
        gt = np.array(gt, dtype=np.float32)
        if len(gt.shape) == 3:
            gt = gt[:, :, 0]

        #NEW: Invert to comply with DexiNed GT format
        gt = 255.0 - gt

        gt /= 255. # for DexiNed input and BDCN

        img = np.array(img, dtype=np.float32)
        # img -= self.mean_bgr
        i_h, i_w,_ = img.shape
        # data = []
        # if self.scale is not None:
        #     for scl in self.scale:
        #         img_scale = cv2.resize(img, None, fx=scl, fy=scl, interpolation=cv2.INTER_LINEAR)
        #         data.append(torch.from_numpy(img_scale.transpose((2, 0, 1))).float())
        #     return data, gt
        #  400 for BIPEd and 352 for BSDS check with 384
        crop_size = self.img_height if self.img_height == self.img_width else None#448# MDBD=480 BIPED=480/400 BSDS=352

        # for BSDS 352/BRIND
        if self.arg.crop_img and i_w> crop_size and i_h>crop_size:
            i = random.randint(0, i_h - crop_size)
            j = random.randint(0, i_w - crop_size)
            img = img[i:i + crop_size , j:j + crop_size ]
            gt = gt[i:i + crop_size , j:j + crop_size ]

        # # for BIPED/MDBD
        # if np.random.random() > 0.4: #l
        #     h,w = gt.shape
        #     if i_w> 500 and i_h>500:
        #
        #         LR_img_size = crop_size #l BIPED=256, 240 200 # MDBD= 352 BSDS= 176
        #         i = random.randint(0, h - LR_img_size)
        #         j = random.randint(0, w - LR_img_size)
        #         # if img.
        #         img = img[i:i + LR_img_size , j:j + LR_img_size ]
        #         gt = gt[i:i + LR_img_size , j:j + LR_img_size ]
        #     else:
        #         LR_img_size = 352#256  # l BIPED=208-352, # MDBD= 352-480- BSDS= 176-320
        #         i = random.randint(0, h - LR_img_size)
        #         j = random.randint(0, w - LR_img_size)
        #         # if img.
        #         img = img[i:i + LR_img_size, j:j + LR_img_size]
        #         gt = gt[i:i + LR_img_size, j:j + LR_img_size]
        #         img = cv2.resize(img, dsize=(crop_size, crop_size), )
        #         gt = cv2.resize(gt, dsize=(crop_size, crop_size))

        else:
            # New addidings
            img = cv2.resize(img, dsize=(crop_size, crop_size))
            gt = cv2.resize(gt, dsize=(crop_size, crop_size))
        # BRIND
        #gt[gt > 0.1] +=0.2#0.4
        gt = np.clip(gt, 0., 1.)
        # gt[gt > 0.1] =1#0.4
        # gt = np.clip(gt, 0., 1.)
        # # for BIPED
        # gt[gt > 0.2] += 0.6# 0.5 for BIPED
        # gt = np.clip(gt, 0., 1.) # BIPED
        # # for MDBD
        # gt[gt > 0.1] +=0.7
        # gt = np.clip(gt, 0., 1.)
        # # For RCF input
        # # -----------------------------------
        # gt[gt==0]=0.
        # gt[np.logical_and(gt>0.,gt<0.5)] = 2.
        # gt[gt>=0.5]=1.
        #
        # gt = gt.astype('float32')
        # ----------------------------------

        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img.copy()).float()
        
        img_mean=(0.485, 0.456, 0.406)
        img_std=(0.229, 0.224, 0.225)
        img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
        img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]
        
        img /= 255.
        img -= img_mean
        img /= img_std
        
        gt = torch.from_numpy(np.array([gt])).float()
        return img, gt