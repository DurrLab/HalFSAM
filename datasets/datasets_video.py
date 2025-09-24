import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import json
import argparse
from collections import defaultdict
import ipdb

DATASET_NAMES = [
    'BIPED',
    'MDBD',
    'CLASSIC',
    'C3VDv2'
]
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


class VideoDataset(Dataset):
    dataset_types = ['rgbr', ]
    data_types = ['synthetic', ]

    def __init__(self,
                 data_root='Data/c3vdv2',
                 split_list = 'train_pairs_rel.json',
                 img_height=512,
                 img_width=512,
                 mean_bgr=[103.939,116.779,123.68] ,
                 num_frames=3,  # New: Number of temporal frames
                 args=None,
                 dataset_type='rgbr',
                 # Whether to crop image or otherwise resize image to match image height and width.
                 crop_img=False,
                 size_limit = None,
                 ):

        self.data_root = data_root
        self.split_list = split_list
        self.dataset_type = dataset_type
        self.data_type = 'synthetic'  # be aware that this might change in the future
        self.img_height = img_height
        self.img_width = img_width
        self.mean_bgr = mean_bgr
        self.crop_img = crop_img
        self.arg = args
        self.num_frames = num_frames
        
        self.data_index = self._build_index()
        
        if size_limit is not None:
            self.data_index = self.data_index[:size_limit]

    def _build_index(self):
        assert self.dataset_type in self.dataset_types, self.dataset_type
        assert self.data_type in self.data_types, self.data_type

        data_root = os.path.abspath(self.data_root)
        file_path = os.path.join(data_root, self.split_list)
        
        with open(file_path) as f:
            files = json.load(f)
        
        # Group frames by video sequence
        video_sequences = defaultdict(list)
        for pair in files:
            img_path = pair['image']
            # Extract video name from filename pattern: [video_name]_[4-digit-frame].png
            filename = os.path.basename(img_path)
            video_name = "_".join(filename.split('_')[:-1])  # Remove frame number
            video_sequences[video_name].append(pair)

        sequences = []
        for video_name, frames in video_sequences.items():
            # Sort frames numerically by their 4-digit suffix
            frames.sort(key=lambda x: int(os.path.basename(x['image']).split('_')[-1].split('.')[0]))
            
            # Create sliding window of frame sequences
            # 1 frame overlapping window (for temporal consistency over all frames), skips the late partial window
            for i in range(0, len(frames) - self.num_frames + 1, self.num_frames):
                sequence = [
                    (os.path.join(self.data_root, frames[j]['image']),
                    os.path.join(self.data_root, frames[j]['gt']),)
                    for j in range(i, i+self.num_frames)
                ]
                sequences.append(sequence)
        return sequences

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        frame_paths = self.data_index[idx]
        
        # Load sequence of frames
        frames, labels = [], []
        
        for path in frame_paths:
            img = cv2.imread(path[0], cv2.IMREAD_COLOR)
            gt = cv2.imread(path[1], cv2.IMREAD_GRAYSCALE)
            # Change if transform other than resizing applied
            img, gt = self.transform(img, gt)
            frames.append(img)
            labels.append(gt)
            
        # Create 4D tensor (T, C, H, W)
        video_tensor = torch.stack(frames, dim=0)  
        label_tensor = torch.stack(labels, dim=0)

        return {'images': video_tensor, 'labels': label_tensor}

    def transform(self, img, gt):
        gt = np.array(gt, dtype=np.float32)
        if len(gt.shape) == 3:
            gt = gt[:, :, 0]

        #NEW: Invert to comply with DexiNed GT format
        # gt = 255.0 - gt
        gt /= 255. # for DexiNed input and BDCN

        img = np.array(img, dtype=np.float32)
        # img -= self.mean_bgr
        i_h, i_w,_ = img.shape

        if i_h != self.img_height or i_w != self.img_width \
            or gt.shape[0] != self.img_height or gt.shape[1] != self.img_width:
            img = cv2.resize(img, dsize=(self.img_width, self.img_height))
            gt = cv2.resize(gt, dsize=(self.img_width, self.img_height))
        
        gt[gt > 0.2] += 0.6
        gt = np.clip(gt, 0., 1.) 
        
        if self.arg.Encoder == 'DexiNed':
            img -= self.mean_bgr

        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img.copy()).float()

        if self.arg.Encoder == 'SAM':
            img_mean=(0.485, 0.456, 0.406)
            img_std=(0.229, 0.224, 0.225)
            img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
            img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]
            
            img /= 255.
            img -= img_mean
            img /= img_std
        
        gt = torch.from_numpy(np.array([gt])).float()
        return img, gt


class VideoDataset_Test(Dataset):
    def __init__(self,
                 data_root,
                 rank,
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
        
        self.data_root = data_root
        self.test_data = test_data
        self.test_list = test_list
        self.args=arg
        self.mean_bgr = mean_bgr
        self.img_height = img_height
        self.img_width = img_width
        self.num_frames = num_frames
        self.rank = rank
        self.gpus = self.args.gpus
        self.data_index = self._build_index()
            
    def _build_index(self):
        data_root = os.path.abspath(self.data_root)
        data_root = os.path.join(data_root, 'images')
        print(f"Data root: {data_root}")
        video_names = ['Validate/'+vn for vn in os.listdir(data_root+'/Validate')]
        video_names += ['Test/'+vn for vn in os.listdir(data_root+'/Test')]
         
        video_names.sort()
        video_names = video_names[self.rank::self.gpus]
        
        file_names = []
        for v in video_names:
            frames = os.listdir(os.path.join(data_root, v, 'image'))
            frames.sort()
            for f in frames:
                fpath = os.path.join(data_root, v, 'image', f)
                file_names.append(fpath)
        return file_names
    
    def __len__(self):
        return len(self.data_index)
    
    def __getitem__(self, idx):
        path = self.data_index[idx % len(self.data_index)]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = self.transform(img)
        data_split = path.split('/')[-4]
        video_name = path.split('/')[-3]
        frame_name = path.split('/')[-1]
        return {'image': img,
                'video_name': os.path.join(data_split, video_name),
                'frame_name': frame_name,
                }
        
    def transform(self, img):
        # Make images and labels at least 512 by 512
        if img.shape[0] != self.args.test_img_width or  img.shape[1] != self.args.test_img_height:
            if img.shape[0] < 512 or img.shape[1] < 512:
                img = cv2.resize(img, (self.args.test_img_width, self.args.test_img_height)) # 512
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
                flow = cv2.resize(flow, (img_width, img_height))

        img = np.array(img, dtype=np.float32)
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img.copy()).float()
        if self.args.model == 'SAM':
            img_mean=(0.485, 0.456, 0.406)
            img_std=(0.229, 0.224, 0.225)
            img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
            img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]
            
            img /= 255.
            img -= img_mean
            img /= img_std

        return img
        
    
            

class TestDataset(Dataset):
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
        self.num_frames = num_frames
        self.data_index = self._build_index()
        if size_limit is not None:
            self.data_index = self.data_index[:size_limit]

        print(f"mean_bgr: {self.mean_bgr}")

    def _build_index(self):
        sample_indices = []
        if not self.test_list:
            raise ValueError(
                f"Test list not provided for dataset: {self.test_data}")

        data_root = os.path.abspath(self.data_root)
        file_path = os.path.join(data_root, self.args.test_list)
        
        with open(file_path) as f:
            files = json.load(f)
        
        # Group frames by video sequence
        video_sequences = defaultdict(list)
        for pair in files:
            img_path = pair['image']
            # Extract video name from filename pattern: [video_name]_[4-digit-frame].png
            filename = os.path.basename(img_path)
            video_name = "_".join(filename.split('_')[:-1])  # Remove frame number
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
                # Refined
                sequence = [
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
            img = cv2.imread(path[0], cv2.IMREAD_COLOR)
            gt = cv2.imread(path[1], cv2.IMREAD_GRAYSCALE)
            flow = cv2.imread(path[2], cv2.IMREAD_COLOR)
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
            flow = cv2.resize(flow, (img_width, img_height))

        # if self.yita is not None:
        #     gt[gt >= self.yita] = 1
        img = np.array(img, dtype=np.float32)
        
        flow = flow[:,:,1:]
        flow = np.array(flow, dtype=np.float32)
        # if self.rgb:
        #     img = img[:, :, ::-1]  # RGB->BGR
        # img=cv2.resize(img, (400, 464))
        # img -= self.mean_bgr
        
        if self.args.model == 'DexiNed':
            img -= self.mean_bgr

        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img.copy()).float()
        
        if self.args.model == 'SAM':
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
            # gt = 255. - gt
            gt /= 255.
            gt = np.clip(gt, 0., 1.)
            gt = torch.from_numpy(np.array([gt])).float()

        flow = flow.transpose((2, 0, 1))
        flow = torch.from_numpy(flow.copy()).float()
        flow = flow/255.0 * 40.0 - 20.0 

        return img, gt, flow