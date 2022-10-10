from pathlib import Path
from dotmap import DotMap
from torch.utils.data import Dataset

import os
import cv2
import torch
import numpy as np
import pandas as pd
import torchvision.transforms as tf


class VREyeTrackingERP(Dataset):
    def __init__(self, datasetconfig, phase='val'):
        dataset_config = DotMap(datasetconfig)

        self.w = dataset_config.w
        self.h = dataset_config.h
        self.img_path = dataset_config.path_frames
        self.depth_path = dataset_config.path_depth
        self.gts_path = dataset_config.path_gts
        self.gtf_path = dataset_config.path_gtf
        path_split = dataset_config.path_split

        self.pano_shape = (self.h, self.w)

        self.transform = tf.Compose([
            tf.ToTensor(),
            tf.Resize(self.pano_shape),
            tf.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
        ])

        if dataset_config["koc"]:
            fold_list = Path(path_split) / Path(phase + '_frames_koc.csv')
        else:
            fold_list = Path(path_split) / Path(phase + '_frames.csv')

        list_video = pd.read_csv(fold_list)
        self.list_video = [list(row) for row in list_video.values]

    def __len__(self):
        return len(self.list_video)

    def __getitem__(self, idx):
        """
        returns: erp_img, erp_gt, filename
        """
        file_name = str(self.list_video[idx][0]).zfill(3)
        start_idx = self.list_video[idx][1]
        erp_img = cv2.imread(os.path.join(self.img_path, file_name, "%04d.png" % start_idx))
        erp_img = cv2.resize(erp_img, (self.w, self.h))
        erp_img = np.float32(erp_img)
        erp_img = torch.FloatTensor(erp_img)
        erp_img = erp_img.permute(2, 0, 1)

        # saliency map
        erp_gts = cv2.imread(os.path.join(self.gts_path, file_name, '%05d.png' % start_idx), 0)
        erp_gts = (erp_gts - np.min(erp_gts)) / (np.max(erp_gts) - np.min(erp_gts) + np.finfo(np.float).eps)
        erp_gts = torch.from_numpy(erp_gts).contiguous().to(dtype=torch.float32)

        # fixations
        erp_gtf = cv2.imread(os.path.join(self.gtf_path, file_name, '%05d.png' % start_idx), 0)
        erp_gtf = torch.from_numpy(erp_gtf).contiguous().to(dtype=torch.float32)
        erp_gt = torch.stack([erp_gts, erp_gtf])
        return erp_img, erp_gt, f"{file_name}_{str(start_idx).zfill(5)}"
